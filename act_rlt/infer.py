"""Interactive ACT inference: sampled reset poses and repeatable timed episodes."""

import argparse
import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np


ACTION_KEYS = ("dx", "dy", "dz", "drx", "dry", "drz")
DEFAULT_RESET_SPEED_M_S = 0.02
SAMPLE_TRANSITION_LIFT_M = 0.01
DEFAULT_CONTROL_HZ = 30.0
DEFAULT_CAMERA_SHAPES = {"wrist": (480, 640, 3), "front": (480, 640, 3)}
MAX_INFERENCE_S = 0.5
RESIDUAL_LIMIT_M = 0.0003
Z_STALL_WINDOW_S = 0.25
Z_STALL_MIN_COMMAND_M = 0.0002
Z_STALL_MAX_PROGRESS_M = 0.00005
SAMPLE_LOWER_OFFSET_M = np.array([-0.02, -0.02, -0.005], dtype=float)
SAMPLE_UPPER_OFFSET_M = np.array([0.02, 0.02, 0.005], dtype=float)
WORKSPACE_LOWER_OFFSET_M = np.array([-0.10, -0.10, -0.03], dtype=float)
WORKSPACE_UPPER_OFFSET_M = np.array([0.10, 0.10, 0.01], dtype=float)


@dataclass(frozen=True)
class PreparedAction:
    action: dict[str, float]
    values: np.ndarray
    translation_triggered: bool
    rotation_triggered: bool


class ResidualMotionAccumulator:
    """Track requested-but-not-measured TCP translation within a small lead."""

    def __init__(self):
        self.residual_xyz = np.zeros(3, dtype=float)
        self.prev_measured_xyz = None
        self.negative_z_samples = deque()
        self.z_stalled = False

    def step(self, measured_pose, action, now):
        measured = np.asarray(measured_pose, dtype=float)
        delta = np.asarray([action[key] for key in ACTION_KEYS[:3]], dtype=float)
        if measured.shape != (6,) or not np.isfinite(measured).all():
            raise ValueError(f"invalid measured TCP pose for residual control: {measured}")
        if not np.isfinite(delta).all() or not np.isfinite(now):
            raise ValueError("residual control requires finite action and time")
        if self.prev_measured_xyz is not None:
            self.residual_xyz -= measured[:3] - self.prev_measured_xyz
        self.residual_xyz += delta
        self.prev_measured_xyz = measured[:3].copy()

        newly_stalled = False
        if delta[2] > 0:
            # Retraction remains allowed, but the negative-Z latch lasts this episode.
            self.negative_z_samples.clear()
        elif delta[2] < 0 and not self.z_stalled:
            self.negative_z_samples.append((float(now), float(measured[2]), float(delta[2])))
            # Keep the oldest sample spanning at least the full window, even
            # when the tick period does not divide 0.25 s exactly.
            while (len(self.negative_z_samples) > 1
                   and now - self.negative_z_samples[1][0] >= Z_STALL_WINDOW_S):
                self.negative_z_samples.popleft()
            if self.negative_z_samples:
                window_s = now - self.negative_z_samples[0][0]
                commanded_down = -sum(item[2] for item in self.negative_z_samples)
                measured_down = self.negative_z_samples[0][1] - measured[2]
                if (window_s >= Z_STALL_WINDOW_S
                        and commanded_down >= Z_STALL_MIN_COMMAND_M
                        and measured_down < Z_STALL_MAX_PROGRESS_M):
                    self.z_stalled = True
                    newly_stalled = True
                    self.negative_z_samples.clear()
        else:
            self.negative_z_samples.clear()

        self.residual_xyz = np.clip(
            self.residual_xyz, -RESIDUAL_LIMIT_M, RESIDUAL_LIMIT_M
        )
        if self.z_stalled:
            self.residual_xyz[2] = max(0.0, self.residual_xyz[2])
        return self.residual_xyz.copy(), newly_stalled


class TemporalActionEnsembler:
    """Blend overlapping ACT chunks that predict the same control timestep.

    The paper defines ``w_i = exp(-m * i)`` with ``i=0`` assigned to the
    oldest applicable prediction. We normalize those weights in log space:
    ``log_w_i = -m*i`` and ``w_i = exp(log_w_i - logsumexp(log_w))``.
    """

    def __init__(self, chunk_size, coefficient):
        if chunk_size <= 0:
            raise ValueError("temporal ensemble chunk_size must be positive")
        if not np.isfinite(coefficient) or coefficient < 0:
            raise ValueError("temporal ensemble coefficient must be finite and non-negative")
        self.chunk_size = int(chunk_size)
        self.coefficient = float(coefficient)
        self._chunks = []
        self._lock = threading.Lock()

    def add_chunk(self, start_step, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (self.chunk_size, 7) or not np.isfinite(values).all():
            raise ValueError(
                f"expected temporal ensemble chunk ({self.chunk_size}, 7), got {values.shape}"
            )
        with self._lock:
            self._chunks.append((int(start_step), values.copy()))
            self._chunks.sort(key=lambda item: item[0])

    def action_for(self, step, max_translation, max_rotation):
        with self._lock:
            # Chunks ending before this step can never contribute again.
            self._chunks = [
                item for item in self._chunks
                if item[0] + self.chunk_size > step
            ]
            votes = [
                values[step - start]
                for start, values in self._chunks
                if start <= step < start + self.chunk_size
            ]
        if not votes:
            return None, 0

        log_weights = -self.coefficient * np.arange(len(votes), dtype=float)
        log_normalizer = np.logaddexp.reduce(log_weights)
        weights = np.exp(log_weights - log_normalizer)
        ensembled = np.sum(np.stack(votes) * weights[:, None], axis=0)
        action, translation_triggered, rotation_triggered = bounded_action_with_triggers(
            ensembled, max_translation, max_rotation
        )
        return PreparedAction(
            action=action,
            values=ensembled,
            translation_triggered=translation_triggered,
            rotation_triggered=rotation_triggered,
        ), len(votes)


def checkpoint_camera_shapes(config):
    """Validate the ACT feature contract and return required HWC image shapes."""
    features = config.input_features
    if set(features) - {"observation.state", "observation.images.wrist", "observation.images.front"}:
        raise ValueError(f"Unsupported ACT input features: {sorted(features)}")
    if "observation.state" not in features or tuple(features["observation.state"].shape) != (7,):
        raise ValueError("Checkpoint must use a 7D joint state")
    if "action" not in config.output_features or tuple(config.output_features["action"].shape) != (7,):
        raise ValueError("Checkpoint must output a 7D action")

    cameras = {}
    for name in ("wrist", "front"):
        key = f"observation.images.{name}"
        if key not in features:
            continue
        shape = tuple(features[key].shape)
        if len(shape) != 3 or shape[0] != 3 or any(dim <= 0 for dim in shape):
            raise ValueError(f"{key}: expected a positive 3-channel CHW shape, got {shape}")
        cameras[name] = (shape[1], shape[2], 3)
    if not cameras:
        raise ValueError("Checkpoint must use at least one wrist/front RGB camera")
    # FrankaRobot currently preserves wrist pixels at native size, but always
    # converts the front stream to VGA. Reject unsupported front shapes early.
    if "front" in cameras and cameras["front"] != (480, 640, 3):
        raise ValueError("FrankaRobot currently supports only 640x480 front images")
    return cameras


def robot_camera_kwargs(cameras, args):
    """Configure only the camera streams required by the checkpoint."""
    wrist_height, wrist_width, _ = cameras.get("wrist", (480, 640, 3))
    return {
        "camera_serial": args.wrist_camera,
        "camera_width": wrist_width,
        "camera_height": wrist_height,
        "enable_wrist_camera": "wrist" in cameras,
        "front_camera_serial": args.front_camera,
        "front_camera_width": 640,
        "front_camera_height": 480,
        "enable_front_camera": "front" in cameras,
    }


def observation_frame(observation, camera_shapes=None):
    import torch

    if camera_shapes is None:
        camera_shapes = DEFAULT_CAMERA_SHAPES
    state = np.array([observation[f"joint_{i}"] for i in range(7)], dtype=np.float32)
    if not np.isfinite(state).all():
        raise ValueError("Non-finite joint state")
    frame = {"observation.state": torch.from_numpy(state)}
    for camera, shape in camera_shapes.items():
        pixels = np.asarray(observation[camera])
        if pixels.shape != shape or pixels.dtype != np.uint8:
            raise ValueError(f"{camera}: expected uint8 RGB {shape}")
        frame[f"observation.images.{camera}"] = (
            torch.from_numpy(pixels.copy()).permute(2, 0, 1).float().div(255)
        )
    return frame


def resolve_checkpoint(path):
    """Accept pretrained_model itself, a checkpoint directory, or a run directory."""
    path = Path(path).expanduser().resolve()
    candidates = (path, path / "pretrained_model", path / "checkpoints/last/pretrained_model")
    for candidate in candidates:
        required = (
            "config.json", "model.safetensors", "policy_preprocessor.json",
            "policy_postprocessor.json",
        )
        if all((candidate / name).is_file() for name in required):
            return candidate
    raise FileNotFoundError(
        f"No complete local pretrained_model found under {path}. "
        "Pass the downloaded run directory or checkpoints/last/pretrained_model."
    )


def bounded_action_with_triggers(values, max_translation, max_rotation):
    values = np.asarray(values, dtype=float).reshape(-1).copy()
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("Expected a finite 7D action")
    triggered = []
    for start, limit in ((0, max_translation), (3, max_rotation)):
        norm = np.linalg.norm(values[start:start + 3])
        triggered.append(bool(norm > limit))
        if norm > limit:
            values[start:start + 3] *= limit / norm
    # The gripper remains held; do not send repetitive gripper commands.
    return dict(zip(ACTION_KEYS, values[:6], strict=True)), triggered[0], triggered[1]


def bounded_action(values, max_translation, max_rotation):
    action, _, _ = bounded_action_with_triggers(values, max_translation, max_rotation)
    return action


def sample_workspace_pose(workspace_min, workspace_max, orientation, xy_padding, rng):
    """Sample XYZ inside the workspace, with padding on X/Y and fixed rotvec orientation."""
    lower = np.asarray(workspace_min, dtype=float)
    upper = np.asarray(workspace_max, dtype=float)
    orientation = np.asarray(orientation, dtype=float)
    if lower.shape != (3,) or upper.shape != (3,) or orientation.shape != (3,):
        raise ValueError("workspace bounds and reset orientation must be 3D")
    if not np.isfinite(np.concatenate([lower, upper, orientation])).all():
        raise ValueError("workspace bounds and reset orientation must be finite")
    if np.any(lower >= upper):
        raise ValueError("workspace min must be smaller than workspace max")
    if not np.isfinite(xy_padding) or xy_padding < 0:
        raise ValueError("reset XY padding must be finite and non-negative")
    sample_lower = lower.copy()
    sample_upper = upper.copy()
    sample_lower[:2] += xy_padding
    sample_upper[:2] -= xy_padding
    if np.any(sample_lower[:2] >= sample_upper[:2]):
        raise ValueError(
            f"workspace X/Y spans must each exceed {2 * xy_padding:.3f} m "
            f"for {xy_padding:.3f} m reset padding"
        )
    return np.concatenate([rng.uniform(sample_lower, sample_upper), orientation])


def centered_bounds(measured_pose):
    """Return sample and workspace XYZ bounds around a measured base-frame TCP pose."""
    pose = np.asarray(measured_pose, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError(f"invalid measured TCP pose: {pose}")
    center = pose[:3]
    return (
        center + SAMPLE_LOWER_OFFSET_M,
        center + SAMPLE_UPPER_OFFSET_M,
        center + WORKSPACE_LOWER_OFFSET_M,
        center + WORKSPACE_UPPER_OFFSET_M,
    )


def confirm_centered_bounds(robot):
    """Capture the TCP pose at Enter, then enable its centered workspace."""
    while True:
        pose = np.asarray(robot.robot.get_tool_pose(), dtype=float)
        sample_min, sample_max, workspace_min, workspace_max = centered_bounds(pose)
        print(f"Current TCP center XYZ (base frame, m): {format_pose(pose[:3])}", flush=True)
        print(f"Current TCP orientation (rotation vector, rad): {format_pose(pose[3:])}", flush=True)
        print(f"Sample XYZ min/max (m): {format_pose(sample_min)} / {format_pose(sample_max)}", flush=True)
        print(f"Workspace XYZ min/max (m): {format_pose(workspace_min)} / {format_pose(workspace_max)}", flush=True)
        decision = prompt_choice(
            "[Enter]=confirm current center, r=refresh measured pose, q=quit: ",
            {"": "confirm", "r": "refresh", "q": "quit"},
        )
        if decision == "quit":
            return None
        if decision == "confirm":
            # The arm may have been repositioned while the prompt was open.
            pose = np.asarray(robot.robot.get_tool_pose(), dtype=float)
            sample_min, sample_max, workspace_min, workspace_max = centered_bounds(pose)
            print(f"Confirmed TCP pose (m, rad): {format_pose(pose)}", flush=True)
            print(f"Confirmed sample XYZ min/max (m): {format_pose(sample_min)} / {format_pose(sample_max)}", flush=True)
            print(f"Confirmed workspace XYZ min/max (m): {format_pose(workspace_min)} / {format_pose(workspace_max)}", flush=True)
            robot.config.workspace_min_xyz = tuple(workspace_min)
            robot.config.workspace_max_xyz = tuple(workspace_max)
            return sample_min, sample_max, pose[3:].copy()


def format_pose(pose):
    return " ".join(f"{float(value):+.6f}" for value in pose)


def move_to_pose(robot, target, speed, *, label):
    target = np.asarray(target, dtype=float)
    print(f"{label}: {format_pose(target)}", flush=True)
    measured = np.asarray(
        robot.robot.move_tool(target.tolist(), speed=float(speed)), dtype=float
    )
    if measured.shape != (6,) or not np.isfinite(measured).all():
        raise RuntimeError(f"invalid measured pose after {label}: {measured}")
    robot.resync_command_pose()
    print(f"Reached {label}: {format_pose(measured)}", flush=True)
    return measured


def move_to_next_sample(robot, target, speed):
    """Lift the measured TCP 1 cm in base +Z before moving to a new sample."""
    current = np.asarray(robot.robot.get_tool_pose(), dtype=float)
    if current.shape != (6,) or not np.isfinite(current).all():
        raise RuntimeError(f"invalid measured pose before sample transition: {current}")
    lift_target = current.copy()
    lift_target[2] += SAMPLE_TRANSITION_LIFT_M
    move_to_pose(robot, lift_target, speed, label="lift before next sampled inference start")
    return move_to_pose(robot, target, speed, label="sampled inference start")


def prompt_choice(prompt, choices):
    while True:
        choice = input(prompt).strip().lower()
        if choice in choices:
            return choices[choice]
        print("Invalid choice; please use one of the keys shown.", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_CONTROL_HZ,
        help="servo command rate in Hz (default: 30; use 15 for a 15 Hz model)",
    )
    parser.add_argument("--n-action-steps", type=int)
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=0.01,
        metavar="M",
        help="ACT paper exponential temporal-ensemble coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--no-temporal-ensemble",
        action="store_const",
        const=None,
        dest="temporal_ensemble_coeff",
        help="disable temporal ensembling and execute n-action-steps open loop",
    )
    parser.add_argument(
        "--residual-control",
        action="store_true",
        help=(
            "accumulate unexecuted XYZ motion with a +/-0.3 mm lead and Z-stall "
            "guard (default: disabled; use legacy measured-TCP anchoring)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--log-dir", type=Path, default=Path("logs/act_rlt_infer"),
        help="directory for per-episode, per-step JSONL traces (default: logs/act_rlt_infer)",
    )
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--wrist-camera", default="349622072679")
    parser.add_argument("--front-camera", default="233522075778")
    parser.add_argument("--max-step-m", type=float, default=0.002)
    parser.add_argument("--max-step-rad", type=float, default=0.02)
    parser.add_argument("--reset-speed", type=float, default=DEFAULT_RESET_SPEED_M_S)
    parser.add_argument("--sampling-seed", type=int)
    return parser


def validate_args(parser, args):
    if not all(np.isfinite(x) and x > 0 for x in (args.duration, args.max_step_m, args.max_step_rad)):
        parser.error("duration and step limits must be finite and positive")
    if not np.isfinite(args.fps) or args.fps <= 0 or args.fps > 30:
        parser.error("fps must be finite and in (0, 30], matching the camera stream")
    if not np.isfinite(args.reset_speed) or args.reset_speed <= 0:
        parser.error("reset-speed must be finite and positive")
    if (args.temporal_ensemble_coeff is not None
            and (not np.isfinite(args.temporal_ensemble_coeff)
                 or args.temporal_ensemble_coeff < 0)):
        parser.error("temporal-ensemble-coeff must be finite and non-negative")


def _predict_action_chunk(policy, pre, post, observation, args, camera_shapes=None) -> list[PreparedAction]:
    """Run preprocessing once, then drain one ACT action chunk.

    LeRobot's ``select_action`` performs a model forward only when its internal
    action queue is empty. Reusing the same batch for the remaining calls avoids
    recapturing, converting, and uploading images that the policy will ignore.
    """
    batch = pre(observation_frame(observation, camera_shapes))
    n_action_steps = int(policy.config.n_action_steps)
    prepared = []
    for _ in range(n_action_steps):
        values = post(policy.select_action(batch)).detach().cpu().numpy().reshape(-1)
        action, translation_triggered, rotation_triggered = bounded_action_with_triggers(
            values, args.max_step_m, args.max_step_rad
        )
        prepared.append(PreparedAction(
            action=action,
            values=values.copy(),
            translation_triggered=translation_triggered,
            rotation_triggered=rotation_triggered,
        ))
    return prepared


def _predict_full_action_chunk(policy, pre, post, observation, camera_shapes=None) -> np.ndarray:
    """Predict and unnormalize the full ACT chunk for temporal ensembling."""
    batch = pre(observation_frame(observation, camera_shapes))
    values = post(policy.predict_action_chunk(batch)).detach().cpu().numpy()
    values = np.asarray(values, dtype=float).reshape(-1, 7)
    expected = int(policy.config.chunk_size)
    if values.shape != (expected, 7) or not np.isfinite(values).all():
        raise ValueError(f"expected ACT chunk ({expected}, 7), got {values.shape}")
    return values


def run_inference_episode(robot, policy, pre, post, args, torch, camera_shapes=None,
                          *, episode_index=None):
    """Run camera, ACT, and servo as three independently scheduled workers."""
    trace_file = None
    trace_queue = None
    log_dir = getattr(args, "log_dir", None)
    if log_dir is not None:
        log_dir = Path(log_dir).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        trace_path = log_dir / f"episode_{stamp}_{uuid4().hex[:8]}.jsonl"
        trace_file = trace_path.open("x", encoding="utf-8")
        trace_queue = queue.SimpleQueue()
        print(f"Full inference trace: {trace_path.resolve()}", flush=True)

    def trace(event, **fields):
        if trace_queue is not None:
            trace_queue.put({
                "event": event,
                "wall_time_ns": time.time_ns(),
                "monotonic_s": time.monotonic(),
                **fields,
            })

    def drain_trace():
        if trace_file is None:
            return
        while True:
            try:
                row = trace_queue.get_nowait()
            except queue.Empty:
                break
            trace_file.write(json.dumps(row, allow_nan=False) + "\n")
        trace_file.flush()

    policy.reset()
    n_action_steps = int(policy.config.n_action_steps)
    temporal_coefficient = args.temporal_ensemble_coeff
    temporal_ensembler = None
    if temporal_coefficient is not None:
        temporal_ensembler = TemporalActionEnsembler(
            policy.config.chunk_size, temporal_coefficient
        )
    residual_control = bool(getattr(args, "residual_control", False))
    residual_controller = ResidualMotionAccumulator() if residual_control else None
    # Start producing the next chunk with half of the current chunk still buffered.
    # This gives inference time to finish without keeping a full extra chunk stale.
    low_watermark = max(0, n_action_steps // 2)
    action_queue = queue.Queue(maxsize=max(2, n_action_steps + low_watermark))
    log_queue = queue.SimpleQueue()

    stop_event = threading.Event()
    first_actions_ready = threading.Event()
    refill_event = threading.Event()
    temporal_request_event = threading.Event()
    servo_finished = threading.Event()
    observation_ready = threading.Condition()
    latest_observation = {"seq": 0, "value": None, "captured_at": None}
    latest_temporal_request = {"start_step": 0}
    failures = []
    failure_lock = threading.Lock()
    stats = {
        "ticks": 0,
        "translation": 0,
        "rotation": 0,
        "either": 0,
        "both": 0,
        "deadline_misses": 0,
        "max_lateness_s": 0.0,
        "z_stall_events": 0,
    }
    trace(
        "episode_start", episode_index=episode_index,
        checkpoint=str(args.checkpoint) if hasattr(args, "checkpoint") else None,
        fps=float(args.fps), duration_s=float(args.duration),
        temporal_ensemble_coeff=temporal_coefficient,
        n_action_steps=n_action_steps, chunk_size=int(policy.config.chunk_size),
        residual_control=residual_control,
        residual_limit_m=RESIDUAL_LIMIT_M if residual_control else None,
        z_stall_window_s=Z_STALL_WINDOW_S if residual_control else None,
        z_stall_min_command_m=Z_STALL_MIN_COMMAND_M if residual_control else None,
        z_stall_max_progress_m=Z_STALL_MAX_PROGRESS_M if residual_control else None,
        dry_run=bool(args.dry_run),
        workspace_min_xyz=list(robot.config.workspace_min_xyz)
        if getattr(getattr(robot, "config", None), "workspace_min_xyz", None) is not None else None,
        workspace_max_xyz=list(robot.config.workspace_max_xyz)
        if getattr(getattr(robot, "config", None), "workspace_max_xyz", None) is not None else None,
    )

    def fail(exc):
        with failure_lock:
            if not failures:
                failures.append(exc)
        stop_event.set()
        first_actions_ready.set()
        refill_event.set()
        temporal_request_event.set()
        with observation_ready:
            observation_ready.notify_all()

    def camera_worker():
        try:
            while not stop_event.is_set():
                observation = robot.get_observation()
                with observation_ready:
                    latest_observation["seq"] += 1
                    latest_observation["value"] = observation
                    latest_observation["captured_at"] = time.monotonic()
                    observation_ready.notify_all()
        except BaseException as exc:
            fail(exc)

    def inference_worker():
        last_observation_seq = 0
        try:
            with torch.inference_mode():
                while not stop_event.is_set():
                    if temporal_ensembler is None:
                        refill_event.wait()
                        refill_event.clear()
                    else:
                        temporal_request_event.wait()
                        temporal_request_event.clear()
                    if stop_event.is_set():
                        break
                    if (temporal_ensembler is None and not action_queue.empty()
                            and action_queue.qsize() > low_watermark):
                        continue

                    with observation_ready:
                        observation_ready.wait_for(
                            lambda: stop_event.is_set()
                            or latest_observation["seq"] > last_observation_seq
                        )
                        if stop_event.is_set():
                            break
                        observation = latest_observation["value"]
                        last_observation_seq = latest_observation["seq"]
                        observation_captured_at = latest_observation["captured_at"]

                    inference_start = time.monotonic()
                    if temporal_ensembler is None:
                        chunk = _predict_action_chunk(
                            policy, pre, post, observation, args, camera_shapes
                        )
                    else:
                        start_step = latest_temporal_request["start_step"]
                        chunk = _predict_full_action_chunk(
                            policy, pre, post, observation, camera_shapes
                        )
                    inference_elapsed = time.monotonic() - inference_start
                    trace(
                        "model_chunk", observation_seq=last_observation_seq,
                        observation_monotonic_s=observation_captured_at,
                        start_step=start_step if temporal_ensembler is not None else None,
                        inference_s=inference_elapsed,
                        actions=[item.values.tolist() for item in chunk]
                        if temporal_ensembler is None else chunk.tolist(),
                    )
                    if inference_elapsed > MAX_INFERENCE_S:
                        message = f"Preprocessing/inference took {inference_elapsed:.2f}s"
                        if not args.dry_run:
                            raise RuntimeError(message + "; stopping")
                        log_queue.put("WARNING: " + message + "; dry-run continues")

                    if temporal_ensembler is None:
                        for prepared in chunk:
                            while not stop_event.is_set():
                                try:
                                    action_queue.put(prepared, timeout=0.05)
                                    break
                                except queue.Full:
                                    continue
                    else:
                        temporal_ensembler.add_chunk(start_step, chunk)
                    if len(chunk) and not stop_event.is_set():
                        first_actions_ready.set()
        except BaseException as exc:
            fail(exc)

    def servo_worker():
        try:
            first_actions_ready.wait()
            if stop_event.is_set():
                return

            period = 1.0 / args.fps
            start = time.monotonic()
            deadline = start
            end = start + args.duration
            control_step = 0
            while deadline < end and not stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining > 0 and stop_event.wait(remaining):
                    break

                now = time.monotonic()
                lateness = max(0.0, now - deadline)
                if lateness > 0.001:
                    stats["deadline_misses"] += 1
                    stats["max_lateness_s"] = max(stats["max_lateness_s"], lateness)

                if temporal_ensembler is None:
                    try:
                        prepared = action_queue.get_nowait()
                    except queue.Empty:
                        refill_event.set()
                        message = (
                            "ACT action queue underrun at the servo deadline; "
                            "stopping instead of repeating a relative action"
                        )
                        if not args.dry_run:
                            raise RuntimeError(message)
                        log_queue.put("WARNING: " + message + "; dry-run skips this tick")
                        deadline += period
                        control_step += 1
                        continue
                    if action_queue.qsize() <= low_watermark:
                        refill_event.set()
                    queue_status = f"queue={action_queue.qsize()}"
                else:
                    prepared, vote_count = temporal_ensembler.action_for(
                        control_step, args.max_step_m, args.max_step_rad
                    )
                    if prepared is None:
                        raise RuntimeError(
                            "temporal ensemble has no prediction for the servo deadline; stopping"
                        )
                    queue_status = f"ensemble_votes={vote_count}"

                stats["translation"] += int(prepared.translation_triggered)
                stats["rotation"] += int(prepared.rotation_triggered)
                stats["either"] += int(
                    prepared.translation_triggered or prepared.rotation_triggered
                )
                stats["both"] += int(
                    prepared.translation_triggered and prepared.rotation_triggered
                )
                step_index = stats["ticks"]
                stats["ticks"] += 1

                measured_pose = None
                measured_joint = None
                external_wrench = None
                target_pose = None
                residual_xyz = None
                newly_stalled = False
                if trace_queue is not None or (not args.dry_run and residual_control):
                    # ControlClient reads this from its state cache, not a control RPC.
                    measured_pose = robot.robot.get_tool_pose()
                if trace_queue is not None:
                    measured_joint = robot.robot.get_joint_angles()
                    external_wrench = robot.robot.get_tool_force_raw()
                    state_sample_monotonic_s = time.monotonic()
                send_start = time.monotonic()
                if not args.dry_run:
                    if residual_control:
                        residual_xyz, newly_stalled = residual_controller.step(
                            measured_pose, prepared.action, now
                        )
                        robot.send_action(
                            prepared.action, position_residual_xyz=residual_xyz
                        )
                        if newly_stalled:
                            stats["z_stall_events"] += 1
                            log_queue.put(
                                "WARNING: Z stall detected; cleared Z residual and "
                                "suppressed further negative Z commands for this episode; "
                                "upward retreat remains allowed"
                            )
                    else:
                        # Original behavior: each relative action is anchored to
                        # the latest measured TCP pose; no residual is retained.
                        robot.resync_command_pose()
                        robot.send_action(prepared.action)
                    if trace_queue is not None:
                        target_pose = np.asarray(robot._cmd_pose, dtype=float).tolist()
                send_elapsed = time.monotonic() - send_start
                trace(
                    "control_step", step=control_step, elapsed_s=now - start,
                    deadline_lateness_ms=1000 * lateness,
                    action_unbounded=prepared.values.tolist(),
                    action_sent=[float(prepared.action[key]) for key in ACTION_KEYS],
                    gripper_prediction_m=float(prepared.values[6]),
                    translation_limited=prepared.translation_triggered,
                    rotation_limited=prepared.rotation_triggered,
                    ensemble_votes=vote_count if temporal_ensembler is not None else None,
                    queue_remaining=action_queue.qsize() if temporal_ensembler is None else None,
                    state_sample_monotonic_s=state_sample_monotonic_s
                    if trace_queue is not None else None,
                    measured_tcp_before=measured_pose,
                    measured_joint_before=measured_joint,
                    external_wrench_base_before=external_wrench,
                    residual_xyz=residual_xyz.tolist() if residual_xyz is not None else None,
                    z_stall_latched=residual_controller.z_stalled
                    if residual_control and not args.dry_run else None,
                    z_stall_triggered=newly_stalled,
                    target_tcp_sent=target_pose,
                    send_action_s=send_elapsed,
                )
                if step_index % max(1, round(args.fps)) == 0:
                    log_queue.put(
                        f"t={time.monotonic()-start:.1f}s "
                        f"delta={list(prepared.action.values())} "
                        f"gripper_prediction={prepared.values[6]:.4f}m (held) "
                        f"{queue_status}"
                    )
                if temporal_ensembler is not None:
                    latest_temporal_request["start_step"] = control_step + 1
                    temporal_request_event.set()
                deadline += period
                control_step += 1
        except BaseException as exc:
            fail(exc)
        finally:
            stop_event.set()
            refill_event.set()
            temporal_request_event.set()
            with observation_ready:
                observation_ready.notify_all()
            servo_finished.set()

    threads = [
        threading.Thread(target=camera_worker, daemon=True, name="act-camera"),
        threading.Thread(target=inference_worker, daemon=True, name="act-inference"),
        threading.Thread(target=servo_worker, daemon=True, name="act-servo"),
    ]
    if temporal_ensembler is None:
        refill_event.set()
    else:
        temporal_request_event.set()
    for thread in threads:
        thread.start()

    stop_error = None
    try:
        while not servo_finished.wait(timeout=0.1):
            while True:
                try:
                    print(log_queue.get_nowait(), flush=True)
                except queue.Empty:
                    break
    finally:
        stop_event.set()
        first_actions_ready.set()
        refill_event.set()
        temporal_request_event.set()
        with observation_ready:
            observation_ready.notify_all()

        # Stop the physical stream as soon as the fixed-rate publisher exits;
        # camera shutdown can take up to its bounded frame timeout.
        if robot.is_connected and not args.dry_run:
            try:
                if not robot.robot.stop_servo():
                    stop_error = RuntimeError(
                        "control server did not acknowledge stop_servo after inference"
                    )
                robot.resync_command_pose()
            except BaseException as exc:
                stop_error = exc

        for thread in threads:
            thread.join()
        drain_trace()
        while True:
            try:
                print(log_queue.get_nowait(), flush=True)
            except queue.Empty:
                break

        denominator = max(stats["ticks"], 1)
        print(
            "Action-limit report: "
            f"steps={stats['ticks']}, "
            f"max_step_m_triggered={stats['translation']} "
            f"({100 * stats['translation'] / denominator:.1f}%), "
            f"max_step_rad_triggered={stats['rotation']} "
            f"({100 * stats['rotation'] / denominator:.1f}%), "
            f"either={stats['either']} ({100 * stats['either'] / denominator:.1f}%), "
            f"both={stats['both']} ({100 * stats['both'] / denominator:.1f}%), "
            f"deadline_misses={stats['deadline_misses']}, "
            f"max_lateness_ms={1000 * stats['max_lateness_s']:.2f}, "
            f"z_stall_events={stats['z_stall_events']}",
            flush=True,
        )
        trace(
            "episode_end", stats=stats.copy(),
            error=str(failures[0]) if failures else str(stop_error) if stop_error else None,
            measured_tcp_after_stop=robot.robot.get_tool_pose()
            if trace_queue is not None else None,
        )
        drain_trace()
        if trace_file is not None:
            trace_file.close()
        policy.reset()

    if failures:
        raise failures[0]
    if stop_error is not None:
        raise stop_error


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from evo_rlt.adapters.lerobot.franka_robot import FrankaRobot, FrankaRobotConfig

    checkpoint = str(resolve_checkpoint(args.checkpoint))
    print(f"Loading checkpoint: {checkpoint}", flush=True)
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if config.type != "act":
        raise ValueError(f"Expected an ACT checkpoint, got {config.type!r}")
    config.device = args.device
    camera_shapes = checkpoint_camera_shapes(config)
    if args.n_action_steps is not None:
        config.n_action_steps = args.n_action_steps
    if not 1 <= config.n_action_steps <= config.chunk_size:
        parser.error("n-action-steps must be between 1 and checkpoint chunk_size")
    # The three-thread runtime performs temporal ensembling over complete chunks
    # itself, so it does not use LeRobot's select_action(n_action_steps=1) wrapper.
    config.temporal_ensemble_coeff = args.temporal_ensemble_coeff
    policy = ACTPolicy.from_pretrained(
        checkpoint, config=config, local_files_only=True
    ).to(args.device).eval()
    pre, post = make_pre_post_processors(
        config, pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    robot = FrankaRobot(FrankaRobotConfig(
        robot_ip=args.robot_ip,
        **robot_camera_kwargs(camera_shapes, args),
        enable_fci_keepalive=False,
    ))
    print(
        "TCP translation control: "
        + (
            "residual accumulation (+/-0.3 mm lead, Z-stall guard enabled)"
            if args.residual_control
            else "legacy measured-TCP anchoring (residual disabled)"
        ),
        flush=True,
    )
    rng = np.random.default_rng(args.sampling_seed)
    try:
        robot.connect()
        # Warm up kernels using real observations before allowing motion.
        with torch.inference_mode():
            post(policy.select_action(pre(observation_frame(robot.get_observation(), camera_shapes))))
        policy.reset()
        if args.dry_run:
            episode_index = 0
            while True:
                action = prompt_choice(
                    f"DRY RUN: [Enter/s]=start {args.duration:g}s inference, q=quit: ",
                    {"": "start", "s": "start", "start": "start", "q": "quit", "quit": "quit"},
                )
                if action == "quit":
                    break
                episode_index += 1
                run_inference_episode(
                    robot, policy, pre, post, args, torch, camera_shapes,
                    episode_index=episode_index,
                )
            return

        sample_bounds = confirm_centered_bounds(robot)
        if sample_bounds is None:
            return
        sample_min, sample_max, orientation = sample_bounds
        print(
            "Sample reset orientation (confirmed TCP rotation vector, rad): "
            f"{format_pose(orientation)}",
            flush=True,
        )
        episode_start = None
        need_new_sample = True
        lift_before_next_sample = False
        first_move = True
        episode_index = 0
        while True:
            if need_new_sample:
                target = sample_workspace_pose(
                    sample_min,
                    sample_max,
                    orientation,
                    0.0,
                    rng,
                )
                if first_move:
                    decision = prompt_choice(
                        "First sampled reset target is "
                        f"{format_pose(target)}. [Enter/y]=move, n=new sample, q=quit: ",
                        {
                            "": "move", "y": "move", "yes": "move",
                            "n": "new", "new": "new",
                            "q": "quit", "quit": "quit",
                        },
                    )
                    if decision == "quit":
                        break
                    if decision == "new":
                        continue
                    first_move = False
                if lift_before_next_sample:
                    episode_start = move_to_next_sample(robot, target, args.reset_speed)
                    lift_before_next_sample = False
                else:
                    episode_start = move_to_pose(
                        robot, target, args.reset_speed, label="sampled inference start"
                    )
                need_new_sample = False

            decision = prompt_choice(
                "At inference start "
                f"{format_pose(episode_start)}. [Enter/s]=start inference, "
                "n=move to next sample, q=quit: ",
                {
                    "": "start", "s": "start", "start": "start",
                    "n": "new", "new": "new",
                    "q": "quit", "quit": "quit",
                },
            )
            if decision == "quit":
                break
            if decision == "new":
                need_new_sample = True
                continue

            episode_index += 1
            print(f"Starting inference episode {episode_index} for {args.duration:g}s.", flush=True)
            run_inference_episode(
                robot, policy, pre, post, args, torch, camera_shapes,
                episode_index=episode_index,
            )
            print(f"Inference episode {episode_index} finished.", flush=True)
            decision = prompt_choice(
                "Next action: r=return to this episode's start, "
                "n=move to a new sampled point, q=quit: ",
                {
                    "r": "return", "return": "return",
                    "n": "new", "new": "new",
                    "q": "quit", "quit": "quit",
                },
            )
            if decision == "quit":
                break
            if decision == "new":
                # Avoid sweeping laterally from the final inference pose to the
                # next sample: retreat 1 cm in base +Z first.
                need_new_sample = True
                lift_before_next_sample = True
                continue
            episode_start = move_to_pose(
                robot, episode_start, args.reset_speed, label="previous inference start"
            )
    except KeyboardInterrupt:
        print("Stopped by operator.")
    finally:
        try:
            if robot.is_connected and not args.dry_run:
                robot.robot.stop_move()
        finally:
            policy.reset()
            if robot.is_connected:
                robot.disconnect()


if __name__ == "__main__":
    main()
