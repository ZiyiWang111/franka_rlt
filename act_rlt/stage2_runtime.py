"""Persistent camera/inference/servo workers for fixed-chunk Stage-2 collection."""

from __future__ import annotations

from collections import deque
import copy
import queue
import threading
import time
from dataclasses import dataclass

import torch
from scipy.spatial.transform import Rotation

from act_rlt.infer import MAX_INFERENCE_S, bounded_action, observation_frame
from act_rlt.stage2 import ACTStage2Policy, ChunkExecution
from act_rlt.stage2_timing import Stage2Timing
from act_rlt.replay_overlap import ReplayObservation
from act_rlt.exploration_bias import DOWNWARD_BIAS_STEP_M


@dataclass
class PreparedActions:
    commands: list[dict]
    executed: torch.Tensor
    clipped: list[bool]
    delta_xyz: torch.Tensor | None = None


@dataclass
class Prediction:
    state: torch.Tensor
    reference: torch.Tensor
    commands: list[dict]
    reference_commands: list[dict]
    mean_commands: list[dict]
    executed: torch.Tensor
    clipped: list[bool]
    human: bool = False
    gripper_values: list[float] | None = None
    timing: Stage2Timing | None = None
    ready_at: float = 0.0
    bias_step_xyyaw: torch.Tensor | None = None  # X/Y meters, local yaw radians
    bias_delta: torch.Tensor | None = None
    downward: PreparedActions | None = None


class FixedChunkRuntime:
    """One runtime per episode/collection phase; learning runs in the caller.

    No temporal blending or mid-chunk replanning. In latest mode, use the newest
    cached observation not yet consumed, waiting only when no newer one exists.
    Boundary mode instead requires exposure/state time (or legacy call-start
    time) after the last command. Either mode shares one encoding between
    replay's next state and the next prediction; latest mode does not promise a
    physically post-command observation.
    Worker queues never repeat an action or silently discard a transition.
    """

    def __init__(self, env, policy, initial_batch, *, warmup: bool, step_budget: int | None,
                 deterministic: bool = False):
        self.env = env
        self.policy = policy
        self.warmup = warmup
        self.step_budget = step_budget
        self.deterministic = deterministic
        self.replay_stride = (
            env.config.chunk_length if deterministic else
            (getattr(env.config, "replay_stride", None) or env.config.chunk_length)
        )
        self._last_replay_sequence = 0
        self.observation_selection = getattr(env, "observation_selection", "boundary")
        if self.observation_selection not in {"latest", "boundary"}:
            raise ValueError("observation_selection must be latest or boundary")
        self.human = getattr(env, "_control_mode", "policy") == "human"
        self.actor = None if warmup else copy.deepcopy(policy.actor).eval().requires_grad_(False)
        self.actor_lock = threading.Lock()
        self.actor_state = None
        self.stop = threading.Event()
        self.finished = threading.Event()
        self.observation_ready = threading.Condition()
        self.observation = None
        self.observation_started = -1.0
        self.observation_seq = 0
        self.last_observation_seq = 0
        self.observation_history = deque(maxlen=8)
        self.requests = queue.Queue(maxsize=1)
        self.predictions = queue.Queue(maxsize=1)
        # Bound learner lag without dropping robot experience.
        self.results = queue.Queue(maxsize=64)
        self.failure = None
        self.failure_lock = threading.Lock()
        self.initial_batch = initial_batch
        self.threads = [
            threading.Thread(target=self._camera, name="stage2-camera", daemon=True),
            threading.Thread(target=self._inference, name="stage2-inference", daemon=True),
            threading.Thread(target=self._servo, name="stage2-servo", daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def publish_actor(self, actor):
        started = time.monotonic()
        if not self.warmup:
            # Called only by the learner thread, between optimizer steps. CPU
            # copies synchronize CUDA before another worker loads the snapshot.
            snapshot = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
            with self.actor_lock:
                self.actor_state = snapshot
        return (time.monotonic() - started) * 1000

    def _fail(self, error):
        with self.failure_lock:
            if self.failure is None:
                self.failure = error
        self.stop.set()
        with self.observation_ready:
            self.observation_ready.notify_all()

    def _camera(self):
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                timed_reader = getattr(self.env.robot, "get_timed_observation", None)
                if callable(timed_reader):
                    timed = timed_reader()
                    observation, fresh_after = timed.observation, timed.fresh_after
                    diagnostics = timed.diagnostics
                else:
                    observation = self.env.robot.get_observation()
                    fresh_after = started
                    diagnostics = {"camera_boundary_mode": "legacy",
                                   "camera_boundary_fallback_reason": "timed_observation_unavailable"}
                capture_auxiliary = getattr(self.env, "_capture_auxiliary_state", None)
                if capture_auxiliary is not None:
                    observation = capture_auxiliary(observation)
                if self.replay_stride < self.env.config.chunk_length:
                    # Own sensor buffers before the next camera read can reuse
                    # them. CPU copying stays outside the servo thread/lock.
                    observation = copy.deepcopy(observation)
                completed = time.monotonic()
                with self.observation_ready:
                    self.observation_seq += 1
                    diagnostics = {**diagnostics, "observation_seq": self.observation_seq}
                    self.observation = observation
                    self.observation_started = started
                    self.observation_completed = completed
                    self.observation_history.append(
                        (started, completed, fresh_after, observation, diagnostics)
                    )
                    self.observation_ready.notify_all()
        except BaseException as error:
            self._fail(error)

    def _replay_observation(self, after: float) -> ReplayObservation | None:
        """Select an owned cache snapshot without waiting or advancing control."""
        with self.observation_ready:
            minimum_seq = max(self.last_observation_seq, self._last_replay_sequence)
            selected = next((item for item in reversed(self.observation_history)
                             if item[4]["observation_seq"] > minimum_seq
                             and (self.observation_selection == "latest" or item[2] >= after)), None)
            if selected is None:
                return None
            _, _, fresh_after, observation, diagnostics = selected
            self._last_replay_sequence = diagnostics["observation_seq"]
            return ReplayObservation(observation, self._last_replay_sequence,
                                     time.monotonic(), fresh_after)

    def _boundary_batch(self, after, timing=None):
        started = time.monotonic()
        selection = getattr(self, "observation_selection", "boundary")

        def select():
            history = getattr(self, "observation_history", None)
            if selection == "latest":
                if self.observation_seq <= self.last_observation_seq:
                    return None
                return history[-1] if history else None
            if history is None:  # Compatibility with minimal test/custom collectors.
                if self.observation_started >= after:
                    return (self.observation_started,
                            getattr(self, "observation_completed", self.observation_started),
                            self.observation_started, self.observation, {})
                return None
            return next((item for item in history if item[2] >= after
                         and item[4].get("observation_seq", float("inf"))
                         > getattr(self, "last_observation_seq", 0)), None)

        with self.observation_ready:
            cache_hit = select() is not None
            ready = self.observation_ready.wait_for(
                lambda: self.stop.is_set() or select() is not None,
                timeout=2.0,
            )
            if self.stop.is_set():
                return None
            if not ready:
                raise RuntimeError(
                    f"Stage-2 camera did not supply a newer observation ({selection} selection)"
                )
            observation_started, observation_completed, fresh_after, observation, diagnostics = select()
            selected_at = time.monotonic()
            sequence = diagnostics.get("observation_seq", getattr(self, "observation_seq", 1))
            sequence_gap = sequence - getattr(self, "last_observation_seq", 0)
            self.last_observation_seq = sequence
        if timing is not None:
            timing.values.update(diagnostics)
            timing.values["observation_selection"] = selection
            timing.values["observation_seq"] = sequence
            timing.values["observation_seq_gap"] = sequence_gap
            timing.values["camera_cache_hit"] = cache_hit
            # Delivery age and exposure/state age differ; log both, not just IO time.
            timing.values["observation_cache_age_ms"] = (selected_at - observation_completed) * 1000
            timing.values["observation_age_ms"] = (selected_at - fresh_after) * 1000
            timing.values["observation_fresh_after_request_ms"] = (fresh_after - after) * 1000
            timing.values["camera_wait_wall_ms"] = (time.monotonic() - started) * 1000
            timing.values["observation_start_after_request_ms"] = (observation_started - after) * 1000
            timing.values["camera_capture_wall_ms"] = (observation_completed - observation_started) * 1000
        started = time.monotonic()
        camera_shapes = getattr(self.env, "camera_shapes", None)
        frame = self.env._observation_frame(observation) if hasattr(self.env, "_observation_frame") else (
            observation_frame(observation)
            if camera_shapes is None
            else observation_frame(observation, camera_shapes)
        )
        batch = self.env.pre(frame)
        if timing is not None:
            timing.values["preprocess_wall_ms"] = (time.monotonic() - started) * 1000
        return batch

    def _prediction(self, state, reference, full, *, human=False, timing=None):
        timing = timing or Stage2Timing()
        started = time.monotonic()
        if self.warmup or human:
            actions = reference
            mean_actions = reference
        else:
            with self.actor_lock:
                snapshot, self.actor_state = self.actor_state, None
            with timing.measure("actor_load", state.device):
                if snapshot is not None:
                    self.actor.load_state_dict(snapshot)
            actor_started = time.monotonic()
            actor_events = timing.start_cuda(state.device)
            if self.deterministic:
                mean, _ = self.actor(
                    state, reference.flatten(start_dim=-2), training=False
                )
                selected = mean
            else:
                selected, mean = self.actor.sample(
                    state, reference.flatten(start_dim=-2), training=False
                )
            timing.finish_cuda("actor", actor_events)
            timing.values["actor_wall_ms"] = (time.monotonic() - actor_started) * 1000
            actions = selected.reshape(
                1, self.env.config.chunk_length, self.env.config.action_dim
            )
            mean_actions = mean.reshape_as(actions)

        def physical_actions(action_chunk):
            normalized = full[:, :self.env.config.chunk_length].clone()
            normalized[:, :, :self.env.config.action_dim] = action_chunk
            physical = self.env.post(normalized).detach().cpu().reshape(
                self.env.config.chunk_length, -1
            )
            if physical.shape[1] != full.shape[-1] or not torch.isfinite(physical).all():
                raise RuntimeError("invalid Stage-2 physical action chunk")
            return physical

        post_started = time.monotonic()
        physical = physical_actions(actions)
        unbiased_physical = physical.clone()
        bias_step = physical.new_zeros(3)
        bias = getattr(self.env, "_exploration_bias", None)
        if bias is not None and not (self.warmup or human or self.deterministic):
            bias_step = physical.new_tensor([*bias.step_m, bias.step_yaw_rad])
            physical[:, [0, 1, 5]] += bias_step
        reference_physical = physical_actions(reference)
        mean_physical = physical_actions(mean_actions)
        timing.values["postprocess_cpu_wall_ms"] = (time.monotonic() - post_started) * 1000
        commands_started = time.monotonic()
        commands, reference_commands, mean_commands, executed, clipped = [], [], [], [], []
        gripper_values = []
        bias_deltas = []
        for row, unbiased_row, reference_row, mean_row in zip(
            physical, unbiased_physical, reference_physical, mean_physical, strict=True
        ):
            command = bounded_action(row.numpy(), self.env.max_step_m, self.env.max_step_rad)
            reference_command = bounded_action(
                reference_row.numpy(), self.env.max_step_m, self.env.max_step_rad
            )
            mean_command = bounded_action(
                mean_row.numpy(), self.env.max_step_m, self.env.max_step_rad
            )
            bounded = torch.tensor(list(command.values()), dtype=row.dtype)
            unbiased_command = bounded_action(
                unbiased_row.numpy(), self.env.max_step_m, self.env.max_step_rad
            )
            # Measure the offset after translation/rotation norm clipping.
            bias_deltas.append(bounded[[0, 1, 5]] - bounded.new_tensor(list(unbiased_command.values()))[[0, 1, 5]])
            clipped.append(not torch.allclose(bounded, row[:self.env.config.action_dim]))
            actual = row.clone()
            actual[:self.env.config.action_dim] = bounded
            commands.append(command)
            reference_commands.append(reference_command)
            mean_commands.append(mean_command)
            executed.append(self.env._normalize_executed_action(actual))
            if reference_row.numel() > self.env.config.action_dim:
                gripper_values.append(float(reference_row[self.env.config.action_dim]))
        downward = None
        if getattr(self.env, "_downward_bias", None) is not None and not (human or self.deterministic):
            # Prepare both choices in the inference worker. The servo can then
            # respond mid-chunk without normalization/CUDA work at a tick.
            down_commands, down_executed, down_clipped, down_deltas = [], [], [], []
            for row, baseline in zip(physical, commands, strict=True):
                raw = row.clone()
                raw[2] += DOWNWARD_BIAS_STEP_M
                command = bounded_action(raw.numpy(), self.env.max_step_m, self.env.max_step_rad)
                bounded = raw.new_tensor(list(command.values()))
                down_clipped.append(not torch.allclose(bounded, raw[:self.env.config.action_dim]))
                down_deltas.append(bounded[:3] - bounded.new_tensor(list(baseline.values()))[:3])
                raw[:self.env.config.action_dim] = bounded
                down_commands.append(command)
                down_executed.append(self.env._normalize_executed_action(raw))
            downward = PreparedActions(down_commands, torch.stack(down_executed),
                                          down_clipped, torch.stack(down_deltas))
        timing.values["command_prepare_wall_ms"] = (time.monotonic() - commands_started) * 1000
        timing.values["prediction_wall_ms"] = (time.monotonic() - started) * 1000
        return Prediction(
            state, reference, commands, reference_commands, mean_commands,
            torch.stack(executed), clipped, human, gripper_values, timing, time.monotonic(),
            bias_step, torch.stack(bias_deltas), downward,
        )

    def _encode(self, batch, timing):
        with timing.measure("encode", batch.get("observation.state", torch.empty(0)).device):
            if isinstance(self.policy, ACTStage2Policy):
                return self.policy.encode_and_reference(batch, timing=timing)
            return self.policy.encode_and_reference(batch)

    def _inference(self):
        try:
            with torch.inference_mode():
                timing = Stage2Timing()
                state, reference, full = self._encode(self.initial_batch, timing)
                self.predictions.put_nowait(self._prediction(state, reference, full, human=self.human, timing=timing))
                while not self.stop.is_set():
                    try:
                        previous, execution, after, last, next_human = self.requests.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    timing = Stage2Timing()
                    timing.values["request_queue_wall_ms"] = (time.monotonic() - after) * 1000
                    batch = self._boundary_batch(after, timing)
                    if batch is None:
                        break
                    state, reference, full = self._encode(batch, timing)
                    execution.info["boundary_inference_ms"] = (time.monotonic() - after) * 1000
                    execution.next_batch = batch
                    execution.state_vec = previous.state
                    execution.ref_chunk = previous.reference
                    execution.next_state_vec = state
                    execution.next_ref_chunk = reference
                    if not last:
                        # Prepare the next command before CPU replay/logging work.
                        self.predictions.put_nowait(
                            self._prediction(state, reference, full, human=next_human, timing=timing)
                        )
                    timing.values["boundary_total_wall_ms"] = (time.monotonic() - after) * 1000
                    execution.info["_boundary_timing"] = timing
                    execution.info["result_ready_monotonic_s"] = time.monotonic()
                    try:
                        self.results.put_nowait(execution)
                    except queue.Full:
                        raise RuntimeError("Stage-2 learner fell behind by 64 chunks; stopping collection")
                    if last:
                        self.finished.set()
                        self.stop.set()
                        break
        except BaseException as error:
            self._fail(error)
        finally:
            with self.observation_ready:
                self.observation_ready.notify_all()

    def _servo(self):
        deadline = None
        previous_send = None
        collected = 0
        servo_stopped = False
        human_moving = False
        try:
            while not self.stop.is_set():
                waiting_since = time.monotonic()
                while not self.stop.is_set():
                    try:
                        prediction = self.predictions.get(timeout=0.02)
                        break
                    except queue.Empty:
                        if self.human and human_moving and not self.env._monitor.motion_velocity().any():
                            if not self.env.robot.robot.stop_servo():
                                raise RuntimeError("control server did not acknowledge human stop_servo")
                            human_moving = False
                            servo_stopped = True
                        limit = 2.0 if deadline is None else MAX_INFERENCE_S
                        if time.monotonic() - waiting_since > limit:
                            raise RuntimeError(
                                f"Stage-2 action queue underrun: no prediction within {limit:g}s"
                            )
                else:
                    break
                if prediction.human != self.human:
                    raise RuntimeError("Stage-2 prediction belongs to an obsolete control mode")
                if deadline is None:
                    deadline = time.monotonic()
                rewards = torch.zeros(self.env.config.chunk_length)
                executed = torch.zeros_like(prediction.executed)
                info = {
                    "success": False, "safety_clip_steps": 0,
                    "workspace_violation": False, "workspace_error": None,
                    "execution_mode": "persistent_three_thread_fixed_chunk",
                    "observation_selection": self.observation_selection,
                    "deadline_misses": 0, "max_lateness_ms": 0.0,
                    "max_command_interval_ms": 0.0, "warmup": self.warmup,
                    "control_source": "human" if self.human else "policy",
                    "_prediction_timing": prediction.timing,
                    "prediction_queue_wait_wall_ms": (time.monotonic() - waiting_since) * 1000,
                    "control_period_ms": self.env.period_s * 1000,
                    "send_action_wall_ms_sum": 0.0,
                    "send_action_wall_ms_max": 0.0,
                    "resync_wall_ms_sum": 0.0,
                }
                actual = 0
                replay_observations = {}
                replay_missing = 0
                sampled_sum = torch.zeros(self.env.config.action_dim)
                reference_sum = torch.zeros(self.env.config.action_dim)
                mean_sum = torch.zeros(self.env.config.action_dim)
                bias_sum = torch.zeros(3)
                bias_requested_sum = torch.zeros(3)
                downward_bias = getattr(self.env, "_downward_bias", None)
                downward_requested_z = 0.0
                downward_applied_xyz = torch.zeros(3)
                downward_steps = 0
                done = terminated = truncated = stop_requested = False
                pending_outcome = None
                for index, policy_command in enumerate(prediction.commands):
                    if self.stop.wait(max(0.0, deadline - time.monotonic())):
                        return
                    now = time.monotonic()
                    lateness = max(0.0, now - deadline)
                    if lateness > 0.001:
                        info["deadline_misses"] += 1
                        info["max_lateness_ms"] = max(info["max_lateness_ms"], lateness * 1000)
                    # Do not send a burst of relative actions to catch up after
                    # an underrun. Hold the last target while waiting; never replay it.
                    deadline = max(deadline, now) + self.env.period_s
                    if index and index % self.replay_stride == 0:
                        snapshot = self._replay_observation(previous_send or now)
                        if snapshot is None:
                            replay_missing += 1
                        else:
                            replay_observations[index] = snapshot
                    command = policy_command
                    downward_step = 0.0
                    downward_delta = torch.zeros(3)
                    downward_executed = None
                    step_clipped = prediction.clipped[index]
                    if downward_bias is not None and not self.deterministic:
                        consume = getattr(self.env._monitor, "consume_downward_bias", None)
                        if consume is not None and consume():
                            downward_bias.enable()
                        if not self.human:
                            downward_step = downward_bias.step_m
                        if downward_step:
                            # Read the latch at every servo tick, including
                            # inside an already predicted four-action chunk.
                            command = prediction.downward.commands[index]
                            downward_delta = prediction.downward.delta_xyz[index]
                            step_clipped = prediction.downward.clipped[index]
                            downward_executed = prediction.downward.executed[index]
                    if self.human:
                        monitor = self.env._monitor
                        if monitor is None or not hasattr(monitor, "motion_velocity"):
                            raise RuntimeError("human control requires the Stage-2 SpaceMouse input")
                        delta = monitor.motion_velocity() * self.env.period_s
                        moving = bool((delta != 0).any())
                        if moving and not human_moving:
                            self.env.robot.resync_command_pose()
                        # Official bridge applies angular velocity in the working
                        # frame (R_new = dR * R). send_action expects tool-local
                        # increments (R_new = R * dR_local). Replay stores exactly
                        # these converted local increments sent to the adapter.
                        if delta[3:].any():
                            # The adapter integrates its target while the mouse
                            # is held, just like the standalone bridge. Convert
                            # against that target, not a lagging measured pose.
                            pose = getattr(self.env.robot, "_cmd_pose", None)
                            if pose is None:
                                pose = self.env.robot.robot.get_tool_pose()
                            delta[3:] = Rotation.from_rotvec(pose[3:6]).inv().apply(delta[3:])
                        command = dict(zip(
                            ("dx", "dy", "dz", "drx", "dry", "drz"),
                            (float(value) for value in delta),
                            strict=True,
                        ))
                        if not moving and human_moving:
                            if not self.env.robot.robot.stop_servo():
                                raise RuntimeError("control server did not acknowledge human stop_servo")
                            servo_stopped = True
                        human_moving = moving
                    try:
                        if not self.human or human_moving:
                            resync_started = time.monotonic()
                            if not self.human:
                                self.env.robot.resync_command_pose()
                            send_started = time.monotonic()
                            info["resync_wall_ms_sum"] += (send_started - resync_started) * 1000
                            self.env.robot.send_action(command)
                            send_ms = (time.monotonic() - send_started) * 1000
                            info["send_action_wall_ms_sum"] += send_ms
                            info["send_action_wall_ms_max"] = max(info["send_action_wall_ms_max"], send_ms)
                            servo_stopped = False
                    except RuntimeError as error:
                        if "refusing TCP target outside workspace" not in str(error):
                            raise
                        info["workspace_violation"] = True
                        info["workspace_error"] = str(error)
                        truncated = done = True
                        break
                    if not self.human or human_moving:
                        sent = time.monotonic()
                        if index == 0:
                            info["prediction_ready_to_first_send_ms"] = (sent - prediction.ready_at) * 1000
                            if previous_send is not None:
                                info["chunk_boundary_command_interval_ms"] = (sent - previous_send) * 1000
                        if previous_send is not None:
                            info["max_command_interval_ms"] = max(
                                info["max_command_interval_ms"], (sent - previous_send) * 1000
                            )
                        previous_send = sent
                    else:
                        info["human_idle_steps"] = info.get("human_idle_steps", 0) + 1
                    if self.human:
                        values = list(command.values())
                        if prediction.gripper_values:
                            values.append(prediction.gripper_values[index])
                        full_action = torch.tensor(values, dtype=executed.dtype)
                        executed[index] = self.env._normalize_executed_action(full_action)
                    else:
                        executed[index] = (prediction.executed[index] if downward_executed is None
                                           else downward_executed)
                        if prediction.bias_delta is not None:
                            bias_sum += prediction.bias_delta[index]
                            if prediction.bias_step_xyyaw is not None:
                                bias_requested_sum += prediction.bias_step_xyyaw
                    if downward_step:
                        downward_requested_z += downward_step
                        downward_applied_xyz += downward_delta
                        downward_steps += 1
                    sampled_sum += torch.tensor(list(command.values()))
                    reference_sum += torch.tensor(
                        list(prediction.reference_commands[index].values())
                    )
                    mean_sum += torch.tensor(list(prediction.mean_commands[index].values()))
                    info["safety_clip_steps"] += int(step_clipped) if not self.human else 0
                    actual += 1
                    collected += 1
                    key = self.env._monitor.poll() if self.env._monitor is not None else None
                    if (key in {"s", "f"} or (key == "d" and getattr(self.env, "review_episode_replay", False))) and pending_outcome is None:
                        pending_outcome = key
                    if key == "q":
                        truncated = done = stop_requested = True
                    elif pending_outcome is None and time.monotonic() - self.env._episode_start >= self.env.episode_time_s:
                        info["timed_out"] = done = True
                    if done:
                        break
                # Private chunk summaries are consumed by the environment to
                # create one episode-level diagnostic; never written directly
                # into per-chunk JSON logs.
                info["_sampled_tcp_command_sum"] = sampled_sum.tolist()
                info["_reference_tcp_command_sum"] = reference_sum.tolist()
                info["_mean_tcp_command_sum"] = mean_sum.tolist()
                info["exploration_bias_requested_xy_m"] = (
                    bias_requested_sum[:2].tolist()
                )
                info["exploration_bias_applied_xy_m"] = bias_sum[:2].tolist()
                info["exploration_bias_requested_yaw_rad"] = (
                    float(bias_requested_sum[2])
                )
                info["exploration_bias_applied_yaw_rad"] = float(bias_sum[2])
                info["downward_bias_requested_z_m"] = downward_requested_z
                info["downward_bias_applied_xyz_m"] = downward_applied_xyz.tolist()
                info["downward_bias_steps"] = downward_steps
                info["replay_observations_captured"] = len(replay_observations)
                info["replay_observations_missing"] = replay_missing
                info["replay_observation_offsets"] = list(replay_observations)
                info["replay_observation_sequences"] = [obs.sequence for obs in replay_observations.values()]
                info["replay_observation_age_ms"] = [
                    (obs.selected_at - obs.fresh_after) * 1000 for obs in replay_observations.values()
                ]
                # Normal chunks submit camera/inference work immediately after
                # the fourth send, overlapping its remaining control period.
                # Only an outcome or actual handoff must finish that period
                # before stopping. Enabling intervention alone adds no wait.
                boundary_wait_started = time.monotonic()
                boundary_waited = False
                if pending_outcome is not None and not done:
                    if self.stop.wait(max(0.0, deadline - time.monotonic())):
                        return
                    boundary_waited = True
                    key = self.env._monitor.poll() if self.env._monitor is not None else None
                    if (key in {"s", "f"} or (key == "d" and getattr(self.env, "review_episode_replay", False))) and pending_outcome is None:
                        pending_outcome = key
                    if key == "q":
                        truncated = done = stop_requested = True
                    elif pending_outcome is None and time.monotonic() - self.env._episode_start >= self.env.episode_time_s:
                        info["timed_out"] = done = True
                if pending_outcome is not None and not done:
                    rewards[actual - 1] = float(pending_outcome == "s")
                    info["success"] = pending_outcome == "s"
                    info["discard_episode_replay"] = pending_outcome == "d"
                    terminated = done = True
                last = done or (self.step_budget is not None and collected >= self.step_budget)
                info["collector_paused"] = last
                monitor = self.env._monitor
                handoff = (
                    not done and monitor is not None and hasattr(monitor, "consume_toggle")
                    and monitor.consume_toggle(human=self.human)
                )
                # A Space request arriving after this decision stays pending
                # for the next chunk boundary; it cannot change a chunk whose
                # source/state has already been committed to inference/replay.
                if last or handoff:
                    # Budget boundaries allow the final command its control
                    # period; handoffs and s/f episode outcomes do too.
                    if not done and not boundary_waited and self.stop.wait(max(0.0, deadline - time.monotonic())):
                        return
                    if not self.env.robot.robot.stop_servo():
                        raise RuntimeError("control server did not acknowledge Stage-2 stop_servo")
                    servo_stopped = True
                next_human = self.human
                if handoff:
                    requested_at = getattr(monitor, "last_consumed_toggle_at", None)
                    if requested_at is not None:
                        info["handoff_request_to_stop_ms"] = (
                            time.monotonic() - requested_at
                        ) * 1000
                    next_human = not self.human
                    self.human = next_human
                    self.env._control_mode = "human" if next_human else "policy"
                    info["handoff_to"] = self.env._control_mode
                    info["handoff_after_steps"] = actual
                    deadline = None
                    human_moving = False
                    print(f"Stage-2 control: {self.env._control_mode.upper()}", flush=True)
                result = ChunkExecution(
                    next_batch={}, exec_chunk=executed, reward_seq=rewards,
                    actual_steps=actual, done=done, terminated=terminated,
                    truncated=truncated, stop_requested=stop_requested, info=info,
                    intervention=prediction.human,
                    bc_target_chunk=executed.clone() if prediction.human else None,
                    replay_observations=replay_observations,
                )
                info["boundary_finalize_wall_ms"] = (time.monotonic() - boundary_wait_started) * 1000
                self.requests.put_nowait((prediction, result, time.monotonic(), last, next_human))
                if last:
                    break
        except BaseException as error:
            self._fail(error)
        finally:
            try:
                # Stop before waiting for camera/inference shutdown or human labels.
                if not servo_stopped and not self.env.robot.robot.stop_servo():
                    raise RuntimeError("control server did not acknowledge Stage-2 stop_servo")
            except BaseException as error:
                self._fail(error)

    def next_result(self):
        while True:
            if self.failure is not None:
                raise self.failure
            try:
                result = self.results.get(timeout=0.05)
                result.info["result_queue_wait_wall_ms"] = (time.monotonic() - result.info["result_ready_monotonic_s"]) * 1000
                result.info["result_queue_depth"] = self.results.qsize()
                for key, prefix in (("_prediction_timing", "current_"), ("_boundary_timing", "next_")):
                    timing = result.info.pop(key, None)
                    if timing is not None:
                        result.info.update({prefix + k: v for k, v in timing.resolve().items()})
                return result
            except queue.Empty:
                if self.finished.is_set():
                    raise RuntimeError("Stage-2 collector exhausted its phase budget")

    def close(self):
        self.stop.set()
        with self.observation_ready:
            self.observation_ready.notify_all()
        # Servo stops motion before slow camera capture is joined.
        for thread in reversed(self.threads):
            thread.join()
        if self.failure is not None:
            raise self.failure
