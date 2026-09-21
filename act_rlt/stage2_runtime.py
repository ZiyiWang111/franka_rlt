"""Persistent camera/inference/servo workers for fixed-chunk Stage-2 collection."""

from __future__ import annotations

import copy
import queue
import threading
import time
from dataclasses import dataclass

import torch

from act_rlt.infer import MAX_INFERENCE_S, bounded_action, observation_frame
from act_rlt.stage2 import ChunkExecution


@dataclass
class Prediction:
    state: torch.Tensor
    reference: torch.Tensor
    commands: list[dict]
    reference_commands: list[dict]
    mean_commands: list[dict]
    executed: torch.Tensor
    clipped: list[bool]


class FixedChunkRuntime:
    """One runtime per episode/collection phase; learning runs in the caller.

    No temporal blending or mid-chunk replanning. The observation captured after
    the last command supplies BOTH replay's next state and the next prediction.
    Worker queues never repeat an action or silently discard a transition.
    """

    def __init__(self, env, policy, initial_batch, *, warmup: bool, step_budget: int):
        self.env = env
        self.policy = policy
        self.warmup = warmup
        self.step_budget = step_budget
        self.actor = None if warmup else copy.deepcopy(policy.actor).eval().requires_grad_(False)
        self.actor_lock = threading.Lock()
        self.actor_state = None
        self.stop = threading.Event()
        self.finished = threading.Event()
        self.observation_ready = threading.Condition()
        self.observation = None
        self.observation_started = -1.0
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
        if not self.warmup:
            # Called only by the learner thread, between optimizer steps. CPU
            # copies synchronize CUDA before another worker loads the snapshot.
            snapshot = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
            with self.actor_lock:
                self.actor_state = snapshot

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
                observation = self.env.robot.get_observation()
                with self.observation_ready:
                    self.observation = observation
                    self.observation_started = started
                    self.observation_ready.notify_all()
        except BaseException as error:
            self._fail(error)

    def _boundary_batch(self, after):
        with self.observation_ready:
            ready = self.observation_ready.wait_for(
                lambda: self.stop.is_set() or self.observation_started >= after,
                timeout=2.0,
            )
            if self.stop.is_set():
                return None
            if not ready:
                raise RuntimeError("Stage-2 camera did not supply a fresh boundary observation")
            observation = self.observation
        return self.env.pre(observation_frame(observation))

    def _prediction(self, state, reference, full):
        if self.warmup:
            actions = reference
            mean_actions = reference
        else:
            with self.actor_lock:
                snapshot, self.actor_state = self.actor_state, None
            if snapshot is not None:
                self.actor.load_state_dict(snapshot)
            sampled, mean = self.actor.sample(
                state, reference.flatten(start_dim=-2), training=False
            )
            actions = sampled.reshape(1, self.env.config.chunk_length, self.env.config.action_dim)
            mean_actions = mean.reshape_as(actions)

        def physical_actions(action_chunk):
            normalized = full[:, :self.env.config.chunk_length].clone()
            normalized[:, :, :self.env.config.action_dim] = action_chunk
            physical = self.env.post(normalized).detach().cpu().reshape(
                self.env.config.chunk_length, -1
            )
            if physical.shape[1] != self.env.config.action_dim + 1 or not torch.isfinite(physical).all():
                raise RuntimeError("invalid Stage-2 physical action chunk")
            return physical

        physical = physical_actions(actions)
        reference_physical = physical_actions(reference)
        mean_physical = physical_actions(mean_actions)
        commands, reference_commands, mean_commands, executed, clipped = [], [], [], [], []
        for row, reference_row, mean_row in zip(
            physical, reference_physical, mean_physical, strict=True
        ):
            command = bounded_action(row.numpy(), self.env.max_step_m, self.env.max_step_rad)
            reference_command = bounded_action(
                reference_row.numpy(), self.env.max_step_m, self.env.max_step_rad
            )
            mean_command = bounded_action(
                mean_row.numpy(), self.env.max_step_m, self.env.max_step_rad
            )
            bounded = torch.tensor(list(command.values()), dtype=row.dtype)
            clipped.append(not torch.allclose(bounded, row[:self.env.config.action_dim]))
            actual = row.clone()
            actual[:self.env.config.action_dim] = bounded
            commands.append(command)
            reference_commands.append(reference_command)
            mean_commands.append(mean_command)
            executed.append(self.env._normalize_executed_action(actual))
        return Prediction(
            state, reference, commands, reference_commands, mean_commands,
            torch.stack(executed), clipped,
        )

    def _inference(self):
        try:
            with torch.inference_mode():
                state, reference, full = self.policy.encode_and_reference(self.initial_batch)
                self.predictions.put_nowait(self._prediction(state, reference, full))
                while not self.stop.is_set():
                    try:
                        previous, execution, after, last = self.requests.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    batch = self._boundary_batch(after)
                    if batch is None:
                        break
                    state, reference, full = self.policy.encode_and_reference(batch)
                    execution.info["boundary_inference_ms"] = (time.monotonic() - after) * 1000
                    execution.next_batch = batch
                    execution.state_vec = previous.state
                    execution.ref_chunk = previous.reference
                    execution.next_state_vec = state
                    execution.next_ref_chunk = reference
                    if not last:
                        # Prepare the next command before CPU replay/logging work.
                        self.predictions.put_nowait(self._prediction(state, reference, full))
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
        try:
            while not self.stop.is_set():
                waiting_since = time.monotonic()
                while not self.stop.is_set():
                    try:
                        prediction = self.predictions.get(timeout=0.02)
                        break
                    except queue.Empty:
                        limit = 2.0 if deadline is None else MAX_INFERENCE_S
                        if time.monotonic() - waiting_since > limit:
                            raise RuntimeError(
                                f"Stage-2 action queue underrun: no prediction within {limit:g}s"
                            )
                else:
                    break
                if deadline is None:
                    deadline = time.monotonic()
                rewards = torch.zeros(self.env.config.chunk_length)
                executed = torch.zeros_like(prediction.executed)
                info = {
                    "success": False, "safety_clip_steps": 0,
                    "workspace_violation": False, "workspace_error": None,
                    "execution_mode": "persistent_three_thread_fixed_chunk",
                    "deadline_misses": 0, "max_lateness_ms": 0.0,
                    "max_command_interval_ms": 0.0, "warmup": self.warmup,
                }
                actual = 0
                sampled_sum = torch.zeros(self.env.config.action_dim)
                reference_sum = torch.zeros(self.env.config.action_dim)
                mean_sum = torch.zeros(self.env.config.action_dim)
                done = terminated = truncated = stop_requested = False
                for index, command in enumerate(prediction.commands):
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
                    self.env.robot.resync_command_pose()
                    try:
                        self.env.robot.send_action(command)
                    except RuntimeError as error:
                        if "refusing TCP target outside workspace" not in str(error):
                            raise
                        info["workspace_violation"] = True
                        info["workspace_error"] = str(error)
                        truncated = done = True
                        break
                    sent = time.monotonic()
                    if previous_send is not None:
                        info["max_command_interval_ms"] = max(
                            info["max_command_interval_ms"], (sent - previous_send) * 1000
                        )
                    previous_send = sent
                    executed[index] = prediction.executed[index]
                    sampled_sum += torch.tensor(list(command.values()))
                    reference_sum += torch.tensor(
                        list(prediction.reference_commands[index].values())
                    )
                    mean_sum += torch.tensor(list(prediction.mean_commands[index].values()))
                    info["safety_clip_steps"] += int(prediction.clipped[index])
                    actual += 1
                    collected += 1
                    key = self.env._monitor.poll() if self.env._monitor is not None else None
                    if key == "s":
                        rewards[index] = 1.0
                        info["success"] = terminated = done = True
                    elif key == "f":
                        terminated = done = True
                    elif key == "q":
                        truncated = done = stop_requested = True
                    elif time.monotonic() - self.env._episode_start >= self.env.episode_time_s:
                        info["timed_out"] = done = True
                    if done:
                        break
                # Private chunk summaries are consumed by the environment to
                # create one episode-level diagnostic; never written directly
                # into per-chunk JSON logs.
                info["_sampled_tcp_command_sum"] = sampled_sum.tolist()
                info["_reference_tcp_command_sum"] = reference_sum.tolist()
                info["_mean_tcp_command_sum"] = mean_sum.tolist()
                last = done or collected >= self.step_budget
                info["collector_paused"] = last
                if last:
                    # Budget boundaries allow the final command its control
                    # period; episode outcomes stop immediately. Capture the
                    # final replay observation only after stopping the stream.
                    if not done and self.stop.wait(max(0.0, deadline - time.monotonic())):
                        return
                    if not self.env.robot.robot.stop_servo():
                        raise RuntimeError("control server did not acknowledge Stage-2 stop_servo")
                    servo_stopped = True
                result = ChunkExecution(
                    next_batch={}, exec_chunk=executed, reward_seq=rewards,
                    actual_steps=actual, done=done, terminated=terminated,
                    truncated=truncated, stop_requested=stop_requested, info=info,
                )
                self.requests.put_nowait((prediction, result, time.monotonic(), last))
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
                return self.results.get(timeout=0.05)
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
