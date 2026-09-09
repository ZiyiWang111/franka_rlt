# FR3 remote π0.5 inference

This deployment keeps cameras and all robot execution on the robot host. The
GPU server only accepts a validated observation and returns a π0.5 action
chunk. The robot client starts in `shadow` mode and never sends an action in
that mode.

## Wire schema

- Cameras: `wrist`, then `front`, each RGB `uint8 [480, 640, 3]`.
- State: `float32 [15]`, ordered as 7 joint positions, 6 TCP pose values,
  `gripper_width`, and `gripper_grasped`.
- Task: non-empty UTF-8 text.
- Response: `float32 [50, 7]` with
  `[dx, dy, dz, drx, dry, drz, gripper_target_width]`.

The protocol uses ZMQ multipart frames without pickle. Both sides validate the
protocol version, camera order, shapes, dtypes, action names, request id, and
finite values.

## Install

Install the repository on both machines:

```bash
pip install -e '.[lerobot]'
```

The robot host additionally needs its existing `pyrealsense2` and robotLab
environment. The GPU server needs the final checkpoint and PaliGemma tokenizer
cache; the robot host does not.

## 1. Start the GPU inference server

Check that the selected physical GPU is idle before launching. For physical
GPU 4:

```bash
nvidia-smi --id=4
CUDA_VISIBLE_DEVICES=4 python scripts/fr3_pi05_inference_server.py \
  --device cuda \
  --bind tcp://0.0.0.0:5559
```

Wait for the `READY` line. Loading the checkpoint can take about two minutes.

## 2. Check only the network from the robot host

```bash
python scripts/fr3_remote_robot_client.py \
  --server tcp://GPU_SERVER_IP:5559 \
  --network-only
```

This exits before importing any Franka or RealSense module.

## 3. Run live shadow inference

The default is one cycle. It connects to the hardware and reads live inputs,
but does not call `send_action`, `open_gripper`, or `close_gripper`. It also
disables the Franka FCI keepalive thread so shadow mode emits no servo command.

```bash
python scripts/fr3_remote_robot_client.py \
  --server tcp://GPU_SERVER_IP:5559 \
  --mode shadow \
  --max-cycles 1 \
  --log /tmp/fr3_shadow.jsonl
```

Use `--max-cycles 0` only after the one-cycle check passes; it runs until
Ctrl-C.

## Motion modes

Every motion-capable mode refuses to start without `--allow-motion`:

```text
gripper-only  execute gripper state changes while holding the arm
arm-only      execute clipped TCP deltas and ignore gripper predictions
integrated    execute both with arm hold and replanning at gripper transitions
```

Initial integrated settings execute at most five of the 50 predicted steps,
limit each translation vector to 5 mm and rotation vector to 0.03 rad, then
request a fresh observation and action chunk. These limits are configurable.
`arm-only` and `integrated` additionally require explicit base-frame bounds:

```text
--workspace-min MIN_X MIN_Y MIN_Z
--workspace-max MAX_X MAX_Y MAX_Z
```

The client refuses to enable arm motion without both bounds, and the Franka
adapter rejects any integrated TCP target outside them. Choose these values on
the robot host from the cleared physical work area; do not infer them only from
the training dataset.

The gripper action is an absolute target width in metres. Predictions at or
below 0.040 m request close, predictions at or above 0.055 m request open, and
the dead band preserves the current state. A transition requires two
consecutive predictions. The arm is held from the first transition candidate;
after the blocking gripper RPC, the remaining chunk is discarded and the TCP
target is resynchronized before replanning.
