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

Install the inference dependencies on the GPU server:

```bash
pip install -e '.[lerobot]'
```

Install the vendored Franka backend as well on the robot host:

```bash
pip install -e '.[lerobot,franka]'
```

The robot host additionally needs its existing `pyrealsense2` and RealSense
system setup. It no longer needs a RobotLab checkout or `ROBOTLAB_PATH`. The GPU
server needs the final checkpoint and PaliGemma tokenizer cache; the robot host
does not.

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

## 3. Start Evo-RLT's Franka control server

In a dedicated robot-host terminal, start the copied controller. This process
owns franky/libfranka and the FCI session:

```bash
evo-franka-control-server --ip 172.16.0.2
```

Wait for it to connect, then verify the local IPC path from another terminal:

```bash
python -m evo_franka.control_client --ping
```

The expected result is `PONG`. Keep the control-server terminal running while
using the remote client.

## 4. Run live shadow inference

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

The vanilla loop synchronously predicts a chunk, executes the first eight rows
in order at 15 Hz, then requests a fresh observation and chunk. It does not use
async prefetch, time skipping, chunk blending, low-pass filtering, or
acceleration shaping. Each translation vector still has a 5 mm hard bound and
each rotation vector a 0.03 rad hard bound; these safety bounds are configurable.
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
the confirmed command starts a native Franky Hand Future and returns immediately.
While that Future runs, camera/robot observations and inference continue, but all
resulting action chunks are discarded and the arm stays stopped. Width/grasped are
explicitly last-known values during this interval. After completion, the TCP target
is resynchronized and the first fresh observation is replanned before arm motion resumes.
