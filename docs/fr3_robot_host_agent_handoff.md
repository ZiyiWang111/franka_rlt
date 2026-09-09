# FR3 机器人主机 Agent Handoff

更新时间：2026-09-09

## 目标与部署结构

本项目采用分离式真机推理：机器人主机负责硬件采集、安全判断和动作执行，GPU
服务器只负责加载微调后的 π0.5 并返回 action chunk。

```text
机器人主机                                      GPU 服务器
RealSense wrist/front                           π0.5 checkpoint 030000
FR3 state / control client       ZMQ             PaliGemma tokenizer
22D observation -> 15D state  ---------->        inference
安全执行器                    <----------        float32 action [50, 7]
control_server -> FR3
```

模型、checkpoint、tokenizer 和训练数据都保留在 GPU 服务器。机器人主机不运行
π0.5，也不需要 CUDA。

## 本机 Agent 已完成的代码

### GPU inference server

文件：`scripts/fr3_pi05_inference_server.py`

- 默认加载
  `outputs/fr3_pi05_sft_60ep_state15_action7_bs4_30k/checkpoints/030000/pretrained_model`。
- 在 CUDA 上加载 checkpoint 自带的 preprocessor 和 postprocessor。
- 只接受固定 schema：
  - `wrist`: RGB `uint8 [480, 640, 3]`
  - `front`: RGB `uint8 [480, 640, 3]`
  - state: `float32 [15]`
  - 非空 task instruction
- 返回 `float32 [50, 7]`：
  `[dx, dy, dz, drx, dry, drz, gripper_target_width]`。
- 每次请求检查版本、shape、dtype、相机顺序、request id 和 NaN/Inf。
- GPU server 不导入 Franka、RealSense 或 control-server 代码。

### Robot-host client

文件：`scripts/fr3_remote_robot_client.py`

- 在机器人主机本地创建 `FrankaRobot`，读取 FR3 状态和两台 RealSense。
- 将原始 22D observation 转换为 checkpoint 所需的 15D state：
  - 7 个 joint position
  - 6 个 TCP pose
  - `gripper_width`
  - `gripper_grasped`
- 关节速度保留在硬件 observation 中，但不会发送给模型。
- 将两张原始 RGB 图、15D state 和 task 发送到 GPU server。
- 支持四种模式：
  - `shadow`：只读 observation 和预测，绝不发送动作。
  - `gripper-only`：机械臂不动，只执行夹爪状态变化。
  - `arm-only`：夹爪保持现状，只执行 TCP。
  - `integrated`：执行 TCP 和夹爪。
- 所有非 shadow 模式必须显式提供 `--allow-motion`。
- `arm-only` 和 `integrated` 还必须显式提供工作空间上下界。
- 默认每次只执行 50 步 chunk 的前 5 步，然后重新观察和推理。
- 网络请求超时后不执行动作，REQ socket 会重建。
- 可将 state、完整预测 chunk、实际执行动作和推理时间写入 JSONL。

### Shared protocol and safety logic

目录：`src/evo_rlt/adapters/lerobot/franka_remote/`

- `protocol.py`：ZMQ multipart wire format，使用 JSON header 和原始数组 bytes，
  不使用 pickle。
- `state.py`：22D 命名 observation 到 15D 模型 state 的显式转换。
- `execution.py`：TCP 向量限幅和夹爪状态机。

夹爪状态机约定：

- 第 7 维是绝对目标宽度，单位为米。
- `target <= 0.040 m` 请求闭合。
- `target >= 0.055 m` 请求打开。
- 中间 dead band 保持当前状态。
- 默认需要连续两步预测相同变化才执行。
- 从第一步变化候选开始就冻结 TCP。
- blocking gripper RPC 完成后丢弃 chunk 剩余动作、重置 TCP 积分目标并重新推理。

### Franka adapter changes

文件：

- `src/evo_rlt/adapters/lerobot/franka_robot/configuration_franka.py`
- `src/evo_rlt/adapters/lerobot/franka_robot/franka_robot.py`

改动：

- `robotLab` 路径可由 `robotlab_path` 或 `ROBOTLAB_PATH` 配置，不再只能使用硬编码路径。
- 增加 `enable_fci_keepalive`。远程 client 在 shadow 模式将它关闭，因此 shadow
  不会发出周期性 `servo_joint`。
- 增加可选的 `workspace_min_xyz` / `workspace_max_xyz`。
- TCP 积分目标超出 workspace 时拒绝发送。
- 增加 `resync_command_pose()`，夹爪 blocking RPC 后重新锚定 TCP 目标。

### Dependency and tests

- `pyproject.toml` 的 `lerobot` extra 增加 `pyzmq>=26.0`。
- `tests/test_fr3_remote_inference.py` 覆盖协议往返、schema、state 顺序、TCP
  限幅、夹爪去抖，以及夹爪切换期间禁止 arm action。
- 本机已运行相关测试：9 passed。
- 尚未在机器人主机上导入 robotLab/RealSense，也尚未进行任何真机通信或动作。

## 机器人主机需要同步什么

### 必须同步

同步整个 Git 仓库的源代码，至少必须包含：

```text
pyproject.toml
scripts/fr3_remote_robot_client.py
src/evo_rlt/adapters/lerobot/franka_remote/
src/evo_rlt/adapters/lerobot/franka_robot/configuration_franka.py
src/evo_rlt/adapters/lerobot/franka_robot/franka_robot.py
docs/fr3_remote_inference.md
docs/fr3_robot_host_agent_handoff.md
tests/test_fr3_remote_inference.py
```

机器人主机还需保留它原有的：

- `/home/embint/robotLab` 或实际 robotLab 路径。
- robotLab control server 和 Python control client。
- `pyrealsense2`、RealSense udev 配置和两个相机的 USB 连接。
- 能够成功数采时使用的 Python/系统环境。

### 不需要同步到机器人主机

- `datasets/franka_m1_manual_demo_state15_action7`：不需要。它只用于训练和离线
  dataset dry run。
- `outputs/.../checkpoints/030000`：不需要。模型只在 GPU server 加载。
- `/workspace/wangziyi/models/pi05_base`：不需要。
- Hugging Face PaliGemma tokenizer cache：不需要。
- checkpoint 的 `training_state`：两边推理都不需要。

如果将来想在机器人主机复现 dataset offline dry run，才需要复制数据集；当前方案 B
不做这件事。

## 机器人主机 Agent 必须按以下顺序执行

### 1. 只检查代码与依赖

在机器人主机仓库中：

```bash
git status --short
python -m pip install -e '.[lerobot]'
python -c 'import zmq; print(zmq.__version__)'
python -c 'import pyrealsense2; print("pyrealsense2 OK")'
test -d /home/embint/robotLab && echo 'robotLab OK'
```

如果 robotLab 不在 `/home/embint/robotLab`，后续命令必须加入：

```text
--robotlab-path /实际/robotLab/路径
```

### 2. 只做 GPU server 网络检查

GPU server 必须已经打印 `READY`。在机器人主机运行：

```bash
python scripts/fr3_remote_robot_client.py \
  --server tcp://GPU_SERVER_IP:5559 \
  --network-only
```

这一模式在导入 Franka 或 RealSense 前退出，不连接 control server，不读取相机，
不发送任何机器人命令。

期望输出包含：

```text
Inference server ready
action_chunk_shape=[50, 7]
Network-only check passed
```

失败时检查 GPU server IP、端口 5559、防火墙和两端 `pyzmq`，不要进入后续模式。

### 3. 运行一次 live shadow

确认 FR3、control server 和两个相机处于数采时的正常状态，然后运行：

```bash
python scripts/fr3_remote_robot_client.py \
  --server tcp://GPU_SERVER_IP:5559 \
  --robot-ip 172.16.0.2 \
  --mode shadow \
  --max-cycles 1 \
  --log /tmp/fr3_shadow.jsonl
```

shadow 模式只允许：

- 连接 control client 和相机。
- 读取关节、TCP、夹爪和图像。
- 请求 GPU inference。
- 打印并保存预测。

shadow 模式不得调用：

- `robot.send_action()`
- `open_gripper()`
- `close_gripper()`
- FCI keepalive `servo_joint`

将终端完整输出和 `/tmp/fr3_shadow.jsonl` 返回给本机 Agent 检查。重点检查：

- server 返回的 checkpoint 是 `030000/pretrained_model`。
- inference 输出为 `[50, 7]`。
- `state15` 长度为 15，顺序与本文一致。
- 当前 `gripper_width` 合理。
- TCP delta 和 gripper target 没有 NaN/Inf 或明显越界。
- 网络往返与 `inference_ms` 是否支持后续控制。

### 4. 当前禁止直接执行的模式

在 shadow 日志检查完成前，不运行：

```text
--mode gripper-only
--mode arm-only
--mode integrated
--allow-motion
```

进入 arm motion 前还需要用户根据真实清空工作区提供机器人 base frame 下的
`workspace-min` 和 `workspace-max`。不要只用训练数据的 min/max 代替现场安全边界。

## GPU server 启动参考

此命令在 GPU server 执行，不在机器人主机执行：

```bash
nvidia-smi --id=GPU_ID
CUDA_VISIBLE_DEVICES=GPU_ID \
python scripts/fr3_pi05_inference_server.py \
  --device cuda \
  --bind tcp://0.0.0.0:5559
```

确认所选 GPU 在启动前空闲，并等待模型加载完成后的 `READY`。GPU server 需要：

- 本仓库代码。
- checkpoint `030000/pretrained_model`。
- PaliGemma tokenizer cache。
- CUDA、PyTorch、LeRobot 和 pyzmq。

GPU server 不需要连接 RealSense、robotLab 或 FR3。
