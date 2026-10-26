# Coarse 数采

```bash
bash act_rlt/data_collection/collect_coarse.sh --dataset act_coarse --episodes 100
```

默认写入 `datasets/ethernet_coarse/act_coarse`。启动脚本启动独立 control server，
退出时关闭自己启动的 server；已有 server 占用端口时会报错，不会终止其他 server。
直接运行 Python 模块时需自行启动 control server。

只查看参数和坐标，不连接机械臂或相机、不创建数据：

```bash
bash act_rlt/data_collection/collect_coarse.sh --dry-run
bash act_rlt/data_collection/collect_coarse.sh --help
```

## 流程与坐标

1. 将 TCP 放到需要的目标 XY，按 Enter。只读取 **实际 XY**。
2. 点2的 Z、姿态取 `sampling.py`：Z = 402 mm，intrinsic Euler XYZ =
   `[179.7, 1.4, 90.6]°`。当前位置的 Z、姿态不会被示教为点2目标。
3. 移动到点1，不录制。点1中心为 `[553.4, 54.8, 559.4] mm`，
   intrinsic Euler XYZ = `[174.0, 3.9, 92.0]°`。
4. 开始录制点1到点2的单目标运动，`--move-speed` 默认 **0.06 m/s**。
   沿用现有 `move_tool_traj -> JointMotion/Ruckig` 路径，起停有加减速。
5. 等待真实轨迹完成，并检查实测 TCP 误差（默认 2 mm / 1°以内），保存 episode。
6. 录制结束后在点2停留至少 **0.5 秒**（保存数据的时间计入其中），再以默认
   **0.05 m/s**（`--return-speed`）返回下次采样的点1，到位后直接开始下一次。
   回程、停留、暂停编辑均不录制，无额外下插。
   达到 `--episodes` 数量后，不录制地返回最后一次点1并退出。
7. 本次启动每成功保存 **5 条新 episode**，自动留在点2暂停，行为与按 P 相同；
   可 WASD 调整 XY，Enter 更新点2并继续。手动暂停不会重置五条的计数，丢弃的 episode 不计数。

开启 `--randomize-point1-yaw` 后，每次点1采样还会独立均匀采样 yaw：按现有
intrinsic XYZ Euler 定义，Rz = **92° ±25°**（67°–117°），Rx/Ry 保持 174°/3.9°。
XY 的采样范围照常生效，Z 保持 559.4 mm，点2的 Z 和姿态仍按 `sampling.py` 固定。
此模式**关闭每五条自动暂停**，手动 P 暂停、WASD/Enter 更新目标及 0.5 秒停留均保留。

```bash
bash act_rlt/data_collection/collect_coarse.sh --dataset act_coarse --resume \
  --episodes 100 --randomize-point1-yaw
```

新建数据集时去掉 `--resume`。每条 episode 的元数据包含实际点1姿态及
`point1_randomize_yaw`，允许在同一 coarse 数据集中继续采集。

位置和方向均在机器人基座坐标系表达；程序内位置用 m，姿态用旋转向量 rad。

## 按键

- **P**：锁存暂停请求；当前 episode 完成并保存后，留在点2暂停。
  录制结束后的 0.5 秒停留期间按 P 也生效。每成功录制五条会自动进入同一暂停模式。
- **W / S**：暂停期间沿基座 −X / +X 移动。
- **A / D**：暂停期间沿基座 −Y / +Y 移动。Z、姿态锁定为点2目标。
- **Enter**：暂停时松开 WASD 后按下，停止 servo、读取实际 XY 更新点2，恢复自动采集。
- **= / −**：点1的 XY 随机半范围每次增减 0.5 cm，范围 0–5 cm。
  默认 0 cm；也可通过 `--xy-range-cm 2` 设置。修改只影响下次采样。
  X/Y 独立在 `中心 ± 半范围` 内连续均匀采样，并非 0.5 cm 的离散位置网格。
- **X**：运动期间立即停止；若正在录制，丢弃当前 episode，Enter 后重试。
- **Q / Esc / Ctrl+C**：停止采集并退出。

键盘自动选择：有 `DISPLAY` 时使用 pynput/X11，支持按住连续移动、松开停止，
沿用现有 teleop 的按键方向与 60 ms 松键去抖。
没有 `DISPLAY` 时使用终端模式，WASD 每次按键/重复输入走 1 mm，不模拟持续按住。

```bash
# SSH：终端步进控制
bash act_rlt/data_collection/collect_coarse.sh --input-backend terminal

# 物理键盘：真实按下/松开事件
bash act_rlt/data_collection/collect_coarse.sh --input-backend evdev --keyboard /dev/input/event4
```

连续 teleop 默认 0.02 m/s（`--teleop-speed`），终端步长通过 `--teleop-step-mm` 设置。
指令 XY 相对实际 XY 的超前限制为 3 mm；点2相对本次启动时示教 XY 的移动限制
默认 15 cm（`--teleop-max-shift-cm`）。暂停时仍维持既有 FCI 会话续约。

## 数据内容

- 默认双 RGB：wrist + front，各 640×480；采集默认 15 Hz，可选 `--fps 30`。
- 支持 `--wrist-only`（1280×720）或 `--front-only`（640×480）；不含 depth。
- `observation.state`：22 维，7 维关节角 + 6 维实测 TCP 位姿 + 7 维关节速度 +
  夹爪宽度 + 是否夹持。
- `action`：相邻实测 TCP 的 6 维位姿增量 + 下一帧夹爪宽度；末帧 TCP action 为零。
- `complementary_info.command_action`：6 维目标 TCP 位姿 + 指令速度。
- 保留原数采的任务、时间戳、帧/episode 索引、策略来源等字段。
- **不读取、不写入任何力、关节力矩或外部 wrench 特征。**
- `meta/coarse_collection.json`：采集类型与固定几何定义。
- `meta/coarse_episodes/episode_XXXXXX.json`：本次点1、点2、随机半范围、速度和帧数。

继续采集：

```bash
bash act_rlt/data_collection/collect_coarse.sh --dataset act_coarse --resume --episodes 100
```

恢复时会重新确认当前目标 XY，只允许本脚本创建的无力反馈 coarse 数据集。
旧普通数采的默认行为及数据格式不变；二者不能互相追加。
