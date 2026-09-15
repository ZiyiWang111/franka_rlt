# Franka FR3 数据采集操作说明

本文档介绍两种数据采集方式：

1. 手动示教数采：操作者手拖机械臂完成整段轨迹。
2. 两点自动数采：启动时示教两个固定点，之后机械臂重复执行固定流程。

两种方式都会自动管理项目内的 Evo-RLT Franka control server，并在正式数采前检查两台 RealSense 相机。

## 共同环境与默认硬件配置

项目目录：

```text
/home/embint/wzy/Evo-RLT
```

默认配置：

| 配置 | 默认值 |
|---|---|
| 机器人 IP | `172.16.0.2` |
| wrist 相机序列号 | `349622072679` |
| front 相机序列号 | `233522075778` |
| 数采 Python | `/home/embint/miniconda3/envs/evo-rlt/bin/python` |
| control server Python | `/home/embint/robolab-venv/bin/python` |
| control server 日志 | `/tmp/evo-rlt-franka-control-server.log` |

启动器执行以下公共流程：

1. 调用项目内的 `scripts/start_evo_franka_control_server.sh`，使用 robolab Python 环境启动与 client 同版本的 `evo_franka.control_server`。该过程不会修改 robotLab。
2. 分别启动 wrist 和 front RealSense，并从每台相机实际读取若干帧。
3. 相机检查通过后启动数采程序。
4. 正常结束、异常退出或收到 `Ctrl+C` 时，终止本次启动的 control server。

如果 control server 在数采期间断开，录制器会停止录制并丢弃当前不完整 episode。硬断开的检测受客户端通信超时影响，最坏约需 10 秒。相机在 episode 中停止出帧时，也会丢弃当前 episode 并退出。

## 方式一：手动示教数采

启动脚本：

```text
scripts/record_franka_manual_demo.sh
```

### 基本命令

```bash
cd /home/embint/wzy/Evo-RLT

./scripts/record_franka_manual_demo.sh \
  --dataset franka_manual_demo_v1 \
  --fps 30 \
  --episodes 20 \
  --episode-time 300
```

默认数据目录为：

```text
/home/embint/wzy/Evo-RLT/datasets/<dataset名字>
```

也可以显式指定：

```bash
./scripts/record_franka_manual_demo.sh \
  --dataset franka_manual_demo_v1 \
  --root /home/embint/wzy/Evo-RLT/datasets/franka_manual_demo_v1 \
  --fps 15 \
  --episodes 50 \
  --task "Insert the copper screw into the black sleeve."
```

### 操作流程

1. 在 Franka Desk 中将机械臂置于可手拖的 Guiding 状态。
2. 将机械臂移动到 episode 起始位置。
3. 按 `Enter` 开始录制。
4. 手拖机械臂完成任务。
5. 根据结果保存或丢弃当前 episode。

按键说明：

| 按键 | 功能 |
|---|---|
| `Enter` | 开始一个 episode |
| `Right` | 结束并保存当前 episode |
| `Left` | 结束、丢弃并重新采当前 episode |
| `Esc` | 退出数采；正在录制的部分不会保存 |
| `C` | 闭合夹爪 |
| `O` | 打开夹爪 |

`C/O` 使用阻塞式夹爪命令。夹爪运动期间采样会暂停，动作完成后继续采样。

### 继续已有数据集

```bash
./scripts/record_franka_manual_demo.sh \
  --dataset franka_manual_demo_v1 \
  --fps 30 \
  --episodes 10 \
  --resume
```

恢复录制时，`--fps` 必须与已有数据集一致。

### 手动数采数据语义

每一帧记录：

- Franka 关节位置、关节速度和 TCP 位姿。
- 夹爪实测宽度和 `gripper_grasped` 状态。
- wrist RGB 和 front RGB。
- 7维 action：6维实测 TCP 增量和下一帧实测夹爪宽度。

对于相邻两帧：

```text
action[t] = observation[t] 到 observation[t+1] 的实测运动
```

最后一帧的 TCP action 为零，夹爪 action 保持最后测得的宽度。

## 方式二：示教两点后自动数采

启动脚本：

```text
scripts/record_franka_waypoint_demo.sh
```

该模式在进程启动时依次记录三个完整的6D TCP位姿：初始抓取点、点2和初始起点。它们只保存在当前进程内：初始抓取点作为后续随机点1的固定中心，点2保持不变，初始起点作为后续随机起点的固定中心。退出并重新启动后，需要重新示教。

### 基本命令

```bash
cd /home/embint/wzy/Evo-RLT

./scripts/record_franka_waypoint_demo.sh \
  --dataset franka_auto_demo_v1 \
  --fps 15 \
  --episodes 20 \
  --episode-time 300 \
  --move-speed 0.05 \
  --final-move-speed 0.02 \
  --start-square-side 0.15 \
  --grasp-square-side 0.10 \
  --start-yaw-range-deg 15 \
  --gripper-width 0.0 \
  --gripper-force 40
```

### 首次示教两个目标点和初始起点

启动后按照终端提示操作：

1. 将机械臂切换到 Guiding 状态。
2. 手拖到点1，即夹取位置，按第一次 `Enter` 记录。该位置作为后续抓取点随机采样的固定中心。
3. 手拖到点2，即放置或插入前位置，按 `Enter` 记录。
4. 手拖到初始起点，第三次按 `Enter` 记录。
5. 程序打印点1、点2、最终偏移点以及初始起点的6D TCP坐标。

选择三个示教点、等待 episode 开始以及 episode 结束后的保存/丢弃界面中，以下单键均可直接使用，不需要再按 `Enter`：

| 按键 | 功能 |
|---|---|
| `C` | 使用 `--gripper-width` 和 `--gripper-force` 闭合夹爪 |
| `O` | 完全打开夹爪 |
| `1` | 移动到当前点1；每轮结束后它会被新采样的抓取点覆盖 |
| `2` | 移动到点2 |
| `3` | 移动到固定的初始起点 |
| `4` | 移动到本次显示的 `NEXT START` |
| `Enter` | 确认当前示教点，或者从当前位姿开始 episode |
| `Q` | 退出数采 |

某个位置尚未记录或采样时，按对应数字键只会显示 `NULL`，不会发送运动指令。数字键运动发生在 episode 外，不会写入数据集。若 Franka 仍处于 Programming、Guiding或UserStopped状态，程序会提醒切换到 Execution并停留在当前等待界面；切换完成后再次按相同数字即可，不会因为这次拒绝而退出数采。开始 episode 时若仍处于上述状态，`Enter`同样只会提醒并等待再次确认。

最终点计算方式为：

```text
final_point = point2 + [0, -0.01, 0]
```

默认沿机器人基座坐标系的 `-Y` 方向移动 1 cm，不是沿末端工具坐标系移动。

### 每个 episode 的流程

1. 第一次 episode 的 `NEXT START` 就是第三次 `Enter` 记录的初始起点。
2. 按 `4` 让机械臂移动到 `NEXT START`，或者自行移动到希望的实际起始位姿。
3. 在 Franka Desk 中退出 Guiding，并切换到 Execution。
4. 按 `Enter`。
5. 程序保留等待阶段最后一次 `C/O` 操作后的夹爪状态，并开始录制：

```text
当前位置
  → 点1
  → 闭合夹爪
  → 点2
  → 沿基座坐标系 -Y 移动指定距离
```

执行期间按 `X` 不需要按 `Enter`，程序会立即请求停止当前机械臂或夹爪动作、丢弃当前 episode，然后为下一轮重新采样点1和起点，并返回非录制等待界面。最初记录的抓取中心、点2和初始起点不会被清除。

6. 每次轨迹完成后，程序同时生成下一轮的两个随机点：

   - 当前 `POINT 1`：以第一次 `Enter` 记录的抓取点为固定中心，在水平XY平面边长10 cm的正方形内均匀采样，即X/Y分别变化±5 cm；Z和完整末端姿态保持第一次抓取点的值。新结果会覆盖数字键 `1`。
   - `NEXT START`：以固定初始起点为中心，在水平XY平面边长15 cm的正方形内均匀采样；X/Y分别变化±7.5 cm，Z不变，并绕机器人基座Z轴随机偏航 `-15°～+15°`。

   两种采样始终相对各自最初记录的中心，不会相对上一轮随机点累积。
7. 程序在 episode 外打开夹爪，然后等待保存或丢弃；此时数字键和 `C/O` 仍可使用，移动不计入已经完成的 episode。

| 输入 | 功能 |
|---|---|
| `s` | 保存当前 episode；只有该按键可以保存 |
| `d` | 丢弃当前 episode，并重新采相同编号 |
| `q` | 丢弃当前 episode并退出 |
| `Enter` | 不执行保存或丢弃；提醒上一条 episode 尚未做决定并继续等待 |

自动打开动作不会写入刚完成的 episode；之后在结果界面进行的夹爪或机械臂操作也不会写入。

### 自动数采运动参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--move-speed` | `0.05` | TCP移动速度，单位 m/s |
| `--final-move-speed` | `0.05` | 点2到最终 `-Y` 点的独立速度，单位 m/s |
| `--final-minus-y` | `0.01` | 到达点2后沿基座 `-Y` 移动的距离，单位 m |
| `--start-square-side` | `0.15` | 以固定初始起点为中心的XY起点采样正方形边长，单位 m |
| `--grasp-square-side` | `0.10` | 以第一次记录的抓取点为中心的XY抓取采样正方形边长，单位 m |
| `--start-yaw-range-deg` | `15` | 新起点相对初始起点绕基座Z轴的最大偏航角；可设0–15°，设0关闭角度扰动 |
| `--gripper-width` | `0.0` | 点1处的夹取目标宽度，单位 m |
| `--gripper-force` | `40` | 点1处的持续夹取力，单位 N；允许范围20–70 N |
| `--abort-key` | `x` | episode 内立即停止动作并丢弃的单键，无需回车 |
| `--episode-time` | `300` | 单次自动执行的安全超时，单位秒 |

自动数采默认频率为 `15 Hz`。其中：

- 当前位置到点1使用 `--move-speed`。
- 点1到点2使用 `--move-speed`。
- 点2到最终 `-Y` 点使用独立的 `--final-move-speed`。

例如，将最终移动距离改成 8 mm、夹爪目标宽度改成 5 mm：

```bash
./scripts/record_franka_waypoint_demo.sh \
  --dataset franka_auto_demo_v2 \
  --fps 15 \
  --episodes 20 \
  --move-speed 0.05 \
  --final-move-speed 0.015 \
  --final-minus-y 0.008 \
  --gripper-width 0.005 \
  --gripper-force 30
```

### 自动数采数据语义

移动到点1、夹爪闭合、移动到点2和最终 `-Y` 运动均采用异步执行，因此运动过程中会按照指定 FPS 持续采集 RGB 和机器人状态。

保存的 action 与手动示教保持同一种格式：

```text
action[t] = observation[t] 到 observation[t+1] 的实测运动
```

数据集中保存的是机械臂实际走出的相邻帧增量和实测夹爪宽度，而不是直接把点1、点2坐标写成 action。

### 自动脚本采集的数据

每一帧主要包含：

| 数据 | 内容 |
|---|---|
| `observation.state` | 7维关节角、6维TCP位姿、7维关节速度、夹爪宽度、夹取状态，共22维 |
| `observation.images.wrist` | wrist RGB，`640×480` |
| `observation.images.front` | front RGB；相机以 `1920×1080` 采集，再将完整画面缩放为 `640×480`，不裁剪 |
| `action` | 6维相邻帧实测TCP增量，加下一帧实测夹爪宽度，共7维 |
| complementary info | policy action占位、intervention、phase和collector ID等统一schema字段 |
| 数据集字段 | task、frame index、episode index和按FPS生成的timestamp等 |

`gripper_grasped` 是 observation state 的最后一个值，由 Franka Hand 返回，记录为 `0.0` 或 `1.0`。

点1处使用的是 Franka `grasp` 命令，默认闭合力为 **40 N**。可以通过 `--gripper-force` 调整持续夹取力，脚本将其限制在20–70 N；`--gripper-width` 独立控制目标宽度。

front 原图宽高比为16:9，输出数据为4:3。完整画面直接缩放到 `640×480` 后不会丢失左右视野，但图像几何比例会发生拉伸。

该预处理与旧版“中心裁剪640×480”不同。不要用 `--resume` 把新缩放图像追加到旧裁剪数据集，应该创建新的 dataset 名称，避免同一数据集混入两种视觉分布。

三个机械臂移动阶段使用：

```python
move_tool(target_pose, speed=..., is_async=True)
```

因此它们是异步 `move_tool`，不是逐帧 `servo_tool`。底层会对目标TCP位姿做IK，并执行关节空间的 waypoint motion；主线程在运动期间以指定FPS持续采集 observation。因为是关节空间规划，两个TCP端点之间的真实路径不保证严格直线。

## 公共参数

两种快捷方式都支持以下主要参数：

| 参数 | 含义 |
|---|---|
| `--dataset NAME` | 数据集名称/repo ID |
| `--root PATH` | 本地数据集目录 |
| `--fps HZ` | 采集频率，目前要求正整数 |
| `--episodes N` | 需要保存的 episode 数量；丢弃的 episode 不计数 |
| `--episode-time SEC` | 单个 episode 的最大时长或安全超时 |
| `--task TEXT` | 写入数据集的任务描述 |
| `--resume` | 继续已有数据集 |
| `--robot-ip IP` | Franka IP |
| `--wrist-camera SERIAL` | wrist RealSense 序列号 |
| `--front-camera SERIAL` | front RealSense 序列号 |
| `--collect-python PATH` | 数采程序使用的 Python |
| `--control-python PATH` | control server 使用的 Python |
| `--control-log PATH` | control server 日志路径 |

查看完整帮助：

```bash
./scripts/record_franka_manual_demo.sh --help
./scripts/record_franka_waypoint_demo.sh --help
```

## 安全注意事项

- 自动模式运行前，确认点1、点2以及点2之后的 `-Y` 路径没有障碍物。
- 示教点保存的是完整位置和姿态；不要只检查 XYZ，也要留意末端姿态。
- 自动运动开始前必须退出 Guiding 状态，并确认 Franka 已允许运动。
- `move_tool` 在关节空间规划，TCP在两个端点之间可能不是严格直线，尤其应检查高处障碍和奇异位形。
- 第一次正式批量采集前，建议使用较低速度和少量 episode 做验证，例如 `--move-speed 0.02 --episodes 1`。
- `--gripper-width 0.0` 表示尝试完全闭合。夹取有厚度的物体时，可以设置接近物体厚度的目标宽度。
- `--gripper-force` 是持续夹持参数，默认40 N；只在确认物体和夹具能承受时提高夹取力。
- 如果自动运动、夹爪、相机或 server 出现异常，不要继续使用同一进程；退出、检查日志并重新启动。

control server 日志默认位于：

```text
/tmp/evo-rlt-franka-control-server.log
```
