# ACT 数采、推理与 Stage 2：起点和执行流程

本文对照 2026-10-12 的代码整理。命令从仓库根目录执行，路径请替换为实际数据集和配套 checkpoint。

## 1. 坐标与起点概览

所有位置、偏移和运动方向均采用**机器人基坐标系**。位置单位为米；TCP 位姿中的后三维为旋转向量，单位为弧度。

共享采样常量见 [sampling.py](data_collection/sampling.py)：

- 固定起点高度：`SAMPLE_Z_M = 0.402`，是绝对 Z，不是相对参考点抬高 0.402 m。
- 固定起点姿态：内禀 XYZ 欧拉角 `(179.7°, 1.4°, 90.6°)`，转换为旋转向量后发送给机器人。
- 数采默认插入距离：沿基坐标系 −Z 前进 0.006 m（0.6 cm）。

| 路径 | 启动时确定的参考 | episode 起点 X/Y | 起点 Z / 姿态 |
| --- | --- | --- | --- |
| ACT 数采 | 手动示教完整参考位姿 p0 | X=p0.X+[0, 2.5] cm；Y=p0.Y+[0.3, 2] cm | Z=0.402 m；固定采样姿态 |
| ACT infer | 按 Enter 时的实测 TCP X/Y 中心 | X/Y 各在中心 ±1 cm 内均匀采样 | Z=0.402 m；固定采样姿态 |
| train Stage 2 | 启动采集的 p1 参考位姿；续训恢复保存值 | 默认半范围为 0，固定在当前 reference 的 X/Y；可用 =/- 调整范围 | Z=0.402 m；固定采样姿态 |
| infer Stage 2 | 选择训练参考位姿或当前 TCP | 固定起点；可用 WASD 微调 | 保留确认后的完整起点位姿 |

**数采的 p0 是轨迹经过的参考点；Stage 2 的 p1 是起点生成的 XY 基准。两者不是程序自动关联的同一个点。**当前数采、ACT infer 和 Stage 2 的起点分布也不同，比较效果时应记录实际起点。

## 2. ACT 数采

入口：[record_sample_space.py](data_collection/record_sample_space.py)；也可使用 [collect_sample_space.sh](data_collection/collect_sample_space.sh)。

### 起点与轨迹

1. 手动引导机器人到参考点 p0，按 Enter 保存完整 TCP 位姿，包括位置和姿态。
2. 每个 episode 均匀采样一个起点：
   - `X = p0.X + U(0, 0.025)`。
   - `Y = p0.Y + U(0.003, 0.02)`。
   - `Z = 0.402`。
   - 使用固定采样姿态。
3. 移动到起点，默认等待 1 s，然后开始录制。
4. 录制两段运动：**采样起点 → 完整 p0 位姿 → 相对 p0 沿基坐标系 −Z 移动 0.6 cm、同时沿 −Y 移动 0.1 cm 的终点**；插入时姿态保持 p0 的值。
5. 成功完成录制后保存 episode；默认等待 1 s，再回到 p0，继续下一次采样。

移动到起点和保存后的回退不属于示范轨迹。由于起点使用固定姿态，而 p0 保留示教姿态，第一段运动可能同时包含平移和旋转。

### 录制流程与数据

这条路径通过机器人运动接口执行预设轨迹，并采集相机、机器人状态和力反馈，不调用 ACT 推理。动作标签由相邻实测 TCP 位姿生成，包含 TCP 位移/旋转增量与夹爪宽度；运动目标和速度另存为 command 信息。

默认采集频率为 15 Hz，移动速度为 0.02 m/s，插入速度为 0.01 m/s。起点随机种子可通过 `--seed` 指定。暂停界面支持 `1` 返回 p0、`2` 返回当前采样点；活动录制默认用 `x` 中止。

```bash
python -m act_rlt.data_collection.record_sample_space \
  --dataset act_demo --root datasets/ethernet_fine/act_demo \
  --wrist-only --fps 30 --episodes 10
```

`--wrist-only` 使用 1280×720 腕部图像；`--front-only` 使用 640×480 前置图像。未指定单相机选项时采集双相机。

## 3. ACT infer

入口：[infer.py](infer.py)。

### 支持的模型输入与输出

infer 根据 checkpoint 配置选择输入：无 `observation.state` 时仅输入相机；6 维 state 表示原始基座坐标系 wrench `[fx, fy, fz, mx, my, mz]`（N / Nm）；7 维 state 表示关节位置。wrench 与图像在同一观测采集线程读取，使用 checkpoint 保存的 preprocessor 归一化，不额外清零或变换坐标系。相机与 wrench 是顺序读取，并非严格时间同步。

1 维 state 对应当前 TCP 的绝对 Z 高度：直接取机器人观测中的 `ee_z`（基座坐标系，米），由 checkpoint 保存的 processor 做均值/标准差归一化。该输入与数采、RGB＋Z ACT 训练一致；不是固定采样高度、相对起点高度或动作 `dz`。缺失、非标量或非有限 Z 会拒绝推理。

动作支持 6 维 TCP 增量，以及附带夹爪宽度预测的 7 维动作。两者均支持默认 temporal ensemble 和 `--no-temporal-ensemble`；7 维动作的夹爪预测仍只记录、不执行。纯相机模型自动应用项目已有的 LeRobot latent device 兼容修复。

```bash
# 纯相机；--dry-run 仍会连接机器人与相机，但不发送运动指令
python -m act_rlt.infer --checkpoint outputs/ethernet_act/act_dataset_928_camera_only_action6_act --dry-run --dry-run-episodes 1

# 相机 + 六维 wrench
python -m act_rlt.infer --checkpoint outputs/ethernet_act/act_1010_rgb_wrench_action6_act --dry-run --dry-run-episodes 1

# 相机 + 当前 TCP 绝对 Z；自动加载本机 last 指向的 20k 模型
python -m act_rlt.infer \
  --checkpoint outputs/ethernet_act/act_dataset_928_addon_rgb_z_action6_act \
  --no-temporal-ensemble --n-action-steps 4 --fps 30 --duration 10
```

### 起点与重置

启动后加载 ACT checkpoint 和保存的 pre/postprocessor，连接机器人，用真实观测预热模型。运动模式下，按 Enter 采集当前完整 TCP 参考位姿，workspace 设为参考位置 XYZ 各 ±5 cm；起点 Z 和姿态使用上述固定采样常量。

XY 采样半范围默认 **0 cm**。`=` / `-` 以 0.5 cm 步长调整半范围（0–5 cm），只影响下次采样；`r` 重新采样并移动。首次移动前需确认，抵达起点后按 Enter/y 开始推理，`q` 退出；`n` 也可重采样。首次移动确认时调整范围会重新生成目标。

首次移动确认、等待开始和回合结束菜单中，均可按住 WASD 连续移动（W=-X、S=+X、A=-Y、D=+Y，3 cm/s），松开停止。选定菜单操作时先停止 servo，再用实测 TCP 更新 reference 和当前起点；之后的随机采样围绕新的 XY。首次确认菜单若发生移动，会重新显示更新后的采样目标供确认。S 仅用于移动，开始推理请按 Enter/y。键盘后端默认 `auto`（X11 用 pynput，否则 evdev），可通过 `--teleop-input-backend` 和 `--teleop-keyboard` 指定。

加 `--use-reference-pose` 时，起点 Z 和姿态也使用参考点，并随 WASD 后的实测参考更新，同时取消 XYZ workspace 限制；不加时仍使用固定采集 Z/姿态。

episode 结束后：

- `r`：先从当前实测 TCP 沿基座 +Z 上抬 1 cm，再返回本次 episode 保存的实测起点。
- `n`：从结束位姿先沿 +Z 抬高 1 cm，再移动到新的采样点。
- `q`：退出。

未启用 `--use-reference-pose` 时，workspace 保持启动确认的 XYZ ±5 cm，WASD 和采样均受该范围约束；回退 +Z 1 cm 前也检查 workspace，越界则停止移动。回合结束菜单的 `r` 表示返回当前保存的起点（WASD 移动后更新），`n` 表示采新点。

### 推理与控制流程

```text
相机持续采集 → 选择更新的缓存观测
  → 图像转换 + checkpoint preprocessor
  → 完整 ACT 输出 action chunk
  → postprocessor 还原物理动作
  → temporal ensemble 或动作队列
  → 单步平移/旋转限幅 → TCP 指令
```

相机、推理和伺服由独立线程运行。默认控制频率 30 Hz、episode 时长 10 s：

- 默认启用 temporal ensemble，系数为 0.01，融合不同预测中对应同一执行时刻的动作。
- 使用 `--no-temporal-ensemble` 时，执行 `n_action_steps` 个动作；队列剩余动作达到半段长度时触发补充推理。
- 默认每步将 TCP 目标重新锚定到实测位姿。`--residual-control` 可启用未完成 XYZ 位移的累积，带 ±0.3 mm 残差边界和 Z 停滞保护。
- 默认每步平移增量范数上限 2 mm，旋转增量范数上限 0.02 rad；夹爪保持，不重复发送夹爪指令。

```bash
python -m act_rlt.infer \
  --checkpoint outputs/ethernet_act/act_dataset_928_state7_act \
  --fps 30 --duration 10
```

此入口的 `--dry-run` 仍会连接机器人和相机、运行推理，但不发送动作，也不执行起点移动。

### 提前终止与耗时统计

在交互终端运行 infer 时，按 **`s`** 提前结束本轮并标记 success，无需回车或额外启动参数。`--duration` 仍是最大控制时长。

```bash
python -m act_rlt.infer \
  --checkpoint outputs/ethernet_act/act_dataset_928_state7_act \
  --fps 30 --duration 10
```

infer 过程中按 **`x` 或 `Esc`** 可停止本轮并标记 fail，无需按 Enter。程序按约 20 ms 间隔检查 `s / x / Esc` 按键，收到后停止动作发布并调用 `stop_servo()`，然后返回 `r / n / q` 菜单；实际停止还取决于控制服务响应。按 `Ctrl+C` 则停止并退出程序。

每轮结束都会输出 `elapsed`（本轮推理耗时）、`run_average`（本次进程计入统计的轮次平均耗时）、轮数和结束原因。按 `s` 结束会记录 `end_reason=operator_success`，并计入平均耗时。通过 `x`、`Esc` 或 `Ctrl+C` 停止的本轮仍记录耗时，但**不计入平均值及平均耗时的统计轮数**；日志记录 `end_reason=operator_stop` 和 `included_in_run_average=false`。尚无计入统计的轮次时，平均值显示 `N/A`。其他正常结束、dry-run 和异常结束的轮次仍计入平均值。

每轮还输出本次 run 的 `success/total episodes` 和 `fail` 数量。按指定规则，正常达到时长或按 `s` 提前结束计为 success；`x`、`Esc`、`Ctrl+C` 停止及异常结束计为 fail。所有这些轮次都计入 total，包括不计入平均耗时的手动停止轮次。JSONL 的 `episode_end` 记录 `episode_success`、`run_success_episodes`、`run_total_episodes` 和 `run_fail_episodes`。重启进程后计数归零；此 success 来自结束规则或人工标记。

计时从启动本轮相机/推理/控制线程到停止伺服并同步 TCP 目标之后，包含首个预测等待与停止确认，不包含起点移动、人工交互、启动预热以及后续线程回收/GC。默认 JSONL 日志的 `episode_end` 保存 `inference_duration_s`、`control_duration_s`、`run_average_inference_s`、`run_episode_count` 和 `end_reason`；按 `s` 时还记录 `operator_success` 事件。重启进程后平均值重新计算。

### 相机 + Z 的 RL Token 训练（Stage 1）

训练入口自动读取 ACT 的输入配置，支持一维 TCP 绝对 Z（米）；使用配套的相机 + Z 数据集。ACT 编码器冻结，只训练 RL Token 重建模块。

```bash
python -m act_rlt.train_rl_token \
  --root datasets/ethernet_fine/act_dataset_928_addon_rgb_z_action6 \
  --act-checkpoint outputs/ethernet_act/act_dataset_928_addon_rgb_z_action6_act \
  --output outputs/ethernet_token/act_dataset_928_addon_rgb_z_action6_stage1_pos
```

默认训练 10,000 步，batch size 8，每 2,000 步保存。断点续训使用 `--output <原输出目录> --resume`，恢复保存的训练配置、优化器和调度器状态。新训练和续训都会启用一维 Z 的数据读取兼容。

## 4. train Stage 2

入口：[train_stage2.py](train_stage2.py)；学习循环见 [stage2.py](stage2.py)，实时执行见 [stage2_runtime.py](stage2_runtime.py)。

### Z 插入模式的随机起点

Stage 2 训练默认采用 Z 插入配置，已移除模式选择参数。新训练在启动时按 Enter 采集最初的 reference pose；workspace 限制已取消，旧 checkpoint 中的边界也不再生效。默认起始 XY 半范围为 **0**，固定在当前 reference 的 X/Y；通过 `=` 调大范围后，在该范围内均匀采样，Z 固定为 0.402 m，姿态固定为采样姿态。三个固定 reset 点及 `1/2/3` 选择已移除。XY 采样无需额外参数；半范围为 0 时固定 XY。

每轮开始前（包括首次移动确认界面），可直接按键，无需回车：

| 按键 | 行为 |
| --- | --- |
| `=` | 下一次 X/Y reset 半范围增加 0.5 cm |
| `-` | 下一次 X/Y reset 半范围减少 0.5 cm |
| `r` | 按当前设置重新采样并移动 |
| Enter / `y` | 从当前已抵达的位置开始 episode |
| `q` | 停止 |

半范围限制为 **0–5 cm**，初始为 0；例如 2 cm 表示 X/Y 各在 reference ±2 cm 内采样。范围为 0 时，X/Y 固定为 reference 的 X/Y。按 `=` / `-` 不移动当前 TCP，新范围用于下一次 reset 或按 `r` 重采样，并在本次进程后续 episode 中保留。首次移动确认时调整范围会重新生成待确认的目标。

在起点界面用 WASD 移动后，结束该次输入时先停止伺服，再读取实际 TCP XY 更新当前 reference；后续 reset 和 `r` 重采样围绕更新后的 XY。仅更新 XY，reference 的 Z/姿态保持不变。未进行 WASD 移动时，随机 reset 或 episode 动作不会改变 reference。首次移动确认前的 WASD 调整也会更新待移动目标 XY，避免返回旧位置。当前 reference 和范围随 checkpoint 保存，并同步写入 `startup_reference.json`；续训恢复当前 reference，最初采集的信息仍保存在 `initial_reference` 中，旧 workspace 限制不恢复。

首次移动需确认；每轮结束后沿当前实测位置 +Z 回退 1 cm。reset、episode 动作和 WASD 均不受 workspace 限制；单步动作限幅及 WASD 会话距离限制仍保留。

### Episode 控制

Critic 默认使用 **1 chunk TD**。添加 `--three-chunk-td` 才开启 **3 chunk TD**：累计连续三个 chunk（通常 12 个控制 step）的折扣奖励，从该跨度末端的状态 bootstrap。episode 提前结束时缩短跨度并停止 bootstrap；尚未收集满三个 chunk 的非终止样本等待后续数据。Actor/Critic 的动作输入仍为一个四步 chunk，BC 目标也只对应第一个 chunk。重叠 replay 按 episode 内真实 step 索引衔接；旧 v3 checkpoint 可直接续训并保留网络和优化器。新训练和续训均以命令行开关为准：传入 `--three-chunk-td` 使用 3 chunk，不传则使用 1 chunk，即使 checkpoint 原先使用 3 chunk。

- online 采集使用带高斯探索噪声的 Actor；warmup 使用 ACT 参考动作。
- 成功终止奖励为 +1，其余步骤为 0。按 `s/f` 在当前 chunk 结束后标记成功/失败，`q` 停止；超时后也会询问结果。奖励时刻应尽量贴近实际完成时刻。
- `--enable-human-intervention` 使用 SpaceMouse 六轴人工接管，Space 在 chunk 边界接管/归还，S/F 标记结果并保留 replay，D 结束并删除整轮 replay，Esc 停止。episode 内移动键停用，起点界面保留 WASD。默认每轴平移 13.5 mm/s、roll/pitch 0.09 rad/s、yaw 0.108 rad/s（角速度参数的 1.2 倍）、死区 0.05；可通过 `--teleop-speed-m-s`、`--teleop-angular-speed-rad-s`、`--spacemouse-deadzone` 调节。训练自动启动独立 ROS 输入进程，请勿同时运行独立 SpaceMouse 控制脚本；设备默认 `/dev/hidraw4`，可设置 `SPACEMOUSE_DEVICE`。松开设备再按 Space 归还策略；未请求切换时不因接管开关额外等待，下一轮观测/推理与第 4 步剩余周期重叠。边界切换判定之后收到的 Space 留到下一 chunk 边界处理。输入超过 0.2 s 未更新时停止人工运动。人工实际平移及转换后的局部旋转增量写入 replay，并作为 BC 目标。

相机、推理、伺服持续运行，学习在调用线程中进行。推理使用发布的 Actor 快照，避免读取正在更新的参数。固定 chunk 不做 temporal ensemble，也不在 chunk 中途重新规划；沿用每步实测 TCP 锚定。

当前 `--observation-selection` 默认是 `latest`：使用比上次选中观测更新的最新缓存观测，不要求曝光晚于最后一条动作指令。`boundary` 模式要求观测 freshness 位于 chunk 请求边界之后。`--camera-boundary-mode` 默认 `timestamp`，启用机器人提供的时间戳观测路径；`legacy` 使用普通观测。下一观测的编码同时用作 replay 的 next state 和下一 chunk 的 current state。预测未就绪时保持最后目标，不重发相对动作或突发补发。

```bash
python -m act_rlt.train_stage2 \
  --stage1-checkpoint outputs/ethernet_token/act_dataset_928_state7_stage1_pos \
  --act-checkpoint outputs/ethernet_act/act_dataset_928_state7_act \
  --output outputs/ethernet_rlt/stage2_new_run \
  --fps 30 --allow-motion
```

加 `--use-reference-pose` 后，reset 的 Z 和姿态使用首次采集或续训恢复的 reference；XY 继续按现有半范围采样（新训练默认 0）。不加时仍固定 Z=0.402 m 和采集姿态。该参数以本次启动命令为准，续训需要继续传入；checkpoint 的 startup reference 会记录当前选择和实际 reset Z/姿态。

Online 更新中每更新 2 次 Critic，更新 1 次 Actor（`actor_update_interval=2`）；默认每个保留 chunk 更新 5 次 Critic，Actor 按该间隔更新。续训也采用当前的更新频率，历史更新计数保留。warmup 后的 BC 初始化、Critic 初始化阶段保持原规则。

默认 30 Hz，可通过 `--fps` 调整，episode 默认 5 s。`--total-env-steps` 是累计 online 步数目标，不包含 warmup。`--dry-run` 只检查模型和 processor，不连接机器人。

## 5. infer Stage 2

入口：[infer_stage2.py](infer_stage2.py)。直接复用训练的 `FrankaInsertionStage2Env` 和 `FixedChunkRuntime`。

### 起点是否与训练相同

infer 默认使用 Z insertion 行为，默认 30 Hz，结束沿 +Z 回退 1 cm。已移除 `--z-insertion-mode` 参数和 workspace 限制：选择 `t` 仅恢复训练参考位姿，选择 `c` 立即采集当前 TCP，不生成或恢复空间边界。

infer 无随机采样。WASD 调整后按 Enter/y 开始，保存实际完整位姿作为后续起点。WASD 仍保留单次键盘会话距离起点 15 cm 的独立限制。

### 推理流程

1. 加载已初始化的 v3 Stage 2 checkpoint，从保存的配置解析配套 ACT 和 Stage 1 checkpoint，加载 Actor 权重及 ACT processor。
2. 连接机器人、确定 workspace 和起点，移动到选定点，Enter 开始 episode。
3. 完整 ACT 生成参考 chunk，同时提取 encoder hidden 和 position；RL Token 编码后，根据 ACT 输入契约拼接归一化 TCP Z、关节状态或 wrench；纯相机模型只使用 RL Token。
4. Actor 输入状态与参考 chunk，执行**确定性均值动作**，不添加训练时的探索噪声。
5. postprocessor 还原动作、进行物理限幅，按固定 chunk 执行；更新观测并生成下一 chunk。
6. 成功/失败、超时或退出时停止当前 episode、回退；有下一 episode 时重新进入起点选择流程。

推理冻结全部参数，不更新 Actor/Critic，不写入训练 replay。默认 `latest` 观测选择、`timestamp` 相机模式、四步 chunk，无 temporal ensemble。每步 TCP 锚定、起点选择和回退方式与训练环境共用。

```bash
python -m act_rlt.infer_stage2 \
  --checkpoint outputs/ethernet_rlt/stage2_new_run \
  --fps 30 --episodes 10 \
  --episode-time 5 --allow-motion
```

`--checkpoint` 可指向 run 目录或具体 `.pt` 文件；run 目录解析为 `checkpoints/latest.pt`。配套模型路径变化时可传入 `--act-checkpoint` 和 `--stage1-checkpoint`。`--dry-run` 只加载检查模型，不连接机器人。

起点确认界面的 WASD 使用 `--teleop-input-backend auto`：有 X11 `DISPLAY` 时选择 pynput，无图形会话时选择 evdev。ToDesk/X11 可显式传 `--teleop-input-backend pynput`。使用实体键盘且检测到多个设备时，传入 `--teleop-input-backend evdev --teleop-keyboard /dev/input/eventN` 指定实际使用的设备；指定设备时 auto 也会选择 evdev。键盘检查在连接机器人前完成，`--dry-run` 不检查键盘。

## 6. 图像与对比实验

### 新的六维 action 模型进行 Stage 2 训练

Stage 2 根据 ACT checkpoint 自动选择输入：纯相机为 768 维 RL Token；相机＋TCP Z 为 768＋1 维；相机＋wrench 为 768＋6 维；旧 joint-state 模型为 768＋7 维。TCP Z 使用当前观测中的基坐标系实测绝对高度（米），通过 ACT 保存的 processor 归一化后同时输入 ACT 和 Actor/Critic，不使用固定重置高度或相对位移。wrench 使用与 ACT 训练一致的基坐标系原始六维外力/力矩，再通过 ACT 保存的 processor 归一化。纯相机模型的 Actor/Critic 不加入关节状态或 wrench 分支。支持 ACT 的六维 delta 或旧七维 delta＋gripper 输出；机械臂执行六维 delta。上述输入在 Stage 2 train 和 infer 中共用；不同状态维度的 Stage 2 checkpoint 不能互相续训。

相机＋TCP Z（合并数据集，ACT 20k / RL Token 10k）：

```bash
python -m act_rlt.train_stage2 \
  --stage1-checkpoint outputs/ethernet_token/act_dataset_928_addon_rgb_z_action6_stage1_pos \
  --act-checkpoint outputs/ethernet_act/act_dataset_928_addon_rgb_z_action6_act \
  --output outputs/ethernet_rlt/rlt_928_addon_rgb_z_action6 \
  --allow-motion \
  --warmup-steps 2000 --total-env-steps 50000 --episode-time 15
```

仅加载验证可去掉 `--allow-motion`，加 `--dry-run`（无 GPU 时加 `--device cpu`），不连接机器人或创建训练结果。

纯相机：

```bash
python -m act_rlt.train_stage2 \
  --stage1-checkpoint outputs/ethernet_token/act_dataset_928_camera_only_action6_stage1_pos \
  --act-checkpoint outputs/ethernet_act/act_dataset_928_camera_only_action6_act \
  --output outputs/ethernet_rlt/rlt_camera_only_action6 \
  --allow-motion \
  --warmup-steps 2000 --total-env-steps 50000 --episode-time 15
```

相机＋六维 wrench：

```bash
python -m act_rlt.train_stage2 \
  --stage1-checkpoint outputs/ethernet_token/act_1010_rgb_wrench_action6_stage1_pos \
  --act-checkpoint outputs/ethernet_act/act_1010_rgb_wrench_action6_act \
  --output outputs/ethernet_rlt/rlt_rgb_wrench_action6 \
  --allow-motion \
  --warmup-steps 2000 --total-env-steps 50000 --episode-time 15
```

两者均需新建 Stage 2 run，不能复用旧 joint-state Stage 2 的 Actor/Critic 或 replay。默认四步 chunk、sigma 探索和起点 XY 随机逻辑沿用现有设置。加 `--dry-run --device cpu` 可仅检查模型加载，不连接机械臂。

### 图像预处理

ACT 和 Stage 2 均按 ACT checkpoint 的相机键与图像尺寸处理输入。腕部相机保持配置的采集尺寸；前置相机输出统一为 640×480。图像转为 CHW、像素除以 255，再通过 checkpoint 的 processor 标准化。当前 ACT 路径不自动缩小为 224×224。

对比 ACT 与 Stage 2 时应记录起点绝对位姿、控制频率、观测选择、temporal ensemble、residual control、单步限幅、成功标记时刻以及实际完成时间。ACT 默认 temporal ensemble，Stage 2 固定 chunk；数采和两类推理的起点分布也不同。这些差异都会影响任务完成速度和成功率。


### Episode replay 保留 / 删除

episode 内按 S（成功）或 F（失败）在当前 4-step chunk 结束后自动保留数据，
没有二次确认。按 D 则在当前 chunk 结束后结束本轮并丢弃整轮数据；普通
终端模式可用小写 d，SpaceMouse 接管的键盘监听也支持 D。超时后选择结果时
同样可输入 d 删除。起点菜单的 D 仍用于 WASD 移动，不会删除数据。

为保证删除的数据未参与训练，本轮所有 transition 暂存在独立列表中，
不进入 replay；S/F 确认后一次性提交，并在机器人停止后按保留 chunk 数
执行对应更新（默认每 chunk 5 次 Critic）。episode 内不更新 Actor/Critic，
下一轮使用更新后的 Actor。D 会清除普通 chunk 和全部重叠窗口，不产生
本轮训练更新，也不计入 warmup/online 有效步数。实际执行总步数、episode
数及日志仍记录，删除时终端输出 `REPLAY=DELETED`。

中途停止或异常退出时，尚未提交的本轮缓存不会写入 checkpoint。
达到 online 总预算而暂停的最后一段默认提交，不弹出额外确认。
