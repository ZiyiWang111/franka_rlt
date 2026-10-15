# ACT 数采、推理与 Stage 2：起点和执行流程

本文对照 2026-10-12 的代码整理。命令从仓库根目录执行，路径请替换为实际数据集和配套 checkpoint。

## 1. 坐标与起点概览

所有位置、偏移和运动方向均采用**机器人基坐标系**。位置单位为米；TCP 位姿中的后三维为旋转向量，单位为弧度。

共享采样常量见 [sampling.py](data_collection/sampling.py)：

- 固定起点高度：`SAMPLE_Z_M = 0.402`，是绝对 Z，不是相对参考点抬高 0.402 m。
- 固定起点姿态：内禀 XYZ 欧拉角 `(179.7°, 1.4°, 90.6°)`，转换为旋转向量后发送给机器人。
- 数采默认插入距离：沿基坐标系 −Z 前进 0.01 m。

| 路径 | 启动时确定的参考 | episode 起点 X/Y | 起点 Z / 姿态 |
| --- | --- | --- | --- |
| ACT 数采 | 手动示教完整参考位姿 p0 | X=p0.X±2 cm；Y=p0.Y+[1.5, 2.5] cm | Z=0.402 m；固定采样姿态 |
| ACT infer | 按 Enter 时的实测 TCP X/Y 中心 | X/Y 各在中心 ±1 cm 内均匀采样 | Z=0.402 m；固定采样姿态 |
| train Stage 2，Z 插入模式 | 启动采集的 p1 参考位姿 | 默认围绕首次 reference 的 X/Y 随机采样，范围可用 =/- 调整 | Z=0.402 m；固定采样姿态 |
| infer Stage 2，Z 插入模式 | 默认恢复 checkpoint 中首次训练采集的 p1 参考位姿；无记录时手动采集 | 与训练共用随机 XY reset；范围可用 =/- 调整 | Z=0.402 m；固定采样姿态 |
| train / infer Stage 2，普通模式 | 显式 workspace，以及首次重置时的实测姿态 | 在 sample box 内采样；未指定时用 workspace 并留 XY 边距 | Z 在采样范围内随机；保留首次实测姿态 |

**数采的 p0 是轨迹经过的参考点；Stage 2 的 p1 是起点生成的 XY 基准。两者不是程序自动关联的同一个点。**当前数采、ACT infer 和 Stage 2 的起点分布也不同，比较效果时应记录实际起点。

## 2. ACT 数采

入口：[record_sample_space.py](data_collection/record_sample_space.py)；也可使用 [collect_sample_space.sh](data_collection/collect_sample_space.sh)。

### 起点与轨迹

1. 手动引导机器人到参考点 p0，按 Enter 保存完整 TCP 位姿，包括位置和姿态。
2. 每个 episode 均匀采样一个起点：
   - `X = p0.X + U(-0.02, 0.02)`。
   - `Y = p0.Y + U(0.015, 0.025)`。
   - `Z = 0.402`。
   - 使用固定采样姿态。
3. 移动到起点，默认等待 1 s，然后开始录制。
4. 录制两段运动：**采样起点 → 完整 p0 位姿 → p0 沿 −Z 移动 1 cm 的终点**。
5. 成功完成录制后保存 episode；默认等待 1 s，再回到 p0，继续下一次采样。

移动到起点和保存后的回退不属于示范轨迹。由于起点使用固定姿态，而 p0 保留示教姿态，第一段运动可能同时包含平移和旋转。

### 录制流程与数据

这条路径通过机器人运动接口执行预设轨迹，并采集相机、机器人状态和力反馈，不调用 ACT 推理。动作标签由相邻实测 TCP 位姿生成，包含 TCP 位移/旋转增量与夹爪宽度；运动目标和速度另存为 command 信息。

默认采集频率为 15 Hz，移动速度为 0.02 m/s，插入速度为 0.01 m/s。起点随机种子可通过 `--seed` 指定。暂停界面支持 `1` 返回 p0、`2` 返回当前采样点；活动录制默认用 `x` 中止。

```bash
python -m act_rlt.data_collection.record_sample_space \
  --dataset act_demo --root datasets/act_demo \
  --wrist-only --fps 30 --episodes 10
```

`--wrist-only` 使用 1280×720 腕部图像；`--front-only` 使用 640×480 前置图像。未指定单相机选项时采集双相机。

## 3. ACT infer

入口：[infer.py](infer.py)。

### 起点与重置

启动后加载 ACT checkpoint 和保存的 pre/postprocessor，连接机器人，用真实观测预热模型。运动模式下，按 Enter 采集当前完整 TCP 参考位姿，workspace 设为参考位置 XYZ 各 ±5 cm；起点 Z 和姿态使用上述固定采样常量。

XY 采样半范围默认 **0 cm**，起点 XY 固定为首次参考点。`=` / `-` 以 0.5 cm 步长调整半范围（0–5 cm），只影响下次采样；`r` 重新采样并移动。首次移动前需确认，抵达起点后按 Enter/y 开始推理，`q` 退出；保留 `s` 开始、`n` 重采样兼容按键。首次移动确认时调整范围会重新生成目标。

episode 结束后：

- `r`：先从当前实测 TCP 沿基座 +Z 上抬 1 cm，再返回本次 episode 保存的实测起点。
- `n`：从结束位姿先沿 +Z 抬高 1 cm，再移动到新的采样点。
- `q`：退出。

采样中心始终固定在首次参考点；采样区域须位于 workspace 内。回退 +Z 1 cm 前也检查 workspace，越界则停止移动。回合结束菜单的 `r` 仍表示返回本回合起点，`n` 表示采新点。

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
  --checkpoint outputs/act_dataset_928_state7_act \
  --fps 30 --duration 10
```

此入口的 `--dry-run` 仍会连接机器人和相机、运行推理，但不发送动作，也不执行起点移动。

### 提前终止与耗时统计

在交互终端运行 infer 时，按 **`s`** 提前结束本轮并标记 success，无需回车或额外启动参数。`--duration` 仍是最大控制时长。

```bash
python -m act_rlt.infer \
  --checkpoint outputs/act_dataset_928_state7_act \
  --fps 30 --duration 10
```

infer 过程中按 **`x` 或 `Esc`** 可停止本轮并标记 fail，无需按 Enter。程序按约 20 ms 间隔检查 `s / x / Esc` 按键，收到后停止动作发布并调用 `stop_servo()`，然后返回 `r / n / q` 菜单；实际停止还取决于控制服务响应。按 `Ctrl+C` 则停止并退出程序。

每轮结束都会输出 `elapsed`（本轮推理耗时）、`run_average`（本次进程计入统计的轮次平均耗时）、轮数和结束原因。按 `s` 结束会记录 `end_reason=operator_success`，并计入平均耗时。通过 `x`、`Esc` 或 `Ctrl+C` 停止的本轮仍记录耗时，但**不计入平均值及平均耗时的统计轮数**；日志记录 `end_reason=operator_stop` 和 `included_in_run_average=false`。尚无计入统计的轮次时，平均值显示 `N/A`。其他正常结束、dry-run 和异常结束的轮次仍计入平均值。

每轮还输出本次 run 的 `success/total episodes` 和 `fail` 数量。按指定规则，正常达到时长或按 `s` 提前结束计为 success；`x`、`Esc`、`Ctrl+C` 停止及异常结束计为 fail。所有这些轮次都计入 total，包括不计入平均耗时的手动停止轮次。JSONL 的 `episode_end` 记录 `episode_success`、`run_success_episodes`、`run_total_episodes` 和 `run_fail_episodes`。重启进程后计数归零；此 success 来自结束规则或人工标记。

计时从启动本轮相机/推理/控制线程到停止伺服并同步 TCP 目标之后，包含首个预测等待与停止确认，不包含起点移动、人工交互、启动预热以及后续线程回收/GC。默认 JSONL 日志的 `episode_end` 保存 `inference_duration_s`、`control_duration_s`、`run_average_inference_s`、`run_episode_count` 和 `end_reason`；按 `s` 时还记录 `operator_success` 事件。重启进程后平均值重新计算。

## 4. train Stage 2

入口：[train_stage2.py](train_stage2.py)；学习循环见 [stage2.py](stage2.py)，实时执行见 [stage2_runtime.py](stage2_runtime.py)。

### Z 插入模式的随机起点

开启 `--z-insertion-mode` 后，新训练在启动时按 Enter 采集最初的 reference pose；workspace 为该位置 XYZ 各 ±5 cm。默认以首次 reference 的 X/Y 为中心，X/Y 各在 **±1 cm** 内均匀采样，Z 固定为 0.402 m，姿态固定为采样姿态。三个固定 reset 点及 `1/2/3` 选择已移除。`--randomize-reset-xy` 仅保留为旧命令兼容选项，无需传入。

每轮开始前（包括首次移动确认界面），可直接按键，无需回车：

| 按键 | 行为 |
| --- | --- |
| `=` | 下一次 X/Y reset 半范围增加 0.5 cm |
| `-` | 下一次 X/Y reset 半范围减少 0.5 cm |
| `r` | 按当前设置重新采样并移动 |
| Enter / `y` | 从当前已抵达的位置开始 episode |
| `q` | 停止 |

半范围限制为 **0–5 cm**，初始为 1 cm；例如 2 cm 表示 X/Y 各在 reference ±2 cm 内采样。范围为 0 时，X/Y 固定为 reference 的 X/Y。按 `=` / `-` 不移动当前 TCP，新范围用于下一次 reset 或按 `r` 重采样，并在本次进程后续 episode 中保留。首次移动确认时调整范围会重新生成待确认的目标。

采样中心始终为首次 reference，不随 TCP 当前位置、上次 reset 位置或范围变化而移动。采集的完整参考位姿、workspace 和初始随机范围会写入 `startup_reference.json` 及 checkpoint 的 `startup_reference` 字段，首次信息保存在 `initial_reference` 中。使用 `--resume` 时恢复首次 reference 和 workspace，不重新采集；旧 checkpoint 没有该记录时需重新采集，此次采集成为可追溯的初始 reference。

首次移动需确认；每轮结束后沿当前实测位置 +Z 回退 1 cm。所有 reset 目标仍需通过 workspace 校验。

### 普通模式

未开启 Z 插入模式时，需要传入 `--workspace-min/max`。可通过 `--sample-min/max` 指定起点采样区域；否则在 workspace 中采样，并在 X/Y 两侧各留 2 cm 边距。Z 随机采样，姿态固定为首次重置时的实测 TCP 姿态。episode 结束后沿 +Y 回退 1 cm。

### 在线训练与推理流程

ACT 和 Stage 1 RL Token 编码器被冻结并保持 eval，在线更新 Actor 和 Critic。

### 按 `p` 保存独立 episode 快照

每轮结束后，在下一轮开始前的起点等待菜单按 **`p`**（无需回车），保存当前完整训练状态，并继续留在菜单；按 Enter 继续原训练。此按键在训练等待界面可用，活动 episode 中不执行快照保存。

快照创建在原 output 的同级目录，名字为 `<原output名字>_step<累计env_steps>`。例如原 output 是 `outputs/stage2_new_run`、当前累计 1234 步，则创建：

```text
outputs/stage2_new_run_step1234/
  checkpoints/latest.pt
  config.json
  startup_reference.json   # Z 插入模式
  snapshot.json
```

包含 Actor/Critic、目标网络、优化器、replay、训练进度、随机状态、reference 和当前 XY reset 范围。保存通过临时目录完成后再发布，同一步数已存在的快照不会覆盖。原训练仍继续写 `outputs/stage2_new_run`，后续自动保存不会覆盖这个并列快照。

- **Resume**：在原训练命令中将 `--output` 改为 `outputs/stage2_new_run_step1234`，添加现有 `--resume`，其他训练参数保持兼容；无需 `--resume-from`。这会在快照目录中继续训练，其 `latest.pt` 随后按原规则更新。
- **Infer**：将 `--checkpoint` 指向该快照目录；Actor 必须已经初始化，warmup 阶段的未初始化快照可 resume，不能直接 infer。
- Resume 和 infer 恢复保存的 XY reset 范围；无此字段的旧 checkpoint 使用默认 ±1 cm。

原 output 的自动保存机制不变：每轮 episode 结束、每累计 1000 个环境步，以及退出时更新原目录的 `checkpoints/latest.pt`。

```text
观测 → ACT preprocessor → 完整 ACT
                          ├─ 参考 action chunk
                          └─ encoder hidden + position → RL Token
RL Token + 归一化关节状态 + 参考 chunk → Actor
  → ACT postprocessor → 物理限幅 → 固定 chunk 执行
  → 下一观测 → replay transition → Actor/Critic 更新
```

- 默认先采集 4,000 个低层动作步作为 warmup，只执行 ACT 参考动作、填充 replay，不更新网络。
- warmup 结束后，默认进行 1,000 次 Actor BC 初始化及 1,000 次 Critic 初始化。
- 随后默认采集 10,000 个 online 动作步；每个 chunk 对应一个 transition，执行 5 次 Critic 更新，每两次 Critic 更新执行一次 Actor 更新。
- Stage 2 默认 chunk 长度为 4；ACT 完整推理的参考 chunk 取前四步。Actor 输出六维 TCP 动作，夹爪不在 RL 动作空间内。
- online 采集使用带高斯探索噪声的 Actor；warmup 使用 ACT 参考动作。
- 成功终止奖励为 +1，其余步骤为 0。按 `s/f` 在当前 chunk 结束后标记成功/失败，`q` 停止；超时后也会询问结果。奖励时刻应尽量贴近实际完成时刻。
- `--enable-human-intervention` 支持 Space 在 chunk 边界接管/归还控制，人工 XYZ 控制按键为 W/X/A/D/P/L，S/F 标记结果，Esc 停止。

相机、推理、伺服持续运行，学习在调用线程中进行。推理使用发布的 Actor 快照，避免读取正在更新的参数。固定 chunk 不做 temporal ensemble，也不在 chunk 中途重新规划；沿用每步实测 TCP 锚定。

当前 `--observation-selection` 默认是 `latest`：使用比上次选中观测更新的最新缓存观测，不要求曝光晚于最后一条动作指令。`boundary` 模式要求观测 freshness 位于 chunk 请求边界之后。`--camera-boundary-mode` 默认 `timestamp`，启用机器人提供的时间戳观测路径；`legacy` 使用普通观测。下一观测的编码同时用作 replay 的 next state 和下一 chunk 的 current state。预测未就绪时保持最后目标，不重发相对动作或突发补发。

```bash
python -m act_rlt.train_stage2 \
  --stage1-checkpoint outputs/act_dataset_928_state7_stage1_pos \
  --act-checkpoint outputs/act_dataset_928_state7_act \
  --output outputs/stage2_new_run \
  --z-insertion-mode --fps 30 --allow-motion
```

Z 插入模式默认 30 Hz，普通模式默认 15 Hz，episode 默认 5 s。`--total-env-steps` 是累计 online 步数目标，不包含 warmup。`--dry-run` 只检查模型和 processor，不连接机器人。

## 5. infer Stage 2

入口：[infer_stage2.py](infer_stage2.py)。直接复用训练的 `FrankaInsertionStage2Env` 和 `FixedChunkRuntime`。

### 起点是否与训练相同

开启 `--z-insertion-mode` 后，infer 在连接机器人后选择本次 run 的参考点：`t` 使用 checkpoint 保存的首次训练 reference 和 workspace；`c` 则按 Enter 采集当前 TCP，workspace 设为采集位置 XYZ 各 ±5 cm。整次 run 的采样中心固定为选定参考点。两种选择均沿用 checkpoint 保存的 reset 半范围（旧 checkpoint 默认 1 cm）。每轮开始前用 `=` / `-` 以 0.5 cm 步长调整下一次 reset 半范围（0–5 cm），按 `r` 可立即重新采样，按 Enter 开始本轮。与 train 共用同一套范围调整逻辑，采样中心始终不变。

选择训练参考点时，每轮 infer 开始前还可按 **`4`** 立即移动到训练首次采集的完整 reference pose，含当时的 Z 和姿态。首次移动确认界面也支持 `4`；抵达后按 Enter 才开始 infer。按 `4` 不改变采样中心或范围，之后按 `r` 会继续围绕首次 reference 随机采样。

普通随机 reset 的 Z 仍为 0.402 m、姿态仍固定。选择当前 TCP 或旧 checkpoint 无训练参考记录时，手动采集本次 reference，`4` 不可用；普通模式仍从指定的 workspace/sample box 采样。控制频率和观测选择由 infer 参数指定，比较训练与推理时应使用相同设置。

### 推理流程

1. 加载已初始化的 v3 Stage 2 checkpoint，从保存的配置解析配套 ACT 和 Stage 1 checkpoint，加载 Actor 权重及 ACT processor。
2. 连接机器人、确定 workspace 和起点，移动到选定点，Enter 开始 episode。
3. 完整 ACT 生成参考 chunk，同时提取 encoder hidden 和 position；RL Token 编码后拼接归一化关节状态。
4. Actor 输入状态与参考 chunk，执行**确定性均值动作**，不添加训练时的探索噪声。
5. postprocessor 还原动作、进行物理限幅，按固定 chunk 执行；更新观测并生成下一 chunk。
6. 成功/失败、超时或退出时停止当前 episode、回退；有下一 episode 时重新进入起点选择流程。

推理冻结全部参数，不更新 Actor/Critic，不写入训练 replay。默认 `latest` 观测选择、`timestamp` 相机模式、四步 chunk，无 temporal ensemble。每步 TCP 锚定、起点选择和回退方式与训练环境共用。

```bash
python -m act_rlt.infer_stage2 \
  --checkpoint outputs/stage2_new_run \
  --z-insertion-mode --fps 30 --episodes 10 \
  --episode-time 5 --allow-motion
```

`--checkpoint` 可指向 run 目录或具体 `.pt` 文件；run 目录解析为 `checkpoints/latest.pt`。配套模型路径变化时可传入 `--act-checkpoint` 和 `--stage1-checkpoint`。`--dry-run` 只加载检查模型，不连接机器人。

## 6. 图像与对比实验

ACT 和 Stage 2 均按 ACT checkpoint 的相机键与图像尺寸处理输入。腕部相机保持配置的采集尺寸；前置相机输出统一为 640×480。图像转为 CHW、像素除以 255，再通过 checkpoint 的 processor 标准化。当前 ACT 路径不自动缩小为 224×224。

对比 ACT 与 Stage 2 时应记录起点绝对位姿、控制频率、观测选择、temporal ensemble、residual control、单步限幅、成功标记时刻以及实际完成时间。ACT 默认 temporal ensemble，Stage 2 固定 chunk；数采和两类推理的起点分布也不同。这些差异都会影响任务完成速度和成功率。
