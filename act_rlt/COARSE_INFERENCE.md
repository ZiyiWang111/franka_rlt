# Coarse ACT 推理

入口：`python -m act_rlt.infer_coarse`，在仓库根目录、安装了 LeRobot 和机器人依赖的环境运行。
checkpoint 先下载至本机；支持 run 目录、checkpoint 目录、`pretrained_model` 目录。

起点使用 `act_coarse_1020` 起点统计恢复出的固定关节角，推理启动无需读取数据集。

| 模型 | 输入 | 输出 | 控制接口 |
| --- | --- | --- | --- |
| A：`act_coarse_1020_rgb_joint_abs_joint` | wrist RGB + 当前 7 维关节角 | 7 维绝对关节角，rad | `servo_joint` |
| B：`act_coarse_1020_rgb_z_delta_tcp` | wrist RGB + 当前实测 TCP Z，m | 6 维 TCP 增量 | 原 `send_action` TCP 增量积分 |
| C：`coarse_pose_joint_hold1s` | wrist RGB + 7 维关节角 + 6 维实测 TCP 位姿 | 6 维 TCP 增量 | 同 B |
| `coarse_z_hold1s` | wrist RGB + 当前实测 TCP Z，m | 6 维 TCP 增量 | 同 B |

各模型均使用 checkpoint 保存的 pre/postprocessor 做归一化和反归一化，仅开启 wrist 相机，
图像保持原始 640×480。TCP 增量模型的 XYZ 增量在基座系，旋转增量是工具系 rotvec（rad），
按 `R_next = R_current * R_delta` 积分；不控制夹爪。

`coarse_pose_joint_hold1s` 的 13D state 严格按训练顺序拼接：
`[joint_0..joint_6, ee_x, ee_y, ee_z, ee_rx, ee_ry, ee_rz]`。
关节角及 TCP 旋转向量单位为 rad，XYZ 是基座系绝对坐标，单位 m；输入不转 Euler，
不使用指令目标代替实测状态。`hold1s` 表示训练数据每回合追加了 15 帧静止示例，
不是推理端自动停留一秒。推理端只保留下述 Z gate 停止条件。

## 启动

```bash
# A：wrist + joints -> absolute joints
python -m act_rlt.infer_coarse \
  --checkpoint outputs/ethernet_coarse/act_coarse_1020_rgb_joint_abs_joint

# B：wrist + TCP Z -> delta TCP
python -m act_rlt.infer_coarse \
  --checkpoint outputs/ethernet_coarse/act_coarse_1020_rgb_z_delta_tcp

# C：wrist + measured joints/TCP pose -> delta TCP（本机 last 指向 30k）
python -m act_rlt.infer_coarse \
  --checkpoint outputs/ethernet_coarse/coarse_pose_joint_hold1s

# 只读实时相机和状态，自动执行一回合、两个完整 chunk；不下发运动
python -m act_rlt.infer_coarse \
  --checkpoint outputs/ethernet_coarse/act_coarse_1020_rgb_joint_abs_joint \
  --dry-run --dry-run-episodes 1 --chunks 2
```

默认 `--action-mode auto` 根据 `train_config.json` 的训练任务/数据集名称识别这两个模型，
并校验输入输出维度，也自动识别 `coarse_pose_joint_hold1s` 和 `coarse_z_hold1s`。
其他同结构模型需要显式选择 `--action-mode joint_absolute` 或
`--action-mode tcp_delta`，避免把旧模型的 TCP+gripper 七维输出误当关节角。

## Z gate 与 vanilla 动作执行

模型动作恢复为原始执行路径：TCP 模型先经过原有单步增量限幅，再调用 `send_action`；
关节模型保持原有 `servo_joint` 路径。删除了自定义末段 Z 缓速、加速度/jerk 规划、
XY 参考平滑、固定姿态、目标高度闭环推进，以及模型结束后的独立收敛/停留阶段。
底层原有运动控制和动作限幅仍然生效。

默认开启 Z gate，所有 coarse 模型在**实测绝对 Z ≤ 0.40207 m** 时停止整段 servo，
同时取消后续 ACT 动作和预测请求，XYZ 和旋转一起停止，不执行当前 chunk 的剩余动作。
gate 是停止条件，不会提前接管或减速。

```bash
python -m act_rlt.infer_coarse \
  --checkpoint outputs/ethernet_coarse/coarse_pose_joint_hold1s \
  --z-gate-m 0.40207

# 关闭 Z gate，恢复仅由 duration/chunks/操作员结束的行为
python -m act_rlt.infer_coarse \
  --checkpoint outputs/ethernet_coarse/coarse_pose_joint_hold1s \
  --no-z-gate
```

`--terminal-target-z` 保留为 `--z-gate-m` 的别名；其他旧 `--terminal-*` 参数已删除，
启动命令中需要移除。原来的 20 mm 缓速距离不再使用。

开始预测前、每次下发动作前、等待控制周期和等待模型预测期间都会检查 gate。
控制周期等待最多每 20 ms 检查一次；等待推理时最多每 5 ms 检查一次。
若实测从门限上方直接跨到下方，仍会触发；若启动时已低于门限，不发出模型动作。
触发后先调用 `stop_servo`，收到停止确认后再等待相机/模型线程退出。
已经开始的模型计算不能强行中断，但其输出不会再执行。

Z gate 使用客户端新鲜状态缓存；默认最大接收年龄 0.25 s，参数为
`--z-gate-state-max-age-s`。状态过期或位姿无效则报错并停止，避免基于失联缓存继续运动。
离散观测、状态传输以及机器人实际制动都有延迟，因此 gate 不保证物理停下位置恰好等于门限。
需要更早停止下降时调高 `--z-gate-m`。dry-run 不下发运动，也不按 gate 提前结束。

日志记录 `episode_start.z_gate`、触发事件 `z_gate_reached` 和结束原因 `z_gate`；
回合结束仍打印实测绝对 Z 和 Euler 姿态。时间/chunk 预算先耗尽时直接停止，不追加收尾动作。

## 完整 chunk 与控制时序

默认每次调用 `predict_action_chunk` 得到 **完整 10 个动作**，逐个反归一化后按顺序处理；
TCP 动作通过原 `send_action` 执行；Z gate 可以在 chunk 中途停止。
不调用 `select_action` 的缓存路径；即使训练配置保存了 `n_action_steps=4`，也会覆盖为 10。
默认关闭 temporal ensemble，不做动作重叠融合，也不丢弃 chunk 尾部。

默认频率 **15 Hz**，与数据集一致。与 ACT infer 关闭 ensemble 时的队列模式一致，
相机持续采集最新 RGB/状态，模型在后台预测，控制线程按固定节拍消费动作。
当前 chunk **剩余 2 步时**（已执行第 8 个动作），模型使用最新观测预测下一组完整 10 步；下一组准备好后等待
当前十步全部执行完，在下一控制周期接上。正常交接不再额外等待相机或模型。

`--prefetch-steps N` 调节提前量，默认 2，可选 1–9：数值越大，留给推理的时间越长，
下一组动作使用的观测也越早。只缓存一组下一 chunk，不进行多组提前规划。
保留全部十步的含义是：预取 chunk 的第 0–9 个动作会依次执行，起点对应预取时的观测；
提前观测到实际交接之间的延迟会记录在日志中。

加 **`--no-rtc`** 可关闭上述提前预测机制（默认 `--rtc` 开启）。关闭后，完整执行十步并等待
最后一步的控制周期结束，才开始新的观测读取与下一段预测；不使用执行期间的观测缓存。
此模式由推理线程按请求读取相机，不启动持续采集线程；`--prefetch-steps` 不参与调度。
两段之间会增加观测与推理等待时间，等待期间不重复发送增量、不补发追赶动作；停止键和
错误检查仍保持响应。这里的“执行完”指十条指令及其控制周期结束，不额外等待实测到位，
相机曝光时间也不保证与控制周期硬件同步。

```bash
python -m act_rlt.infer_coarse \
  --checkpoint outputs/ethernet_coarse/act_coarse_1020_rgb_z_delta_tcp \
  --no-rtc
```

日志的 `episode_start` 记录 `rtc_enabled`；关闭时 `camera_mode=post_chunk`、
`prefetch_steps=0`，每次请求后的观测读取另记为 `observation_captured`。

RTC 开启时，下一组尚未准备好会在 chunk 边界停止并报告 `queue underrun`，不会等待模型返回
后突然继续，也不会补发密集动作或重复 TCP 增量。单次推理超过 0.5 秒、相机没有新帧、
观测过期或后台线程异常都会停止；关闭 RTC 时，观测读取超出 0.5 秒或整段准备等待超出
1 秒也会停止。自动 GC 延后至回合之间；日志由独立线程写入。

默认 `--duration 10`，模型预算只在完整 chunk 边界检查时间；Z gate 触发可以中断任意一步。
`--chunks N` 额外限制最多执行 N 个完整模型 chunk；两个限制中先满足的生效。
预算耗尽后直接停止，不追加闭环收敛或停留阶段。
操作员停止、模型输出异常及控制错误允许中断当前 chunk。

## 可选 temporal ensembling

加 `--temporal-ensemble` 开启跨预测融合；默认关闭，可用 `--no-temporal-ensemble` 显式关闭。
上述模型均支持，无需重训或修改 checkpoint。启用示例：

```bash
python -m act_rlt.infer_coarse \
  --checkpoint outputs/ethernet_coarse/act_coarse_1020_rgb_z_delta_tcp \
  --temporal-ensemble --temporal-ensemble-coeff 0.01
```

此模式每个控制周期请求一次新的完整 10 步预测，使用持续采集的更新观测。
预测跨度仍是 10 步，但执行方式切换为逐步融合，不再把每组预测独立执行完十步。
`--prefetch-steps` 不参与调度。`--no-rtc` 要求十步结束才观测，无法产生重叠预测，
因此与 `--temporal-ensemble` 同时使用会报错。

实现按全局控制步对齐：请求时记录该预测的起始步 `s`，第 `k` 步只取其 `prediction[k-s]`。
例如在第 5 步，起始步 0、3、5 的预测分别贡献第 5、2、0 个动作。
迟到的预测保留原始起始步，跳过已过去的前缀，不把第 0 个动作重新放到到达时刻。
这是控制步索引对齐；持续相机缓存、模型延迟及真实曝光时间仍需通过观测年龄日志评估。

对覆盖当前步的预测按生成顺序从旧到新编号 `i=0,1,...`，计算：

```text
w_i = exp(-m * i) / sum_j exp(-m * j)
action = sum_i w_i * prediction_i[k - start_i]
```

`m` 由 `--temporal-ensemble-coeff` 指定，默认 `0.01`，必须有限且非负。
此权重方向与现有 ACT infer 实现一致：`m=0` 为等权平均，正值越大越偏向较早预测；
它不是“系数越大越偏向最新预测”。使用 log-sum-exp 归一化。每步最多有十个预测参与，
过期预测自动移除，有效零动作也参与融合。

融合发生在 checkpoint 后处理、动作反归一化之后。A 对绝对关节目标做加权平均，再执行
原有关节限位与单步限幅；单个预测越出绝对关节限位时仍报错，避免平均掩盖非法输出。
B/C 对六维 TCP 增量做逐分量加权平均，再做平移/旋转范数限幅；
随后交给原 `send_action` 积分，并由 Z gate 检查停止条件。
旋转融合仍针对小幅工具系 rotvec 增量。

相机、推理、控制独立运行。模型忙时只保留最新的待处理请求，当前推理继续完成；
控制端消费已经完成且仍覆盖当前步的预测，不等待每个请求都完成。没有任何有效预测
覆盖当前步时立即报错停止，不重复上一步 TCP 增量。观测和推理超时、操作员停止、
dry-run 禁止运动以及退出时先停止 servo 再等待线程的行为保持有效。
`--chunks N` 在此模式表示最多执行 `N * 10` 个控制步，时间限制仍每十步检查。

日志保留完整 `chunk_ready.predicted`，并记录 `prediction_accepted`、被更新请求替代的
`refill_superseded`。每步 `action` 增加限幅前的 `ensemble_action`、投票数 `ensemble_votes`
和各贡献预测的 `start_step / offset / weight`，便于复算融合结果。

融合会平滑预测切换，也可能因旧预测权重而延迟终点修正；不能据此保证消除超调。

## 交互和动作限制

固定起点保存在 `infer_coarse.py` 的 `COARSE_START_JOINTS`，七个关节角（rad）为：
`[-0.2029924393, -0.0072414274, 0.3428563774, -1.4956223965, -0.1103802323, 1.5422743559, 2.5215561390]`。
来源是原始 `act_coarse_1020` 的 episode 8 第一帧实测值。统计全部 266 个 episode
的第一帧后，精确重复数为零；采用每个关节误差不超过 0.005 rad 的邻域计数，
该起点属于最密集邻域，共包含 episode 5–14 的 10 个起点，计数并列时取最早 episode。
这是一个明确容差下的密度代表，不能称为精确重复的众数；容差变化可能改变选择。
完整来源及统计保存在 `coarse_start_joints.json`，运行推理无需加载数据集。
该帧实测 TCP XYZ 约为 `[0.553269, 0.054814, 0.559411] m`，与此前 TCP 起点接近。

加载模型后，在当前位姿读取真实观测、预热模型，此时不移动机器人。
首次按 Enter 或 `r`，直接移动到上述数据集起点；抵达后再按 Enter 开始推理。
回合结束后可再次 Enter/`r` 返回起点，再按 Enter 开始下一回合，`q` 退出。
`r` 使用一次阻塞的 `move_joint`，各模型共用上述固定关节起点，避免按 TCP 回位造成冗余关节配置漂移。
返回后检查实测关节值，最大单关节误差超过 0.01 rad 则报错，不能开始推理。
回位关节速度上限默认 **0.5 rad/s**，可用 `--reset-joint-speed` 覆盖；底层 dynamics factor
还会缩放实际运动限制。原 coarse 参数 `--reset-speed`（TCP m/s）已移除，启动命令如包含它需改用新参数，数值不能直接按相同单位沿用。
回合中 `s` 标记成功并停止，`x` / `q` / Esc 停止回合，Ctrl+C 退出。
停止键在动作之间和等待周期中检查，不再等待后台相机/模型调用返回。
停止时先下发 `stop_servo`，再等待后台线程退出，最后重置模型队列。
dry-run 禁止重置移动及 servo/stop_servo 命令，并关闭 FCI keepalive。

每回合确认 `stop_servo` 成功、后台线程退出后，读取客户端最新状态缓存中的实测 TCP，
向终端打印绝对 Z 高度（m 和 mm）及 **intrinsic XYZ Euler（度）**，不打印 XY。
Z 沿控制器工作坐标系定义，当前默认是机器人 base；姿态由实测 rotvec 转换，顺序与采集一致。
这是停止后读取的最新实测状态，不是模型动作或积分目标，也未额外等待新的硬件采样或稳定到位。
`episode_end.final_tcp` 同时保存 `z_m`、`euler_xyz_deg`、旋转约定及状态来源，供后续逐轮/平均统计。
读取失败或停止未确认时打印 `End TCP unavailable`，并记录 `final_tcp_error`，不掩盖原有控制错误。
dry-run 打印 `Dry-run end observation`，明确它仅是结束时的观测。

A 校验 FR3 关节限位（留 0.01 rad 边界），再将每关节目标限制在实测角度和上次命令的
`--max-joint-step-rad` 范围内，默认 **0.04 rad**；超过绝对关节限位的预测会直接停止。
B/C 默认将模型单步平移向量范数限制为 **0.01 m**（`--max-step-m`）、旋转向量范数限制为
**0.05 rad**（`--max-step-rad`）。
限幅可能改变模型预测值；未开启 temporal ensemble 时，每个 chunk 的十个动作仍依次处理。

每回合日志位于 `logs/act_coarse_infer/*.jsonl`，包含模式、预取请求对应的控制步、
观测序号与年龄、预测开始/完成、完整原始预测、推理耗时、各动作实际下发值、chunk 内索引、
结束原因。`--dry-run` 的日志记录候选动作，`sent=false`。
