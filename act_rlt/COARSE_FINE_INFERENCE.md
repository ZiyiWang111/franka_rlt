# Coarse → fine 串联推理

```bash
python -m act_rlt.infer_coarse_fine --allow-motion
```

默认 coarse：`outputs/ethernet_coarse/coarse_z_hold1s`，15 Hz，5 秒预算，
Z gate 为绝对高度 `0.40207 m`，RTC 开启。默认 fine：
`outputs/rlt_928_addon_rgb_z_action6`，Z insertion，30 Hz，每次尝试最长 15 秒。
旧模型路径沿用已有 artifact resolver，可定位到整理后的 `outputs/ethernet_rlt/`。

串联入口将 wrist 相机固定为 **1280×720 RGB**，coarse 和 fine 均直接使用该尺寸，
不缩放、不裁剪，阶段切换不重启相机。coarse 的训练输入仍为 640×480；
仅在串联入口覆盖运行时图像尺寸检查，不修改 checkpoint 或独立入口。
coarse 在 720p 输入下的预测效果与推理耗时尚未验证；启动时会先用实际图像预热两个模型。

两个模型在启动时加载并预热，共用机器人和相机。首次确认后以默认 `0.6 rad/s`
回到 coarse 固定关节起点，再按 Enter 开始一轮。默认共 10 轮，每轮开始均需确认。
IPC 回位完成后，先等待服务端序号确认的后续两帧状态（最多 2 秒），再校验关节到位；
不使用回位前缓存作判断，关节容差仍为每关节 `0.01 rad`。未收到新状态则明确报错停止。
TCP 上抬 1 cm、返回 fine 起点及 entry gate 每次调整后，同样先等待后续两帧、
且运动已停止的有效状态（最多 2 秒），直接用这份新 TCP 状态校验到位或 gate，
不会再拿运动前的缓存作判断；各项到位误差容限保持不变。

coarse 触发 Z gate 或达到 5 秒预算后，均自动进入 fine entry gate。
coarse 时限沿用完整 action chunk 边界检查，因此实际停止时间可能略超过设定值。
手动 s/f、退出和异常不触发阶段切换。停止并清理 coarse 工作线程后，
先检查 fine entry gate：目标绝对 Z 为 **0.40187 m**，姿态为
**rotvec [rad] `[-2.19967, 2.22472, -0.02892]`**，不是 Euler 角。
默认允许 Z 误差 ±0.2 mm、姿态角误差 1°；两项均通过才进入 fine。
姿态误差采用 SO(3) 相对旋转角，避免等价旋转向量在 π 附近换符号造成误判。
若不通过，保持当前 XY 目标不变，将 Z 和姿态调整至目标，以 `0.01 m/s` 调用已有 blocking move_tool，
随后重新检查实测状态，最多调整三次；仍不通过或运动异常则停止会话，不进入 fine。
这是已有 IK/关节运动接口，运动中间轨迹不保证 XY 严格恒定。终端打印每次 PASS/FAIL、
有符号 Z 误差（mm）、姿态角误差（deg）、实测姿态（rotvec rad），同时记录 JSONL。

通过 gate 后，记录实测 TCP 六维位姿作为本轮不可变的 fine 起点，同步命令位姿、重新观测并执行 fine。
不做随机采点、不移动到训练参考点、不更改起点姿态，阶段之间无需按键。
切换仍需要停止确认、线程清理、采集观测及首次预测的实际耗时。

| 阶段 / 按键 | 行为 |
| --- | --- |
| coarse 按 s | 全局成功标记，结束本轮，回 coarse 起点 |
| coarse 按 f | 全局失败标记，结束本轮，回 coarse 起点 |
| fine 按 s | 全局成功标记，结束本轮，从实测位置沿基座 +Z 上抬 1 cm，再回 coarse 起点 |
| fine 按 f | 记录本次失败，返回本轮 fine 起点的完整 TCP 位姿，按 Enter 后重试 fine |
| q / x / Esc / Ctrl+C | 停止会话，不执行自动回位 |

fine 的 s/f 沿用原来的当前 action chunk 完成后停止语义；coarse 按键沿用逐步轮询停止。
fine 达到 15 秒时停止运动，再等待 s/f/q；f 仍然返回同一次 coarse 终点后重试。
coarse 超时自动进入 entry gate，通过后执行 fine。机器人异常或 fine 越界则停止会话。

全局用时从本轮 coarse 开始到读取 s/f 的时刻计算，包含阶段切换。
fine 失败后重试继续累计同一轮用时，但扣除失败后的回位和确认等待时间。
超时后等待标记的时间包含在标记用时中。终端输出每次标记的全局用时，
`logs/act_coarse_fine/` 中的 JSONL 记录全局阶段事件、fine 尝试次数、fine 起点和 coarse 明细。

fine 工作空间以每轮通过 fine entry gate 后的实测 TCP 位置为中心，沿机器人基座坐标轴设置
**X ±5 cm、Y ±10 cm、Z ±5 cm**。不再读取 checkpoint 的训练工作空间，
也不接受 `--workspace-min/--workspace-max` 覆盖。边界在每轮进入 fine 时重新计算并打印、记录到日志；
同一轮按 f 返回 fine 起点后重试，沿用该轮边界，不随运动重新居中。
重试 fine 前也会检查 entry gate；该轮记录的 fine 返回点及 workspace 中心保持不变。
启动预热时仅临时使用当前 TCP 周围同尺寸边界，不发出运动命令；该边界不会用于正式 fine 推理。

常用覆盖参数：

```bash
python -m act_rlt.infer_coarse_fine --allow-motion \
  --episodes 10 \
  --coarse-checkpoint outputs/ethernet_coarse/coarse_z_hold1s \
  --fine-checkpoint outputs/rlt_928_addon_rgb_z_action6 \
  --coarse-fps 15 --fine-fps 30 \
  --coarse-duration 5 --fine-duration 15 \
  --z-gate-m 0.40207 --reset-joint-speed 0.6
```

支持 `--no-rtc`、`--temporal-ensemble`、`--temporal-ensemble-coeff` 和 `--prefetch-steps`，
仅作用于 coarse。`--recovery-speed` 控制 fine 上抬及返回 fine 起点的 TCP 速度，默认 `0.04 m/s`。
`--dry-run` 仅加载模型并检查配置，不连接机器人；不会实际运行串联动作。
两个原有独立推理入口保持原行为。

fine entry gate 参数：`--fine-entry-z-m`、`--fine-entry-rotvec-rad RX RY RZ`、
`--fine-entry-z-tolerance-m`、`--fine-entry-angle-tolerance-deg`、
`--fine-entry-adjust-speed`、`--fine-entry-max-adjustments`。
其中 `--fine-entry-max-adjustments 0` 表示仅检查、不自动调整；未通过则停止。
该 gate 与 coarse 的 `--z-gate-m` 分开设置，调整耗时包含在全局推理用时中。
