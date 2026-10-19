# ACT soft approach 数据采集

新入口独立于原来的 `record_sample_space.py`，不修改 `evo_franka`。

```bash
cd /home/embint/wzy/Evo-RLT
bash act_rlt/data_collection/collect_soft_approach.sh \
  --dataset act_soft_rgbd_v2 --episodes 20
```

始终采集 RGB 和深度，无需深度参数；上例存到 `datasets/act_soft_rgbd_v2`。已有 control server 时复用；没有时启动一个，
退出时只停止本启动器创建的 server。使用新数据集，避免把不同采集方式混在原有数据目录里。

流程沿用原 sample-space collector：

1. 启动后手动示教完整参考位姿 p0，按 Enter 保存。最终目标设为 p0 沿基座 -Z 偏移 1 cm，XY 和朝向不变。
2. 自动在原采样空间生成起点，并在录制之外移到起点。
3. 等待 1 秒，开始录制：**一段连续的 soft approach 直接到 p0 下方 1 cm**，到达最终目标后减速停止。
4. 确认最终目标实际到位并稳定后自动保存。途中不在 p0 停车，没有独立的固定下插阶段。
5. 等待 1 秒，停止录制状态下退回 p0，再采样下一轮。

起点采样仍是：X 为 p0 ±2 cm，Y 为 p0+[1.5,2.5] cm，绝对 Z=0.402 m，
固定 XYZ Euler `[179.7,1.4,90.6]` 度；最终目标使用手动示教的朝向。
录制中按 `x` 停止并丢弃当前轮，之后 Enter 恢复；非录制窗口支持 `1` 返回 p0、`2` 返回
当前采样点，数字键操作后 Enter 恢复自动流程，`q` 退出。Ctrl+C 停止。

默认相机、22D 状态、7D 实测 TCP 增量/夹爪 action、力反馈字段都复用原采集器。
双相机为原有 640×480；`--wrist-only` 为 1280×720 腕部图像，`--front-only` 为 640×480 前置图像。
不运行 ACT 模型。

默认同时开启每台相机的 RGB 和 Z16 深度流（深度原生 640×480、30 Hz）。
每台相机从同一个 frameset 读取 RGB/depth，再将 depth 对齐到该相机的 RGB。
深度以无损 16 位 PNG 保存，不进入 RGB/H.264 编码器，不改变现有 ACT 的 RGB 输入或动作字段。
`--wrist-only` 时深度对齐到 1280×720 RGB，输出深度尺寸也为 1280×720。
深度固定开启；旧 RGB 数据集不能用此脚本继续录制，请创建新数据集。

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--fps` | 15 | 数据/图像录制频率，可选 15 或 30 |
| `--control-fps` | 30 | 独立控制线程的目标更新频率 |
| `--approach-speed` | 0.04 m/s | soft approach 的参考速度上限 |
| `--acceleration` | 0.05 m/s² | 参考平移加速度上限，比原 soft approach 默认值低一半 |
| `--jerk` | 0.2 m/s³ | 参考平移加速度变化率上限，使启动更缓和 |
| `--slowdown-distance` | 0.008 m | 最终目标附近的制动距离尺度；实际制动还考虑当前速度、加速度和 jerk |
| `--move-speed` | 0.016 m/s | 非录制起点移动/退回速度 |
| `--insert-minus-z` | 0.01 m | 最终目标相对于示教 p0 的基座 -Z 偏移，不是额外移动阶段 |
| `--episode-time` | 60 s | 每轮总超时 |
| `--stage-timeout` | 40 s | 这一段 approach 的超时，同时受本轮剩余时间限制 |

采集起点的 XY 距离通常只有 15–32 mm，不能直接套用独立脚本的 60→10 mm 放行区间，
否则一开始就几乎完全放行 Z，路径接近直线。现在每轮 approach 开始时确定一次区间：
放行起点为 `min(--lateral-gate-distance, 初始 XY 距离)`，完全放行阈值为
`min(--lateral-tolerance, 放行起点 × 15%)`。例如初始 XY 距离 20 mm 时，
Z 放行比例从 20 mm 处的 15%，平滑增加到 3 mm 处的 100%，形成先侧向靠近、后逐渐下降的弯曲路径。
这是按误差分配速度的曲线，不是固定半径圆弧；纯 Z 起点没有可形成弯曲的侧向位移。
每轮的实际区间及速度、加速度参数写入控制日志的 `stage_start.motion_parameters`。
整个速度分配过程直接以偏移后的最终目标计算误差；不会先瞄准 p0 再切换到更低的目标。

整段运动使用有限时间的速度曲线：平滑加速、近似匀速、末端平滑制动。
中段仍按指令误差和 Z 放行系数确定运动方向，但归一化方向后保持近似恒定的参考速度，
不再使用“速度正比于剩余距离”的方式一直微动逼近。短距离会自动下调巡航速度，
给加速和制动留空间；`--approach-speed` 是上限，不保证短距离能够达到 0.04 m/s。
末段接入五次多项式，与当前指令位置、速度、加速度连续衔接，检查速度、加速度、jerk
以及各坐标不反向后，有限时间到达目标并归零速度/加速度；不会靠终点位置截断来刹车。
初始距离不超过 8 mm 的短修正可直接使用有限时间多项式，不安排巡航段。
姿态沿最短 SO(3) 路径使用有限时间曲线，独立于平移尽早完成，避免姿态微动拖尾。
跟踪滞后保护可减缓或暂停参考推进；最终完成仍以实测到位为准。
日志的 `finite_braking` 标记制动阶段，`cruise_speed_m_s` 给出这轮采用的巡航速度。
这些更改只用于新采集器；独立 `act_rlt.soft_approach` 保持原速度算法。
旧的 `--insertion-speed`、`--insertion-slowdown-distance` 已移除。

到位确认使用 250 ms 测量窗口：拟合 TCP 平移/旋转速度，检查窗口内的位置/朝向摆幅及
各关节速度 RMS，避免相邻帧微小抖动被差分放大、反复清零稳定计时。
仍要求指令完成、实测误差不超过 1 mm / 0.3° 且持续稳定 0.4 s，才结束该阶段。

`--vertical-speed-floor`、`--lag-limit` 等沿用 `soft_approach.py` 当前默认值。
只在脚本中维护参考位置和速度，不设置共享速度倍率、阻抗或底层 Ruckig 参数。
参考位置只在每阶段开始时从实际 TCP 初始化；之后根据指令误差推进，实际位置用于滞后保护和到位确认。

相机/RGB 采集运行在录制主线程，在线目标更新在独立控制线程；深度 PNG 使用有界后台队列写入。
队列持续阻塞超过 1 秒会中止当前轮，避免静默丢帧。若目标到位失败、控制超时、相机异常
或按下中止键，会停止运动，并等控制线程退出后再丢弃数据，不会自动保存失败轮次。

数据标签：

- `action` 保持原有定义：相邻实测 TCP 的位移/旋转增量及实测夹爪宽度。
- `complementary_info.command_action` 记录观测采集开始时最新的在线参考位姿和参考平移速度。
  30 Hz 控制、15 Hz 录制时，一帧动作期间可能发出多个参考目标，这个字段不是完整控制序列。
- `soft_approach_traces/<run>/attempt_*.jsonl` 另存每次控制更新的参考位姿、实测状态、滞后、
  Z 放行比例和阶段标签，以及最终是否保存。`run.json` 保存目标和参数。
- `meta/soft_approach_collection.json` 标记采集方法；`--resume` 只接受当前单段版本创建的数据集。
  旧两段版本请保留并改用新数据集目录，避免混合两种采集逻辑。

深度文件示例：

```text
depth/episode_000000/
  episode.json
  frames.jsonl
  wrist/frame_000000.png
  front/frame_000000.png
soft_approach_traces/<run>/cameras.json
```

`episode_000000` 对应 LeRobot 的 episode_index，PNG 编号对应该 episode 内 frame_index，
包括最终 terminal frame。`frames.jsonl` 为每帧记录各相机的文件路径、RGB/depth 时间戳及其
时钟域、帧编号、主机接收时间、深度单位和图像形状。`cameras.json` 保存序列号、原生 RGB/depth
内参、depth 到 color 的相机内部外参、保存深度的对齐后内参及深度尺度。

读取深度：

```python
import cv2
depth_u16 = cv2.imread("depth/episode_000000/wrist/frame_000000.png", cv2.IMREAD_UNCHANGED)
# depth_scale_m 从 frames.jsonl 或 cameras.json 读取；不要把深度按 RGB 归一化。
depth_m = depth_u16.astype("float32") * depth_scale_m
valid = depth_u16 != 0  # 0 表示无效/缺失深度
```

深度 PNG 与数据帧严格按索引对应，但双机未做硬件同步；同一 frameset 的 RGB/depth 也记录
各自原始时间戳，不声称曝光时刻完全相同。SDK 对齐造成的无效像素保留为 0。
这只是新增采集，ACT 训练不会自动使用旁路深度文件。

写入时先在 `depth/.pending/` 暂存，每轮完成后核对 RGB/深度帧数，再发布目录并保存数据集。
`episode.json` 的 `dataset_saved` 在数据集保存成功后才置为 true。正常中止会删除该轮暂存文件；
若数据集保存过程中出错，完整深度目录会保留并标为未完成，禁止静默 resume，需先核对恢复。

可查看完整选项：

```bash
bash act_rlt/data_collection/collect_soft_approach.sh --help
```

离线验证覆盖直接到偏移目标且只执行一次运动、录制外退回、中止后控制线程退出、16 位 PNG 无损保存、
帧数对应和失败清理，以及真实在线规划函数配合模拟机器人运行。本次单段改动仅做离线验证；
实际 TCP 轨迹与目标误差仍受底层控制及到位容差影响。
两台 D435I 的 RGB+depth 配置已验证可解析，短时实拍检查遇到设备占用，未停止其他进程。
请先退出旧采集/推理程序，再用新入口验证 RGB-D 实采。
