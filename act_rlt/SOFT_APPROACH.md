# 两次 Enter 示教与在线速度分配

当前版本只在实验脚本中实现在线参考轨迹，使用既有 `servo_tool` 接口。
本次不修改 `evo_franka` 实现、共享 server 的速度倍率、stiffness 或底层 Ruckig 配置。
已经运行的 Evo-RLT control server 不用重启；若服务尚未启动，可运行：

```bash
cd /home/embint/wzy/Evo-RLT
bash scripts/start_evo_franka_control_server.sh 172.16.0.2 /home/embint/robolab-venv/bin/python
```

先结束 inference 或其他运动脚本，再运行实验：

```bash
cd /home/embint/wzy/Evo-RLT
PYTHONPATH=src /home/embint/robolab-venv/bin/python -m act_rlt.soft_approach
```

1. 将机械臂放在目标处，松开引导按钮，确认 Execution / FCI active，按 Enter 记录完整 TCP 位姿。
2. 脚本释放 FCI；手动移到起点，松开按钮，确认 Execution / FCI active，再按 Enter 返回目标。
3. 到位后释放 FCI 并继续等待，可以反复移动到起点再按 Enter。输入 `q` 后 Enter 退出。
   Ctrl+C 中断运动，使用现有 stop_move；中断不走参考轨迹的柔和减速。

## 在线规划逻辑

指令位姿 `p_cmd` 只在每轮开始时用实测 TCP 初始化。随后约 30 Hz：

- 用 `goal - p_cmd` 的 XY 横向误差、Z 高度误差计算目标速度。
- 横向指令误差 >= 60 mm 时仍允许少量 Z 运动（放行系数 15%）；降到 60–10 mm 时
  用 smoothstep 逐渐放行；<= 10 mm 时完全放行，避免先横移再下降形成近似直角。
  这对上升/下降两个方向都有效。坐标轴是 server 的工作坐标系，当前默认机器人 base。
- 在剩余距离 30 mm 内逐渐降低参考速度，并限制参考加速度及 jerk。
- 积分得到下一帧指令位姿，通过 `servo_tool` 发送。不预生成密集 waypoint 列表。
- 旋转沿最短 SO(3) 路径独立推进，并在接近目标朝向时减速；不强制按固定路径百分比完成。

实际 TCP 只用于计算跟踪滞后、检查状态和确认到位，不每周期重置 `p_cmd`。
位置滞后超过 4 mm 开始降低推进速度，达到 8 mm 暂停位置推进；朝向滞后在 0.05–0.1 rad
之间逐渐降低推进速度，达到 0.1 rad 暂停旋转。暂停仍发送保留的目标，等待实际 TCP 追上，
恢复时从原指令继续；不会把指令吸附到测量位置。硬滞后暂停和微小终点吸附优先于参考平滑。

指令到达终点后仍维持目标发送，直到实测位置误差 <= 1 mm、朝向误差 <= 0.3°，
TCP 速度 < 2 mm/s、角速度 < 0.02 rad/s、所有关节速度 < 0.01 rad/s，连续保持 0.4 s，
才确认完成并结束 servo。server 的 worker_status=idle 不作为到位条件。
如果本来就已经到位，则检查稳定性，不启动零距离 servo 运动。

## 参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--speed` | 0.05 m/s | 参考平移速度上限，实际速度还受底层限制 |
| `--fps` | 30 Hz | 指令更新频率 |
| `--acceleration` | 0.1 m/s² | 脚本参考加速度限制 |
| `--jerk` | 0.5 m/s³ | 脚本参考 jerk 限制 |
| `--slowdown-distance` | 0.03 m | 接近目标时减速的距离尺度 |
| `--lateral-gate-distance` | 0.06 m | Z 放行比例开始增加的横向指令误差 |
| `--lateral-tolerance` | 0.01 m | Z 完全放行的横向指令误差 |
| `--vertical-speed-floor` | 0.15 | 横向误差较大时的最低 Z 放行比例 |
| `--lag-limit` | 0.008 m | 参考位置推进的跟踪滞后上限 |
| `--max-angular-velocity` | 0.4 rad/s | 参考旋转速度上限 |
| `--position-tolerance` | 0.001 m | 实测终点位置容差 |
| `--orientation-tolerance-deg` | 0.3° | 实测终点朝向容差 |
| `--settle-time` | 0.4 s | 连续到位、低速的确认时间 |
| `--timeout` | 40 s | 本轮运动和到位确认的总超时 |

旧版本的 `--acceleration-scale`、`--jerk-scale`、`--y-align-fraction`、
`--orientation-finish`、`--waypoint-spacing` 已不适用，运行时请移除。
当前的 acceleration/jerk 是脚本参考轨迹参数，并非 robot 或 Ruckig 参数。
默认 Z 放行系数在 XY 误差为 60、40、20、10 mm 时约为 15%、45%、91%、100%；
实际 Z 速度还受剩余高度、总速度限制、平滑器和跟踪滞后影响。可将
`--vertical-speed-floor` 设为 0，让横向误差较大时不推进 Z。

链路为 `指令误差 -> XY/Z 速度分配 -> 平滑与积分 -> servo_tool -> IK -> JointMotion/Ruckig`
`-> JointPositions -> FR3 内置 joint impedance`。参考轨迹与实际轨迹不是同一信号，
底层的关节插值、动力学限制和跟踪误差仍会影响实际路径；实机效果需要测量确认。

日志位于 `logs/act_rlt_soft_approach/*.jsonl`，可用 `--log` 指定。
每个 sample 包含实测 state、command_pose、指令误差、Z 放行比例、跟踪滞后、暂停状态和参考速度，
便于区分参考推进完成和实测到位。状态失效、新的 servo 错误、更新间隔超过 150 ms 或超时都会中断，
不会在故障后自动继续运动。
