# FR3 主控制路径与关节阻抗控制

检查日期：2026-09-30。本文总结当前 ACT / RL 主 servo 路径；频率为代码配置或标称值，不是现场实测值。

## 核心结论

当前系统属于 **位置/位姿目标控制 + FR3 内部关节阻抗控制**。

底层 joint impedance controller **不是本项目自写的，也不是 franky/libfranka 在电脑上实现的**，而是运行在 FR3 机器人控制固件中的 Franka 官方内置控制器。

```text
ACT / RL inference（生成 action chunk）
    ↓ 按配置的 action rate 播放
TCP delta
    ↓ LeRobot FrankaRobot adapter：生成目标位姿、限制目标超前量
FrankaArmControllerClient
    ↓ ZMQ
evo_franka.control_server
    ↓ FrankaArmController / MotionMixin.servo_tool
warm-start DLS IK → joint target q
    ↓ servo_joint → JointMotion
franky / Ruckig：连续关节位置轨迹
    ↓ franka::JointPositions
libfranka：control(callback, ControllerMode::kJointImpedance)
    ↓ FCI
FR3 内部 joint impedance controller
    ↓ 电机力矩
FR3
```

当前主调用链使用 LeRobot adapter、自定义 ZMQ client/server、franky 和 libfranka；没有经过 RobotLab client/server 或 ROS。`robolab-venv` 是 Python 环境目录名，不代表控制链使用 RobotLab。

## 各层职责与代码证据

| 层 | 职责 | 代码位置 |
| --- | --- | --- |
| FrankaRobot adapter | 接收 TCP delta，生成 TCP 目标 | [`franka_robot.py`](../evo_rlt/adapters/lerobot/franka_robot/franka_robot.py)，`send_action()` |
| ZMQ client | 发送 `servo_tool` 命令 | [`control_client.py`](control_client.py)，`servo_tool()` |
| ZMQ server | 收取并合并最新 streaming 目标，调用本地 driver | [`control_server.py`](control_server.py)，`_dispatch()` / `_controller_loop()` |
| 项目 driver | IK 求解关节目标，创建 `JointMotion(JointState(q))` | [`motion.py`](motion.py)，`servo_tool()` / `servo_joint()` |
| franky / Ruckig | 生成平滑关节轨迹，callback 返回 `franka::JointPositions` | franky `src/motion/joint_waypoint_motion.cpp`，`getControlSignal()` |
| franky → libfranka | 调用 `control(m, params_.controller_mode)` | franky `include/franky/robot.hpp`，`move(Motion<franka::JointPositions>)` |
| libfranka | 构造 `ControlLoop<JointPositions>`，发送位置命令及控制模式 | libfranka `src/robot.cpp` / `src/control_loop.cpp` |
| FR3 固件 | 执行内部关节阻抗控制、计算电机力矩 | 不在本项目或 libfranka 开源实现中 |

启动脚本默认使用 `/home/embint/robolab-venv/bin/python`。本次检查该环境安装的是 **franky-control 1.1.3**，扩展实际链接到随包提供的 **libfranka 0.18.0**。本机对应源码位于 `/home/embint/franky` 和 `/home/embint/libfranka`。

## 最终 command 与 ControllerMode

- 主路径最终 command：**`franka::JointPositions`**，不是速度或 torque command。
- 项目构造 `Robot(robot_ip)` 时没有覆盖控制模式。
- franky 默认 `controller_mode` 为 **`kJointImpedance`**，并将它显式传入 libfranka `control(...)`。
- libfranka 的 `JointPositions` control 重载默认模式也为 `kJointImpedance`。
- 主路径在电脑端生成运动目标，关节控制力矩由 FR3 内部控制器计算。

## 当前阻抗参数：已知与未知

| 参数 | 确认结果 |
| --- | --- |
| 内部 Cartesian stiffness | connect 时设置 `[3000, 3000, 3000, 300, 300, 300]`；前三项 N/m，后三项 Nm/rad |
| 内部 joint stiffness，7 个值 | **未知**；当前项目没有显式设置 |
| 内部 joint damping | 机器人内部根据 stiffness 自动确定；具体值和公式未确认 |
| 参数来源 | FR3 机器人端当前有效配置；不是 franky/libfranka 自动写入的默认增益 |

已检查项目、franky 和 libfranka 的初始化、connect、recover、move 路径，没有发现主路径实际调用 `setJointImpedance()`。项目 connect 只设置 Cartesian impedance；franky 构造设置碰撞阈值；libfranka recovery 发送 `AutomaticErrorRecovery` 命令。

常见的 `[3000, 3000, 3000, 2500, 2500, 2000, 2000] Nm/rad` 出现在 libfranka 示例 `examples/examples_common.cpp` 的 `setDefaultBehavior()` 中，由示例**显式设置**。当前项目未调用这个函数，不能将这组值当成当前 FR3 的实际刚度。

现有证据无法判断机器人端值是出厂配置、Desk 配置还是此前程序设置的值，也无法确认其跨断连、恢复或重启的保存规则。

当前 franky/libfranka 0.18.0 提供阻抗 setter，没有公开 joint impedance getter；`RobotState` 也不包含 stiffness/damping。显式设置并记录成功响应可以确定后续配置，但不能反查原来的值。本次检查没有修改机器人阻抗配置。

### Cartesian stiffness 对主路径的作用

libfranka 将 `setJointImpedance()` 和 `setCartesianImpedance()` 作为不同配置命令发送，启动控制时明确选择 joint 或 Cartesian impedance 模式。

按接口定义，当前 `kJointImpedance` 路径使用 joint impedance 参数，设置的 Cartesian stiffness **不作为该控制律的增益**。不能用 `3000 N/m` 推算当前关节阻抗路径的有效末端刚度或精确力阈值。

内部控制器在机器人固件中；libfranka 开源实现不能证明固件所有环节都与 Cartesian 参数完全无关联。

## 主路径控制律的大致形式

```text
Ruckig → q_d → FR3 内部 joint impedance controller → τ → 电机
```

概念上可以写为：

```text
τ ≈ K_q (q_d − q) + D_q (dq_d − dq) + τ_compensation
```

这是解释结构的近似形式，不是已公开并核实的 FR3 固件精确公式。

- `q_d`：由 franky/Ruckig 生成，经 libfranka 位置接口处理后发送。
- `K_q`：机器人端当前有效配置决定；本项目没有覆盖。
- `D_q`：机器人内部根据 stiffness 自动确定；不能套用外部示例的 `2√K` 作为其精确公式。
- 力矩及内部补偿：FR3 固件计算。

## 轨迹生成与频率

每个 action step 刷新 TCP 目标，经 IK 得到关节目标。franky 的 `JointMotion` 使用 `JointWaypointMotion` 和 Ruckig 在线规划；新目标从当前 commanded 状态继续规划，低频目标之间仍有高频关节轨迹输出。

| 层 | 标称频率或代码行为 |
| --- | --- |
| libfranka realtime callback | 控制期间 1000 Hz |
| franky / Ruckig | 名义步长 1 ms，在 realtime callback 中运行 |
| ZMQ server command | 事件驱动，合并最新 streaming 目标；没有独立固定 command rate |
| ZMQ server 状态发布 | 默认 100 Hz |
| ACT action 播放 | 默认 30 Hz，可由 `--fps` 修改 |
| ACT-RLT Stage-2 action 播放 | 默认 15 Hz；`z_insertion_mode` 默认 30 Hz |
| π0.5 remote action 播放 | 默认 15 Hz |

模型推理按 chunk 进行，实际推理频率不等于 action 播放频率，取决于 checkpoint、chunk 长度、计算耗时和运行配置。

## 可选分支：不要与主 servo 路径混淆

- **`move_tool_impedance()`**：创建 franky `CartesianImpedanceMotion`，callback 返回 `franka::Torques`，属于外部笛卡尔阻抗 torque control。默认平移刚度 2000 N/m、旋转刚度 200 Nm/rad，外部实现的阻尼对角项为 `2√K_i`。
- 该外部分支控制律为 `w = −K e −D J dq`，`τ = Jᵀw + coriolis`。`force_constraints` 会将选定轴 wrench 替换为指定值，不是 clamp 上限，也不是测量力误差闭环控制。
- **`move_until_force()`**：首选 `CartesianMotion` / `franka::CartesianPose` 加测力触发停止；拒绝后可回退关节路径。它不维持目标接触力。
- 当前 ACT/RL 主 servo 路径没有调用上述外部 torque 分支。

## 官方参考

- [libfranka 0.18.0 Robot API](https://frankarobotics.github.io/libfranka/0.18.0/classfranka_1_1Robot.html)
- [FCI 控制接口与内部控制模式](https://frankarobotics.github.io/docs/doc/libfranka/docs/overview.html)
- [内部 stiffness 与自动 damping 的说明](https://frankarobotics.github.io/docs/doc/franka_ros2_humble/franka_bringup/doc/index.html)
