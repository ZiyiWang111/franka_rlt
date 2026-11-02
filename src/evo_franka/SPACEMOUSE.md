# 官方 SpaceMouse 输入接入

复用 https://github.com/frankarobotics/franka_spacemouse 的未修改发布器，
本次核对版本为 `1d041da5a34bb040f8a97192504c9137eb0b8923`。
官方源码和隔离依赖保存在同级 `spacemouse_probe/official`、`python_deps`。
依赖版本：pyspacemouse 1.1.4、easyhid 0.0.10；使用 ROS Humble 和 Python 3.10。
这些外部目录不属于 Evo-RLT 仓库，迁移机器时需单独下载和安装。

日常真机启动脚本（默认无时间限制，在运行终端按 Esc 或 Ctrl+C 停止并退出，无需回车）：

```bash
bash scripts/fr3_spacemouse_control.sh
# 可选：指定运行时间
bash scripts/fr3_spacemouse_control.sh --seconds 300
```

默认每轴满偏转平移 9 mm/s，roll/pitch 为 0.06 rad/s，yaw 为 0.072 rad/s（角速度参数的 1.2 倍）。
需要现有 control_server 已启动。真机模式启动时自动调用客户端 connect()，
由已有服务建立或恢复机器人连接；收到新鲜 TCP 状态后才允许发送目标。
无需另行手动执行连接命令，也不会启动第二个 control server。

链路：USB HID → 官方 spacemouse_publisher → 归一化 Twist →
evo_franka.spacemouse_bridge → 原 ZMQ client/server → servo_tool。
不启动官方 arm controller 或 gripper manager；不能让两个 FCI 控制器同时连接机器人。
Evo-RLT 当前 servo_tool 是 IK → JointMotion，与官方关节阻抗控制器行为不同。

仅预览，不创建机器人客户端（此例指定 30 秒）：

```bash
bash scripts/fr3_spacemouse_teleop.sh --seconds 30
```

若插拔导致编号变化，用 `SPACEMOUSE_DEVICE=/dev/hidrawN` 指定。
启动脚本加载 ROS 环境，不要用默认 conda Python 3.13 运行 Humble。

低速真机入口（本机已完成一次 30 秒试控）：

```bash
bash scripts/fr3_spacemouse_teleop.sh --execute --host 127.0.0.1 --seconds 30 \
  --linear-speed 0.003 --angular-speed 0.02 --max-distance 0.02 --max-rotation 0.1
```

需要已启动的 Evo-RLT control_server；桥接会请求该服务连接机器人，
连接失败或连接完成后 3 秒内没有新鲜位姿时退出，不发送运动目标。
退出时停止本次伺服并关闭客户端通信，不向共享服务发送 disconnect。
操作前先在预览模式核对每个轴方向，并核对 server 的工作坐标系：
官方前方操作员映射是 `[-state.y, state.x, state.z, -roll, -pitch, -yaw]`，
桥接在 server 的工作坐标系内积分，未自动转换为不同的工作坐标系。
官方发布器的 operator_position_front 默认为 true。

默认每轴满偏转平移 0.009 m/s，roll/pitch 为 0.06 rad/s，yaw 为 0.072 rad/s，死区 0.05、目标更新约 30 Hz。
角速度用旋转矩阵对应的旋转复合积分，不能直接累加旋转向量。
松手或 ROS 消息超过 0.2 秒不更新时发送 stop_servo；退出也停止。
状态过期、非有限输入或命令领先实际 TCP 超过 1 cm / 0.1 rad 时退出并尝试停止。
默认不限制相对启动 TCP 的累计平移距离和姿态变化。
需要限制时可显式传入 --max-distance（米）/ --max-rotation（弧度）。
实际位姿与目标偏差检查、状态超时停止及机械臂底层保护仍保留。

本版只桥接六轴，夹爪消息尚未接入。

Stage-2 训练加 `--enable-human-intervention` 后，自动通过
`scripts/fr3_spacemouse_input.sh` 启动官方发布器和只读 JSON 输入转发进程，
无需另外运行上面的独立控制脚本。该进程使用同一组 `SPACEMOUSE_DEVICE`、
`SPACEMOUSE_SUPPORT_DIR`、`SPACEMOUSE_PYTHON` 环境变量，不连接机器人。
Space 在 4-step chunk 边界接管/归还，SpaceMouse 控制六轴，S/F 标记结果，
Esc 停止；episode 内不再使用键盘方向键移动。起点菜单的 WASD 保留。
归还前松开 SpaceMouse。实际人工动作按 chunk 写入 replay 并作为 BC 目标；
空间旋转增量转换成训练动作定义的 TCP 局部旋转增量后保存。
速度参数是 `--teleop-speed-m-s`（训练默认 0.0135）、
`--teleop-angular-speed-rad-s`（训练默认 roll/pitch 0.09，yaw 0.108；yaw 始终取该参数的 1.2 倍），
死区为 `--spacemouse-deadzone`（默认 0.05）。
ROS 消息超时只能检测发布器停止发消息；官方库持续发布缓存状态时，
不能据此判定 USB 断开，物理断连行为还需单独验证。

本次验证：实际 HID 打开、官方发布器到桥接消息链路；
9 项测试通过，覆盖位姿积分、目标范围、Esc/超时停止、启动连接顺序、
连接失败不发指令及预览模式不建立机器人连接。
低速真机试控 30 秒收到 2983 条消息（1407 条非零），发送 401 个目标；
实测 TCP 发生变化，结束时 Idle / is_running=false，servo_fault_seq 未增加。
尚未逐轴确认操作方向，夹爪和物理 USB 断连停止尚未验证。
注意：get_state 可返回断开前的旧缓存；建立连接后必须确认新鲜状态流，
不能仅凭缓存中的 Idle / state_stale=false 判断当前机器人连接状态。
