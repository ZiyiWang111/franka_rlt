在本机启动网页服务（仅需 Linux 和 Python 3.10+，无额外依赖）：

```bash
cd /home/embint/wzy/Evo-RLT
python3 scripts/control_server_web.py
```

本机访问 `http://127.0.0.1:8080`；其他电脑访问 `http://本机IP:8080`，需网络可达且防火墙允许该端口。默认监听 `0.0.0.0`，无身份验证。可通过 `--port 端口` 更换端口。

“启动 teleop”先开启 control server，再执行：

```bash
PYTHONPATH=src /home/embint/robolab-venv/bin/python scripts/fr3_keyboard_teleop.py \
  --allow-motion --input-backend pynput --fps 15 --speed-m-s 0.02 --gripper-force-n 30
```

control server 使用原启动脚本，机器人 IP 为 `172.16.0.2`。teleop 使用运行网页服务进程的 DISPLAY/XAUTHORITY 环境，需要在可用的 Linux X11 桌面会话中启动网页服务。

键盘输入来自 Linux 桌面或 ToDesk，不接收外部浏览器按键。W/S、A/D、P/L 控制 XYZ，C 夹紧、O 张开，ESC 退出。

“退出 teleop”先发送 SIGINT，让 teleop 停止运动并断开连接，再关闭 control server；按 ESC、异常退出或停止网页服务也会关闭对应的 control server。关闭浏览器页面不会退出 teleop。terminal 每秒更新 teleop 和 control server 日志。
