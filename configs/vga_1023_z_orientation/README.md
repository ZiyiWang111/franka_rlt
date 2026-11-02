已确认并启用：server 6（113）与 server 7（163）自动等待同一份 ACT 训练。

模型名：vga_1023_z_orientation_50k。
源数据：datasets/vga_fine/act_dataset_1023，701 episodes / 55434 frames。
输入 wrist RGB 原始 1280×720，state=[ee_z, ee_rx, ee_ry, ee_rz]；姿态为原始绝对旋转向量。
输出 action=[dx,dy,dz,drx,dry,drz]；平移是基座系下一帧减当前帧，旋转是当前 TCP
局部系 rotvec(R_current.T @ R_next)，单位 m/rad；末帧零增量，不含夹爪。

关闭 CVAE，chunk=4，50000步，全局batch=64（单卡64，双卡每卡32）。
ResNet18 ImageNet，ACT dim768/heads8/encoder4/decoder1/FF3200/dropout0.1。
AdamW主学习率3e-5、backbone1e-5、weight_decay1e-4、clip10，
warmup1000、cosine50000、末尾3e-6，FP32。
轻度 ColorJitter 同之前配置，全数据训练，无验证集。
每10000步保存 checkpoint，保留10000/20000/30000/40000/50000及last链接。

两台服务器使用相同部署路径：

```text
/workspace/wangziyi/projects/vga_z_orientation_1029
```

每120秒查询，逐GPU UUID累计连续3次空闲。
空闲要求无计算进程、显存低于1024MiB、利用率不超过5%；查询失败清零。
有1张满足即可单卡启动；同一次检测有2张满足则双卡启动。
不跨服务器拼接两张卡；启动前再次核验占用。
跨服务器认证协调服务保存在server7，持久化全局启动权。
第一台领取启动权后另一台退出；协调服务不可达时不启动。
领取后失败不自动释放启动权或重复训练，需检查失败原因后人工处理。
本脚本无法阻止其他用户同时自行启动任务。

主要文件：

- configs/vga_1023_z_orientation/train_config.json：训练配置，batch_size表示全局batch。
- logs/orientation/status.json：本台监控/训练状态、GPU、进程和启动命令。
- logs/orientation/watcher.log：配置预检及每次检测记录。
- logs/orientation/train.log：获得启动权的服务器上的训练日志。
- logs/orientation/global_claim.json：只存在于server7，记录获得启动权的机器和状态。
- outputs/vga_1023_z_orientation_50k/checkpoints/：实际训练机器的模型及续训状态。

协调服务配置包含随机认证令牌，仅存放在服务器权限600的配置文件中。
本机不保存该令牌到仓库；不要把服务器的 coordinator.json 或 watcher_server*.json 提交到git。
旧的只读 monitor_act_gpu_pairs 已停用并标记为被新监控替代。
后台进程与SSH会话独立；机器重启后不自动恢复。
