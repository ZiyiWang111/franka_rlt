两份独立的 CVAE ACT，部署在 server 7 / 10.0.40.163。

| 模型 | 原始数据集 | episodes / frames |
| --- | --- | --- |
| vga_1023_z_cvae_50k | datasets/vga_fine/act_dataset_1023 | 699 / 55125 |
| usb_1027_z_cvae_50k | datasets/usb_fine/usb_dataset_1027 | 362 / 38827 |

输入 wrist RGB（1280×720）和当前实测 ee_z；输出 dx, dy, dz, drx, dry, drz。
每份单卡 batch=32，50000 步，chunk=n_action_steps=4。全部数据训练，无验证集。
CVAE latent=32、posterior encoder=4 层、KL weight=10。
ACT dim=768、heads=8、encoder=4、decoder=1、FF=3200、dropout=0.1，ResNet18 ImageNet。
AdamW lr=3e-5、backbone lr=1e-5、weight decay=1e-4、clip=10。
Warmup 1000 步，cosine 到 50000 步，末尾 lr=3e-6；FP32。
ColorJitter：亮度/对比度/饱和度 0.9–1.1，hue ±0.02，最多 3 项。
仅最终保存 checkpoint，日志每 100 步；seed=1000，workers=4。

服务器启动（只能运行一个队列实例）：

```bash
cd /workspace/wangziyi/projects/franka_rlt
mkdir -p logs/act_idle_queue
nohup /workspace/wangziyi/miniconda3/envs/evo-rlt/bin/python -u \
  scripts/run_act_idle_queue.py \
  --project /workspace/wangziyi/projects/franka_rlt \
  --python /workspace/wangziyi/miniconda3/envs/evo-rlt/bin/python \
  --configs configs/act_fine_1027/vga_1023_z_cvae_50k.json \
            configs/act_fine_1027/usb_1027_z_cvae_50k.json \
  > logs/act_idle_queue/queue.log 2>&1 < /dev/null &
```

添加 `--check-only` 可只做配置预检，不等待也不启动训练。

每 120 秒采样一次，按 GPU UUID 分别累计；没有计算进程、显存低于 1024 MiB、
利用率不超过 5% 时算空闲。同一张卡连续 5 次空闲才启动（首次空闲样本后约 8 分钟）；
任何忙碌、缺失或查询失败都会清零。启动前立即再次核验占用。
两张卡同时满足条件时，两份单卡训练可以并行；只有一张满足时，先运行 VGA，
USB 等另一张卡或 VGA 结束后重新累计空闲次数。GPU 用 UUID 指定，避免编号变化。

`logs/act_idle_queue/status.json` 保存队列 PID、状态、空闲计数、每份训练的 PID/GPU/日志位置。
`logs/act_idle_queue/queue.log` 保存每次检测记录，每份训练有独立日志。
模型保存在 `outputs/act_fine/<模型名>/checkpoints/050000`，`last` 指向最终 checkpoint。
进程退出为 0 且最终权重、50000 步训练状态均存在才标记 COMPLETE。
失败任务标记 FAILED，不自动重试或覆盖；另一份任务继续排队。

停止等待：从 status.json 读取队列 pid，执行 `kill -TERM <pid>`。
这只停止调度，已经启动的训练继续运行。SSH 退出或本机关机不影响服务器后台任务。
本脚本不提供机器重启后的自动恢复，也不提供对其他用户启动器的 GPU 资源预留。
项目锁避免重复启动同一队列，GPU 文件锁用于本脚本实例之间协调。
