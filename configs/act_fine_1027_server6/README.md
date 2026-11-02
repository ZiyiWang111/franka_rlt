server 6（10.0.40.113）GPU 4 顺序训练，先 USB，后 VGA。
server 7 的原等待队列已经停止，两份原 50000 步任务均未启动。

项目：`/workspace/wangziyi/projects/act_fine_1027`
Python：`/workspace/wangziyi/projects/franka_rlt/.venv/bin/python`

| 顺序 | 模型 | 数据 |
| --- | --- | --- |
| 1 | usb_1027_z_cvae_30k | 362 episodes / 38827 frames |
| 2 | vga_1023_z_cvae_30k | 699 episodes / 55125 frames |

两份均为 CVAE ACT，wrist RGB + ee_z -> TCP delta6。
USB 单卡 batch=16，VGA 单卡 batch=32；均为30000 步，chunk=4。
无验证集、ColorJitter 开启、仅最终保存。
USB 原 batch32 训练未产生 checkpoint，已保留旧输出及日志并从0步重启。
AdamW 3e-5（backbone 1e-5），warmup1000，cosine30000，末尾3e-6。
其他模型结构和参数沿用 server 7 已确认配置。

启动参数如下（已经启动时不要再次执行）：

```bash
cd /workspace/wangziyi/projects/act_fine_1027
/workspace/wangziyi/projects/franka_rlt/.venv/bin/python -u scripts/run_act_idle_queue.py \
  --project /workspace/wangziyi/projects/act_fine_1027 \
  --python /workspace/wangziyi/projects/franka_rlt/.venv/bin/python \
  --configs configs/act_fine_1027_server6/usb_1027_z_cvae_30k.json \
            configs/act_fine_1027_server6/vga_1023_z_cvae_30k.json \
  --gpus 4 --idle-samples 1
```

启动前已核验 GPU4 无计算任务，后续每份任务启动前也会重新核验。
只使用 GPU4，不会自动使用其他卡；若有其他任务抢先占用，则每120秒等待检查。
后台启动信息：`configs/act_fine_1027_server6/launcher.json`。
状态：`logs/act_idle_queue/status.json`，队列日志：`logs/act_idle_queue/queue.log`。
两份模型有独立训练日志，路径在 status.json 中。
输出：`outputs/act_fine/<模型名>/checkpoints/030000`，以及 `last` 链接。
SSH 断开后后台训练继续。调度器 SIGTERM 仅停止调度，不中止已运行的训练。
