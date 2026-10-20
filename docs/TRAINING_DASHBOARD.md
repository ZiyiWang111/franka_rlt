# ACT 双任务训练监控

本机运行，无需额外 Python/JavaScript 依赖，不依赖 CDN。通过只读 SSH 增量读取服务器日志，每 10 秒同步一次；浏览器每 5 秒读取本机缓存。

```bash
cd /home/embint/wzy/Evo-RLT
python scripts/training_dashboard.py
```

打开 <http://127.0.0.1:8765>。

默认任务来自 `logs/*20261019.launch.json` 的两个启动记录：

- `act_dataset_928_addon_camera_only_action6_act`
- `act_dataset_928_addon_rgb_z_action6_act`

页面显示 loss、学习率、裁剪前梯度范数、模型更新时间、数据等待时间、进度、近期速度估算的剩余时间、GPU 利用率与显存、已保存 checkpoint。支持 EMA 平滑、对数 Y 轴、范围筛选、隐藏单条曲线、悬停读数和原始 CSV 导出。

图表使用 tqdm 的精确步数，避免 LeRobot 的 `step:2K` 等缩写破坏横轴。EMA 只影响显示；CSV 导出原始记录。数据范围是两条曲线共用的真实 step 范围；两个任务开始时间不同，最近窗口可能暂时没有较慢任务的数据。

曲线是训练 loss，并非独立验证集性能。剩余时间根据最近 25 次记录估算，受视频解码和服务器负载影响。

## SSH 与断线

默认复用 `/tmp/evo-act-camera-%C` 控制连接及 `/home/embint/.ssh/id_ed25519_ziyi`。网页不接收或保存密码、验证码。SSH 认证失效时，页面保留历史数据并明确显示断线。可先在终端完成一次登录：

```bash
ssh -i /home/embint/.ssh/id_ed25519_ziyi \
  -o IdentitiesOnly=yes -o ControlMaster=auto -o ControlPersist=2h \
  -o ControlPath=/tmp/evo-act-camera-%C wangziyi@10.0.40.163 true
```

随后网页自动恢复同步。缓存存于 `logs/training_dashboard_cache.json`，重启服务可继续读取；日志替换或截短时会自动重建对应曲线。

默认仅监听本机 `127.0.0.1`。通过 `--port`、`--interval`、`--host`、`--key`、`--control-path` 或 `--cache` 调整。服务没有暂停、终止或修改训练的接口。

## 保存停止与续训

网页会读取各模型输出目录的 `stop_at_checkpoint.json`，显示 10k 保存后停止、双卡排队、续训到 30k 等计划。续训仍追加同一日志，`USER_RESUME_CHECKPOINT` 标记会清除上次人工停止产生的错误状态，保留历史曲线并累计训练耗时；续训入口使 tqdm 显示绝对训练步数。

2026-10-19 的续训队列由服务器 `scripts/resume_act_addon_queue.py` 管理，记录在 `logs/act_addon_resume_20261019.status.json`。两任务都使用 GPU 0、1，纯相机任务完成 30k 后自动启动 RGB＋Z。通过 LeRobot `resume=true` 恢复模型、优化器、scheduler 和 RNG，不重新 warmup，也不改变原定 30k 的余弦计划。
