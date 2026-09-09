# π0.5 SFT 启动耗时说明

## 结论

当前 π0.5 SFT 每次启动约需 2 分 20 秒才进入第一个训练 step。这不是数据格式错误、
GPU 冲突或网络下载造成的卡死，主要是 41 亿参数模型的实例化、14.47 GB base 权重读取、
权重名称转换、加载和设备迁移。

训练开始后，默认 `log_freq=200`，以约 1.63 step/s 运行时还要约 2 分 4 秒才输出第一条
loss。因此，从执行启动命令到看到第一条 loss，通常需要约 4 分 26 秒。

## 两次实际启动记录

以下时间来自本仓库的实际训练日志：

| 训练 | 开始创建 policy | 开始训练 | policy 初始化 | step 200 | 启动到首条 loss |
|---|---:|---:|---:|---:|---:|
| 15D state / 6D action | 14:33:28 | 14:35:52 | 144 秒 | 14:37:56 | 268 秒 |
| 15D state / 7D action | 16:13:20 | 16:15:42 | 142 秒 | 16:17:46 | 266 秒 |

两次 policy 初始化只相差 2 秒，说明 7 维 action 没有增加可观察的启动开销，耗时具有稳定
的重复性。

对应日志：

```text
outputs/fr3_pi05_sft_60ep_state15_bs4_30k.log
outputs/fr3_pi05_sft_60ep_state15_action7_bs4_30k.log
```

## 启动阶段实际做了什么

### 1. 数据集初始化

LeRobot 读取 `info.json`、`stats.json`、episode metadata 和 parquet 索引，并根据 schema
创建 processor。当前 7390 帧、60 episodes 的数据集很小；日志中 `Creating dataset` 和
`Creating policy` 出现在同一秒，数据集初始化不是主要耗时。

### 2. 创建 41 亿参数模型

当前模型共有：

```text
num_learnable_params = 4,143,404,816
num_total_params     = 4,143,404,816
```

这是全参数训练，需要先在内存中构建完整的 PaliGemma 和 action expert 网络。

### 3. 读取并转换 base 权重

base checkpoint 位于：

```text
/workspace/wangziyi/models/pi05_base/model.safetensors
```

文件大小为 `14,467,165,872` bytes，约 14.47 GB。加载代码还会执行：

1. 从 safetensors 读取完整 state dict。
2. 修正 OpenPI 与 LeRobot π0.5 之间的 key 差异。
3. 重映射 812 个 state-dict key。
4. 将权重复制到已实例化模型并迁移到训练设备。
5. 创建 AdamW optimizer 和 scheduler。

这段过程主要受共享磁盘读取、CPU 内存带宽、模型对象创建和 CPU→GPU 传输影响。GPU 在
前一部分时间仍可能显示接近空闲，不能据此判断训练进程已经卡死。

以下日志表示权重加载成功：

```text
✓ Loaded state dict from model.safetensors
Remapped 812 state dict keys
All keys loaded successfully!
Creating optimizer and scheduler
Start offline training on a fixed dataset
```

启动期间出现的两条 `Vision embedding key might need handling` 是当前 LeRobot π0.5 loader
发出的提示。本次加载随后报告 `All keys loaded successfully!`，所以它不是缺失权重错误。

### 4. 等待第一条训练指标

训练配置默认：

```text
log_freq = 200
speed    ≈ 1.63 step/s
```

训练已经进入 step 1 后，约 124 秒才会打印 step 200 的 loss。进度条会通过回车符持续刷新；
某些日志查看方式不会立即显示这些刷新，看起来像是日志停止增长。

## 为什么不是网络问题

训练通过本地路径加载 base model，并且 PaliGemma tokenizer 已存在 Hugging Face 本地缓存。
两次日志都没有下载进度或远端请求错误。当前约 142 秒的稳定耗时符合本地大模型加载过程，
没有证据表明它在等待网络。

## 如何判断正常加载还是异常卡住

正常启动顺序应为：

```text
Creating dataset
Creating policy
（约 2 分 20 秒模型加载窗口）
All keys loaded successfully!
Creating optimizer and scheduler
Start offline training...
（约 2 分 4 秒后出现 step:200）
```

满足以下任一情况时再按异常排查：

- `Creating policy` 后超过 5 分钟仍没有 `All keys loaded successfully!`，并且进程 CPU、内存
  和磁盘读取都没有活动。
- tmux pane 已退出，或训练 Python 进程不存在。
- 日志出现 `Traceback`、`CUDA out of memory`、`Killed`、`NaN` 或明确的模型加载错误。
- 开始训练后超过 3 分钟仍没有到 step 200，且 GPU 利用率持续为 0%。

检查命令：

```bash
tail -f outputs/fr3_pi05_sft_60ep_state15_action7_bs4_30k.log
tmux attach -t fr3_pi05_state15_action7_30k
nvidia-smi
```

## checkpoint 保存也可能出现暂停

旧的 step-5000 checkpoint 总大小约 23 GB，其中：

```text
pretrained_model/model.safetensors       约 9.35 GB
training_state/optimizer_state.safetensors 约 15.13 GB
```

每 5000 steps 保存 checkpoint 时需要把这些内容写入磁盘。在共享存储繁忙时，进度条可能
短暂停留；只要 checkpoint 文件仍在增长且进程存在，这通常是正常保存过程。

## 可以采取的改进

以下方法只改善可观察性，不改变模型训练结果：

- 将 `--log_freq=50` 加到未来训练命令中。开始训练后约 30 秒即可看到第一条 loss。
- 设置 `PYTHONUNBUFFERED=1`，让普通 `print` 日志立即写入文件。
- 避免频繁停止和重启；每次重启都必须重新完成大模型加载。

可能缩短实际加载时间、但需要单独验证的方案：

- 保存一份经过验证的本地 bfloat16 base checkpoint，减少 base 权重读取量。
- 将 base model 和输出 checkpoint 放在吞吐更高、竞争更少的本地 SSD。
- 从完整训练 checkpoint 恢复时同时读取模型和 optimizer state；它保证训练连续性，但总读取
  量可能超过从 base model 重新开始，因此不一定更快。

减小模型、只训练部分参数或切换 action expert 会改变训练方法和最终效果，不能只作为启动
加速选项直接替换当前配置。
