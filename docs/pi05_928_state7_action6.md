# π0.5：七维关节状态、六维动作、固定夹爪

数据集：`datasets/pi05_dataset_928_state7_action6`，由 `datasets/act_dataset_928_state7` 独立复制生成。
任务文本统一为 `insert the ethernet cable into the hole`。

- 输入：当前 wrist RGB、`joint_0` 至 `joint_6`、任务文本。使用标准 PI05 state tokenizer，保留关节状态。
- 输出：`dx, dy, dz, drx, dry, drz`，默认预测 10 步。内部保留预训练模型的 32 维动作投影，监督与对外输出均为 6 维。
- 夹爪：不由此模型预测；部署控制器应独立维持现有夹持状态，不应把“闭合”解释为强制宽度归零。
- `complementary_info.*` 保留为原始采集记录，不作为模型输入或监督动作。

转换会重写 action、任务索引/文本及相关元数据。state/action 的全局统计直接使用全部帧计算，episode 统计使用各自全部帧计算，分位数采用 NumPy linear 方法，标准差使用 ddof=0。不会再次把 episode 分位数加权平均成全局分位数。若模型使用的全局分位数范围退化，转换会报错，不会静默除以 epsilon。

```bash
python scripts/prepare_pi05_franka.py \
  datasets/act_dataset_928_state7 datasets/pi05_dataset_928_state7_action6
```

先检查启动命令（此操作不加载模型、不训练）：

```bash
python scripts/train_pi05_franka.py \
  --base-model /workspace/wangziyi/models/pi05_base \
  --gpu 2 --dry-run
```

确认后去掉 `--dry-run` 启动。使用装有本项目和 LeRobot 0.5.1 的 Python 环境。基础模型目录是显式参数；脚本不会写死代理或读取 Hugging Face token 文件。训练前检查 GPU 占用和模型文件。复制到服务器时还需同步新数据、该启动脚本以及 `src/evo_rlt/cli/train_pi05_masked.py` 和 `src/evo_rlt/adapters/lerobot/policies/pi05_masked_loss.py`。

双卡使用 `--gpus 2 4 --batch-size 16`（替换 `--gpu 2`）。`batch-size` 表示全局 batch，脚本通过 torchrun 启动两个进程，各自 batch=8；LeRobot 的 Accelerator 负责数据分片、梯度同步和仅主进程保存。默认训练步数和学习率调度不随卡数改变。

默认配置：30,000 steps、batch size 16、chunk size/n_action_steps 为 10/10、AdamW 主学习率 5e-5、1,000 步 warmup、到训练末尾余弦衰减至 5e-6，每 10,000 步保存、每 50 步记录日志。BF16、梯度检查点开启；图像增强关闭；保留完整 VLM 微调。`--train-expert-only` 是可选冻结 VLM 实验，不默认开启。没有自动验证集，训练 loss 不等于任务成功率。

动作是已记录的 TCP 增量，显式设置 `use_relative_actions=false`。标准图像预处理等比例缩放并补边到 224×224。视频使用 PyAV，时间戳容差为 1e-4 秒。

专用入口仅在当前训练进程内替换 PI05 的 loss reduction：保留原始 flow-matching 模型，只监督真实动作维度，并按 `action_is_pad` 屏蔽 episode 末尾补齐动作。每个样本对有效时刻/动作维度取平均，再对样本取平均。支持 `reduction=none`，并同步屏蔽逐维 loss 日志。不会修改 site-packages，也不会改变模型 checkpoint 的标准 PI05 格式；普通推理仍可读取它。

恢复训练也需要这个入口，以保留 padding mask 行为：

```bash
PYTHONPATH=src python -m evo_rlt.cli.train_pi05_masked \
  --config_path=outputs/pi05_dataset_928_state7_action6/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

新模型的远程推理入口为 `scripts/pi05_state7_inference_server.py` 和 `scripts/pi05_state7_robot_client.py`。旧 FR3 远程入口仍保留 state15/action7 协议，新入口使用版本 2 的 state7/wrist720p/action10×6 协议，两者会拒绝互连。模型使用 checkpoint 自带的预处理、分位数统计与反归一化；内部标准 PI05 投影不变，无需训练 loss 补丁。

服务器（先选择空闲 GPU；以下只是启动命令，修改代码不会自动启动服务）：

```bash
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=2 python scripts/pi05_state7_inference_server.py \
  --checkpoint outputs/pi05_dataset_928_state7_action6/checkpoints/030000/pretrained_model
```

机器人本机先只验证网络，再观察 shadow 推理；网络检查不会初始化硬件，shadow 不发送动作：

```bash
PYTHONPATH=src:. python scripts/pi05_state7_robot_client.py \
  --server tcp://10.0.40.163:5559 --network-only
PYTHONPATH=src:. python scripts/pi05_state7_robot_client.py \
  --server tcp://10.0.40.163:5559 --mode shadow --max-cycles 1
```

客户端默认 checkpoint 路径是服务器的 030000 目录，必须与服务器报告的绝对路径完全一致；使用 020000 等其他模型时传入 `--expected-checkpoint`。实机执行用 `--mode arm-only --allow-motion`，并传入实际工作空间的 `--workspace-min X Y Z --workspace-max X Y Z`，不能照抄其他任务的边界。默认 chunk=10，每轮按 30 Hz 执行前 4 步再观测，可通过 `--execute-steps 1..10` 调整；这是同步推理，实际平均频率会受推理与网络延迟影响，并非持续实时 30 Hz。

夹爪不接受模型动作，不自动重新闭合、不把目标宽度设为零；执行前必须已有 `gripper_grasped=true` 的夹持状态，每轮及推理返回后检查，丢失夹持会退出。保留 TCP 增量限幅、工作空间限制、动作时效检查和可选 `--force-stop-n` 力变化保护；触发力保护时沿 +Z 撤退并结束。

离线入口 `scripts/fr3_pi05_offline_inference.py` 的默认数据/模型已改为本次新模型，显式指定旧路径仍可检查旧模型。只读样本验证命令：

```bash
PYTHONPATH=src:. python scripts/fr3_pi05_offline_inference.py --schema-only
```

去掉 `--schema-only` 可进行真实模型预测，但不连接机器人。原 ACT/RL Token 数据集与 checkpoint 不受本次推理改动影响。
