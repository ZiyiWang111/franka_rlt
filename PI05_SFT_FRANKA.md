# FR3 π0.5 SFT 数据与 README 方案对比

本文记录当前工作区已经实际跑通的 FR3 π0.5 监督微调（SFT）。结论以
`/workspace/wangziyi/SFT_HANDOFF.md`、本地 LeRobot 数据集元信息，以及
`outputs/vla_ft_smoke/checkpoints/000200/pretrained_model/train_config.json` 为准。

## 1. 实际使用的数据

已完成的 200-step smoke run 只使用了一个本地数据集，没有混合其他数据：

```text
repo_id: franka_m1_manual_demo
root:    /workspace/wangziyi/projects/franka_rlt/datasets/franka_m1_manual_demo
```

保存的训练配置中 `episodes=null`，因此数据集内全部 5 个 episode 均参与训练。

数据内容如下：

| 属性 | 实际值 |
|---|---|
| 机器人 | Franka FR3 |
| 任务 | `pick up the vga and insert into the motherboard` |
| episode 数 | 5 |
| episode 长度 | 181、156、130、124、157 帧 |
| 总帧数 | 748 |
| 采样频率 | 30 Hz |
| 相机 | `wrist`、`front` |
| 图像 | RGB，640×480，AV1 MP4 |
| state | 22 维 `float32` |
| action | 6 维 `float32` |

## 2. 磁盘格式

数据采用 LeRobot v3 布局：

```text
datasets/franka_m1_manual_demo/
├── data/
│   └── chunk-000/file-000.parquet
├── videos/
│   ├── observation.images.front/
│   │   └── chunk-000/file-000.mp4
│   └── observation.images.wrist/
│       └── chunk-000/file-000.mp4
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/chunk-000/file-000.parquet
└── recovery_frames.jsonl
```

`data/chunk-000/file-000.parquet` 每帧一行，主要列为：

| 列 | Parquet 类型 | 含义 |
|---|---|---|
| `action` | `fixed_size_list<float32>[6]` | 当前帧的末端增量动作 |
| `observation.state` | `fixed_size_list<float32>[22]` | 机器人状态 |
| `complementary_info.policy_action` | `fixed_size_list<float32>[6]` | 采集时策略动作记录 |
| `complementary_info.is_intervention` | `float32` | 是否发生人工干预 |
| `complementary_info.state` | `float32` | 干预状态 |
| `complementary_info.phase` | `float32` | RLT 阶段标记 |
| `complementary_info.collector_policy_id` | `int64` | 采集策略来源；当前 `0=human` |
| `timestamp` | `float32` | episode 内时间 |
| `frame_index` | `int64` | episode 内帧号 |
| `episode_index` | `int64` | episode 编号 |
| `index` | `int64` | 数据集全局帧号 |
| `task_index` | `int64` | 指向 `meta/tasks.parquet` 的任务编号 |

图像不直接放在 Parquet 中。LeRobot 根据 episode 视频时间范围和当前帧时间戳，
从两路 MP4 中解码对应图像。

## 3. 训练样本格式

训练时，LeRobot 将一个基础帧组装成近似如下的样本：

```python
{
    "observation.images.wrist": FloatTensor[3, 480, 640],
    "observation.images.front": FloatTensor[3, 480, 640],
    "observation.state": FloatTensor[22],
    "task": "pick up the vga and insert into the motherboard",
    "action": FloatTensor[50, 6],
    "action_is_pad": BoolTensor[50],
}
```

模型只使用两路图像、state、task 和 action。数据中的 `complementary_info.*`、
时间戳和索引用于记录与数据组织，不是本次 π0.5 SFT 声明的模型输入或监督目标。

### 3.1 State

22 维 state 的固定顺序是：

```text
[joint_0 ... joint_6,                 # 7 个关节角
 ee_x, ee_y, ee_z,                    # 3 维末端位置
 ee_rx, ee_ry, ee_rz,                 # 3 维末端旋转向量
 joint_vel_0 ... joint_vel_6,         # 7 个关节速度
 gripper_width, gripper_grasped]      # 2 个夹爪状态
```

### 3.2 Action

每帧保存一个 6 维末端增量动作：

```text
[dx, dy, dz, drx, dry, drz]
```

前三维是平移增量，后三维是旋转向量增量。这里的动作在数据中已经是 delta，
训练配置中的 `use_relative_actions=false` 表示预处理器不会再次把它转换成相对动作。

π0.5 的 `chunk_size=50`。以当前帧为起点，loader 读取连续 50 个动作，得到
`[50, 6]` 的监督目标；在 30 Hz 下覆盖约 1.67 秒。episode 尾部不足 50 帧时，
loader 补齐动作并通过 `action_is_pad` 标出补齐位置，不会跨到下一条 episode。

### 3.3 预处理

- 原始图像解码为 `[3, 480, 640]` 的 `float32` 张量，随后按 π0.5 流程调整到 224×224。
- state 和 action 使用 `meta/stats.json` 中的 `q01`、`q99` 做 quantile normalization。
- 模型内部固定最大 state/action 维度为 32，实际的 22/6 维会补齐到模型维度。
- task 使用 `google/paligemma-3b-pt-224` tokenizer，最大长度为 200。
- 本次 smoke run 的图像增强配置为 `enable=false`。

## 4. 实际跑通的 smoke 配置

实际验证使用：

```text
base checkpoint: /workspace/wangziyi/models/pi05_base
policy type:     pi05
dtype:           bfloat16
batch size:      2
steps:           200
save frequency:  100
eval frequency:  0
tolerance:       0.04 s
optimizer:       AdamW
learning rate:   2.5e-5
output:          outputs/vla_ft_smoke
```

这次配置是全参数微调：`use_peft=false`、`freeze_vision_encoder=false`、
`train_expert_only=false`。它的目的主要是验证数据加载、训练、保存以及 checkpoint
重新加载的完整链路。200 steps 不能视为已经完成的正式任务训练。

必须保留以下两个命令行参数：

```bash
--policy.input_features=null
--policy.push_to_hub=false
```

前者让 π0.5 根据当前数据集推断两路相机、22 维 state 和 6 维 action；否则 base
checkpoint 中原有的三相机 schema 会导致 feature mismatch。

## 5. 与仓库 README 的微调方案对比

README 中的命令是可替换数据集和 checkpoint 的通用示例；页面底部列出的训练数据和
模型则是原项目公开的双臂旋螺丝资源。两套数据都采用 LeRobot v3 的目录组织和
`observation.*` / `action` 字段约定，但字段数量和张量维度并不相同。当前 FR3 smoke
run 是在同一 LeRobot π0.5 训练入口上，换成了本地 FR3 数据和本地 base checkpoint。

| 对比项 | README 中的方案 | 当前实际 FR3 SFT |
|---|---|---|
| 数据集 | 命令使用 `<HF_ORG>/<DATASET>` 占位符；公开资源列出 `Elvinky/bi-so101-insert-screw-562ep` | 本地 `franka_m1_manual_demo` |
| 机器人/任务 | 双臂 SO101 插螺丝数据 | 单臂 Franka FR3，将 VGA 插入主板 |
| 数据量 | 公开数据名表明为 562 episodes | 5 episodes、748 帧 |
| 视觉输入 | `left_wrist`、`right_wrist`、`right_front`，3 路 RGB | `wrist` + `front`，2 路 RGB |
| 单帧 state | `[12]`：左右臂各 6 个关节位置 | `[22]`：7 个关节角、6 维末端位姿、7 个关节速度、2 个夹爪状态 |
| 单帧 action | `[12]`：左右臂各 6 个目标关节位置 | `[6]`：末端位姿增量 `[dx,dy,dz,drx,dry,drz]` |
| 50 步训练目标 | `[50, 12]` | `[50, 6]` |
| base 模型 | `<BASE_PI05_CHECKPOINT_DIR>` 占位符 | `/workspace/wangziyi/models/pi05_base` |
| batch size | 16 | 2 |
| steps | 30,000 | 200（smoke 验证） |
| 保存间隔 | 5,000 | 100 |
| 输出目录 | `outputs/vla_ft` | `outputs/vla_ft_smoke` |
| 图像增强 | README 未说明 | 关闭 |
| 参数更新范围 | README 未显式说明，默认由 checkpoint/config 决定 | 全参数更新，无 PEFT，不冻结视觉编码器 |
| 共同设置 | `bfloat16`、`eval_freq=0`、`tolerance_s=0.04`、自动推断输入特征、不上传 Hub | 相同 |

所以，两套 SFT 数据的“外层格式”相同：每个训练项都有当前 observation、任务文本和
未来 50 步 action chunk；“内层 schema”不同。README 数据的模型输入是 3 张图像加
12 维双臂关节状态，监督目标是 `[50,12]` 的双臂目标关节位置。FR3 数据的模型输入是
2 张图像加 22 维混合状态，监督目标是 `[50,6]` 的单臂末端增量。

两者另一个核心区别是实验阶段：README 展示的是面向公开双臂任务的正式 30k-step
命令模板，而当前记录的是面向本地 FR3 单任务数据的 200-step 端到端 smoke。
若要进行正式 FR3 微调，应继续使用当前数据 schema 和已验证命令，只调整新的
`output_dir`、训练步数、保存间隔，并按可用显存选择 batch size。

## 6. 证据位置

- 实际操作交接：`/workspace/wangziyi/SFT_HANDOFF.md`
- 数据集 schema：`datasets/franka_m1_manual_demo/meta/info.json`
- 归一化统计：`datasets/franka_m1_manual_demo/meta/stats.json`
- 任务表：`datasets/franka_m1_manual_demo/meta/tasks.parquet`
- episode 索引：`datasets/franka_m1_manual_demo/meta/episodes/chunk-000/file-000.parquet`
- 实际训练配置：`outputs/vla_ft_smoke/checkpoints/000200/pretrained_model/train_config.json`
- 原 README 微调示例：`README.md` 的 `Finetune VLA` 与 `Model & Dataset` 两节
