# 数据集与模型目录

原数据集及备份保存在 `datasets/ethernet_fine/<dataset>`；新的粗定位数采保存至
`datasets/ethernet_coarse/<dataset>`，见 [Coarse 数采](../act_rlt/data_collection/COARSE_COLLECTION.md)。
模型及其训练状态按用途保存：

| 目录 | 内容 |
|---|---|
| `outputs/ethernet_act/<run>` | ACT 模型；PI05 相关测时文件也保存在本类 |
| `outputs/ethernet_token/<run>` | Stage 1 / RL Token 模型 |
| `outputs/ethernet_rlt/<run>` | Stage 2 RL 模型和快照 |

每类目录中的 `MODEL_INVENTORY.md` 和 `model_inventory.json` 为本机模型清单。

普通数采启动方式不变，默认输出位置已更新：

```bash
bash act_rlt/data_collection/collect_sample_space.sh --dataset act_demo --insertion-cm 1
```

以上命令写入 `datasets/ethernet_fine/act_demo`。显式 `--root` 仍使用指定位置。

ACT inference 示例：

```bash
python -m act_rlt.infer \
  --checkpoint outputs/ethernet_act/act_dataset_928_state7_act \
  --no-temporal-ensemble --n-action-steps 4 --fps 30 --duration 7
```

训练、数据预处理和推理命令中的路径示例均已更新。ACT、Token 和 Stage 2
加载器也兼容旧 checkpoint 中保存的旧目录及训练服务器绝对路径：原路径不存在时，
在本机新分类目录下查找同名模型。不创建旧目录的符号链接；模型权重、优化器和
replay 二进制文件保持原样，保存为 JSON 的路径配置已更新。
