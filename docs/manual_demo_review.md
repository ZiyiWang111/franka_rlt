# 删除数据集末尾的 episode

在项目根目录运行：

```bash
PYTHONPATH=src /home/embint/miniconda3/envs/evo-rlt/bin/python -m evo_rlt.cli.review_manual_demo
```

已安装项目命令时，也可以运行 `evo-rlt-review-manual-demo`。

1. 从项目的 `datasets/` 目录开始，输入目录编号逐层选择，直到找到包含 `meta/info.json` 的具体 LeRobot 数据集。
2. 输入正整数 N，立即删除最后 N 个 episode。例如有 30 个 episode，输入 3 会删除索引 27、28、29，保留 0–26。
3. 完成后脚本退出。目录选择时可输入 `b` 返回上一级；输入 `q` 或按 Ctrl+C 退出。

必须至少保留 1 个 episode（LeRobot 编辑工具的限制）。请先停止向该数据集写入的采集程序。
删除使用 LeRobot 工具重建数据、视频和元数据，验证 episode 数量后替换原目录，成功后清理临时备份。
该脚本只提供目录选择和末尾批量删除，不再包含可视化、异常扫描或逐条审查功能。
