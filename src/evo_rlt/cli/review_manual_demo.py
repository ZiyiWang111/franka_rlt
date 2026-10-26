"""Choose a local dataset and delete its last N episodes."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DATASETS_ROOT = Path(__file__).resolve().parents[3] / "datasets"


def _select_dataset(base: Path) -> Path | None:
    base = base.resolve()
    current = base
    while True:
        if (current / "meta" / "info.json").is_file():
            return current
        directories = sorted(
            (path for path in current.iterdir()
             if path.is_dir() and not path.name.startswith(".")
             and path.resolve().is_relative_to(base)),
            key=lambda path: path.name,
        )
        print(f"\n当前目录：{current}")
        for index, directory in enumerate(directories, 1):
            print(f"  {index}. {directory.name}/")
        if not directories:
            print("此目录下没有数据集或子目录。")
        answer = input("输入编号进入目录，b 返回上一级，q 退出：").strip().lower()
        if answer == "q":
            return None
        if answer == "b":
            if current != base:
                current = current.parent
            continue
        if answer.isdecimal() and 1 <= int(answer) <= len(directories):
            current = directories[int(answer) - 1].resolve()
        else:
            print("请输入有效目录编号、b 或 q。")


def _read_count(total: int) -> int | None:
    if total < 2:
        print("LeRobot 数据集至少需要保留 1 个 episode，当前无法删除。")
        return None
    while True:
        answer = input(f"输入要删除的末尾 episode 数量（1–{total - 1}，q 退出）：").strip().lower()
        if answer == "q":
            return None
        if answer.isdecimal() and 1 <= int(answer) < total:
            return int(answer)
        print(f"请输入 1–{total - 1} 的整数；至少保留 1 个 episode。")


def _resolve_editor() -> str:
    command = "lerobot-edit-dataset"
    resolved = shutil.which(command)
    if resolved is not None:
        return resolved
    sibling = Path(sys.executable).parent / command
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    raise FileNotFoundError(f"当前 Python 环境中找不到 {command}")


def _delete_tail(root: Path, count: int, snapshot: bytes) -> None:
    info_path = root / "meta" / "info.json"
    total = int(json.loads(snapshot)["total_episodes"])
    if not 1 <= count < total:
        raise ValueError("删除数量必须大于 0 且小于 episode 总数")
    if info_path.read_bytes() != snapshot:
        raise RuntimeError("数据集已发生变化，未执行删除；请停止采集后重试。")
    indices = list(range(total - count, total))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    staging = root.with_name(f".{root.name}.filtering-{stamp}-{os.getpid()}")
    backup = root.with_name(f".{root.name}.backup-{stamp}-{os.getpid()}")
    if staging.exists() or backup.exists():
        raise FileExistsError("临时目录已存在，未执行删除")
    command = [
        _resolve_editor(),
        "--repo_id", root.name,
        "--root", str(root),
        "--new_repo_id", root.name,
        "--new_root", str(staging),
        "--operation.type", "delete_episodes",
        "--operation.episode_indices", json.dumps(indices),
    ]
    print(f"正在删除 episode {indices[0]}–{indices[-1]}，重建数据集……")
    subprocess.run(command, check=True)
    if info_path.read_bytes() != snapshot:
        raise RuntimeError(f"重建期间数据集发生变化，原目录未替换；临时结果：{staging}")
    new_info = json.loads((staging / "meta" / "info.json").read_bytes())
    if int(new_info["total_episodes"]) != total - count:
        raise RuntimeError(f"重建后的 episode 数量不正确，原目录未替换；临时结果：{staging}")
    root.rename(backup)
    try:
        staging.rename(root)
    except Exception:
        backup.rename(root)
        raise
    shutil.rmtree(backup)
    print(f"已删除最后 {count} 个 episode，剩余 {total - count} 个。\n数据集：{root}")


def main() -> None:
    try:
        root = _select_dataset(DATASETS_ROOT)
        if root is None:
            return
        snapshot = (root / "meta" / "info.json").read_bytes()
        total = int(json.loads(snapshot)["total_episodes"])
        print(f"\n已选择数据集：{root}\nepisode 总数：{total}")
        count = _read_count(total)
        if count is not None:
            _delete_tail(root, count, snapshot)
    except (KeyboardInterrupt, EOFError):
        print("\n已退出。")
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
