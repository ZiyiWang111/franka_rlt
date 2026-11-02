#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$repo_dir:$repo_dir/src:${PYTHONPATH:-}"
exec "${SPACEMOUSE_PYTHON:-/home/embint/robolab-venv/bin/python}" -u \
    "$repo_dir/scripts/fr3_spacemouse_record_replay.py" "$@"
