#!/usr/bin/env bash
# Read-only input worker for Stage-2: never creates a robot client.
set -eo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
support_dir=${SPACEMOUSE_SUPPORT_DIR:-$(dirname "$repo_dir")/spacemouse_probe}
python_bin=${SPACEMOUSE_PYTHON:-/home/embint/robolab-venv/bin/python}
source /opt/ros/humble/setup.bash
export PYTHONPATH="$repo_dir/src:$support_dir/python_deps:$support_dir/official/src/spacemouse_publisher:${PYTHONPATH:-}"
exec "$python_bin" -u -m evo_franka.spacemouse_input "$@"
