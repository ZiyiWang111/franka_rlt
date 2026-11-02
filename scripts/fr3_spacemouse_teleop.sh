#!/usr/bin/env bash
set -eo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
support_dir=${SPACEMOUSE_SUPPORT_DIR:-$(dirname "$repo_dir")/spacemouse_probe}
python_bin=${SPACEMOUSE_PYTHON:-/home/embint/robolab-venv/bin/python}
source /opt/ros/humble/setup.bash
export PYTHONPATH="$repo_dir/src:$support_dir/python_deps:$support_dir/official/src/spacemouse_publisher:${PYTHONPATH:-}"
publisher_pid=
cleanup() {
    if [[ -n "$publisher_pid" ]]; then
        kill -INT "$publisher_pid" 2>/dev/null || true
        wait "$publisher_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
"$python_bin" -u -m spacemouse_publisher.pyspacemouse_publisher --ros-args \
    -p "device_path:=${SPACEMOUSE_DEVICE:-/dev/hidraw4}" &
publisher_pid=$!
"$python_bin" -u -m evo_franka.spacemouse_bridge "$@"
