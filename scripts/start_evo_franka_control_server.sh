#!/usr/bin/env bash
# Start the Evo-RLT control server with the Franka-capable robotLab Python.
# This script is self-contained in Evo-RLT and never modifies robotLab.
set -Eeuo pipefail

ROBOT_IP="${1:?robot IP required}"
CONTROL_PYTHON="${2:-/home/embint/robolab-venv/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTROL_LOG="${CTRL_LOG:-/tmp/evo-rlt-franka-control-server.log}"
CONTROL_PID_FILE="${CTRL_PIDF:-/tmp/evo-rlt-franka-control-server.pid}"

[[ -x "$CONTROL_PYTHON" ]] || { echo "control Python is not executable: $CONTROL_PYTHON" >&2; exit 2; }

CPU_COUNT="$(nproc --all 2>/dev/null || nproc)"
if [[ "$CPU_COUNT" -ge 8 ]]; then
    DEFAULT_CORES="$((CPU_COUNT - 4))-$((CPU_COUNT - 1))"
else
    DEFAULT_CORES="0-$((CPU_COUNT - 1))"
fi
CONTROL_CORES="${CTRL_CORES:-$DEFAULT_CORES}"

# Free the fixed ZMQ ports from either protocol version. The bracketed regexes
# do not match these pkill command lines themselves.
pkill -9 -f "[e]vo_franka\.control_server" 2>/dev/null || true
pkill -9 -f "[r]obots\.franka\.control_server" 2>/dev/null || true
sleep 2

[[ -f "$CONTROL_LOG.prev" ]] && mv -f "$CONTROL_LOG.prev" "$CONTROL_LOG.prev2"
[[ -f "$CONTROL_LOG" ]] && mv -f "$CONTROL_LOG" "$CONTROL_LOG.prev"

LAUNCH=(taskset -c "$CONTROL_CORES")
if chrt -f 80 true 2>/dev/null; then
    LAUNCH+=(chrt -f 80)
    RT_STATUS=yes
else
    RT_STATUS=no
    echo "WARN: SCHED_FIFO not permitted; control server will run non-RT" >&2
fi
LAUNCH+=(
    env "PYTHONPATH=${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
    "$CONTROL_PYTHON" -u -m evo_franka.control_server --ip "$ROBOT_IP"
)

nohup "${LAUNCH[@]}" </dev/null >"$CONTROL_LOG" 2>&1 &
CONTROL_PID=$!
echo "$CONTROL_PID" >"$CONTROL_PID_FILE"

for _ in {1..30}; do
    if grep -q "control server up" "$CONTROL_LOG" 2>/dev/null; then
        echo "evo_franka control server READY pid=${CONTROL_PID} cores=${CONTROL_CORES} rt=${RT_STATUS}"
        exit 0
    fi
    if ! kill -0 "$CONTROL_PID" 2>/dev/null; then
        echo "evo_franka control server DIED on start:" >&2
        tail -n 10 "$CONTROL_LOG" >&2 || true
        exit 1
    fi
    sleep 0.5
done

echo "evo_franka control server NOT ready in time:" >&2
tail -n 10 "$CONTROL_LOG" >&2 || true
kill "$CONTROL_PID" 2>/dev/null || true
exit 1
