#!/usr/bin/env bash
# Coarse collection owns the server it starts and stops it on exit.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COLLECT_PYTHON="${COLLECT_PYTHON:-/home/embint/miniconda3/envs/evo-rlt/bin/python}"
CONTROL_PYTHON="${CONTROL_PYTHON:-/home/embint/robolab-venv/bin/python}"
CONTROL_LOG="${CONTROL_LOG:-/tmp/act-coarse-control-server.log}"
ROBOT_IP="${ROBOT_IP:-172.16.0.2}"
RECORDER_ARGS=()
READ_ONLY=false
CONTROL_PID=""
CONTROL_PID_FILE=""
while (($#)); do
    case "$1" in
        --collect-python) COLLECT_PYTHON="${2:?--collect-python requires a value}"; shift 2 ;;
        --control-python) CONTROL_PYTHON="${2:?--control-python requires a value}"; shift 2 ;;
        --control-log) CONTROL_LOG="${2:?--control-log requires a value}"; shift 2 ;;
        --robot-ip) ROBOT_IP="${2:?--robot-ip requires a value}"; shift 2 ;;
        --robot-ip=*) ROBOT_IP="${1#*=}"; shift ;;
        -h|--help|--dry-run|--validate-only) READ_ONLY=true; RECORDER_ARGS+=("$1"); shift ;;
        *) RECORDER_ARGS+=("$1"); shift ;;
    esac
done
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[[ -x "$COLLECT_PYTHON" ]] || { echo "Collection Python is not executable: $COLLECT_PYTHON" >&2; exit 2; }
RECORDER_ARGS+=(--robot-ip "$ROBOT_IP")
if [[ "$READ_ONLY" == true ]]; then
    exec "$COLLECT_PYTHON" -m act_rlt.data_collection.record_coarse "${RECORDER_ARGS[@]}"
fi
# Validate arguments, root/resume compatibility before opening the robot or cameras.
"$COLLECT_PYTHON" -m act_rlt.data_collection.record_coarse "${RECORDER_ARGS[@]}" --validate-only
[[ -x "$CONTROL_PYTHON" ]] || { echo "Control Python is not executable: $CONTROL_PYTHON" >&2; exit 2; }
cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [[ -z "$CONTROL_PID" && -n "$CONTROL_PID_FILE" && -s "$CONTROL_PID_FILE" ]]; then
        CONTROL_PID="$(<"$CONTROL_PID_FILE")"
    fi
    if [[ "$CONTROL_PID" =~ ^[1-9][0-9]*$ ]] && kill -0 "$CONTROL_PID" 2>/dev/null; then
        local cmdline=""
        [[ ! -r "/proc/${CONTROL_PID}/cmdline" ]] || cmdline="$(tr '\0' ' ' < "/proc/${CONTROL_PID}/cmdline")"
        if [[ "$cmdline" == *"evo_franka.control_server"* ]]; then
            echo "Stopping owned coarse control server pid=${CONTROL_PID} ..."
            kill "$CONTROL_PID" 2>/dev/null || true
            for _ in {1..30}; do
                kill -0 "$CONTROL_PID" 2>/dev/null || break
                sleep 0.1
            done
            kill -0 "$CONTROL_PID" 2>/dev/null && kill -9 "$CONTROL_PID" 2>/dev/null || true
        fi
    fi
    [[ -z "$CONTROL_PID_FILE" ]] || rm -f -- "$CONTROL_PID_FILE"
    exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
CONTROL_PID_FILE="$(mktemp /tmp/act-coarse-server.XXXXXX.pid)"
echo "Starting coarse control server; RGB cameras are opened by the recorder."
CTRL_LOG="$CONTROL_LOG" CTRL_PIDF="$CONTROL_PID_FILE" \
    bash "$PROJECT_ROOT/scripts/start_evo_franka_control_server.sh" "$ROBOT_IP" "$CONTROL_PYTHON"
CONTROL_PID="$(<"$CONTROL_PID_FILE")"
[[ "$CONTROL_PID" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid control-server PID" >&2; exit 1; }
"$COLLECT_PYTHON" -m act_rlt.data_collection.record_coarse "${RECORDER_ARGS[@]}"
