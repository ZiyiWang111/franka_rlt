#!/usr/bin/env bash
# Borrow an idle Evo-RLT server, or start one and stop only that owned process.
set -Eeuo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COLLECT_PYTHON="${COLLECT_PYTHON:-/home/embint/miniconda3/envs/evo-rlt/bin/python}"
CONTROL_PYTHON="${CONTROL_PYTHON:-/home/embint/robolab-venv/bin/python}"
ROBOT_IP="${ROBOT_IP:-172.16.0.2}"
DATASET_NAME="${DATASET_NAME:-act_soft_approach}"
DATASET_ROOT="${DATASET_ROOT:-}"
CONTROL_PID=""
CONTROL_PID_FILE=""
FORWARDED=()
HELP=false
while (($#)); do
    case "$1" in
        --dataset) DATASET_NAME="${2:?--dataset requires a value}"; shift 2 ;;
        --root) DATASET_ROOT="${2:?--root requires a value}"; shift 2 ;;
        --robot-ip) ROBOT_IP="${2:?--robot-ip requires a value}"; shift 2 ;;
        --collect-python) COLLECT_PYTHON="${2:?--collect-python requires a value}"; shift 2 ;;
        --control-python) CONTROL_PYTHON="${2:?--control-python requires a value}"; shift 2 ;;
        -h|--help) HELP=true; FORWARDED+=("--help"); shift ;;
        *) FORWARDED+=("$1"); shift ;;
    esac
done
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/datasets/${DATASET_NAME}}"
RECORDER_ARGS=(--dataset "$DATASET_NAME" --root "$DATASET_ROOT" --robot-ip "$ROBOT_IP" "${FORWARDED[@]}")
if [[ "$HELP" == true ]]; then
    echo "Launcher options: --dataset NAME --root PATH --collect-python PATH --control-python PATH"
    exec "$COLLECT_PYTHON" -m act_rlt.data_collection.record_soft_approach "${RECORDER_ARGS[@]}"
fi
# Validate without touching a robot, camera or dataset directory.
"$COLLECT_PYTHON" -c '
import sys
from act_rlt.data_collection.record_soft_approach import build_parser, validate_args, verify_dataset_mode
args = build_parser().parse_args(sys.argv[1:])
validate_args(args)
verify_dataset_mode(args)
' "${RECORDER_ARGS[@]}"

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [[ -z "$CONTROL_PID" && -n "$CONTROL_PID_FILE" && -s "$CONTROL_PID_FILE" ]]; then
        CONTROL_PID="$(<"$CONTROL_PID_FILE")"
    fi
    if [[ "$CONTROL_PID" =~ ^[1-9][0-9]*$ ]] && kill -0 "$CONTROL_PID" 2>/dev/null; then
        local command_line=""
        [[ ! -r "/proc/${CONTROL_PID}/cmdline" ]] || command_line="$(tr '\0' ' ' < "/proc/${CONTROL_PID}/cmdline")"
        if [[ "$command_line" == *"evo_franka.control_server"* ]]; then
            echo "Stopping this launcher's control server pid=$CONTROL_PID ..."
            kill "$CONTROL_PID" 2>/dev/null || true
        fi
    fi
    [[ -z "$CONTROL_PID_FILE" ]] || rm -f -- "$CONTROL_PID_FILE"
    exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if "$CONTROL_PYTHON" -m evo_franka.control_client --ping >/dev/null 2>&1; then
    echo "Using existing control server; it will remain running when collection ends."
else
    CONTROL_PID_FILE="$(mktemp /tmp/act-soft-control-server.XXXXXX.pid)"
    CTRL_LOG="${CTRL_LOG:-/tmp/act-soft-control-server.log}" CTRL_PIDF="$CONTROL_PID_FILE" \
        bash scripts/start_evo_franka_control_server.sh "$ROBOT_IP" "$CONTROL_PYTHON"
    CONTROL_PID="$(<"$CONTROL_PID_FILE")"
fi
"$COLLECT_PYTHON" -m act_rlt.data_collection.record_soft_approach "${RECORDER_ARGS[@]}"
