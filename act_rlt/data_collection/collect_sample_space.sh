#!/usr/bin/env bash
# Start a fresh control server, check both cameras, and run sample-space collection.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTROL_SERVER_SCRIPT="${PROJECT_ROOT}/scripts/start_evo_franka_control_server.sh"
RECORDER_SCRIPT="${SCRIPT_DIR}/record_sample_space.py"
CAMERA_CHECK_SCRIPT="${PROJECT_ROOT}/scripts/check_franka_cameras.py"

ROBOT_IP="${ROBOT_IP:-172.16.0.2}"
CONTROL_PYTHON="${CONTROL_PYTHON:-/home/embint/robolab-venv/bin/python}"
COLLECT_PYTHON="${COLLECT_PYTHON:-/home/embint/miniconda3/envs/evo-rlt/bin/python}"
WRIST_CAMERA_SERIAL="${WRIST_CAMERA_SERIAL:-349622072679}"
FRONT_CAMERA_SERIAL="${FRONT_CAMERA_SERIAL:-233522075778}"
CONTROL_LOG="${CONTROL_LOG:-/tmp/act-rlt-control-server.log}"

DATASET_NAME="${DATASET_NAME:-act_rlt_sample_space}"
DATASET_ROOT="${DATASET_ROOT:-}"
FPS="${FPS:-15}"
EPISODES="${EPISODES:-10}"
EPISODE_TIME_S="${EPISODE_TIME_S:-60}"
TASK="${TASK:-Move from a sampled workspace pose to the reference point and advance along -Y.}"
MOVE_SPEED="${MOVE_SPEED:-0.02}"
INSERTION_SPEED="${INSERTION_SPEED:-0.01}"
PRE_EPISODE_SLEEP="${PRE_EPISODE_SLEEP:-1.0}"
POST_EPISODE_SLEEP="${POST_EPISODE_SLEEP:-1.0}"
X_HALF_RANGE="${X_HALF_RANGE:-0.01}"
Y_RANGE="${Y_RANGE:-0.02}"
Z_HALF_RANGE="${Z_HALF_RANGE:-0.01}"
INSERT_MINUS_Y="${INSERT_MINUS_Y:-0.01}"
ABORT_KEY="${ABORT_KEY:-x}"
SEED="${SEED:-}"
RESUME="${RESUME:-false}"
CAMERA_CHECK_FRAMES="${CAMERA_CHECK_FRAMES:-5}"

CONTROL_PID=""
CONTROL_PID_FILE=""

usage() {
    cat <<'EOF'
Usage: act_rlt/data_collection/collect_sample_space.sh [options]

Dataset:
  --dataset NAME          Dataset repo/name (default: act_rlt_sample_space)
  --root PATH             Dataset root (default: PROJECT/datasets/NAME)
  --fps HZ                Recording rate (default: 15)
  --episodes N            Number of newly saved episodes (default: 10)
  --episode-time SEC      Per-attempt timeout (default: 60)
  --task TEXT             LeRobot task description
  --resume                Append to an existing compatible dataset

Trajectory (metres, robot-base frame):
  --move-speed MPS        Reposition and sample->reference speed (default: 0.02)
  --insertion-speed MPS   Reference->final speed (default: 0.01)
  --pre-episode-sleep SEC Delay after reaching sample (default: 1.0)
  --post-episode-sleep SEC Delay after saving episode (default: 1.0)
  --x-half-range M        Sample x0 +/- range (default: 0.01)
  --y-range M             Sample y0 through y0+range (default: 0.02)
  --z-half-range M        Sample z0 +/- range (default: 0.01)
  --insert-minus-y M      Recorded -Y advance (default: 0.01)
  --abort-key KEY         Stop and discard active episode (default: x)
  --seed N                Optional reproducible sample sequence

Hardware/runtime:
  --robot-ip IP
  --wrist-camera SERIAL
  --front-camera SERIAL
  --collect-python PATH
  --control-python PATH
  --control-log PATH
  -h, --help

Workflow:
  1. Hand-guide to reference p0 and press ENTER once.
  2. The recorder samples and moves to a start pose outside the episode.
  3. After 1.0 s it records sample -> p0 -> p0-1cm(Y) and saves automatically.
  4. After another 1.0 s it retreats to p0, moves to the next sample, and repeats.

No per-episode ENTER is required. During either 0.5 s transition window,
1 moves to p0, 2 moves to the current sample, and Q quits; numeric keys pause
automation until ENTER. Ctrl+C stops collection at any time.
Inside an episode: X (or --abort-key) stops motion and discards the take.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

while (($#)); do
    case "$1" in
        --dataset) DATASET_NAME="${2:?--dataset requires a value}"; shift 2 ;;
        --root) DATASET_ROOT="${2:?--root requires a value}"; shift 2 ;;
        --fps) FPS="${2:?--fps requires a value}"; shift 2 ;;
        --episodes) EPISODES="${2:?--episodes requires a value}"; shift 2 ;;
        --episode-time) EPISODE_TIME_S="${2:?--episode-time requires a value}"; shift 2 ;;
        --task) TASK="${2:?--task requires a value}"; shift 2 ;;
        --move-speed) MOVE_SPEED="${2:?--move-speed requires a value}"; shift 2 ;;
        --insertion-speed) INSERTION_SPEED="${2:?--insertion-speed requires a value}"; shift 2 ;;
        --pre-episode-sleep) PRE_EPISODE_SLEEP="${2:?--pre-episode-sleep requires a value}"; shift 2 ;;
        --post-episode-sleep) POST_EPISODE_SLEEP="${2:?--post-episode-sleep requires a value}"; shift 2 ;;
        --x-half-range) X_HALF_RANGE="${2:?--x-half-range requires a value}"; shift 2 ;;
        --y-range) Y_RANGE="${2:?--y-range requires a value}"; shift 2 ;;
        --z-half-range) Z_HALF_RANGE="${2:?--z-half-range requires a value}"; shift 2 ;;
        --insert-minus-y) INSERT_MINUS_Y="${2:?--insert-minus-y requires a value}"; shift 2 ;;
        --abort-key) ABORT_KEY="${2:?--abort-key requires a value}"; shift 2 ;;
        --seed) SEED="${2:?--seed requires a value}"; shift 2 ;;
        --resume) RESUME=true; shift ;;
        --robot-ip) ROBOT_IP="${2:?--robot-ip requires a value}"; shift 2 ;;
        --wrist-camera) WRIST_CAMERA_SERIAL="${2:?--wrist-camera requires a value}"; shift 2 ;;
        --front-camera) FRONT_CAMERA_SERIAL="${2:?--front-camera requires a value}"; shift 2 ;;
        --collect-python) COLLECT_PYTHON="${2:?--collect-python requires a value}"; shift 2 ;;
        --control-python) CONTROL_PYTHON="${2:?--control-python requires a value}"; shift 2 ;;
        --control-log) CONTROL_LOG="${2:?--control-log requires a value}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (use --help)" ;;
    esac
done

is_positive_number() {
    [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ && "$1" =~ [1-9] ]]
}

[[ -n "$DATASET_NAME" ]] || die "dataset name cannot be empty"
[[ "$FPS" =~ ^[1-9][0-9]*$ ]] || die "--fps must be a positive integer"
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || die "--episodes must be a positive integer"
for pair in \
    "episode-time:${EPISODE_TIME_S}" \
    "move-speed:${MOVE_SPEED}" \
    "insertion-speed:${INSERTION_SPEED}" \
    "pre-episode-sleep:${PRE_EPISODE_SLEEP}" \
    "post-episode-sleep:${POST_EPISODE_SLEEP}" \
    "x-half-range:${X_HALF_RANGE}" \
    "y-range:${Y_RANGE}" \
    "z-half-range:${Z_HALF_RANGE}" \
    "insert-minus-y:${INSERT_MINUS_Y}"; do
    name="${pair%%:*}"
    value="${pair#*:}"
    is_positive_number "$value" || die "--${name} must be positive"
done
[[ ${#ABORT_KEY} -eq 1 ]] || die "--abort-key must be exactly one character"
[[ -z "$SEED" || "$SEED" =~ ^-?[0-9]+$ ]] || die "--seed must be an integer"
[[ "$RESUME" == true || "$RESUME" == false ]] || die "RESUME must be true or false"
[[ -x "$COLLECT_PYTHON" ]] || die "collection Python is not executable: $COLLECT_PYTHON"
[[ -x "$CONTROL_PYTHON" ]] || die "control-server Python is not executable: $CONTROL_PYTHON"
[[ -f "$CONTROL_SERVER_SCRIPT" ]] || die "control-server launcher not found: $CONTROL_SERVER_SCRIPT"
[[ -f "$RECORDER_SCRIPT" ]] || die "recorder not found: $RECORDER_SCRIPT"

if [[ -z "$DATASET_ROOT" ]]; then
    DATASET_ROOT="${PROJECT_ROOT}/datasets/${DATASET_NAME}"
fi

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [[ -z "$CONTROL_PID" && -n "$CONTROL_PID_FILE" && -s "$CONTROL_PID_FILE" ]]; then
        CONTROL_PID="$(<"$CONTROL_PID_FILE")"
        [[ "$CONTROL_PID" =~ ^[1-9][0-9]*$ ]] || CONTROL_PID=""
    fi
    if [[ -n "$CONTROL_PID" ]] && kill -0 "$CONTROL_PID" 2>/dev/null; then
        local cmdline=""
        if [[ -r "/proc/${CONTROL_PID}/cmdline" ]]; then
            cmdline="$(tr '\0' ' ' < "/proc/${CONTROL_PID}/cmdline")"
        fi
        if [[ "$cmdline" == *"evo_franka.control_server"* ]]; then
            echo "Stopping control server pid=${CONTROL_PID} ..."
            kill "$CONTROL_PID" 2>/dev/null || true
            for _ in {1..30}; do
                kill -0 "$CONTROL_PID" 2>/dev/null || break
                sleep 0.1
            done
            if kill -0 "$CONTROL_PID" 2>/dev/null; then
                echo "Control server did not stop after 3 seconds; sending SIGKILL." >&2
                kill -9 "$CONTROL_PID" 2>/dev/null || true
            fi
        else
            echo "WARN: pid ${CONTROL_PID} is not the expected control server; not killing it." >&2
        fi
    fi
    if [[ -n "$CONTROL_PID_FILE" ]]; then
        rm -f -- "$CONTROL_PID_FILE"
    fi
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "ACT-RLT sample-space collection:"
echo "  dataset=${DATASET_NAME}, root=${DATASET_ROOT}"
echo "  fps=${FPS}, new_episodes=${EPISODES}, timeout=${EPISODE_TIME_S}s"
echo "  sample x=+/-${X_HALF_RANGE}m, y=[0,+${Y_RANGE}]m, z=+/-${Z_HALF_RANGE}m from p0"
echo "  move_speed=${MOVE_SPEED}m/s, insertion_speed=${INSERTION_SPEED}m/s"
echo "  pre_episode_sleep=${PRE_EPISODE_SLEEP}s, post_episode_sleep=${POST_EPISODE_SLEEP}s"
echo "  final=-Y ${INSERT_MINUS_Y}m, abort_key=${ABORT_KEY}"

CONTROL_PID_FILE="$(mktemp /tmp/act-rlt-control-server.XXXXXX.pid)"
echo "Starting a fresh Evo-RLT control server ..."
CTRL_LOG="$CONTROL_LOG" CTRL_PIDF="$CONTROL_PID_FILE" \
    bash "$CONTROL_SERVER_SCRIPT" "$ROBOT_IP" "$CONTROL_PYTHON"
CONTROL_PID="$(<"$CONTROL_PID_FILE")"
[[ "$CONTROL_PID" =~ ^[1-9][0-9]*$ ]] || die "invalid control-server PID: $CONTROL_PID"
kill -0 "$CONTROL_PID" 2>/dev/null || die "control server exited; see $CONTROL_LOG"

echo "Checking wrist and front RealSense cameras ..."
"$COLLECT_PYTHON" "$CAMERA_CHECK_SCRIPT" \
    --wrist-serial "$WRIST_CAMERA_SERIAL" \
    --front-serial "$FRONT_CAMERA_SERIAL" \
    --wrist-width 640 \
    --wrist-height 480 \
    --front-width 640 \
    --front-height 480 \
    --frames "$CAMERA_CHECK_FRAMES"

RECORDER_ARGS=(
    --dataset "$DATASET_NAME"
    --root "$DATASET_ROOT"
    --fps "$FPS"
    --episodes "$EPISODES"
    --episode-time "$EPISODE_TIME_S"
    --task "$TASK"
    --robot-ip "$ROBOT_IP"
    --wrist-camera "$WRIST_CAMERA_SERIAL"
    --front-camera "$FRONT_CAMERA_SERIAL"
    --move-speed "$MOVE_SPEED"
    --insertion-speed "$INSERTION_SPEED"
    --pre-episode-sleep "$PRE_EPISODE_SLEEP"
    --post-episode-sleep "$POST_EPISODE_SLEEP"
    --x-half-range "$X_HALF_RANGE"
    --y-range "$Y_RANGE"
    --z-half-range "$Z_HALF_RANGE"
    --insert-minus-y "$INSERT_MINUS_Y"
    --abort-key "$ABORT_KEY"
)
if [[ -n "$SEED" ]]; then
    RECORDER_ARGS+=(--seed "$SEED")
fi
if [[ "$RESUME" == true ]]; then
    RECORDER_ARGS+=(--resume)
fi

echo "Camera preflight passed. Starting recorder ..."
"$COLLECT_PYTHON" "$RECORDER_SCRIPT" "${RECORDER_ARGS[@]}"
