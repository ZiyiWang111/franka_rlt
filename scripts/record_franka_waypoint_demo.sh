#!/usr/bin/env bash
# Start a fresh Evo-RLT Franka control server and run the two-waypoint auto recorder.
set -Eeuo pipefail

PROJECT_ROOT="/home/embint/wzy/Evo-RLT"
CONTROL_SERVER_SCRIPT="${PROJECT_ROOT}/scripts/start_evo_franka_control_server.sh"

ROBOT_IP="${ROBOT_IP:-172.16.0.2}"
CONTROL_PYTHON="${CONTROL_PYTHON:-/home/embint/robolab-venv/bin/python}"
COLLECT_PYTHON="${COLLECT_PYTHON:-/home/embint/miniconda3/envs/evo-rlt/bin/python}"
WRIST_CAMERA_SERIAL="${WRIST_CAMERA_SERIAL:-349622072679}"
FRONT_CAMERA_SERIAL="${FRONT_CAMERA_SERIAL:-233522075778}"
CONTROL_LOG="${CONTROL_LOG:-/tmp/evo-rlt-franka-control-server.log}"

DATASET_NAME="${DATASET_NAME:-franka_waypoint_demo}"
DATASET_ROOT="${DATASET_ROOT:-}"
FPS="${FPS:-15}"
EPISODES="${EPISODES:-10}"
EPISODE_TIME_S="${EPISODE_TIME_S:-300}"
TASK="${TASK:-Move between two taught waypoints and insert along -Y.}"
MOVE_SPEED="${MOVE_SPEED:-0.05}"
FINAL_MOVE_SPEED="${FINAL_MOVE_SPEED:-0.05}"
FINAL_MINUS_Y="${FINAL_MINUS_Y:-0.01}"
START_SQUARE_SIDE="${START_SQUARE_SIDE:-0.15}"
GRASP_SQUARE_SIDE="${GRASP_SQUARE_SIDE:-0.10}"
START_YAW_RANGE_DEG="${START_YAW_RANGE_DEG:-15}"
GRIPPER_WIDTH="${GRIPPER_WIDTH:-0.0}"
GRIPPER_FORCE="${GRIPPER_FORCE:-40.0}"
ABORT_KEY="${ABORT_KEY:-x}"
RESUME="${RESUME:-false}"
CAMERA_CHECK_FRAMES="${CAMERA_CHECK_FRAMES:-5}"

CONTROL_PID=""
CONTROL_PID_FILE=""

usage() {
    cat <<'EOF'
Usage: scripts/record_franka_waypoint_demo.sh [options]

Dataset options:
  --dataset NAME          Dataset repo/name (default: franka_waypoint_demo)
  --root PATH             Dataset root (default: PROJECT/datasets/NAME)
  --fps HZ                Recording frequency (default: 15)
  --episodes N            Number of saved episodes (default: 10)
  --episode-time SEC      Per-attempt safety timeout (default: 300)
  --task TEXT             Dataset task text
  --resume                Resume an existing dataset

Trajectory options:
  --move-speed MPS        TCP motion speed in m/s (default: 0.05)
  --final-move-speed MPS  Independent point-2 -> -Y speed (default: 0.05)
  --final-minus-y METRES  Distance after point 2 (default: 0.01)
  --start-square-side M   XY start sampling square side length (default: 0.15)
  --grasp-square-side M   XY grasp sampling square side length (default: 0.10)
  --start-yaw-range-deg D Max +/- base-Z yaw variation, 0-15 deg (default: 15)
  --gripper-width METRES  Grasp target width (default: 0.0)
  --gripper-force NEWTONS Grasp force, continuous range 20-70 (default: 40)
  --abort-key KEY         Stop/discard the active episode (default: x)

Hardware/runtime options:
  --robot-ip IP
  --wrist-camera SERIAL
  --front-camera SERIAL
  --collect-python PATH
  --control-python PATH
  --control-log PATH
  -h, --help

Workflow:
  1. ENTER records point 1, then point 2, then INITIAL START.
  2. First episode uses INITIAL START. Later NEXT START poses are sampled in
     a 0.15 m XY square with up to +/-15 deg yaw, both relative to INITIAL START.
     Later POINT 1 poses use a 0.10 m XY square around the first taught grasp.
  3. Outside recording: 1/2/3/4 move to point1/point2/initial/new start;
     C closes, O opens. A missing pose prints NULL.
  4. ENTER starts; robot runs point 1 -> close -> point 2 -> -Y offset.
  During step 4, x immediately stops motion and discards the episode.
  After completion, only s saves, d discards, and q discards and quits. ENTER
  reminds you that the completed episode still needs a decision; numeric pose
  keys and C/O remain available while choosing.

The two taught targets and the initial start remain fixed until this launcher exits.
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
        --final-move-speed) FINAL_MOVE_SPEED="${2:?--final-move-speed requires a value}"; shift 2 ;;
        --final-minus-y) FINAL_MINUS_Y="${2:?--final-minus-y requires a value}"; shift 2 ;;
        --start-square-side) START_SQUARE_SIDE="${2:?--start-square-side requires a value}"; shift 2 ;;
        --grasp-square-side) GRASP_SQUARE_SIDE="${2:?--grasp-square-side requires a value}"; shift 2 ;;
        --start-yaw-range-deg) START_YAW_RANGE_DEG="${2:?--start-yaw-range-deg requires a value}"; shift 2 ;;
        --gripper-width) GRIPPER_WIDTH="${2:?--gripper-width requires a value}"; shift 2 ;;
        --gripper-force) GRIPPER_FORCE="${2:?--gripper-force requires a value}"; shift 2 ;;
        --abort-key) ABORT_KEY="${2:?--abort-key requires a value}"; shift 2 ;;
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

[[ -n "$DATASET_NAME" ]] || die "dataset name cannot be empty"
[[ "$FPS" =~ ^[1-9][0-9]*$ ]] || die "--fps must be a positive integer"
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || die "--episodes must be a positive integer"
[[ "$EPISODE_TIME_S" =~ ^[0-9]+([.][0-9]+)?$ && "$EPISODE_TIME_S" =~ [1-9] ]] \
    || die "--episode-time must be positive"
[[ "$MOVE_SPEED" =~ ^[0-9]+([.][0-9]+)?$ && "$MOVE_SPEED" =~ [1-9] ]] \
    || die "--move-speed must be positive"
[[ "$FINAL_MOVE_SPEED" =~ ^[0-9]+([.][0-9]+)?$ && "$FINAL_MOVE_SPEED" =~ [1-9] ]] \
    || die "--final-move-speed must be positive"
[[ "$FINAL_MINUS_Y" =~ ^[0-9]+([.][0-9]+)?$ && "$FINAL_MINUS_Y" =~ [1-9] ]] \
    || die "--final-minus-y must be positive"
[[ "$START_SQUARE_SIDE" =~ ^[0-9]+([.][0-9]+)?$ && "$START_SQUARE_SIDE" =~ [1-9] ]] \
    || die "--start-square-side must be positive"
[[ "$GRASP_SQUARE_SIDE" =~ ^[0-9]+([.][0-9]+)?$ && "$GRASP_SQUARE_SIDE" =~ [1-9] ]] \
    || die "--grasp-square-side must be positive"
[[ "$START_YAW_RANGE_DEG" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || die "--start-yaw-range-deg must be a number in [0, 15]"
[[ "$GRIPPER_WIDTH" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--gripper-width must be non-negative"
[[ "$GRIPPER_FORCE" =~ ^[0-9]+([.][0-9]+)?$ && "$GRIPPER_FORCE" =~ [1-9] ]] \
    || die "--gripper-force must be positive"
[[ ${#ABORT_KEY} -eq 1 ]] || die "--abort-key must be exactly one character"
[[ "$RESUME" == true || "$RESUME" == false ]] || die "RESUME must be true or false"
[[ -x "$COLLECT_PYTHON" ]] || die "collection Python is not executable: $COLLECT_PYTHON"
[[ -x "$CONTROL_PYTHON" ]] || die "control-server Python is not executable: $CONTROL_PYTHON"
[[ -f "$CONTROL_SERVER_SCRIPT" ]] || die "control-server launcher not found: $CONTROL_SERVER_SCRIPT"

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
                echo "Control server did not stop after 3 s; sending SIGKILL." >&2
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
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "Automatic waypoint collection:"
echo "  dataset=${DATASET_NAME}, root=${DATASET_ROOT}"
echo "  fps=${FPS}, episodes=${EPISODES}, timeout=${EPISODE_TIME_S}s"
echo "  speed=${MOVE_SPEED}m/s, final_speed=${FINAL_MOVE_SPEED}m/s"
echo "  final=-Y ${FINAL_MINUS_Y}m, grasp_width=${GRIPPER_WIDTH}m"
echo "  start_square_side=${START_SQUARE_SIDE}m (centred on immutable initial start)"
echo "  grasp_square_side=${GRASP_SQUARE_SIDE}m (centred on immutable initial grasp)"
echo "  start_yaw_range=+/-${START_YAW_RANGE_DEG}deg about base Z"
echo "  grasp_force=${GRIPPER_FORCE}N, abort_key=${ABORT_KEY}"

CONTROL_PID_FILE="$(mktemp /tmp/evo-rlt-control-server.XXXXXX.pid)"
echo "Starting a fresh control server ..."
CTRL_LOG="$CONTROL_LOG" CTRL_PIDF="$CONTROL_PID_FILE" \
    bash "$CONTROL_SERVER_SCRIPT" "$ROBOT_IP" "$CONTROL_PYTHON"
CONTROL_PID="$(<"$CONTROL_PID_FILE")"
[[ "$CONTROL_PID" =~ ^[1-9][0-9]*$ ]] || die "invalid control-server PID: $CONTROL_PID"
kill -0 "$CONTROL_PID" 2>/dev/null || die "control server exited immediately; see $CONTROL_LOG"

echo "Checking both RealSense cameras ..."
"$COLLECT_PYTHON" "$PROJECT_ROOT/scripts/check_franka_cameras.py" \
    --wrist-serial "$WRIST_CAMERA_SERIAL" \
    --front-serial "$FRONT_CAMERA_SERIAL" \
    --frames "$CAMERA_CHECK_FRAMES"

echo "Camera preflight passed. Starting waypoint recorder ..."
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
    --final-move-speed "$FINAL_MOVE_SPEED"
    --final-minus-y "$FINAL_MINUS_Y"
    --start-square-side "$START_SQUARE_SIDE"
    --grasp-square-side "$GRASP_SQUARE_SIDE"
    --start-yaw-range-deg "$START_YAW_RANGE_DEG"
    --gripper-width "$GRIPPER_WIDTH"
    --gripper-force "$GRIPPER_FORCE"
    --abort-key "$ABORT_KEY"
)
if [[ "$RESUME" == true ]]; then
    RECORDER_ARGS+=(--resume)
fi
"$COLLECT_PYTHON" "$PROJECT_ROOT/scripts/record_franka_waypoint_demo.py" "${RECORDER_ARGS[@]}"
