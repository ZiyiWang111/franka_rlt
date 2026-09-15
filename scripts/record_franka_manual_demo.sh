#!/usr/bin/env bash
# One-command FR3 manual-demo collection launcher.
# Starts a fresh Evo-RLT Franka control server, verifies both RealSense streams,
# runs Evo-RLT recording, and stops the exact server process on every exit path.
set -Eeuo pipefail

PROJECT_ROOT="/home/embint/wzy/Evo-RLT"
CONTROL_SERVER_SCRIPT="${PROJECT_ROOT}/scripts/start_evo_franka_control_server.sh"

ROBOT_IP="${ROBOT_IP:-172.16.0.2}"
CONTROL_PYTHON="${CONTROL_PYTHON:-/home/embint/robolab-venv/bin/python}"
COLLECT_PYTHON="${COLLECT_PYTHON:-/home/embint/miniconda3/envs/evo-rlt/bin/python}"
WRIST_CAMERA_SERIAL="${WRIST_CAMERA_SERIAL:-349622072679}"
FRONT_CAMERA_SERIAL="${FRONT_CAMERA_SERIAL:-233522075778}"
CONTROL_LOG="${CONTROL_LOG:-/tmp/evo-rlt-franka-control-server.log}"

DATASET_NAME="${DATASET_NAME:-franka_manual_demo}"
DATASET_ROOT="${DATASET_ROOT:-}"
FPS="${FPS:-30}"
EPISODES="${EPISODES:-10}"
EPISODE_TIME_S="${EPISODE_TIME_S:-3000}"
TASK="${TASK:-Insert the copper screw into the black sleeve.}"
RESUME="${RESUME:-false}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"
CAMERA_CHECK_FRAMES="${CAMERA_CHECK_FRAMES:-5}"

CONTROL_PID=""
CONTROL_PID_FILE=""
EXTRA_BACKEND_ARGS=()

usage() {
    cat <<'EOF'
Usage: scripts/record_franka_manual_demo.sh [options] [-- backend-options...]

Basic options:
  --dataset NAME          Dataset repo/name (default: franka_manual_demo)
  --root PATH             Local dataset root (default: PROJECT/datasets/NAME)
  --fps HZ                Collection frequency, integer (default: 30)
  --episodes N            Number of episodes to save (default: 10)
  --episode-time SEC      Maximum duration of one episode (default: 3000)
  --task TEXT             Task instruction
  --resume                Resume an existing dataset
  --no-sounds             Disable voice prompts

Hardware/runtime options:
  --robot-ip IP
  --wrist-camera SERIAL
  --front-camera SERIAL
  --collect-python PATH   Python for Evo-RLT (default: evo-rlt conda env)
  --control-python PATH   Franka-capable Python used by the Evo-RLT server
  --control-log PATH
  -h, --help

Environment variables with the uppercase names above are also supported.
Arguments after -- are passed directly to the recording backend.

Manual-demo keys:
  ENTER start | RIGHT save | LEFT discard/re-record | ESC quit
  C close gripper | O open gripper
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
        --resume) RESUME=true; shift ;;
        --no-sounds) PLAY_SOUNDS=false; shift ;;
        --robot-ip) ROBOT_IP="${2:?--robot-ip requires a value}"; shift 2 ;;
        --wrist-camera) WRIST_CAMERA_SERIAL="${2:?--wrist-camera requires a value}"; shift 2 ;;
        --front-camera) FRONT_CAMERA_SERIAL="${2:?--front-camera requires a value}"; shift 2 ;;
        --collect-python) COLLECT_PYTHON="${2:?--collect-python requires a value}"; shift 2 ;;
        --control-python) CONTROL_PYTHON="${2:?--control-python requires a value}"; shift 2 ;;
        --control-log) CONTROL_LOG="${2:?--control-log requires a value}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_BACKEND_ARGS=("$@"); break ;;
        *) die "unknown option: $1 (use --help)" ;;
    esac
done

[[ -n "$DATASET_NAME" ]] || die "dataset name cannot be empty"
[[ "$FPS" =~ ^[1-9][0-9]*$ ]] || die "--fps must be a positive integer"
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || die "--episodes must be a positive integer"
[[ "$EPISODE_TIME_S" =~ ^[1-9][0-9]*$ ]] || die "--episode-time must be a positive integer"
[[ "$CAMERA_CHECK_FRAMES" =~ ^[1-9][0-9]*$ ]] || die "CAMERA_CHECK_FRAMES must be positive"
[[ "$RESUME" == true || "$RESUME" == false ]] || die "RESUME must be true or false"
[[ "$PLAY_SOUNDS" == true || "$PLAY_SOUNDS" == false ]] || die "PLAY_SOUNDS must be true or false"
[[ -x "$COLLECT_PYTHON" ]] || die "collection Python is not executable: $COLLECT_PYTHON"
[[ -x "$CONTROL_PYTHON" ]] || die "control-server Python is not executable: $CONTROL_PYTHON"
[[ -f "$CONTROL_SERVER_SCRIPT" ]] || die "control-server launcher not found: $CONTROL_SERVER_SCRIPT"

if [[ -z "$DATASET_ROOT" ]]; then
    DATASET_ROOT="${PROJECT_ROOT}/datasets/${DATASET_NAME}"
fi

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    # control_server.sh may have spawned the process and then failed its own
    # readiness check. Recover the PID from our private pidfile in that case so
    # the failed launcher cannot leave a server behind.
    if [[ -z "$CONTROL_PID" && -n "$CONTROL_PID_FILE" && -s "$CONTROL_PID_FILE" ]]; then
        CONTROL_PID="$(<"$CONTROL_PID_FILE")"
        if [[ ! "$CONTROL_PID" =~ ^[1-9][0-9]*$ ]]; then
            CONTROL_PID=""
        fi
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
            echo "WARN: pid ${CONTROL_PID} no longer belongs to the control server; not killing it." >&2
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

echo "Collection configuration:"
echo "  dataset=${DATASET_NAME}"
echo "  root=${DATASET_ROOT}"
echo "  fps=${FPS}, episodes=${EPISODES}, episode_time=${EPISODE_TIME_S}s"
echo "  robot=${ROBOT_IP}"
echo "  wrist=${WRIST_CAMERA_SERIAL}, front=${FRONT_CAMERA_SERIAL}"

CONTROL_PID_FILE="$(mktemp /tmp/evo-rlt-control-server.XXXXXX.pid)"
echo "Starting a fresh control server ..."
CTRL_LOG="$CONTROL_LOG" CTRL_PIDF="$CONTROL_PID_FILE" \
    bash "$CONTROL_SERVER_SCRIPT" "$ROBOT_IP" "$CONTROL_PYTHON"

CONTROL_PID="$(<"$CONTROL_PID_FILE")"
[[ "$CONTROL_PID" =~ ^[1-9][0-9]*$ ]] || die "invalid control-server PID: $CONTROL_PID"
kill -0 "$CONTROL_PID" 2>/dev/null || die "control server exited immediately; see $CONTROL_LOG"

echo "Checking both RealSense cameras by streaming ${CAMERA_CHECK_FRAMES} frames ..."
"$COLLECT_PYTHON" "$PROJECT_ROOT/scripts/check_franka_cameras.py" \
    --wrist-serial "$WRIST_CAMERA_SERIAL" \
    --front-serial "$FRONT_CAMERA_SERIAL" \
    --frames "$CAMERA_CHECK_FRAMES"

echo "Camera preflight passed. Starting manual-demo recording ..."
"$COLLECT_PYTHON" -m evo_rlt.adapters.lerobot.record.backend \
    --robot.type=franka \
    --robot.robot_ip="$ROBOT_IP" \
    --robot.camera_serial="$WRIST_CAMERA_SERIAL" \
    --robot.front_camera_serial="$FRONT_CAMERA_SERIAL" \
    --dataset.repo_id="$DATASET_NAME" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.single_task="$TASK" \
    --dataset.fps="$FPS" \
    --dataset.num_episodes="$EPISODES" \
    --dataset.episode_time_s="$EPISODE_TIME_S" \
    --dataset.manual_demo_mode=true \
    --dataset.push_to_hub=false \
    --dataset.vcodec=h264 \
    --resume="$RESUME" \
    --play_sounds="$PLAY_SOUNDS" \
    "${EXTRA_BACKEND_ARGS[@]}"
