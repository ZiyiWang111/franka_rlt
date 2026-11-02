#!/usr/bin/env bash
# SpaceMouse: 9 mm/s per axis; roll/pitch 0.06, yaw 0.072 rad/s; Esc or Ctrl+C to stop.
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/fr3_spacemouse_teleop.sh" --execute "$@"
