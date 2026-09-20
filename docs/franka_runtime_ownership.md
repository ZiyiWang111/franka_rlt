# Franka runtime ownership

Evo-RLT's control server acquires a nonblocking Linux file lock before constructing
its controller. Robot and Hand factories require that process's ownership. The
existing client, training, inference, recording and servo interfaces are unchanged.

Start the server with `scripts/start_evo_franka_control_server.sh ROBOT_IP`, or
`python -m evo_franka.control_server --ip ROBOT_IP`. Use a numeric IP address.
If a cooperating server already owns that robot, startup fails without connecting
or stopping the owner. Stop the existing server deliberately before restarting it.
The launcher no longer kills Evo-RLT or legacy RobotLab servers automatically.
Legacy servers do not participate in this lock; stop them before using this setup.

Locks live at `/tmp/evo-franka-runtime/<normalized-ip>.lock`, shared across project
copies and Python environments on this host. The arm and Hand use the same lock.
Do not delete lock files, change their path, or put cooperating processes in
separate mount namespaces. Metadata is diagnostic; only the kernel lock indicates
ownership. The first user creates the directory; permission failures fail closed.
This default is intended for the existing single-user robot host deployment.

Client disconnect, reconnect, recovery and Hand rebuild retain ownership until
the server exits. SIGINT/SIGTERM request shutdown; the server waits for its control
and motion threads and destroys handles before releasing the lock. If cleanup
fails or a thread is stuck, ownership is retained until process exit. Process
death releases the kernel lock; a leftover file is normal. Forked children do not
inherit authorization. A released software lock is not a statement of robot readiness.

Direct driver or Hand-only connections through Evo-RLT now raise `OwnershipError`
outside the server. Importing modules and running IK require no ownership. Tests
should inject fake controllers or fake native constructors, never enable a global
hardware bypass. Ownership regressions use documentation-only addresses and fakes.

This is a cooperative guard against accidental use. It does not authenticate IPC
clients, prevent direct `franky`/libfranka access, or isolate other hosts. IPC
authorization and OS/network isolation are separate changes.

## Fault diagnostics

Fault formatting and health logging use the last cached robot state, never live
reads. Normal `robot_mode()` reads also capture communication success rate,
current errors and last motion errors from the same state object (no additional
hardware read). A bounded history stores up to 200 samples at at most 10 Hz.
On a command fault, `FCI pre-fault evidence:` in the existing control-server log
contains that history as JSON. For sample-space collection the default log is
`/tmp/act-rlt-control-server.log` (previous runs are `.prev` and `.prev2`).

Samples include robot mode, `ccsr`, errors, wall-clock timestamp, cache age,
active command, busy/servo flags, fallback count and `cycle_gap_max_ms`. The gap
measures the Python control-server loop, NOT the native 1 kHz FCI loop. Idle CCSR
must not be interpreted as active-motion packet loss. History covers roughly
20 seconds when normal polling keeps up; timestamp/age fields reveal gaps.

A surfaced `FrankaSessionLost` latches the fault, suspends routine hardware
polling and rejects ordinary commands. Cached observations retain their old
timestamp and are marked `state_stale` with `session_fault`. Stop, disconnect
and explicit connect/recover/reset remain available; successful explicit
connect/recover/reset clears the latch. No failed movement is automatically
replayed. The original native failing operation and explicit stop/cleanup can
still wait on native-library timeouts; this change removes diagnostic rereads,
not every possible source of delay.

Offline regression command (in an environment with pytest and Franky installed):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_franka_runtime_owner.py tests/test_franka_control_ipc.py \
  tests/test_franka_async_gripper.py tests/test_franka_runtime_ik.py
```
