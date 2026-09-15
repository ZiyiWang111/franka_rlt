# Source provenance

This package was vendored from `/home/embint/robotLab` at Git commit
`c3bb3c25e03f31bfac8091499afd710348ed1e5b` on 2026-09-11.

The package/import paths and deployment defaults were adapted so the backend is
self-contained inside Evo-RLT. The control architecture and runtime behavior
remain the same: a dedicated ZMQ control-server process owns franky/libfranka,
streams cached state, executes branch-continuous IK and joint-space servo
commands, and enforces the original watchdog and recovery behavior.
