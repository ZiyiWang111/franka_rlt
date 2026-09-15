"""Standalone Franka FR3 control backend vendored into Evo-RLT.

The control server owns franky/libfranka in a dedicated process; robot-facing
Evo-RLT code imports ``evo_franka.control_client`` to talk to it. Keeping this
module dependency-free also allows the pure NumPy kinematics to be used alone.
"""
