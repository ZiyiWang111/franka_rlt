"""ACT-RLT: RL Token experiments using ACT as the reference policy."""


def register() -> None:
    """Register the ACT RL-token policy with LeRobot's dynamic factories."""
    # Importing the decorated config class is sufficient for LeRobot 0.5.1.
    from act_rlt.configuration_act_rlt_token import ACTRLTokenConfig  # noqa: F401


__all__ = ["register"]
