"""Speed-fraction validation used by the standalone Franka session."""

MIN_FRACTION = 1e-3
MAX_FRACTION = 1.0


def clamp_fraction(fraction: float) -> float:
    """Clamp a relative dynamics factor to franky's usable range."""
    return max(MIN_FRACTION, min(MAX_FRACTION, float(fraction)))
