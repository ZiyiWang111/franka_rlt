"""Locate local artifacts after the Ethernet dataset/model reorganization."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_CATEGORIES = ("ethernet_act", "ethernet_token", "ethernet_rlt")
DATASET_CATEGORIES = ("ethernet_fine", "ethernet_coarse")


def resolve_artifact_path(value: str | Path) -> Path:
    """Resolve existing paths, including old paths embedded in checkpoints.

    Saved checkpoints can name this checkout's former layout or the training
    machine's absolute checkout path. Only relocate when the original is absent
    and the corresponding artifact exists here; unrelated paths stay unchanged.
    """
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    parts = path.parts
    candidates = []
    for index, part in enumerate(parts):
        suffix = parts[index + 1:]
        if not suffix:
            continue
        if part == "datasets":
            if suffix[0] not in DATASET_CATEGORIES:
                suffix = ("ethernet_fine", *suffix)
            candidates.append(PROJECT_ROOT / "datasets" / Path(*suffix))
        elif part == "outputs":
            if suffix[0] in OUTPUT_CATEGORIES:
                candidates.append(PROJECT_ROOT / "outputs" / Path(*suffix))
            else:
                candidates.extend(
                    PROJECT_ROOT / "outputs" / category / Path(*suffix)
                    for category in OUTPUT_CATEGORIES
                )
    matches = {candidate.resolve() for candidate in candidates if candidate.exists()}
    if len(matches) > 1:
        raise ValueError(f"Ambiguous relocated artifact {value}: {sorted(map(str, matches))}")
    return next(iter(matches)) if matches else path.resolve()
