# Review and remove suspicious manual-demo episodes

`evo-rlt-review-manual-demo` scans a local LeRobot v3 dataset for episodes
that are static, freeze at the end, stop far from the dominant task endpoint,
or do not complete a pick-and-transport sequence. Candidates are reviewed one
at a time with `lerobot-dataset-viz`; no episode is removed without a `y`
answer and a final confirmation.

Do not run the tool while the recorder is active. It refuses to continue when
it sees an incomplete parquet file or detects that `meta/info.json` changed.

## Installation

After adding or updating this checkout's console scripts:

```bash
conda activate evo-rlt
cd /home/embint/Evo-RLT
pip install -e .
```

Without reinstalling, invoke the module directly with
`PYTHONPATH=src python -m evo_rlt.cli.review_manual_demo`.

## Scan without opening the viewer

```bash
evo-rlt-review-manual-demo \
  --repo-id embint/franka_stage1_initial \
  --root /home/embint/Evo-RLT/datasets/franka_stage1_initial \
  --require-pick-sequence \
  --scan-only
```

## Interactive review and in-place filtering

```bash
evo-rlt-review-manual-demo \
  --repo-id embint/franka_stage1_initial \
  --root /home/embint/Evo-RLT/datasets/franka_stage1_initial \
  --require-pick-sequence
```

For every candidate, the tool runs an equivalent of:

```bash
lerobot-dataset-viz \
  --repo-id embint/franka_stage1_initial \
  --root /home/embint/Evo-RLT/datasets/franka_stage1_initial \
  --episode-index EPISODE_INDEX \
  --mode local
```

Close the viewer and answer `y` to exclude the episode, `n` to keep it, or
`q` to stop without modifying anything. Confirmed exclusions are applied in
one batch after review so that videos are only rebuilt once. In-place filtering
preserves the original directory as
`franka_stage1_initial_backup_YYYYMMDD_HHMMSS`.

To keep the source at its original path and write a filtered copy instead:

```bash
evo-rlt-review-manual-demo \
  --repo-id embint/franka_stage1_initial \
  --root /home/embint/Evo-RLT/datasets/franka_stage1_initial \
  --require-pick-sequence \
  --output-root /home/embint/Evo-RLT/datasets/franka_stage1_initial_filtered
```

Run `--help` to tune motion, static-tail, endpoint-distance, and post-grasp
thresholds. Omit `--require-pick-sequence` for tasks that do not require a
gripper pick followed by transport.
