# Franka RGB-only PI05 experiment

`bash scripts/train_fa1_rgb.sh` trains `outputs/fa1_rgb` on `datasets/fa1_s15`.
GPU defaults to 2; override with `GPU_INDEX`. Paths can be overridden with
`PYTHON_BIN` and `BASE_MODEL`. Run the launcher under tmux to survive SSH exit.

The entrypoint is `python -m evo_rlt.cli.train_pi05_rgb
--discrete-state-input=false` followed by standard LeRobot training flags.
This is a project entrypoint option, **not** an upstream LeRobot
`--policy.discrete_state_input` field.

The RGB processor replaces both the state prompt preparation and tokenization
steps after loading the base checkpoint's processors. It emits BOS + cleaned
task tokens + separately encoded newline, following OpenPI's no-state branch:
https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/tokenizer.py

`observation.state` remains in the source dataset and metadata for compatibility,
but is removed from the processed model batch. PI05 has no continuous state
projection. Actions remain TCP deltas plus absolute gripper target width;
relative-action conversion is disabled. Image and action preprocessing retain
the existing LeRobot behavior.

Defaults: batch 16, chunk/execution length 10, peak LR 5e-5, final LR 5e-6,
30000 steps, seed 1000, gradient checkpointing enabled. Uses local offline
Hugging Face assets. This aligns the no-state text construction with OpenPI;
it does not claim that every LeRobot/OpenPI training detail is identical.

The saved `policy_preprocessor.json` contains `Pi05RGBTokenizerStep` and
`discrete_state_input: false`. The policy config stays ordinary `pi05`.
Inference and resume must load the saved processors, not rebuild default PI05
processors. The deployment environment must include this Evo-RLT module so
the processor's full import path can be resolved.

Validation:

```bash
python -m unittest discover -s tests -p test_pi05_rgb.py
```

Server validation additionally loaded the real base checkpoint's processor
and tokenizer: RGB/task/action batches with missing, zero and changed state
produced identical token IDs, masks and normalized actions. Saving/reloading
the full processor preserved the result.

The existing remote robot inference protocol still expects 50 action steps;
these chunk-10 checkpoints require a separate protocol adaptation before
robot deployment. The robot controller may continue using measured state
to execute TCP deltas even though the neural policy does not observe it.
