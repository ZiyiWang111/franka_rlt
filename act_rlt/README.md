# ACT-RLT

This directory is isolated from the original `src/evo_rlt` implementation. It
holds ACT-RLT, in which ACT replaces the original VLA reference policy. The
first component is the FR3 sample-space data collector.

## ACT training on a server

For joint-angle-only state (7D), first create a separate dataset. This keeps
the original dataset and videos intact and slices state statistics as well:

```bash
python act_rlt/prepare_joint_state.py --root datasets/act_rlt_001 \
  --output datasets/act_rlt_001_state7
python act_rlt/train.py --root datasets/act_rlt_001_state7 --output runs/act_state7
```

The resulting policy takes `joint_0` through `joint_6`, in that order, together
with both cameras. Deployment must supply the same 7D state. Actions remain
7D TCP deltas plus gripper width; with VAE disabled they are training targets only.

Copy the finished `datasets/act_rlt_001` directory (including data, meta and
videos) and `act_rlt/train.py` to the server. Activate a Python environment
with LeRobot and its ACT dependencies installed (the local environment uses
LeRobot 0.5.1). This script needs no robot drivers or RoboLab installation.
The ResNet18 ImageNet weights must be cached or downloadable on first use.

```bash
python act_rlt/train.py --root /path/to/act_rlt_001 --dry-run
python act_rlt/train.py --root /path/to/act_rlt_001 --output runs/act_001
```

Choose a preset; explicit `--steps` and `--batch-size` override it:

| Preset | Steps | Batch size | Purpose |
| --- | ---: | ---: | --- |
| `standard` (default) | 20000 | 8 | Full training |
| `lowmem` | 20000 | 2 | Lower memory use, same model |
| `smoke` | 100 | 2 | Check the server training setup |

```bash
python act_rlt/train.py --root /path/to/act_rlt_001 --preset smoke --output runs/smoke
python act_rlt/train.py --root /path/to/act_rlt_001 --preset lowmem --output runs/lowmem
python act_rlt/train.py --root /path/to/act_rlt_001 --steps 40000 --batch-size 16 \
  --lr 1e-5 --no-augment --output runs/custom
python act_rlt/train.py --help
```

Default model: ResNet18, transformer width 768, feedforward width 3200,
4 encoder layers, 1 decoder layer, VAE disabled, chunk size 16 and 4 action
steps per inference. Learning rates are 3e-5 for the main model and 1e-5 for
the backbone. `--chunk-size` and `--n-action-steps` can override the horizon.
Augmentation is enabled; `--no-augment` disables it. W&B is opt-in with
`--wandb`. Checkpoints are saved every 5000 steps and at the end.

Training uses both cameras and the 22D `observation.state` to predict the
7D `action` (measured TCP deltas plus gripper width). The command-action and
other `complementary_info.*` columns are not model inputs or targets.
All episodes are used; this script does not create a validation split.

`--root` is the dataset directory itself, not its parent. `--repo-id` defaults
to that directory's name. `--output` is relative to the current working
directory and must not exist yet; use a different output for each run.
The final policy is at `OUTPUT/checkpoints/last/pretrained_model`.

## Stage 1: ACT encoder hidden to RL Token

Stage 1 freezes the trained ACT policy and reconstructs its main transformer
encoder output with the existing Evo-RLT `RLTokenModule`. For the current
two-camera VGA checkpoint the measured ACT hidden shape is `[B, 602, 768]`.
The sequence length is discovered at runtime rather than configured: changing
camera count, image resolution, or backbone stride can change it.

The reconstruction bottleneck uses `encode_multi`, with shape `[B, 1, 768]`.
The downstream Stage 2 Actor/Critic interface uses `encode`, which mean-pools
RL tokens and returns `[B, 768]`.

First inspect the generated LeRobot command:

```bash
python -m act_rlt.train_rl_token \
  --root datasets/act_rlt_001_state7 \
  --act-checkpoint outputs/act_rlt_001_state7_act \
  --output outputs/act_rlt_001_stage1 \
  --dry-run
```

Then train:

```bash
python -m act_rlt.train_rl_token \
  --root datasets/act_rlt_001_state7 \
  --act-checkpoint outputs/act_rlt_001_state7_act \
  --output outputs/act_rlt_001_stage1
```

Defaults are batch size 8, 10,000 steps, AdamW at `2e-4`, cosine decay with
200 warmup steps, gradient clipping at 1.0, FP32, and checkpoints every 2,000
steps. Image augmentation is disabled. The ACT checkpoint and its saved
normalization processors are the source of truth; ACT parameters are frozen
and excluded from the Stage-1 checkpoint.

On the first batch, training compares the encoder-only helper with an
`ACT.forward()` hook and requires exact equality. This one-time check can be
disabled with `--no-verify-extractor` after the implementation has been
validated on a server.

Resume the complete LeRobot state (RL Token weights, optimizer, scheduler,
step, and RNG) from the `last` checkpoint with:

```bash
python -m act_rlt.train_rl_token \
  --output outputs/act_rlt_001_stage1 \
  --resume
```

## Stage 2: online chunk Actor-Critic

Stage 2 freezes ACT and the Stage-1 RL Token encoder, then trains a two-layer
256-wide Actor and twin two-layer 256-wide critics. The first FR3 version uses
the 768D RL token plus normalized 7D joint state, and predicts a four-step 6D
TCP-delta chunk. The ACT reference is the first four arm actions from its
`[B,16,7]` output; the gripper remains held outside the RL action space.

Load and validate both checkpoints and the saved LeRobot processors without
connecting to the robot:

```bash
python -m act_rlt.train_stage2 \
  --stage1-checkpoint outputs/act_rlt_001_stage1 \
  --act-checkpoint outputs/act_rlt_001_state7_act \
  --output outputs/act_rlt_001_stage2 \
  --dry-run
```

The confirmed scheduling semantics are:

```text
first 4,000 low-level steps: frozen ACT actions, replay collection, no updates
next 10,000 online steps: one collected chunk -> one transition -> five critic updates
actor update: every second critic update
target critics: updated after every critic optimizer step
```

The current repository-faithful order executes the first Actor chunk before
the first replay update, so the Actor is still randomly initialized at that
boundary. Treat the initial motion run as an engineering validation and keep
the physical stop available. If desired, add an explicit replay-only bootstrap
as a separately reviewed deviation before longer training.

Motion is guarded by both `--allow-motion` and explicit base-frame workspace
bounds:

```bash
python -m act_rlt.train_stage2 \
  --stage1-checkpoint outputs/act_rlt_001_stage1 \
  --act-checkpoint outputs/act_rlt_001_state7_act \
  --output outputs/act_rlt_001_stage2 \
  --allow-motion \
  --workspace-min X_MIN Y_MIN Z_MIN \
  --workspace-max X_MAX Y_MAX Z_MAX \
  --sample-min SAMPLE_X_MIN SAMPLE_Y_MIN SAMPLE_Z_MIN \
  --sample-max SAMPLE_X_MAX SAMPLE_Y_MAX SAMPLE_Z_MAX
```

No orientation teaching or orientation argument is required. Before Episode 0,
the program reads the current TCP rotation vector, samples a reset XYZ,
asks for one-time authorization, and moves there. Without a sample box, sampled
X/Y stays at least 2 cm inside the workspace boundaries; Z uses the full
configured range. The optional `--sample-min` / `--sample-max` pair instead
selects reset poses from that exact box, which must lie inside the workspace.
The workspace remains the hard safety boundary for every commanded TCP target.
After each episode, the arm automatically retreats 1 cm along base-frame +Y and
samples the next reset pose, then asks whether to accept it or sample again. During every
warmup and online episode, `s` marks success, `f` marks failure, and `q` stops
training. Inference stops after 5 seconds by default and, if no outcome was
entered during motion, explicitly asks for `s`, `f`, or `q` before reset. Human
action intervention is intentionally disabled; outcome labeling remains enabled.
If an inference action is refused for crossing the TCP workspace, the episode
is truncated and the arm proceeds to the next sampled reset pose instead of
terminating training. A successfully executed prefix is retained; a refusal on
the first action creates no replay transition.
Checkpoints and replay are saved at episode boundaries and every 1,000 low-level
steps. Resume with the same arguments plus `--resume`.

`--total-env-steps` is a cumulative online target. To continue for another
10,000 online steps after finishing 10,000, resume with
`--total-env-steps 20000`; the existing replay, networks, optimizers, and update
counters are retained. Resume permits changing only `warmup_steps` and this
cumulative online target, with progress-safety checks.

## ACT inference (first version)

Run from the repository root in the installed Evo-RLT environment. Start the
existing control server separately, as for collection, and stop collection
before connecting. Use the 7D-state checkpoint including its saved processors.

```bash
python -m act_rlt.infer \
  --checkpoint outputs/act_rlt_001_state7_act \
  --dry-run --duration 10
```

After the model and cameras warm up, Enter starts an episode; the prompt returns
after every episode so another dry-run can be started without reloading the model.
Ctrl+C stops the session.
Dry-run reads live observations and prints predictions without sending arm
motion commands. Both cameras use native 640x480 RGB. Gripper commands are
omitted in this version, so the current grasp is held.

For motion, omit `--dry-run` and supply `--workspace-min X Y Z` and
`--workspace-max X Y Z`: measured bounds for your task in base-frame metres.
Before inference the script samples XYZ inside the workspace, keeping X and Y
3 cm inside their respective boundaries, and moves there at 0.02 m/s. Z uses
the full configured range. The default reset orientation is the frame-level
mean TCP rotation vector from `datasets/act_rlt_001`:
`[-2.19841545, 2.22703212, -0.03082950]` rad. Override these settings with
`--reset-orientation RX RY RZ`, `--reset-xy-padding`, and `--reset-speed`.
The first sampled move requires explicit authorization. At every sampled point,
choose whether to start inference or move to a new sample. After each
`--duration`, choose whether to return to that episode's measured start pose,
move to a new sample, or quit; the process stays alive for subsequent episodes.
Default action limits are 2 mm translation and 0.02 rad rotation per step;
`--max-step-m` and `--max-step-rad` override them.
At the end of every inference episode, an action-limit report prints the total
number of evaluated steps and the counts/percentages that triggered the
translation limit, rotation limit, either limit, or both limits.

The loop runs at 15 Hz, using checkpoint action chunking (override with
`--n-action-steps 1` for replanning every tick). Saved processors normalize
observations and unnormalize actions; no augmentation is applied. Each delta
is anchored to the measured TCP pose, using the shared body-frame rotation
convention. Observation/inference latency above 0.5 seconds stops the run.
Ctrl+C is handled when the current call returns; this is not a hard real-time
emergency stop. Keep the robot's physical stop accessible during execution.

## Sample-space data collection

Run the launcher from the repository root:

```bash
bash act_rlt/data_collection/collect_sample_space.sh \
  --dataset act_rlt_demo_v1 \
  --episodes 20
```

The launcher starts a fresh Evo-RLT Franka control server, verifies the wrist
and front RealSense cameras, starts the recorder, and shuts down the exact
server process when recording exits.

At startup, hand-guide the TCP to the reference pose `p0` and press `Enter`.
The full 6D pose is stored for this process.  Samples preserve its orientation
and are drawn uniformly in the robot base frame:

```text
x in [x0 - 0.01, x0 + 0.01] m
y in [y0,        y0 + 0.02] m
z in [z0 - 0.01, z0 + 0.01] m
```

The bounds above intentionally follow the requested numeric convention.  If
`p0` is physically the maximum-Y boundary but your robot coordinate system
increases toward the workspace interior, change `--y-range` handling before a
real run; the current implementation always samples toward `+Y`.

For each episode, the robot is positioned at the sampled pose outside
recording. After a 0.5-second settling delay, recording starts automatically:

```text
sample -> reference p0 -> p0 + [0, -0.01, 0]
```

The episode is saved automatically, followed by a 0.5-second delay. The robot
then returns `+1 cm` to `p0`, samples the next pose, moves there outside
recording, waits 0.5 seconds, and starts the next episode without confirmation.

During either 0.5-second non-episode window:

- `1`: move to the immutable reference pose.
- `2`: move to the current sampled pose.
- `Q`: stop collection.

Using a numeric key pauses automation; `Enter` resumes it. `Ctrl+C` stops the
collector at any time.

During an episode, press `X` (configurable with `--abort-key`) to request an
immediate motion stop and discard the entire buffered episode.  Automatic
motion stays paused after an abort; press `Enter` when it is safe to resample,
reposition, and continue.

Only `sample -> p0 -> -Y` is recorded. Sampling, repositioning, retreating,
and numeric-key moves are excluded from the dataset. Saved data uses a 22D
observed state and two native `640x480` RGB streams. It records both action
representations:

- `action`: 7D actual action measured between consecutive observations
  (`dx..drz` plus measured gripper width).
- `complementary_info.command_action`: the exact active high-level
  `move_tool` command (`target_x..target_rz` plus `speed_m_s`). This is an
  absolute TCP target because the automatic collector commands waypoint
  targets rather than sending one relative servo action per frame.

`complementary_info.policy_action` remains a zero placeholder because no ACT
policy is running during scripted demonstration collection.

Useful options:

```bash
bash act_rlt/data_collection/collect_sample_space.sh --help
```

Default motion speeds are `0.02 m/s` for repositioning and sample-to-reference,
and `0.01 m/s` for the final `-Y` segment. Adjust them independently with
`--move-speed` and `--insertion-speed`. The transition delays can be changed
with `--pre-episode-sleep` and `--post-episode-sleep`.

Use a one-episode low-speed dry run before a larger collection:

```bash
bash act_rlt/data_collection/collect_sample_space.sh \
  --dataset act_rlt_smoke \
  --episodes 1 \
  --move-speed 0.02 \
  --insertion-speed 0.01
```
