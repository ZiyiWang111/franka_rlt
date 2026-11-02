# ACT-RLT

起点、重置方式以及数采 / ACT infer / Stage 2 训练与推理流程的中文说明见
[ACT 数采、推理与 Stage 2：起点和执行流程](README_START_AND_INFERENCE.md)。

Coarse 两种 ACT 模型（绝对关节位置 / TCP 增量）的完整 10 步 chunk 推理见
[Coarse ACT 推理](COARSE_INFERENCE.md)，入口为 `python -m act_rlt.infer_coarse`。

This directory is isolated from the original `src/evo_rlt` implementation. It
holds ACT-RLT, in which ACT replaces the original VLA reference policy. The
first component is the FR3 sample-space data collector.

## ACT training on a server

For joint-angle-only state (7D), first create a separate dataset. This keeps
the original dataset and videos intact and slices state statistics as well:

```bash
python act_rlt/prepare_joint_state.py --root datasets/ethernet_fine/act_rlt_001 \
  --output datasets/ethernet_fine/act_rlt_001_state7
python act_rlt/train.py --root datasets/ethernet_fine/act_rlt_001_state7 --output outputs/ethernet_act/act_state7
```

The resulting policy takes `joint_0` through `joint_6`, in that order, together
with both cameras. Deployment must supply the same 7D state. Actions remain
7D TCP deltas plus gripper width; with VAE disabled they are training targets only.

Copy the finished `datasets/ethernet_fine/act_rlt_001` directory (including data, meta and
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
`--wandb`. Checkpoints are saved every 10000 steps and at the end.
By default, learning rates warm up for 1000 steps (capped at 10% of the run),
then follow cosine decay to 10% of their initial values at the final step.
The backbone learning rate follows the same factor. Use `--scheduler none`
for constant learning rates, or override `--warmup-steps`, `--min-lr-ratio`,
and `--save-freq`. Copy `act_rlt/train_scheduled.py` alongside `train.py` when
running on another machine; it preserves the explicit scheduler while applying
LeRobot's ACT optimizer preset.

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
encoder output with the existing Evo-RLT `RLTokenModule`. Its RL encoder reads
`encoder_out + encoder_pos_embed`, using the same latent/state and camera
positions that ACT supplies to its decoder. The reconstruction target remains
`encoder_out`. For the current
two-camera VGA checkpoint the measured ACT hidden shape is `[B, 602, 768]`.
The sequence length is discovered at runtime rather than configured: changing
camera count, image resolution, or backbone stride can change it.

The reconstruction bottleneck uses `encode_multi`, with shape `[B, 1, 768]`.
The downstream Stage 2 Actor/Critic interface uses `encode`, which mean-pools
RL tokens and returns `[B, 768]`.

First inspect the generated LeRobot command:

```bash
python -m act_rlt.train_rl_token \
  --root datasets/ethernet_fine/act_rlt_001_state7 \
  --act-checkpoint outputs/ethernet_act/act_rlt_001_state7_act \
  --output outputs/ethernet_token/act_rlt_001_stage1_pos \
  --dry-run
```

Then train:

```bash
python -m act_rlt.train_rl_token \
  --root datasets/ethernet_fine/act_rlt_001_state7 \
  --act-checkpoint outputs/ethernet_act/act_rlt_001_state7_act \
  --output outputs/ethernet_token/act_rlt_001_stage1_pos
```

Defaults are batch size 8, 10,000 steps, AdamW at `2e-4`, cosine decay with
200 warmup steps, gradient clipping at 1.0, FP32, and checkpoints every 2,000
steps. Image augmentation is disabled. The ACT checkpoint and its saved
normalization processors are the source of truth; ACT parameters are frozen
and excluded from the Stage-1 checkpoint. RL Token receives the ACT encoder output directly, without adding positional
embeddings. Its reconstruction target is the same encoder output. Use a new
output directory when switching from a checkpoint trained with added positions.

On the first batch, training compares the encoder-only helper with an
`ACT.forward()` hook and requires exact equality. This one-time check can be
disabled with `--no-verify-extractor` after the implementation has been
validated on a server.

Resume the complete LeRobot state (RL Token weights, optimizer, scheduler,
step, and RNG) from the `last` checkpoint with:

```bash
python -m act_rlt.train_rl_token \
  --output outputs/ethernet_token/act_rlt_001_stage1_pos \
  --resume
```

## Stage 2: online chunk Actor-Critic

Stage 2 freezes ACT and the Stage-1 RL Token encoder, then trains a two-layer
256-wide Actor and twin two-layer 256-wide critics. The first FR3 version uses
the 768D RL token plus normalized 7D joint state, and predicts a four-step 6D
TCP-delta chunk. The ACT reference is the first four arm actions from its
`[B,16,7]` output; the gripper remains held outside the RL action space.
Actor and each Critic project the 768D RL token, auxiliary state (TCP Z,
joints, or ACT wrench input), and flattened action chunk through separate
Linear + LayerNorm + tanh branches. Each branch outputs 128 dimensions in
[-1, 1], giving all three inputs the same width and comparable scale before
concatenation. The resulting 384D vector enters the 256-wide MLP. RGB-only
models omit the auxiliary branch and fuse to 256 dimensions. `--fusion-dim`
controls all branch widths (default 128). Actor, Q1 and Q2 have independent
projection parameters. The Actor action branch receives the ACT reference;
the Critic action branch receives the action being evaluated. Stage-1 weights,
ACT inputs/outputs, raw replay tensors, and Actor/Critic losses are unchanged.

This layout is saved as `input_fusion=balanced`. Checkpoints with the previous
`token_identity` layout cannot resume into this architecture. Start a new output
directory with `--replay-from OLD_RUN/checkpoints/latest.pt` to retain compatible
replay and initialize new networks. Frozen inference still loads old checkpoints
using their saved layout automatically. Earlier equal-width checkpoints remain
compatible when their other architecture settings match.

Load and validate both checkpoints and the saved LeRobot processors without
connecting to the robot:

```bash
python -m act_rlt.train_stage2 \
  --stage1-checkpoint outputs/ethernet_token/act_rlt_001_stage1_pos \
  --act-checkpoint outputs/ethernet_act/act_rlt_001_state7_act \
  --output outputs/ethernet_rlt/act_rlt_001_stage2 \
  --dry-run
```

The confirmed scheduling semantics are:

```text
warmup: frozen ACT actions, replay collection, no updates; at least 4,000 low-level steps
warmup boundary: finish the current episode, then 1,000 BC-only Actor updates and 1,000 Critic-only updates
next 10,000 online steps: one collected chunk -> one transition -> five critic updates
actor update: every second critic update
target critics: updated after every critic optimizer step
target actor: updated after every actor optimizer step
```

Stage-2 v3 initializes the Actor by imitating the ACT reference before its first
robot action. `--bc-init-steps` and `--critic-init-steps` configure this boundary
initialization; no environment steps are consumed and its duration is excluded
from the episode timeout. The frozen ACT and RL Token encoder remain unchanged.

Success supplies a single terminal +1; other steps supply zero. The target is
**1-chunk TD by default** (`td_chunks=1`). Add `--three-chunk-td` to use
**3-chunk TD** (`td_chunks=3` in saved config). With this flag, a replay action and the
Actor/Critic action input still contain one four-step chunk. The TD sampler
follows three consecutive chunk transitions, accumulating up to 12 executed
steps, and uses the final state/reference for the target Actor:
`sum(gamma**i * reward[i], i=0..K-1) + gamma**K * (1-done_at_horizon) * clamp(min_target_Q, 0, 1)`.
Here K is the total actual step count, normally 12. A terminal/truncated episode
ends the horizon early, including short final chunks. Nonterminal starts with
fewer than three known chunks wait for more data; they are not downgraded to
one-chunk TD. Actor BC and its actual-step mask use only the starting chunk.
New replay records episode step indices; sampling follows time coordinates,
never adjacency in storage. Old completed episodes can reuse their contiguous
original chunks after checking state/reference continuity. Legacy overlap
windows and unfinished legacy episodes without provable time positions are
excluded from three-chunk TD sampling, but retained in replay and BC sampling.
Existing v3 network/optimizer checkpoints can resume with either horizon;
the changed `td_chunks` setting is reported among resume overrides. Pass
`--three-chunk-td` on every training/resume command that should use three
chunks; omitting it selects one chunk even if the saved run used three.
Bootstrap Q clipping is enabled by default (`--target-q-clipping`). Use
`--no-target-q-clipping` to use the raw minimum target Q instead. Finite-value
checks remain active. This toggle is saved in checkpoints and may be changed
with `--resume`; older checkpoints default to clipping enabled. Diagnostics
include `target_q_clipping_enabled`; clip fractions are zero when disabled.
Unexecuted reward padding is masked. Truncations (including workspace refusals)
are treated as episode ends with no bootstrap, matching this reset-based task.
The deterministic target Actor supplies next actions. Both target actions and
Actor-loss actions pass through differentiable copies of the robot's physical
translation/rotation norm limits, using the saved ACT mean/std normalization.
The online Actor still outputs final actions, with BC toward ACT and Gaussian
exploration during collection. The Actor Q output itself is not hard-clipped.

Stage 2 now uses persistent camera, inference, and servo threads by default.
Each four-step proposal is executed intact, without temporal ensembling or
mid-chunk replanning. No extra launch flag is needed. The legacy
`--temporal-ensemble-coeff` field is accepted for checkpoint compatibility but
does not enable blending.

The workers run continuously across chunk boundaries while the main thread
inserts completed transitions and performs the same five updates per online
transition. Inference uses a separate Actor snapshot published between updates;
it never reads partially updated parameters. Collection may lead learning by a
few chunks. Warmup executes only ACT references with no learner updates and
finishes its current episode once the minimum step target is reached, before
BC/Critic initialization and Actor control.
Workers also stop at episode boundaries and the online budget. Reset and human
outcome prompts run only after the servo worker stops. A checkpoint contains
consumed replay transitions; in-flight collection is not resumed after restart.

Add `--replay-stride 2` to collect overlapping four-step replay windows while
still executing complete four-step proposals. The default `--replay-stride 4`
keeps the original replay behavior. For example, in addition to `[a0,a1,a2,a3]`
and `[a4,a5,a6,a7]`, stride 2 stores `[a2,a3,a4,a5]` with its own state/reference
at step 2 and next state/reference at step 6. Default one-chunk TD bootstraps at step 6. With `--three-chunk-td`, TD follows
windows starting at steps 2, 6 and 10 to bootstrap at step 14, when all are available.
The actions come from the actual bounded commands, including exploration noise
and directional bias; neither actions nor ACT references are resampled/spliced.

During an episode the camera worker owns CPU image/state snapshots, and the
servo worker selects intermediate cache entries without waiting for a frame or
running extra policy inference. After the workers stop, intermediate states and
ACT references are encoded once, extra transitions are added, and the episode
cache is released. This costs CPU image memory during the episode and encoding
time between episodes. Snapshots follow `--observation-selection`; `latest`
can precede the preceding command, while `boundary` requires a snapshot after
that command and skips capture when none is ready. Missing anchors skip the
affected extra window rather than reusing/interpolating a different state.

Only complete four-step extra windows are generated. Original short terminal
chunks retain their actual-step mask. Windows never cross episodes, warmup/
Actor source changes, or human takeover boundaries; pure human windows retain
their correction targets separately from the ACT reference. Rewards and
terminal/truncation flags follow the exact executed interval. The online step
budget also flushes completed extra windows after stopping, without fabricating
a terminal outcome. Warmup initialization sees all extra windows from its final
episode. Environment steps, chunk counters and UTD update budgets count real
execution only; added windows consume no extra robot steps or update iterations.

The stride is saved in checkpoints and can be changed on `--resume`. Existing
replay is retained without reconstruction; intermediate snapshots are not saved
in checkpoints. Logs include `replay_overlap_chunks` (cumulative extra windows),
`replay_overlap_added`, skipped-window counts, encoded-anchor count and encoding
wall time at flush, plus intermediate frame sequence/age diagnostics per chunk.

Stage-2 training and frozen inference default to `--observation-selection latest`: the camera keeps
capturing during execution, and at each chunk boundary inference consumes the
newest cached observation with a sequence newer than the last one used. It only
waits for the camera if no newer observation exists. One encoding still supplies
both the previous transition's next state and the following chunk's current
state/reference. A cached observation can precede the final command, so this is
not a guarantee of a physically post-chunk replay state. Logs report selection,
sequence, cache hit, and observation ages. Use `--observation-selection boundary`
to restore strict post-command selection. `--camera-boundary-mode` independently
controls exposure/state timestamp pairing and defaults to `timestamp` in both
training and frozen inference; use `legacy` for ordinary observations.

This change does not enable early chunk prediction, temporal blending, or forced
downward actions. Inference still begins after the four-step chunk, but camera
waiting can be overlapped with its execution. Late predictions
hold the last target, never repeat a relative action or trigger a catch-up burst;
an action queue wait over 0.5 seconds stops collection (2 seconds at startup).
GPU contention from learning can still increase latency.

`metrics.jsonl` reports `execution_mode=persistent_three_thread_fixed_chunk`,
`max_command_interval_ms` (including chunk boundaries), `deadline_misses`,
`max_lateness_ms`, and `boundary_inference_ms`. At 15 Hz the intended command
interval is 66.7 ms. These measurements are needed to assess real robot timing.

These are deliberate engineering changes from the earlier Evo-RLT adaptation:
reward-aware target bounds, target Actor, boundary BC/Critic initialization,
and matching physical action projection during learning. They are not claimed
as exact paper reproduction or a guarantee of real-world success.

`metrics.jsonl` includes Q1/Q2, replay and Actor Q, raw target Q, upper/lower
clip fractions, TD targets/errors, reward, terminal fraction, BC loss and
Actor-reference RMSE. Statistics average all UTD updates for each chunk (extrema
retain min/max); Actor statistics use only updates that actually train the Actor.
Gradient norms are measured before clipping. Non-finite gradients abort updates.

Each chunk's complete metrics are queued in memory. A dedicated writer keeps the
JSONL file open and writes/flushes a batch every second; regular terminal metrics
are printed at most once per second using the latest record. Shutdown drains
remaining records, including exception cleanup. Writer failures and a full queue
are surfaced instead of silently dropping metrics.

Use a new output directory for a fresh run, without `--resume` or `--replay-from`.
Old v1/v2 learner checkpoints cannot be resumed as v3: the Actor/Critic network
shape has changed. Use `--replay-from OLD/checkpoints/latest.pt` with a new
output directory to reuse transitions while initializing a new learner.
New checkpoints include initialization
status, target Actor/Critic, optimizers and random states for exact learner resume.

Motion is guarded by both `--allow-motion` and explicit base-frame workspace
bounds:

```bash
python -m act_rlt.train_stage2 \
  --stage1-checkpoint outputs/ethernet_token/act_rlt_001_stage1_pos \
  --act-checkpoint outputs/ethernet_act/act_rlt_001_state7_act \
  --output outputs/ethernet_rlt/act_rlt_001_stage2 \
  --allow-motion
```

Stage-2 training uses Z insertion reset by default at 30 Hz. At startup,
place the TCP at the reference pose and press Enter to capture it. Workspace limits are disabled; resume restores the
saved reference and ignores any old workspace limits. The initial reset XY half range is 0, fixing X/Y at the reference.
Increase the half range with `=` to enable uniform XY sampling, with fixed absolute Z = 0.402 m and intrinsic XYZ Euler
orientation `(179.7, 1.4, 90.6)` degrees. Use `=` / `-` to adjust the next XY
half range, `r` to resample, and Enter/y to start from the current pose.
The arm retreats +Z 1 cm after each episode. Policy, reset and idle WASD
commands do not use workspace bounds. Per-step limits and the WASD session
distance limit remain. Omit workspace/sample bounds;
the mode-selection parameter has been removed. `--fps` can override 30 Hz.
For a wrist-only ACT checkpoint, only its configured wrist camera is opened.
During warmup and online episodes, `s` marks success and `f` marks failure
without waiting for the episode timer. The current four-action chunk finishes
before the robot stops; its final executed step receives the terminal label.
`--episode-time` is the maximum rollout duration (5 seconds by default); at
that limit, if no outcome was entered, the robot stops and asks for `s`, `f`,
or `q` before reset.
Without human intervention, `q` also stops training during rollout.

At the reset confirmation prompts before an episode, hold WASD to continuously
move TCP XY: W=-X, S=+X, A=-Y, D=+Y. This reuses the keyboard teleop servo
controller at 15 Hz and 3 cm/s; release the keys to stop. It uses
`--teleop-input-backend` / `--teleop-keyboard` for real key press/release events,
including X11/ToDesk release debounce. Z and orientation stay fixed; workspace
violations stop the servo and report an error. At the
sample-ready prompt, press Enter/y to start from the adjusted pose, or r to
sample again. These setup movements are not added to replay or training steps.

During Stage-2 training, the same prompts accept arrow keys to set directional
exploration for the next Actor episode: Up=-X, Down=+X, Left=-Y, Right=+Y,
N=-yaw, M=+yaw. Yaw follows the action's `drz` convention: rotation about
the TCP's local Z axis, rather than world-frame Euler yaw.
Each press adds one level; X, Y and yaw independently range from -5 to +5,
opposite keys cancel, and 0 clears all three. Holding a key does not add repeated levels.
Each level adds 11.71875 micrometers per executed control step on that axis,
about 2 mm over the 170.67-step average Actor episode in rlt_1023. The four
actions in each chunk each receive the offset. Each yaw level adds
0.01171875 degrees (0.000204530772 rad) per step, about 2 degrees over
170.67 steps. All three axes can be combined; five levels correspond to
about 10 mm on each XY axis and 10 degrees of yaw at that episode length.
The setting survives reset resampling, is selected when the episode starts,
and clears when that episode ends. ACT warmup, human takeover, and standalone
inference do not receive this bias. WASD still moves the TCP directly.
The offset is added after action denormalization and before safety clipping;
replay stores the bounded, normalized commands actually sent. JSONL metrics
record both requested cumulative XY/yaw bias and its effect after clipping, which
is an integrated command difference rather than a guaranteed endpoint shift.

During ACT warmup and Actor rollout, press **B** once to latch an additional
base-frame `dz = -0.00015 m` (-Z 0.15 mm) on every remaining policy step.
Repeated presses do not stack or disable it. The selected X/Y/yaw directional
bias and ordinary Actor exploration noise remain active. The Z offset pauses
during human takeover and resumes when policy control returns.

The offset applies before the next command, including unsent actions within
the current chunk, and clears at episode end/reset. Setup movements and frozen
inference do not use B. The inference worker prepares bounded/normalized
alternatives so the servo can switch immediately without GPU work. Replay
stores the actual sent actions. Bias metrics count only executed affected steps.

Add `--use-reference-pose` to reset with the captured/restored reference Z and
orientation, keeping the existing XY range rules. Without it, fixed collection
Z/orientation remain the default. Pass the flag again when resuming. The active
reset mode and pose are recorded in startup-reference metadata.

Add `--enable-human-intervention` to use SpaceMouse takeover in warmup and rollout.
Stage-2 automatically starts a read-only input worker using the same official
publisher and axis mapping as `scripts/fr3_spacemouse_control.sh`. The worker
uses ROS Humble's Python separately from the training environment and never
connects to the robot. Do not run the standalone SpaceMouse control script
alongside training. Set `SPACEMOUSE_DEVICE=/dev/hidrawN` when needed; the default
is `/dev/hidraw4`. `SPACEMOUSE_SUPPORT_DIR` and `SPACEMOUSE_PYTHON` have the same
meaning as in the standalone script. Live input is checked before connecting
the robot, and the worker remains open across episodes until training exits.

Keyboard events still provide Space (take over/return), S/F (success/failure),
F8/F9 (alternative outcomes), Esc (stop), and B (policy downward bias).
Episode movement keys W/X/A/D/P/L are disabled. The reset menu retains WASD XY
jogging and exploration-bias keys. Keyboard selection still uses
`--teleop-input-backend {auto,evdev,pynput}` and `--teleop-keyboard`.

Space takes effect after the current four-step chunk's last control period,
with an acknowledged servo stop before switching. Without a pending switch or
outcome, enabling intervention adds no last-period wait: observation/inference
work overlaps the fourth action's remaining period. A Space request arriving
after the boundary decision is handled at the next chunk boundary.
Release SpaceMouse before
pressing Space to return. Releasing the device or receiving no fresh input
for 0.2 s stops human motion while leaving human mode active. A failed input
worker aborts collection and stops the servo. As with the standalone official
publisher, a fresh cached HID value cannot prove the USB device is connected.

Human six-axis velocities use `--teleop-speed-m-s` (default 0.0135 m/s per axis),
`--teleop-angular-speed-rad-s` (default roll/pitch 0.09 rad/s, yaw 0.108 rad/s), and
`--spacemouse-deadzone` (default 0.05). Spatial angular velocity is converted to
the adapter's tool-local rotation increment before sending and replay storage.
Startup checks the maximum combined per-tick translation and rotation against
`--max-step-m` and `--max-step-rad`. Command-lead limits remain active; training
workspace restrictions remain disabled. Gripper buttons are not connected.

Takeover does not cut a policy chunk or create a short transition. Human
control is recorded in four-step chunks with `source=HUMAN_OVERRIDE` and
`intervention=1`. Only terminal outcomes or existing safety/budget boundaries
may yield a shorter chunk. The replay keeps ACT's original `ref_chunk` as the
actor input and separately stores executed human actions as the BC target.
Older replay without that field still uses the ACT reference as its BC target.
`metrics.jsonl` includes the control source and handoff point, and checkpoint
metrics count human chunks and steps. The configured episode timeout includes
time spent under human control; increase `--episode-time` when longer
corrections are needed.
If an inference action is refused for crossing the TCP workspace, the episode
is truncated and the arm proceeds to the next sampled reset pose instead of
terminating training. A successfully executed prefix is retained; a refusal on
the first action creates no replay transition.
Checkpoints and replay are saved at episode boundaries and every 1,000 low-level
steps. Resume with the same arguments plus `--resume`.

`--total-env-steps` is a cumulative online target. To continue for another
10,000 online steps after finishing 10,000, resume with
`--total-env-steps 20000`; the existing replay, networks, optimizers, and update
counters are retained. Resume permits changing `warmup_steps`, this cumulative
online target and `device`, with progress-safety checks.

## Frozen Stage-2 inference

Use `act_rlt.infer_stage2` with an initialized Stage-2 v3 checkpoint. It loads
the saved Actor, the matching Stage-1 RL Token and ACT checkpoints, and ACT's
saved pre/postprocessors. It does not construct a learner, update weights, or
save a training checkpoint. The Actor uses its mean action, without exploration
noise. It executes fixed four-action chunks with the same physical
step limits as Stage-2 training. Start the existing FR3 control server first.

Check all three models without connecting to the robot:

```bash
python -m act_rlt.infer_stage2 \
  --checkpoint outputs/ethernet_rlt/act_rlt_stage2_0921 \
  --dry-run
```

Run inference from a fixed reference pose:

```bash
python -m act_rlt.infer_stage2 \
  --checkpoint outputs/ethernet_rlt/act_rlt_stage2_0921 \
  --allow-motion --episodes 1
```

Inference uses a fixed start pose. Choose the saved training reference or
capture the current TCP, then use WASD and Enter/y to confirm the start.
It has no random reset or workspace bounds and retreats +Z 1 cm after an episode.
During
an episode, `s` or `f` ends after the current chunk, `q` stops the session, and
`--episode-time` sets the maximum rollout duration (default 5 seconds). Gripper
commands remain held. The Stage-2 checkpoint records the original ACT and
Stage-1 paths; if those checkpoints moved, supply `--act-checkpoint` and
`--stage1-checkpoint` with their new locations.
Camera handling matches training by default: `--observation-selection latest`
uses a newer cached observation, and `--camera-boundary-mode timestamp` enables
exposure/state timestamp pairing when available. Both options accept the same
`boundary` and `legacy` alternatives as training.
The inference terminal groups model/control settings at startup, displays reset
coordinates in millimetres, and refreshes running status at most once per second.
Interactive terminals reuse one status line; redirected output uses plain lines.
Each episode ends with a success/failure, workspace, or stop result, cumulative
clipping counts and average boundary timing. The session summary reports outcome
counts and success rate over completed episodes. These summaries are terminal
output only; inference does not append training metrics or save checkpoints.

## ACT inference (first version)

Run from the repository root in the installed Evo-RLT environment. Start the
existing control server separately, as for collection, and stop collection
before connecting. Supported checkpoints include camera-only/action6, RGB + wrench6/action6,
and joint-state7/action7, including their saved processors. Six-dimensional
state is interpreted as raw base-frame wrench in fx/fy/fz/mx/my/mz order.

```bash
python -m act_rlt.infer \
  --checkpoint outputs/ethernet_act/act_rlt_001_state7_act \
  --dry-run --duration 10
```

After the model and cameras warm up, Enter starts an episode; the prompt returns
after every episode so another dry-run can be started without reloading the model.
Ctrl+C stops the session.
Dry-run reads live observations and prints predictions without sending arm
motion commands. Both cameras use native 640x480 RGB. Gripper commands are
omitted in this version, so the current grasp is held.

For motion, omit `--dry-run`. After connecting and warming up, the script shows
the measured TCP XYZ, rotation vector, and derived sample/workspace bounds.
Press Enter to measure the TCP pose again and confirm its XY as the center.
Sample resets use the recorder's fixed intrinsic XYZ Euler orientation
`(179.7, 1.4, 90.6)` degrees, converted to a rotation vector. Press `r` to
refresh the preview, or `q` to quit without moving. The sample
space defaults to fixed reference X/Y (0 cm half range) with absolute Z = 0.402 m.
The workspace is the captured TCP XYZ ±5 cm. Reset selection accepts `=` / `-`
to adjust the next sample half range by 0.5 cm (0–5 cm), `r` to resample,
and Enter/y to start. The initial reference XY remains unchanged. Only after
confirmation are workspace bounds installed; sample bounds and +Z retreats
are checked against the workspace. Resets move at 0.02 m/s.
Place the arm at a safe center and orientation first and inspect the printed
bounds; these limits do not check for fixtures or collisions. Every
sampled reset target uses the fixed collection orientation, including the first
move. The policy may still rotate the TCP during inference. Override reset
speed with `--reset-speed`.
The first sampled move requires explicit authorization. At every sampled point,
choose whether to start inference or move to a new sample. After each
`--duration`, choose whether to return to that episode's measured start pose,
move to a new sample, or quit; the process stays alive for subsequent episodes.
Default action limits are 2 mm translation and 0.02 rad rotation per step;
`--max-step-m` and `--max-step-rad` override them.
At the end of every inference episode, an action-limit report prints the total
number of evaluated steps and the counts/percentages that triggered the
translation limit, rotation limit, either limit, or both limits.
Every episode also writes a complete JSONL trace to `logs/act_rlt_infer/` by
default (`--log-dir DIR` changes the destination). The terminal prints the
absolute file path. Events are buffered in memory and written after the servo
stream stops, so filesystem writes cannot delay control ticks. `model_chunk`
records every postprocessed model prediction
before temporal ensembling; `control_step` records every control tick, including
the ensembled/unbounded action, bounded action, vote count, deadline lateness,
cached TCP/joint state and raw base-frame external wrench immediately before
sending, the attempted servo target TCP, and—when enabled—the bounded
unexecuted-motion residual and Z-stall state. The `episode_start` event records
whether `residual_control` was enabled.
Automatic Python cyclic GC is paused for each episode. A full collection runs
before worker startup and after servo shutdown and worker joins, so collection
cannot stall the servo publisher. Normal reference-counted object/tensor cleanup
continues. The caller's original GC state is restored even on failure.
The episode start/end events record the boundary collection duration and current
process RSS before/after collection for checking memory across repeated episodes.
Add `--profile-timing` to record wall/CPU time for image conversion,
preprocessing, policy calls, postprocessing, CPU copies and action preparation.
It also records Python GC pauses with the executing thread and active inference
phase, camera frame timings, refill requests, chunk enqueue times and underruns.
CUDA events measure the first policy call without adding a CUDA synchronization;
the CUDA interval can include host pauses between GPU launches.
For an unattended live-camera test without arm motion, run several episodes in
the same process (omit `--dry-run-episodes` to keep the interactive prompts):

```bash
python -m act_rlt.infer \
  --checkpoint outputs/ethernet_act/act_dataset_928_state7_act \
  --fps 30 --no-temporal-ensemble --duration 6 \
  --dry-run --dry-run-episodes 8 --profile-timing \
  --log-dir logs/act_rlt_infer/timing_dry_run
```

The measured pose comes from the control client's state cache, and the
target is fire-and-forget: a row does not prove the robot reached that target.
Compare a step's `target_tcp_sent` with subsequent steps' `measured_tcp_before`
to diagnose whether small insertion actions were physically followed. Traces
do not include camera images; the wrench is a cached raw measurement, not a
contact-state decision.

The loop runs at 30 Hz by default, using checkpoint action chunking (override with
`--n-action-steps 1` for replanning every tick). Saved processors normalize
observations and unnormalize actions; no augmentation is applied. By default,
the original non-residual controller re-anchors every relative action to the
latest measured TCP pose. Add `--residual-control` to accumulate unexecuted XYZ
translation instead: measured motion is subtracted from the pending offset,
which is clipped independently on X/Y/Z to ±0.3 mm before sending
`measured TCP + residual`. Rotation keeps its per-step convention. In residual
mode, after about 0.25 s of continuous requested descent with at least 0.2 mm
commanded but less than 0.05 mm measured descent, the script clears the Z
residual and suppresses further downward motion for that episode; upward retreat
remains possible. This is a position-stall guard, not a calibrated force/contact
detector. Observation/inference latency above 0.5 seconds stops the run.
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
Its XY position defines the sample box. Samples use a fixed absolute Z and TCP
orientation, and are drawn uniformly in the robot base frame:

```text
x in [x0, x0 + 0.025] m
y in [y0 + 0.003, y0 + 0.02] m
z = 0.402 m
TCP XYZ Euler angle = [179.7, 1.4, 90.6] degrees
```

The recorder stores the complete hand-taught reference pose, including its
orientation. Only sampled starts use the fixed Euler angle above, converted to
the rotation-vector representation used by the Franka control API. Moving to a
sample changes to that fixed orientation; returning to `p0` restores the taught
orientation. Insertion and retreat keep the taught orientation.

For each episode, the robot is positioned at the sampled pose outside
recording. After a 1.0-second settling delay, recording starts automatically:

```text
sample -> reference p0 -> p0 + [0, -0.001, -0.006]
```

The episode is saved automatically, followed by a 1.0-second delay. The robot
then returns `+0.6 cm` along Z and `+0.1 cm` along Y to `p0`, samples the next pose, moves there outside
recording, waits 1.0 seconds, and starts the next episode without confirmation.

During either 1.0-second non-episode window:

- `1`: move to the immutable reference pose.
- `2`: move to the current sampled pose.
- `Q`: stop collection.

Using a numeric key pauses automation; `Enter` resumes it. `Ctrl+C` stops the
collector at any time.

During an episode, press `X` (configurable with `--abort-key`) to request an
immediate motion stop and discard the entire buffered episode.  Automatic
motion stays paused after an abort; press `Enter` when it is safe to resample,
reposition, and continue.

Only `sample -> p0 -> -Z` is recorded. Sampling, repositioning, retreating,
and numeric-key moves are excluded from the dataset. Saved data uses a 22D
observed state and two native `640x480` RGB streams.

Each frame also records three independent float32 observation vectors:
`observation.joint_torque` (7 measured `tau_J` values, Nm),
`observation.external_joint_torque` (7 filtered `tau_ext_hat_filtered`
estimates, Nm), and `observation.external_wrench` (raw `O_F_ext_hat_K`,
`[fx, fy, fz, mx, my, mz]`, N/Nm, expressed in the robot base frame).
They come from one RobotState read and the latest streamed server snapshot;
the existing 22D `observation.state` is unchanged. They are recorded at the
requested `--fps` (15 or 30), with existing last-good fallback behavior on
transient state-read failures. This is a recording cadence, not a guarantee
of distinct fresh hardware samples at every frame. Restart the updated
control server and create a new dataset; older datasets without these fields
cannot be resumed with this collector.

It records both action representations:

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
and `0.01 m/s` for the final `-Z` segment. Adjust them independently with
`--move-speed` and `--insertion-speed`. The transition delays can be changed
with `--pre-episode-sleep` and `--post-episode-sleep`.
Recording frequency is selected with `--fps`; supported values are `15` and
`30` Hz, with `15` Hz as the default. The Z insertion depth defaults to 0.6 cm
and can be set in centimetres with `--insertion-cm` (original two-stage collector only),
or in metres with `--insert-minus-z`. These options are mutually exclusive.
Insertion also offsets the endpoint by 0.1 cm along robot-base -Y, keeping the taught orientation.
For example, `--insertion-cm 1.5` makes the second stage descend 1.5 cm from p0;
the non-recorded retreat then returns to p0.

Use a one-episode low-speed dry run before a larger collection:

```bash
bash act_rlt/data_collection/collect_sample_space.sh \
  --dataset act_rlt_smoke \
  --episodes 1 \
  --move-speed 0.02 \
  --insertion-speed 0.01
```
