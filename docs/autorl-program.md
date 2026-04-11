# PROVIDER AutoRL

Autonomous PPO research for the PROVIDER supply-chain simulation.
The idea: give an AI agent the RL training setup and let it experiment overnight.
It modifies the network/hyperparameters, trains for 5 minutes, checks if the
defender reward improved, keeps or discards, and repeats.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr9`).
   The branch `autorl/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autorl/<tag>` from current main.
3. **Read the in-scope files** for full context:
   - `experiments/autorl_program.md` — this file, the instructions
   - `experiments/autorl_train.py` — frozen harness, DO NOT MODIFY
   - `provider_sim/rl/network.py` — the file you modify (architecture + hyperparameters)
   - `provider_sim/env/environment.py` — the simulation environment (read-only context)
4. **Initialize results.tsv**: create `experiments/checkpoints/autorl_results.tsv` with
   just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: confirm setup looks good, then kick off experimentation.

## What you are optimizing

The PROVIDER simulation models a supply-chain (S1: Soja/Soy). Two agents compete:
- **Attacker**: disrupts supply-chain entities (reduces supply)
- **Defender**: protects entities (restores supply)

**Your job is to find the best RL policy for the Defender.**

The metric is `val_reward` — mean defender health (fraction of supply-chain intact)
over 3 validation episodes with a greedy policy. **Higher is better** (range: 0 to 1).
A val_reward of 0.7 means the defender kept 70% of the supply-chain healthy on average.

Zero-sum property: `reward_attacker + reward_defender = 1.0` always.

## What you CAN and CANNOT do

**What you CAN do:**
- Modify `provider_sim/rl/network.py` — this is the ONLY file you edit.
  Everything is fair game: network architecture, activation functions, depth,
  width, normalization, initialization, and all PPO hyperparameters at the top.

**What you CANNOT do:**
- Modify `experiments/autorl_train.py` — it is frozen. It defines the training
  harness, evaluation protocol, time budget, and metric.
- Modify `provider_sim/env/environment.py` or any PDL scenario file.
- Install new packages. Use only what's already importable (PyTorch, NumPy, etc.).
- Modify the val_reward computation. It is defined in `autorl_train.py`.

## Running an experiment

```bash
python experiments/autorl_train.py > run.log 2>&1
```

The script trains for a **fixed 5-minute wall-clock budget**, then validates
greedy on seeds 42, 43, 44. Each run produces ~N episodes depending on hardware.

Extract the key metric:
```bash
grep "^val_reward:" run.log
```

Extract the full summary:
```bash
grep -A 10 "^---$" run.log | tail -11
```

## Output format

When the script finishes it prints a summary like this:

```
---
val_reward:       0.623400
mean_train_r:     0.601200
train_episodes:   28
training_seconds: 300.1
total_seconds:    328.4
peak_memory_mb:   142.0
n_params:         75840
n_obs:            99
n_atk_act:        20
```

## Logging results

Log each experiment to `experiments/checkpoints/autorl_results.tsv`
(tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	val_reward	n_params	status	description
```

1. git commit hash (short, 7 chars)
2. val_reward achieved (e.g. 0.623400) — use 0.000000 for crashes
3. n_params from the summary (e.g. 75840) — use 0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_reward	n_params	status	description
a1b2c3d	0.610000	75840	keep	baseline
b2c3d4e	0.628000	75840	keep	lr 3e-4 → 1e-3
c3d4e5f	0.599000	75840	discard	GELU activation (worse than ReLU)
d4e5f6g	0.641000	155648	keep	wider network h1=512 h2=256
e5f6g7h	0.000000	0	crash	added LayerNorm (shape mismatch)
```

**Note:** do NOT commit the results.tsv — leave it untracked by git.

## Simplicity criterion

All else being equal, simpler is better. Weigh complexity cost against improvement:
- val_reward +0.002 from adding 20 lines of hacky code? Probably not worth it.
- val_reward +0.002 from deleting code? Definitely keep.
- val_reward ~unchanged but simpler code? Keep.

## Checkpoint strategy

At the **start** of every run, `autorl_train.py` automatically backs up all
checkpoint files as `.bak` siblings:

```
checkpoints/autorl_attacker.pt     ← current weights (this run)
checkpoints/autorl_attacker.pt.bak ← weights before this run (restore on discard)
checkpoints/autorl_defender.pt
checkpoints/autorl_defender.pt.bak
checkpoints/autorl_obs_norm.npz
checkpoints/autorl_obs_norm.npz.bak
```

The `restore_cmd` line at the end of `run.log` contains the exact cp commands
to restore all three files. Use it when discarding.

## The experiment loop

LOOP FOREVER:

1. Look at the git state: current branch and commit
2. Modify `provider_sim/rl/network.py` with an experimental idea
3. `git commit` (commit the change before running)
4. Run: `python experiments/autorl_train.py > run.log 2>&1`
5. Read metric: `grep "^val_reward:" run.log`
6. If grep output is empty, the run crashed.
   Run `tail -n 50 run.log` to read the Python traceback and attempt a fix.
   After more than a few failed fix attempts, give up — log as "crash" and revert.
7. Log results in autorl_results.tsv (never commit this file)
8. If val_reward **improved** (higher): keep the commit and advance the branch
9. If val_reward is **equal or worse**: revert BOTH the code AND the checkpoints:
   ```bash
   git reset --hard HEAD~1
   # Get the restore command from the log:
   grep "^restore_cmd:" run.log | sed 's/restore_cmd: *//' | bash
   ```
   This ensures the next experiment starts from the same weights as this one did,
   making val_reward comparisons fair across experiments.

**Timeout:** Each run takes ~5 min training + ~30s validation = ~5.5 min total.
If a run exceeds 12 minutes, kill it and treat as failure.

**Crashes:** Fix trivial bugs (missing import, typo) and re-run.
Fundamentally broken ideas (OOM, wrong tensor shapes): log as crash and move on.

## Ideas to try

Architecture:
- Change hidden layer widths (h1, h2 in PPONet.__init__)
- Add a third hidden layer
- Change activation: ReLU → GELU / SiLU / Tanh
- Add BatchNorm or LayerNorm between layers
- Orthogonal weight initialization
- Separate shared backbone into independent actor/critic networks

Hyperparameters (constants at top of network.py):
- Learning rate LR: try 1e-3, 1e-4
- Discount GAMMA: try 0.95, 0.999
- GAE_LAMBDA: try 0.9, 0.98
- Clip epsilon CLIP_EPS: try 0.1, 0.3
- PPO_EPOCHS: try 2, 8
- ENTROPY_COEF: try 0.001, 0.05
- Defender budget DEFENDER_BUDGET: try 0.6, 0.8 (note: attacker auto-adjusts to zero-sum)

**NEVER STOP:** Once the experiment loop has begun, do NOT pause to ask the human
if you should continue. The human might be asleep. You are autonomous. If you run
out of ideas, think harder — try combining previous near-misses, try more radical
changes. The loop runs until the human interrupts you, period.
