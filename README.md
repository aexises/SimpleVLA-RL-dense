# SimpleVLA-RL LIBERO Subgoal Reward Notes

This fork keeps the original SimpleVLA-RL rollout and training pipeline intact. The LIBERO subgoal reward code is an optional extension used for logging, debugging, and controlled reward-shaping experiments.

The baseline path is unchanged when:

```bash
reward.subgoal.enabled=False
```

That is the default in `verl/trainer/config/ppo_trainer.yaml`. With this setting, the LIBERO rollout worker does not create a subgoal reward engine, no subgoal tensors are added to the rollout batch, and the original terminal success reward path is used.

## Relevant Files

Subgoal reward implementation:

- `verl/utils/subgoal_reward/libero_state.py`: best-effort LIBERO / robosuite state extraction.
- `verl/utils/subgoal_reward/phases.py`: subgoal phase definitions.
- `verl/utils/subgoal_reward/task_specs.py`: maps task metadata and extracted state to a supported task spec.
- `verl/utils/subgoal_reward/tracker.py`: online per-env tracker with monotonic phase progression and best-progress deltas.
- `verl/utils/subgoal_reward/dense_reward.py`: dense reward formula and weights.
- `verl/utils/subgoal_reward/engine.py`: facade called by rollout workers.

Integration points:

- `verl/workers/rollout/rob_rollout.py`: creates one LIBERO env process per rollout and updates one subgoal tracker per env process.
- `verl/trainer/main_ppo.py`: keeps original rewards in `log_only`, or optionally applies dense rewards in `add` / `replace`.
- `verl/trainer/ppo/ray_trainer.py`: preserves `group_id` for GRPO grouping by task suite and task id.
- `verl/trainer/config/ppo_trainer.yaml`: config defaults.

Tests:

- `tests/test_subgoal_reward.py`

Run them with:

```bash
python -m unittest tests.test_subgoal_reward
```

## Running The Original Baseline

Use the existing LIBERO script without subgoal overrides:

```bash
bash examples/run_openvla_oft_rl_libero.sh
```

Or make the baseline setting explicit:

```bash
HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
  ...existing overrides... \
  reward.subgoal.enabled=False
```

This should match the original SimpleVLA-RL behavior. Use this for baseline comparisons and sanity checks.

## Log-Only Subgoal Tracking

Use this to compute subgoal phase/progress metrics without changing the training reward:

```bash
HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
  ...existing overrides... \
  reward.subgoal.enabled=True \
  reward.subgoal.mode=log_only \
  reward.subgoal.log=True
```

Expected behavior:

- One independent tracker is created per LIBERO env process / rollout instance.
- Multiple samples for the same LIBERO task do not share tracker state.
- Original terminal success reward remains the training reward.
- Numeric fields such as `subgoal_phase_id`, `subgoal_progress`, `reward_total`, and `success` are added to the rollout batch for logging/debugging.

Unsupported tasks default to terminal-only fallback:

```bash
reward.subgoal.unsupported_task_behavior=terminal_only
```

To fail fast when a task does not match a defined subgoal rule:

```bash
reward.subgoal.unsupported_task_behavior=error
```

## Adding New Subgoal Rules

A subgoal rule has two pieces:

1. Phase classes in `verl/utils/subgoal_reward/phases.py`.
2. Task matching in `verl/utils/subgoal_reward/task_specs.py`.

Each phase must implement:

```python
class MyPhase(Phase):
    name = "my_phase"

    def compute_progress(self, state: LiberoState) -> float:
        # Return a value in [0, 1].
        return progress

    def is_done(self, state: LiberoState) -> bool:
        # Return True once this phase should advance.
        return done
```

Rules for phase logic:

- Use only online information from the current step: current/next observation, action, info, done, and task metadata.
- Do not inspect future trajectory steps.
- Return `0.0` or conservative fallback behavior when required state is unavailable.
- Keep `phase_id` monotonic. The tracker already enforces this, but phase definitions should not depend on going backward.
- Use best-progress deltas for reward: reward should come from new best progress, not movement back and forth.

To attach phases to tasks, update `infer_task_spec(...)` in `task_specs.py`. For example:

```python
def infer_task_spec(state: LiberoState, thresholds: Thresholds) -> TaskSpec | None:
    text = " ".join(part for part in [state.task_name, state.instruction] if part).lower()

    if "open" in text and "drawer" in text:
        return TaskSpec(
            name="open_drawer",
            phases=[
                ReachHandlePhase(thresholds),
                GraspHandlePhase(thresholds),
                PullDrawerPhase(thresholds),
                SuccessPhase(thresholds),
            ],
        )

    ...
```

If a new rule needs extra LIBERO state, add it to `LiberoState` and extract it in `LiberoStateExtractor`. Extraction must be best effort: missing keys should not crash normal training unless `unsupported_task_behavior=error` is explicitly selected.

After adding rules, add or update tests in `tests/test_subgoal_reward.py`.

## Changing Reward Functions

The default dense reward is:

```text
total =
    w_sub * positive_delta_best_progress
    + w_phase * phase_completed
    + w_final * terminal_success
    - w_smooth * action_delta_l2
```

Default weights:

```yaml
reward:
  subgoal:
    weights:
      subgoal_progress: 0.2
      phase_transition: 0.05
      terminal_success: 1.0
      smoothness: 0.0
    clip_dense_reward: 0.05
```

For most experiments, prefer changing config values rather than code:

```bash
reward.subgoal.weights.subgoal_progress=0.1 \
reward.subgoal.weights.phase_transition=0.02 \
reward.subgoal.weights.terminal_success=1.0 \
reward.subgoal.weights.smoothness=0.001 \
reward.subgoal.clip_dense_reward=0.03
```

If the formula itself changes, edit `DenseRewardManager.compute(...)` in `verl/utils/subgoal_reward/dense_reward.py`.

Keep terminal success dominant. Dense reward should separate bad, medium, and good failed rollouts, but finishing the task should remain more valuable than accumulating shaping reward.

## Reward Modes

Log only, no training reward change:

```bash
reward.subgoal.enabled=True \
reward.subgoal.mode=log_only
```

Add dense reward to the original terminal reward:

```bash
reward.subgoal.enabled=True \
reward.subgoal.mode=add
```

Replace original reward with dense subgoal reward:

```bash
reward.subgoal.enabled=True \
reward.subgoal.mode=replace
```

Use `log_only` first for every new rule. Move to `add` or `replace` only after the phase/progress logs look correct.

## Experiment Examples

Baseline:

```bash
bash examples/run_openvla_oft_rl_libero.sh
```

Log default pick/place rules:

```bash
bash examples/run_openvla_oft_rl_libero.sh \
  reward.subgoal.enabled=True \
  reward.subgoal.mode=log_only
```

Different thresholds:

```bash
bash examples/run_openvla_oft_rl_libero.sh \
  reward.subgoal.enabled=True \
  reward.subgoal.mode=log_only \
  reward.subgoal.thresholds.reach_distance=0.04 \
  reward.subgoal.thresholds.target_distance=0.05 \
  reward.subgoal.thresholds.lift_height=0.10
```

Dense reward added to terminal reward:

```bash
bash examples/run_openvla_oft_rl_libero.sh \
  reward.subgoal.enabled=True \
  reward.subgoal.mode=add \
  reward.subgoal.weights.subgoal_progress=0.1 \
  reward.subgoal.weights.phase_transition=0.03 \
  reward.subgoal.clip_dense_reward=0.03
```

Dense reward replacement:

```bash
bash examples/run_openvla_oft_rl_libero.sh \
  reward.subgoal.enabled=True \
  reward.subgoal.mode=replace \
  reward.subgoal.weights.terminal_success=1.0
```

Smoothness penalty:

```bash
bash examples/run_openvla_oft_rl_libero.sh \
  reward.subgoal.enabled=True \
  reward.subgoal.mode=add \
  reward.subgoal.weights.smoothness=0.001
```

LoRA baseline, no subgoal reward:

```bash
bash examples/run_openvla_oft_rl_libero.sh \
  reward.subgoal.enabled=False \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32
```

LoRA with log-only subgoal metrics:

```bash
bash examples/run_openvla_oft_rl_libero.sh \
  reward.subgoal.enabled=True \
  reward.subgoal.mode=log_only \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32
```

LoRA with dense reward:

```bash
bash examples/run_openvla_oft_rl_libero.sh \
  reward.subgoal.enabled=True \
  reward.subgoal.mode=add \
  reward.subgoal.weights.subgoal_progress=0.1 \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32
```

## Suggested Experiment Naming

Make the reward mode, rule set, and LoRA setting visible in `EXPERIMENT_NAME`.

Examples:

```bash
EXPERIMENT_NAME=lib10_baseline_full
EXPERIMENT_NAME=lib10_subgoal_log_pickplace_v1
EXPERIMENT_NAME=lib10_subgoal_add_pickplace_v1_wsub01_clip03
EXPERIMENT_NAME=lib10_lora32_baseline
EXPERIMENT_NAME=lib10_lora32_subgoal_add_pickplace_v1
```

## Metrics To Watch

Useful training logs include:

- `train_verify_score/all`: original terminal success score.
- `train_reward/reward_all`: final training reward after selected reward mode.
- `train_reward/subgoal_dense`: dense reward contribution when subgoal reward is enabled.
- `train_verify_score/subgoal/subgoal_progress`: average phase progress.
- `train_verify_score/subgoal/subgoal_best_progress`: average best progress.
- `train_verify_score/subgoal/subgoal_positive_delta`: new progress gained online.
- `train_verify_score/subgoal/phase_completed`: phase completions per rollout batch.

For GRPO, rollouts are grouped by `group_id`, derived from task suite and task id. This prevents group-relative normalization across unrelated LIBERO tasks.

## Safety Checklist

Before using a new rule for reward shaping:

1. Run the baseline with `reward.subgoal.enabled=False`.
2. Run the same setup with `reward.subgoal.enabled=True` and `reward.subgoal.mode=log_only`.
3. Confirm terminal success metrics are still computed normally.
4. Inspect subgoal progress and phase completion logs.
5. Add or update tests for the new phase/rule.
6. Only then try `mode=add` or `mode=replace`.

Do not enable dense reward by default. Do not modify LIBERO source for rule experiments unless there is no other way to extract required state.
