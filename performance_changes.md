# Performance Changes

This note records the low-risk memory/performance changes made during the RAM-pressure investigation. The intent is to reduce unnecessary object retention and improve observability without changing the training pipeline, rollout structure, rewards, masks, PPO logic, or data shapes.

## Changes Made

### Removed raw action retention from rollout history

`RobHFRollout` still computes `actions` and uses them immediately for environment stepping, but no longer stores them in each `vla_history` step.

Why this should not affect training:

- `_prepare_output_batch()` never returned `action`.
- The downstream trainer and actor update paths consume `responses`, `input_ids`, `attention_mask`, `pixel_values`, `finish_step`, optional `proprio`, optional subgoal metric tensors, and reward tensors.
- Current subgoal metrics are computed during environment stepping and returned as numeric tensors, not as raw action trajectories.

Expected effect:

- Slightly lower temporary rollout memory.
- No change to returned `DataProto` keys or tensor shapes.

### Cleaned up failed Robotwin environment initialization

If Robotwin `setup_demo()` fails after partially creating an environment, the old environment is now closed before retrying initialization.

Why this should not affect training:

- This path only runs after an initialization failure.
- The failed environment was not usable training data.
- Retry behavior is preserved.

Expected effect:

- Less leaked simulator state after transient initialization failures.

### Released Robotwin environment references after close

After `close_env(clear_cache=True)`, `RobotwinEnvWrapper` now sets `self.env = None` and `self.args = None`.

Why this should not affect training:

- The rollout has already collected `complete`, `finish_step`, observations, and model tensors before cleanup.
- This only drops Python references after environment use is complete.

Expected effect:

- Earlier garbage collection of simulator objects.

### Added lightweight memory instrumentation

Added `[memory] ...` log lines in rollout workers and the trainer driver around high-risk memory boundaries:

- rollout start
- after environment cleanup
- after rollout output preparation
- after driver receives rollout generation output
- after driver unions rollout data
- after rollout filtering
- after actor update
- after entropy computation

Why this should not affect training:

- The logger only reads process RSS via `resource.getrusage()` and CUDA allocation stats via PyTorch.
- It does not alter tensors, seeds, rewards, masks, optimizer state, or control flow.

Expected effect:

- Easier identification of whether RAM grows in the rollout worker, Ray transfer/object store, driver union/filtering, actor update, or validation phases.

## Changes Deliberately Not Made

The following optimizations were not implemented because they could change training behavior unless carefully padded/reconstructed:

- stopping rollout early when all environments are inactive
- running VLA only for active environments
- trimming trajectories by `finish_step`
- changing image preprocessing away from TensorFlow
- removing returned tensors used by actor update or reward computation

These may be valid later, but they require a stricter equivalence check because the current actor pipeline expects fixed dense trajectory tensors.
