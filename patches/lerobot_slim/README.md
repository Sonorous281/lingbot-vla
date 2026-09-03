# LeRobot slim patches (`patches/lerobot_slim/`)

These three files are the **load-bearing** slimming overrides applied on top of
the official `lerobot==0.4.2` wheel by `scripts/apply_lerobot_slim.py`.

They are byte-identical to the previously-vendored `vendor/lerobot` sources,
which were themselves the official `0.4.2` wheel plus exactly these three edits.

## Why slimming is required (not cosmetic)

The official `lerobot==0.4.2` `policies/__init__.py` eagerly imports every
policy backend. `deploy.lingbot_vla_policy` imports
`lerobot.policies.pi0.configuration_pi0`, which triggers that eager
`__init__`, which pulls:

```
lerobot.policies.__init__ (eager) -> groot.modeling_groot
  -> lerobot.policies.pretrained:33 (eager) -> lerobot.configs.train
  -> lerobot.envs -> lerobot.robots -> lerobot.motors.motors_bus -> import serial 💥
```

`pyserial` (and the further realsense/dynamixel/feetech hardware deps behind it)
are not in the RoboTwin VLA inference runtime, so the eager chain breaks the
import. The three overrides defer the imports:

| file | change | nature |
|------|--------|--------|
| `policies/__init__.py` | eager exports -> `_LAZY_EXPORTS` + `__getattr__` | load-bearing, behaviour-neutral |
| `policies/pretrained.py` | `TrainPipelineConfig` import guarded behind `TYPE_CHECKING` | load-bearing, behaviour-neutral |
| `processor/__init__.py` | eager -> lazy `__getattr__` | load-bearing, behaviour-neutral |

Only import *timing* changes (lazy vs eager); the exported symbols and their
semantics are unchanged. Verified bitwise-identical VLA inference output
(`max_abs_diff = 0.0`) against the vendored baseline.

## Application

```bash
pip install --no-deps "lerobot==0.4.2"
python scripts/apply_lerobot_slim.py
```

`--no-deps` is required: lerobot 0.4.2 pins `torch>=2.2.1,<2.8.0`, which would
downgrade the RoboTwin CUDA stack (`torch 2.8.0`). The slim files make the
hardware deps unnecessary, so `--no-deps` is safe for inference.

The applier is **fail-closed** and idempotent. For each target it takes a
three-way decision on the file's sha256 against two pinned hashes — the
official `0.4.2` wheel hash and the patched (slim) hash:

| target sha256 matches | action | status |
|-----------------------|--------|--------|
| patched hash | leave untouched (idempotent) | `already_patched` |
| official `0.4.2` hash | write the slim copy | `patched` |
| neither (or version ≠ 0.4.2) | **do not overwrite**; exit non-zero | `unexpected_hash` / `version_mismatch` |

The shipped slim source itself must also hash to the pinned patched value, else
the run fails as `patch_source_drift` (an accidental edit to `patches/lerobot_slim/`
fails loudly instead of installing a different patch).
