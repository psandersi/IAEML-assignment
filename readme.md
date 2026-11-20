# IAEML — Robotaxi: JAX environment and PPO

**Daria Galushko, AI-01** · Assignments 1 and 2 · 20 November 2026

The submitted implementation is `src/env/robotaxi.py` (inherits `BaseEnv`) and
`src/algorithms/ppo.py`. The original starter files below are retained for reference.

| Deliverable | Location |
|---|---|
| Environment, bicycle dynamics, observations, reward | `src/env/robotaxi.py` |
| PPO, GAE, continuous policy, checkpoint I/O | `src/algorithms/ppo.py` |
| Training / matched-seed evaluation / videos | `src/train.py`, `src/evaluate.py` |
| Signed environment report | [reports/assignment1.md](reports/assignment1.md) |
| Signed RL report | [reports/assignment2.md](reports/assignment2.md) |
| Measured results, generated after evaluation | `artifacts/RESULTS.md` |
| Checkpoints, videos, raw episode results | `artifacts/ppo_seed*/` |
| Physics and PPO integration tests | `tests/` |

Reports describe the implementation and clearly distinguish planned outputs from
measured results. The student must add her own reflection after examining the runs.

## Setup (PowerShell, Python 3.12)

```powershell
cd 'C:\Users\dasga\OneDrive\Рабочий стол\pmldl\IAEML-assignment'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-assignment.txt
```

The implementation runs on CPU without a graphics window. On Linux a suitable
GPU build of JAX may be used in a separate environment. CPU/GPU results need not
be bit-identical. `artifacts/environment.freeze.txt` records the versions actually
used by the worker; install it in addition to the requirements file to reproduce
those versions once it exists. The original `requirements.txt` belongs to the
starter demo; the assignment uses `requirements-assignment.txt`.

## Full experiment in the background

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_background.ps1
```

The worker performs tests, a throughput benchmark, then training and evaluation for
seeds 42, 43, 44 sequentially. Each seed uses 300 PPO updates / 2,457,600 transitions.
Evaluation compares random, heuristic, untrained and trained policies on all three
maps with 32 matched-seed episodes each. MP4 and GIF videos show the first evaluation
episode for each trained/reference controller and map. Summaries and plots are
written after each completed seed. The worker stops on any nonzero stage exit code.
Keep the computer awake until it finishes; no interactive renderer is required.

For one shorter, explicitly separate experiment:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_background.ps1 -Updates 100 -Seeds '42' -Episodes 16
```

Do not launch another worker while the first is active: the launcher checks its PID.
Restarting with the same output directories replaces that run's files. Use
`src/train.py --out artifacts/new_run` when preserving a previous experiment.

## Status commands (run yourself when needed)

```powershell
Get-Content logs/pipeline.status.json
Get-Process -Id ([int](Get-Content logs/pipeline.pid)) -ErrorAction SilentlyContinue
Get-Content logs/train_seed42.stdout.log -Tail 5
Get-Content logs/train_seed42.stderr.log -Tail 20
Get-Content artifacts/ppo_seed42/metrics.jsonl -Tail 3
Get-Content artifacts/RESULTS.md
```

The worker is finished only when `pipeline.status.json` says `"state": "complete"`.
Process existence alone does not establish progress or success. A `failed` state
names the failed stage; its stderr is `logs/<stage>.stderr.log`. Training checkpoints
are atomically saved every ten updates as `latest.npz`, with `final.npz` at the end.
`untrained.npz` is the exact initial network of that run. Checkpoints contain model
parameters, not an optimizer state for exact training resumption.

## Local checks and reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --junitxml=artifacts/tests.xml
.\.venv\Scripts\python.exe src/benchmark.py
.\.venv\Scripts\python.exe src/train.py --seed 42 --out artifacts/reproduce_seed42
.\.venv\Scripts\python.exe src/evaluate.py --checkpoint artifacts/ppo_seed42/final.npz --episodes 32
.\.venv\Scripts\python.exe src/summarize.py
.\.venv\Scripts\python.exe scripts/check_submission.py
```

The final command checks required reports, tests, completed runs, checkpoints and
videos before committing. It fails if artifacts are missing or the personal
reflection placeholders are still present. PPO success is measured, never assumed
from the mere existence of a model file.

## Submit to your fork (no automatic push)

After the background pipeline has completed and personal reflection is written:

```powershell
$gitExe = 'C:\Program Files\Git\cmd\git.exe'
& $gitExe status --short
& $gitExe branch --show-current
# The branch reviewed by the instructor must be named exactly Submission.
# If still on main: & $gitExe switch -c Submission
.\.venv\Scripts\python.exe scripts/check_submission.py
if ($LASTEXITCODE -ne 0) { throw 'Fix the listed submission issues before committing.' }
& $gitExe add .gitignore requirements-assignment.txt readme.md src/env/robotaxi.py src/algorithms/ppo.py src/train.py src/evaluate.py src/summarize.py src/benchmark.py scripts tests reports artifacts
& $gitExe commit -m 'Implement robotaxi environment and PPO experiments'
& $gitExe push -u origin Submission
```

`origin` should point to your own fork (currently `git@github.com:psandersi/IAEML-assignment.git`).
Send the repository link to the instructor in a personal message. Deadline in the
assignment PDF: **20 November 2026, 23:59** (timezone unspecified). Do not modify
`Submission` after that deadline and before receiving the grade. Background runs
write working-tree artifacts: only commit finished results before the deadline.

## Original starter documentation

This repository was specifically designed to practice and implement the concepts covered in IAEML course at Innopolis University. [Vehicular automation](https://en.wikipedia.org/wiki/Vehicular_automation) is a notable challenging domain where these concepts can be applied.

Specifically, in this repository we have a simple 2D environment with an agent, destination and obstacles to avoid. The obstacles can be either static or moving. For simplicity, all objects are modeled as circles with radius $r=1$.

`tasks.py` comes in handy when we want to train something like meta-learning.

`base.py` defines an environment which models core elements required in self-driving cars:

- Perception is modeled as $M$ evenly fanned rays casted from the center of the agent.
- Navigation is solved using a dynamic programming algorithm fully executed on a device.

The environment has the following interface

- `init_params()` prepares `env_params` and `init_state`.
- `reset()` can be used to set `env_state` back to `env_state`.
- `step()` executes one step of the simulation.
- `get_observation()` returns the "point-of-view" of the agent.

Notably, `BaseEnv` is just a struct with functions that operates on `BaseEnvState` and takes `BaseEnvParams` as static arguments. Static arguments are "baked" into a function using `jax.jit()` decorator.

One limitation at the moment is that sampling a new task will always result in recompiling the environment functions since it changes the content of `env_params` and shapes in `env_state`. The latter can be solved by forcing all maps to be of the same shape.

`base.py` does not implement a model of vehicle dynamics.

You can play it

```bash
python src/main.py
```

There are some issues with JAX on OSX. This solves it

```bash
JAX_PLATFORM_NAME=cpu python src/main.py
```

...

---

Tested with python 3.11, Ubuntu 24.04 adm64 on Oct 20, 2025.
