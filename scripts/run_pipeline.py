"""Background worker: tests -> training -> evaluation -> results.

Each stage has separate logs. A failure stops the pipeline and records its
exit code. No external process monitoring is required.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    logs = ROOT/"logs"
    logs.mkdir(exist_ok=True)
    (ROOT/"artifacts").mkdir(exist_ok=True)
    status = {"pid": os.getpid(), "state": "running", "stages": [], "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    def write():
        for target in (logs/"pipeline.status.json", ROOT/"artifacts/pipeline.status.json"):
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
            temporary.replace(target)
    def stage(name, command):
        record = {"name": name, "command": command, "state": "running"}
        status["stages"].append(record)
        status["current_stage"] = name
        write()
        print(f"Starting {name}", flush=True)
        start = time.perf_counter()
        env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
        with (logs/(name+".stdout.log")).open("w", encoding="utf-8") as stdout, \
             (logs/(name+".stderr.log")).open("w", encoding="utf-8") as stderr:
            result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        record.update(state="complete" if result.returncode == 0 else "failed",
                      exit_code=result.returncode, elapsed_s=time.perf_counter()-start)
        write()
        if result.returncode:
            raise RuntimeError(f"{name} failed with code {result.returncode}; see logs/{name}.stderr.log")
    write()
    try:
        # Record the exact environment actually used, rather than guessed versions.
        stage("environment", ["scripts/environment.py"])
        (ROOT/"artifacts/environment.freeze.txt").write_text(
            (logs/"environment.stdout.log").read_text(encoding="utf-8"), encoding="utf-8")
        if not args.skip_tests:
            stage("tests", ["-m", "pytest", "tests", "-q", "--junitxml=artifacts/tests.xml"])
        stage("benchmark", ["src/benchmark.py"])
        for seed in args.seeds:
            output = f"artifacts/ppo_seed{seed}"
            stage(f"train_seed{seed}", ["-u", "src/train.py", "--seed", str(seed),
                "--updates", str(args.updates), "--out", output])
            stage(f"evaluate_seed{seed}", ["-u", "src/evaluate.py", "--checkpoint", output+"/final.npz",
                "--episodes", str(args.episodes)])
            stage(f"summarize_seed{seed}", ["src/summarize.py"])
        status["state"] = "complete"
    except Exception:
        status["state"] = "failed"
        status["error"] = traceback.format_exc()
        traceback.print_exc()
        raise
    finally:
        status["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write()


if __name__ == "__main__":
    main()
