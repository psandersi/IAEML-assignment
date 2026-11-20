"""Check concrete deliverables before the user commits and pushes Submission."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def check():
    errors = []
    def require(relative):
        p = ROOT/relative
        if not p.is_file() or p.stat().st_size == 0:
            errors.append(f"Missing/empty: {relative}")
            return None
        return p
    for report in ("reports/assignment1.md", "reports/assignment2.md"):
        p = require(report)
        if p and "Личная рефлексия — заполнить автору перед сдачей" in p.read_text(encoding="utf-8"):
            errors.append(f"Add your own reflection and replace its placeholder heading: {report}")
    for name in ("src/env/robotaxi.py", "src/algorithms/ppo.py", "requirements-assignment.txt",
                 "artifacts/RESULTS.md", "artifacts/learning_curves.png", "artifacts/benchmark.json",
                 "artifacts/environment.freeze.txt"):
        require(name)
    p = require("artifacts/tests.xml")
    if p:
        suites = ET.parse(p).getroot()
        if any(int(s.get("failures", 0))+int(s.get("errors", 0)) for s in suites.iter("testsuite")):
            errors.append("The saved test run contains failures/errors")
    status = require("artifacts/pipeline.status.json")
    if status and json.loads(status.read_text(encoding="utf-8"))["state"] != "complete":
        errors.append("Background pipeline has not completed successfully")
    for seed in (42, 43, 44):
        base = f"artifacts/ppo_seed{seed}"
        for name in ("config.json", "untrained.npz", "final.npz", "metrics.jsonl",
                     "TRAINING_COMPLETE.json", "evaluation/metrics.json", "evaluation/EVALUATION_COMPLETE.json"):
            require(base+"/"+name)
        metrics = ROOT/base/"evaluation/metrics.json"
        if metrics.exists():
            results = json.loads(metrics.read_text(encoding="utf-8"))["results"]
            if len(results) != 12:
                errors.append(f"Incomplete 3-map x 4-policy evaluation: seed {seed}")
        for map_id in (0, 1, 2):
            for policy in ("heuristic", "ppo"):
                require(f"{base}/evaluation/map{map_id}_{policy}.mp4")
    for error in errors:
        print("FAIL:", error)
    if errors:
        print(f"\n{len(errors)} submission checks need attention.")
        return 1
    print("All submission files are present; tests and pipeline completed. Review your results and reflection before pushing.")
    return 0


if __name__ == "__main__":
    sys.exit(check())
