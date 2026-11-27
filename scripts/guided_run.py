"""Run the guided experiment from the repository root. All output is local."""
import json
from pathlib import Path
import subprocess
import sys

root = Path("artifacts/guided")
root.mkdir(parents=True, exist_ok=False)
lines = ["# Результаты на полных маршрутах", "",
         "| Seed | Карта | Политика | Доезды из 32 | Столкновения | Таймауты |",
         "|---:|---:|---|---:|---:|---:|"]
for seed in (42, 43, 44):
    out = root / f"ppo_seed{seed}"
    for args in (["src/train.py", "--guided", "--seed", str(seed), "--updates", "100",
                  "--save-every", "5", "--out", str(out)],
                 ["src/evaluate.py", "--checkpoint", str(out/"best.npz")]):
        print(" ".join(args), flush=True)
        subprocess.run([sys.executable, "-u", *args], check=True)
    results = json.loads((out/"evaluation/metrics.json").read_text(encoding="utf-8"))
    for row in results["results"]:
        lines.append(f"| {seed} | {row['map_id']} | {row['policy']} | {row['successes']} | "
                     f"{row['collision_rate']:.0%} | {row['timeout_rate']:.0%} |")
    (root/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
print("COMPLETE", flush=True)
