"""Matched-seed PPO reward ablation. Run from the repository root."""
import json
from pathlib import Path
import subprocess
import sys


def counts(directory):
    data = json.loads((directory / "evaluation/metrics.json").read_text(encoding="utf-8"))
    rows = sorted((r for r in data["results"] if r["policy"] == "ppo"), key=lambda r: r["map_id"])
    assert [r["map_id"] for r in rows] == [0, 1, 2] and all(r["episodes"] == 32 for r in rows)
    return [r["successes"] for r in rows]


def main():
    root = Path("artifacts/reward_control")
    root.mkdir(parents=True, exist_ok=True)
    for seed in (42, 43, 44):
        for reward in ("progress", "original"):
            out = root / reward / f"ppo_seed{seed}"
            if not (out/"TRAINING_COMPLETE.json").exists():
                subprocess.run([sys.executable, "-u", "src/train.py", "--reward-control", reward,
                                "--seed", str(seed), "--updates", "100", "--save-every", "5",
                                "--out", str(out)], check=True)
            if not (out/"evaluation/EVALUATION_COMPLETE.json").exists():
                subprocess.run([sys.executable, "-u", "src/evaluate.py", "--checkpoint",
                                str(out/"best.npz")], check=True)
    lines = ["# Влияние награды и предварительного обучения", "",
             "Число доездов из 32 на каждой карте.", "",
             "| Seed | Условие | Карта 0 | Карта 1 | Карта 2 |", "|---:|---|---:|---:|---:|"]
    totals = [0, 0, 0]
    for seed in (42, 43, 44):
        name = f"ppo_seed{seed}"
        for i, (label, path) in enumerate([
            ("Старая награда", root/"original"/name), ("Новая награда", root/"progress"/name),
            ("Новая награда и подражание", Path("artifacts/guided")/name)]):
            values = counts(path)
            totals[i] += sum(values)
            lines.append(f"| {seed} | {label} | " + " | ".join(map(str, values)) + " |")
    lines += ["", f"За 288 оценочных эпизодов PPO с нуля достиг цели {totals[0]} раз со старой "
              f"наградой и {totals[1]} раз с новой. При той же новой награде предварительное "
              f"подражание дало {totals[2]} доездов. Разбивка по картам и инициализациям приведена выше."]
    target = root/"RESULTS.md"
    temporary = target.with_suffix(".tmp")
    temporary.write_text("\n".join(lines)+"\n", encoding="utf-8")
    temporary.replace(target)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
