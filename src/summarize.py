"""Build an evidence-based results supplement; never invent missing metrics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def summarize(root):
    root = Path(root)
    runs = sorted(root.glob("ppo_seed*"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    text = ["# Автоматически измеренные результаты", "",
        "Это техническое приложение, построенное из файлов запусков. Автор отчётов: Daria Galushko, AI-01.", "",
        "Обучающие seed представлены отдельно. Оценка использует другие случайные ключи; "
        "карта 0 — обучающая, карты 1 и 2 — перенос без дообучения. "
        "Доверительный интервал Wilson отражает вариативность эпизодов, а не независимых обучений.", ""]
    for run in runs:
        metrics = run/"metrics.jsonl"
        if not metrics.exists():
            continue
        rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            continue
        label = run.name
        for ax, field in zip(axes, ("mean_return", "success_rate", "value_loss")):
            # Episode-weighted aggregation in consecutive ten-update windows.
            xs, ys = [], []
            for start in range(0, len(rows), 10):
                window = rows[start:start+10]
                valid = [r for r in window if r[field] is not None]
                if not valid:
                    continue
                weights = [max(r["episodes"], 1) if field != "value_loss" else 1 for r in valid]
                xs.append(window[-1]["env_steps"])
                ys.append(np.average([r[field] for r in valid], weights=weights))
            ax.plot(xs, ys, label=label)
            ax.set_xlabel("Environment transitions")
            ax.set_title(field)
            ax.grid(alpha=.2)
        last = rows[-1]
        status = "завершено" if (run/"TRAINING_COMPLETE.json").exists() else "не завершено"
        text += [f"## {run.name}", "", f"Обучение: {status}. "
            f"Зафиксировано {last['env_steps']:,} переходов за {last['elapsed_s']:.1f} с "
            f"(включая компиляцию), {last['steps_per_second']:.0f} переходов/с.", ""]
        eval_file = run/"evaluation/metrics.json"
        if not eval_file.exists():
            text += ["Оценка ещё не записана. Численные результаты не заявляются.", ""]
            continue
        evaluation = json.loads(eval_file.read_text(encoding="utf-8"))
        text += ["| Карта | Политика | Эпизоды | Успех | Wilson 95% | Столкновения | Таймаут | Награда |", 
                 "|---|---|---:|---:|---|---:|---:|---:|"]
        for r in evaluation["results"]:
            lo, hi = r["success_wilson95"]
            text.append(f"| {r['map_id']} | {r['policy']} | {r['episodes']} | {r['success_rate']:.1%} | "
                f"{lo:.1%}–{hi:.1%} | {r['collision_rate']:.1%} | {r['timeout_rate']:.1%} | {r['mean_return']:.2f} |")
        text += ["", "Видео фактических траекторий: "
            f"[{run.name}/evaluation]({run.name}/evaluation). "
            "В видео всегда показан первый оценочный эпизод; успешные эпизоды специально не отбираются.", ""]
    for ax in axes:
        if ax.lines:
            ax.legend()
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root/"learning_curves.png", dpi=160)
    plt.close(fig)
    text += ["## Ограничения интерпретации", "", 
        "Наличие файла checkpoint само по себе не доказывает, что агент научился достигать цели. "
        "Успех следует оценивать по таблицам выше и сравнивать с необученной политикой и baseline. "
        "Повышение shaped reward без роста доли успехов не является решением навигации. "
        "Эпизоды здесь варьируют стартовую позицию и ориентацию, но не расположение препятствий; "
        "это ограниченная проверка устойчивости.", "", "![Кривые обучения](learning_curves.png)", ""]
    (root/"RESULTS.md").write_text("\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    summarize(parser.parse_args().root)
