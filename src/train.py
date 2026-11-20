"""Train PPO; writes structured progress, checkpoints, and run provenance."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import time

import jax
import numpy as np

from algorithms.ppo import PPOConfig, init_network, make_trainer, save_checkpoint
from env.robotaxi import RobotaxiEnv, RobotaxiObservation


def main():
    parser = argparse.ArgumentParser()
    for name in ("seed", "num_envs", "rollout_steps", "updates", "epochs", "minibatches", "map_id", "max_steps"):
        parser.add_argument("--"+name.replace("_", "-"), type=int, default=getattr(PPOConfig(), name))
    parser.add_argument("--dynamic-speed", type=float, default=0.7)
    parser.add_argument("--out", type=Path, default=Path("artifacts/ppo_seed42"))
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()
    config = PPOConfig(**{k: getattr(args, k) for k in asdict(PPOConfig()) if hasattr(args, k)})
    if min(config.updates, config.num_envs, config.rollout_steps, args.save_every) <= 0:
        parser.error("Update, environment, rollout and save counts must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = {"config": asdict(config), "python": sys.version, "platform": platform.platform(),
        "devices": [str(d) for d in jax.devices()], "jax_version": jax.__version__,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv}
    (args.out/"config.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    key = jax.random.PRNGKey(config.seed)
    key, env_key, network_key, reset_key = jax.random.split(key, 4)
    ep, initial = RobotaxiEnv.init_params(env_key, map_id=config.map_id,
        max_steps=config.max_steps, dynamic_speed=config.dynamic_speed)
    optimizer, reset, update = make_trainer(config, ep, initial)
    observation, states = reset(jax.random.split(reset_key, config.num_envs))
    params = init_network(network_key, observation.as_vector().shape[-1], config.hidden_size)
    opt_state = optimizer.init(params)
    save_checkpoint(args.out/"untrained.npz", params, config, 0, 0)
    start = time.perf_counter()
    total_episodes = total_success = total_collision = 0
    print(json.dumps({"event": "training_started", **provenance}), flush=True)
    with (args.out/"metrics.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        for i in range(1, config.updates+1):
            params, opt_state, states, observation_vector, key, summary = update(
                params, opt_state, states, observation.as_vector(), key)
            observation = RobotaxiObservation(observation_vector)
            summary = jax.device_get(summary)
            if not np.all(np.isfinite(summary["losses"])):
                raise FloatingPointError(f"Non-finite loss at update {i}")
            episodes = int(summary["episodes"])
            total_episodes += episodes
            total_success += int(summary["successes"])
            total_collision += int(summary["collisions"])
            elapsed = time.perf_counter()-start
            steps = i*config.num_envs*config.rollout_steps
            row = {"update": i, "env_steps": steps, "elapsed_s": elapsed,
                "steps_per_second": steps/max(elapsed, 1e-6), "episodes": episodes,
                "mean_return": float(summary["return_sum"])/episodes if episodes else None,
                "success_rate": int(summary["successes"])/episodes if episodes else None,
                "collision_rate": int(summary["collisions"])/episodes if episodes else None,
                "mean_length": float(summary["length_sum"])/episodes if episodes else None,
                "episodes_total": total_episodes, "successes_total": total_success,
                "collisions_total": total_collision,
                **dict(zip(("loss", "policy_loss", "value_loss", "latent_entropy", "approx_kl", "clip_fraction"),
                           map(float, summary["losses"])))}
            log.write(json.dumps(row)+"\n")
            print(json.dumps(row), flush=True)
            if i % args.save_every == 0 or i == config.updates:
                save_checkpoint(args.out/"latest.npz", params, config, i, steps)
    save_checkpoint(args.out/"final.npz", params, config, config.updates, steps)
    (args.out/"TRAINING_COMPLETE.json").write_text(json.dumps(row, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
