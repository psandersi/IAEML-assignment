"""Matched-seed evaluation and real rendered rollouts, without a GUI."""
import argparse
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from algorithms.ppo import forward, load_checkpoint
from env.robotaxi import RobotaxiEnv, heuristic_action


def wilson(success, count):
    z = 1.96
    p = success/count
    centre = (p+z*z/(2*count))/(1+z*z/count)
    half = z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
    return [centre-half, centre+half]


def run_evaluation(params, config, map_id, policy, episodes, seed):
    ep, initial = RobotaxiEnv.init_params(jax.random.PRNGKey(seed), map_id=map_id,
        max_steps=config.max_steps, dynamic_speed=config.dynamic_speed)
    reset = jax.vmap(lambda k: RobotaxiEnv.reset(k, ep, initial))
    step = jax.vmap(lambda k, s, a: RobotaxiEnv.step(k, s, a, ep))
    @jax.jit
    def simulate(params, key):
        key, reset_key = jax.random.split(key)
        obs, states = reset(jax.random.split(reset_key, episodes))
        def one(carry, _):
            states, obs, key = carry
            key, action_key, step_key = jax.random.split(key, 3)
            if policy == "random":
                actions = jax.random.uniform(action_key, (episodes, 2), minval=-1., maxval=1.)
            elif policy == "heuristic":
                actions = jax.vmap(heuristic_action)(obs)
            else:
                mean, _, _ = forward(params, obs.as_vector())
                actions = jnp.tanh(mean)
            obs, states, reward, done, info = step(jax.random.split(step_key, episodes), states, actions)
            frame = {"position": states.agent_pos[0], "heading": states.heading[0],
                "speed": states.speed[0], "moving": states.kinematic_obstacles[0],
                "done": done[0], "success": info["success"][0],
                "collision": info["collision"][0], "return": states.episode_return[0]}
            return (states, obs, key), frame
        (states, _, _), trajectory = jax.lax.scan(one, (states, obs, key), None, length=config.max_steps)
        return {"success": states.success, "collision": states.collision,
                "truncated": states.truncated, "returns": states.episode_return,
                "lengths": states.time,
                "final_distance": jnp.linalg.norm(states.agent_pos-states.goal_pos, axis=-1)}, trajectory
    start = time.perf_counter()
    result, trajectory = jax.device_get(simulate(params, jax.random.PRNGKey(seed)))
    n = int(np.sum(result["success"]))
    summary = {"map_id": map_id, "policy": policy, "episodes": episodes, "evaluation_seed": seed,
        "successes": n, "success_rate": n/episodes, "success_wilson95": wilson(n, episodes),
        "collision_rate": float(np.mean(result["collision"])),
        "timeout_rate": float(np.mean(result["truncated"])),
        "mean_return": float(np.mean(result["returns"])),
        "std_return": float(np.std(result["returns"])),
        "mean_length": float(np.mean(result["lengths"])),
        "mean_final_distance": float(np.mean(result["final_distance"])),
        "evaluation_wall_s_including_compile": time.perf_counter()-start}
    return summary, result, trajectory, ep, initial


def render_video(trajectory, ep, initial, destination, label):
    """Render actual physics states at 5 fps; both MP4 and browser-friendly GIF."""
    h, w = ep.map_height_width
    scale, margin = 12, 40
    width, height = w*scale, h*scale+margin
    static = np.asarray(initial.static_obstacles)
    path = np.asarray(initial.path_array)
    goal = np.asarray(initial.goal_pos)
    def screen(p):
        return (float(p[0]*scale), float(margin+(h-1-p[1])*scale))
    def circle(draw, p, radius, color):
        x, y = screen(p)
        r = radius*scale
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
    terminals = np.flatnonzero(trajectory["done"])
    last = int(terminals[0])+1 if len(terminals) else len(trajectory["position"])
    frames = []
    for i in range(0, last, 2):
        canvas = Image.new("RGB", (width, height), "#101827")
        draw = ImageDraw.Draw(canvas)
        draw.line([screen(p) for p in path], fill="#436478", width=2)
        for p in static:
            circle(draw, p, ep.radius, "#6a7688")
        for p in trajectory["moving"][i]:
            circle(draw, p, ep.radius, "#f4a261")
        circle(draw, goal, ep.goal_radius, "#368c65")
        circle(draw, trajectory["position"][i], ep.radius, "#60caff")
        x, y = screen(trajectory["position"][i])
        theta = trajectory["heading"][i]
        draw.line((x, y, x+20*np.cos(theta), y-20*np.sin(theta)), fill="white", width=3)
        draw.text((10, 7), f"{label} | t={i*ep.step_size:.1f}s | v={trajectory['speed'][i]:.2f}m/s", fill="white")
        draw.text((10, 22), f"return={trajectory['return'][i]:.2f}", fill="#b7c8de")
        frames.append(np.asarray(canvas))
    frames.extend([frames[-1]]*5)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(destination.with_suffix(".gif"), frames, duration=200, loop=0)
    imageio.mimsave(destination.with_suffix(".mp4"), frames, fps=5, macro_block_size=2,
                    codec="libx264", quality=7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--maps", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    params, config, metadata = load_checkpoint(args.checkpoint)
    out = args.out or args.checkpoint.parent/"evaluation"
    out.mkdir(parents=True, exist_ok=True)
    untrained, _, _ = load_checkpoint(args.checkpoint.parent/"untrained.npz")
    rows = []
    for map_id in args.maps:
        for policy in ("random", "heuristic", "untrained", "ppo"):
            row, results, trajectory, ep, initial = run_evaluation(
                untrained if policy == "untrained" else params, config,
                map_id, policy, args.episodes, args.seed)
            rows.append(row)
            name = f"map{map_id}_{policy}"
            np.savez_compressed(out/(name+"_episodes.npz"), **results)
            np.savez_compressed(out/(name+"_trajectory.npz"), **trajectory)
            (out/"metrics.json").write_text(json.dumps({"checkpoint": str(args.checkpoint),
                "checkpoint_metadata": metadata, "results": rows}, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)
            if not args.no_video and policy in ("heuristic", "ppo"):
                render_video(trajectory, ep, initial, out/name, name)
    (out/"EVALUATION_COMPLETE.json").write_text(json.dumps({"evaluated_policies": len(rows)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
