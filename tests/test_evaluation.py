import jax
import numpy as np

from algorithms.ppo import PPOConfig, init_network
from evaluate import render_video, run_evaluation


def test_evaluation_and_actual_video_encoding(tmp_path):
    config = PPOConfig(max_steps=3)
    params = init_network(jax.random.PRNGKey(0))
    summary, results, trajectory, ep, initial = run_evaluation(
        params, config, map_id=0, policy="ppo", episodes=2, seed=321)
    assert summary["episodes"] == 2
    assert summary["success_rate"]+summary["collision_rate"]+summary["timeout_rate"] == 1.
    assert trajectory["position"].shape == (3, 2)
    assert np.all(results["lengths"] <= 3)
    output = tmp_path/"smoke"
    render_video(trajectory, ep, initial, output, "Untrained policy: encoder smoke test")
    assert output.with_suffix(".mp4").stat().st_size > 100
    assert output.with_suffix(".gif").stat().st_size > 100
