import jax
import jax.numpy as jnp
import numpy as np

from algorithms.ppo import (PPOConfig, forward, generalized_advantage,
    init_network, load_checkpoint, make_trainer, normal_log_prob, save_checkpoint)
from env.robotaxi import RobotaxiEnv


def test_gae_stops_at_reset_and_bootstraps_time_limit():
    column = lambda x: jnp.asarray(x)[:, None]
    advantage, targets = generalized_advantage(column([1., 2., 3.]), column([.5, .6, .7]),
        column([.6, .7, 4.]), column([False, True, False]), column([False, True, True]), .9, .8)
    np.testing.assert_allclose(advantage[:, 0], [2.048, 1.4, 5.9], atol=1e-5)
    np.testing.assert_allclose(targets[:, 0], [2.548, 2., 6.6], atol=1e-5)


def test_normal_log_probability():
    actual = normal_log_prob(jnp.zeros((3, 2)), jnp.zeros((3, 2)), jnp.zeros(2))
    np.testing.assert_allclose(actual, -np.log(2*np.pi), atol=1e-6)


def test_checkpoint_roundtrip(tmp_path):
    config = PPOConfig()
    params = init_network(jax.random.PRNGKey(5))
    path = tmp_path/"model.npz"
    save_checkpoint(path, params, config, 2, 16384)
    restored, restored_config, metadata = load_checkpoint(path)
    assert config == restored_config and metadata["env_steps"] == 16384
    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(a, b)


def test_compiled_ppo_update_is_finite_and_changes_parameters():
    # A single small update validates the integration, gradients and autoreset.
    c = PPOConfig(num_envs=2, rollout_steps=4, updates=1, epochs=1,
                  minibatches=2, max_steps=3)
    ep, initial = RobotaxiEnv.init_params(jax.random.PRNGKey(0), max_steps=3, dynamic_speed=0.)
    optimizer, reset, update = make_trainer(c, ep, initial)
    obs, states = reset(jax.random.split(jax.random.PRNGKey(1), 2))
    params = init_network(jax.random.PRNGKey(2))
    trained, _, states, next_obs, _, metrics = update(params, optimizer.init(params), states,
        obs.as_vector(), jax.random.PRNGKey(3))
    assert np.all(np.isfinite(np.asarray(metrics["losses"])))
    assert int(metrics["episodes"]) >= 2
    assert next_obs.shape == (2, 39)
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(trained)))
