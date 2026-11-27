from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from env.base import BaseEnv
from env.robotaxi import RobotaxiEnv, RobotaxiState


@pytest.fixture(scope="module")
def scene():
    return RobotaxiEnv.init_params(jax.random.PRNGKey(0), dynamic_speed=0.)


def open_scene(scene):
    p, s = scene
    # Keep static shapes: move the obstacles far away for isolated physics tests.
    s = eqx.tree_at(lambda z: (z.agent_pos, z.static_obstacles, z.kinematic_obstacles,
        z.heading, z.agent_forward_dir), s,
        (jnp.array([30., 10.]), jnp.full_like(s.static_obstacles, 1000.),
         jnp.full_like(s.kinematic_obstacles, 1000.), jnp.float32(0), jnp.array([1., 0.])))
    return p, s


def step(p, s, a=(0., 0.)):
    return RobotaxiEnv.step(jax.random.PRNGKey(1), s, jnp.array(a), p)


@pytest.mark.parametrize("map_id", [0, 1, 2])
def test_all_maps_have_feasible_start_and_path(map_id):
    p, s = RobotaxiEnv.init_params(jax.random.PRNGKey(0), map_id=map_id)
    assert issubclass(RobotaxiEnv, BaseEnv)
    assert isinstance(s, RobotaxiState)
    assert np.min(np.linalg.norm(np.asarray(s.agent_pos-s.static_obstacles), axis=-1)) > 2*p.radius
    assert np.linalg.norm(np.asarray(s.path_array[-1]-s.goal_pos)) < 1.5
    obs = RobotaxiEnv.get_observation(s, p).as_vector()
    assert obs.shape == (39,)
    assert np.all(np.isfinite(obs)) and np.max(np.abs(obs)) <= 1


def test_acceleration_drag_and_rate_limits(scene):
    p, s = open_scene(scene)
    _, accelerated, _, _, _ = step(p, s, (1., 1.))
    assert 0 < float(accelerated.speed) <= p.max_acceleration*p.step_size
    assert float(accelerated.agent_pos[0]) > float(s.agent_pos[0])
    assert float(accelerated.heading) > 0
    assert float(accelerated.steering) <= p.max_steering_rate*p.step_size+1e-6
    s = eqx.tree_at(lambda z: z.speed, s, jnp.float32(2))
    _, slowed, _, _, _ = step(p, s)
    assert 0 < float(slowed.speed) < 2


def test_collision_uses_sum_of_radii_and_terminal_is_absorbing(scene):
    p, s = open_scene(scene)
    # Centres 1.8m apart: collision for two r=1 circles.
    s = eqx.tree_at(lambda z: z.static_obstacles, s,
                    s.static_obstacles.at[0].set(s.agent_pos+jnp.array([1.8, 0.])))
    _, terminal, _, done, info = step(p, s)
    assert bool(done) and bool(info["collision"]) and not bool(info["success"])
    _, repeated, reward, done, _ = step(p, terminal, (1., 1.))
    assert bool(done) and float(reward) == 0.
    np.testing.assert_array_equal(repeated.agent_pos, terminal.agent_pos)
    assert int(repeated.time) == int(terminal.time)


def test_swept_collision_prevents_tunnelling(scene):
    p, s = open_scene(scene)
    # One long integration step crosses an obstacle, endpoints are collision-free.
    p = replace(p, step_size=1., substeps=1, max_speed=10., drag_linear=0., drag_quadratic=0.)
    s = eqx.tree_at(lambda z: (z.speed, z.static_obstacles), s,
                   (jnp.float32(10.), s.static_obstacles.at[0].set(s.agent_pos+jnp.array([5., 0.]))))
    _, _, _, done, info = step(p, s)
    assert bool(done) and bool(info["collision"])


def test_timeout_and_success_are_distinct(scene):
    p, s = open_scene(scene)
    p = replace(p, max_steps_in_episode=1)
    _, _, _, done, info = step(p, s)
    assert bool(done) and bool(info["truncated"]) and not bool(info["terminated"])
    s = eqx.tree_at(lambda z: z.goal_pos, s, s.agent_pos)
    _, _, _, done, info = step(p, s)
    assert bool(done) and bool(info["success"]) and bool(info["terminated"])
    assert not bool(info["truncated"])


def test_boundary_and_rays(scene):
    p, s = open_scene(scene)
    s = eqx.tree_at(lambda z: z.static_obstacles, s,
                    s.static_obstacles.at[0].set(s.agent_pos+jnp.array([5., 0.])))
    obs = RobotaxiEnv.get_observation(s, p).as_vector()
    # Ray at index 16 points forward. Inflated contact is 5-2=3m away.
    assert float(obs[7+p.num_ray_sensors//2]) == pytest.approx(3/p.perception_radius, abs=1e-5)
    s = eqx.tree_at(lambda z: z.agent_pos, s, jnp.array([0.5, 10.]))
    assert bool(step(p, s)[4]["collision"])


def test_seeded_reset_and_vectorization(scene):
    p, s = scene
    keys = jax.random.split(jax.random.PRNGKey(20), 3)
    obs, batch = jax.vmap(lambda k: RobotaxiEnv.reset(k, p, s))(keys)
    assert obs.as_vector().shape == (3, 39)
    first, single = RobotaxiEnv.reset(keys[0], p, s)
    np.testing.assert_allclose(batch.agent_pos[0], single.agent_pos)
    result = jax.vmap(lambda k, state: RobotaxiEnv.step(k, state, jnp.zeros(2), p))(keys, batch)
    expected = step(p, single)
    np.testing.assert_allclose(result[0].as_vector()[0], expected[0].as_vector(), atol=1e-6)


def test_stationary_guided_policy_pays_time_cost(scene):
    p, state = scene
    _, _, reward, done, info = step(replace(p, progress_reward=True), state)
    assert not bool(done)
    assert float(info["shaping"]) == pytest.approx(0.)
    assert float(reward) == pytest.approx(-0.015)

