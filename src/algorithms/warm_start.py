"""Fit the actor to reference steering and speed targets before PPO rollouts."""
import jax
import jax.numpy as jnp
import optax

from algorithms.ppo import forward
from env.robotaxi import RobotaxiObservation, heuristic_action


def pretrain(params, key):
    optimizer = optax.adam(0.001)

    def loss(actor, observations, targets):
        mean, _, _ = forward({**params, "actor": actor}, observations)
        return jnp.mean((jnp.tanh(mean) - targets)**2)

    def step(carry, _):
        actor, state, key = carry
        key, sample_key = jax.random.split(key)
        # Vary nuisance features independently to cover unseen ray patterns.
        obs = jax.random.uniform(sample_key, (256, 39))
        angle = (obs[:, 0] - 0.5) * 3.6
        distance = 0.25 + 0.75 * obs[:, 1]
        obs = obs.at[:, 0].set(distance*jnp.cos(angle))
        obs = obs.at[:, 1].set(distance*jnp.sin(angle))
        obs = obs.at[:, 2:4].set(2*obs[:, 2:4]-1)
        obs = obs.at[:, 5].set(2*obs[:, 5]-1)
        targets = jax.vmap(lambda v: heuristic_action(RobotaxiObservation(v)))(obs)
        grad = jax.grad(loss)(actor, obs, targets)
        updates, state = optimizer.update(grad, state, actor)
        return (optax.apply_updates(actor, updates), state, key), None

    actor = params["actor"]
    (actor, _, _), _ = jax.lax.scan(step, (actor, optimizer.init(actor), key), None, length=4000)
    return {**params, "actor": actor, "log_std": jnp.full((2,), -2.)}
