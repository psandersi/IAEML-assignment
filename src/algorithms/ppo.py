"""Continuous-action PPO and GAE implemented with JAX/Optax.

The rollout stores pre-tanh Gaussian actions. Change-of-variable terms cancel
in the importance ratio; latent Normal entropy is used for exploration.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from env.robotaxi import RobotaxiEnv


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 42
    num_envs: int = 64
    rollout_steps: int = 128
    updates: int = 300
    epochs: int = 4
    minibatches: int = 8
    hidden_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    max_grad_norm: float = 0.5
    map_id: int = 0
    dynamic_speed: float = 0.7
    max_steps: int = 600
    guided: bool = False
    reward_control: str = ""

    @property
    def progress_reward(self):
        return self.guided or self.reward_control == "progress"


def init_network(key, obs_dim=39, hidden=64):
    keys = iter(jax.random.split(key, 6))
    def mlp(out_dim, output_scale):
        sizes = (obs_dim, hidden, hidden, out_dim)
        return tuple({"w": jax.random.normal(next(keys), (a, b))*scale/jnp.sqrt(float(a)),
                      "b": jnp.zeros(b)}
                     for a, b, scale in zip(sizes[:-1], sizes[1:], (np.sqrt(2), np.sqrt(2), output_scale)))
    return {"actor": mlp(2, 0.01), "critic": mlp(1, 1.), "log_std": jnp.full((2,), -0.5)}


def forward(params, obs):
    def mlp(layers):
        x = obs
        for layer in layers[:-1]:
            x = jnp.tanh(x @ layer["w"] + layer["b"])
        return x @ layers[-1]["w"] + layers[-1]["b"]
    return mlp(params["actor"]), mlp(params["critic"])[..., 0], jnp.clip(params["log_std"], -5., 1.)


def normal_log_prob(latent, mean, log_std):
    return -0.5*jnp.sum(((latent-mean)*jnp.exp(-log_std))**2 + 2*log_std + np.log(2*np.pi), -1)


def generalized_advantage(rewards, values, next_values, terminated, done, gamma, lam):
    """Time-limit bootstrap uses the final observation, never the reset state.

    True terminations do not bootstrap. Both termination and truncation stop
    the GAE recursion, preventing information leakage across episode resets.
    """
    deltas = rewards+gamma*(1-terminated.astype(jnp.float32))*next_values-values
    def reverse(adv, row):
        delta, boundary = row
        adv = delta+gamma*lam*(1-boundary.astype(jnp.float32))*adv
        return adv, adv
    _, advantages = jax.lax.scan(reverse, jnp.zeros_like(rewards[0]), (deltas, done), reverse=True)
    return advantages, advantages+values


def make_trainer(config, env_params, initial):
    c = config
    batch_size = c.num_envs*c.rollout_steps
    if batch_size % c.minibatches:
        raise ValueError("rollout_steps*num_envs must be divisible by minibatches")
    schedule = optax.linear_schedule(c.learning_rate, c.learning_rate*0.1,
                                    c.updates*c.epochs*c.minibatches)
    optimizer = optax.chain(optax.clip_by_global_norm(c.max_grad_norm), optax.adam(schedule, eps=1e-5))
    reset = jax.vmap(lambda k: RobotaxiEnv.reset(k, env_params, initial))
    step = jax.vmap(lambda k, s, a: RobotaxiEnv.step(k, s, a, env_params))

    def rollout(params, env_state, observation, key):
        def one(carry, _):
            state, obs, key = carry
            key, noise_key, step_key, reset_key = jax.random.split(key, 4)
            mean, value, log_std = forward(params, obs)
            latent = mean+jnp.exp(log_std)*jax.random.normal(noise_key, mean.shape)
            log_prob = normal_log_prob(latent, mean, log_std)
            final_obs, final_state, reward, done, info = step(
                jax.random.split(step_key, c.num_envs), state, jnp.tanh(latent))
            # Value is evaluated BEFORE autoreset, essential for time limits.
            _, next_value, _ = forward(params, final_obs.as_vector())
            reset_obs, reset_state = reset(jax.random.split(reset_key, c.num_envs))
            def choose(a, b):
                mask = done.reshape((c.num_envs,)+(1,)*(a.ndim-1))
                return jnp.where(mask, b, a)
            next_state = jax.tree.map(choose, final_state, reset_state)
            next_obs = jnp.where(done[:, None], reset_obs.as_vector(), final_obs.as_vector())
            data = (obs, latent, log_prob, value, reward, next_value,
                    info["terminated"], done,
                    jnp.where(done, info["episode_return"], 0.),
                    done & info["success"], done & info["collision"],
                    jnp.where(done, info["length"], 0))
            return (next_state, next_obs, key), data
        return jax.lax.scan(one, (env_state, observation, key), None, length=c.rollout_steps)

    def loss(params, batch):
        obs, latent, old_logp, old_value, advantage, target = batch
        mean, value, log_std = forward(params, obs)
        logp = normal_log_prob(latent, mean, log_std)
        ratio = jnp.exp(logp-old_logp)
        policy_loss = -jnp.mean(jnp.minimum(ratio*advantage,
                    jnp.clip(ratio, 1-c.clip_eps, 1+c.clip_eps)*advantage))
        clipped_value = old_value+jnp.clip(value-old_value, -c.clip_eps, c.clip_eps)
        value_loss = 0.5*jnp.mean(jnp.maximum((value-target)**2, (clipped_value-target)**2))
        entropy = jnp.sum(log_std+0.5*np.log(2*np.pi*np.e))
        total = policy_loss+c.value_coef*value_loss-c.entropy_coef*entropy
        approx_kl = jnp.mean((ratio-1)-(logp-old_logp))
        clip_fraction = jnp.mean(jnp.abs(ratio-1) > c.clip_eps)
        return total, jnp.array([total, policy_loss, value_loss, entropy, approx_kl, clip_fraction])

    @jax.jit
    def update(params, opt_state, env_state, observation, key):
        (env_state, observation, key), data = rollout(params, env_state, observation, key)
        obs, latent, old_logp, values, rewards, next_values, terminated, done, ep_returns, success, collision, lengths = data
        adv, targets = generalized_advantage(rewards, values, next_values, terminated, done, c.gamma, c.gae_lambda)
        adv = (adv-jnp.mean(adv))/(jnp.std(adv)+1e-8)
        flatten = lambda x: x.reshape((batch_size,)+x.shape[2:])
        batch = tuple(map(flatten, (obs, latent, old_logp, values, adv, targets)))
        def epoch(carry, _):
            params, opt_state, key = carry
            key, permutation_key = jax.random.split(key)
            order = jax.random.permutation(permutation_key, batch_size)
            mini = tuple(x[order].reshape((c.minibatches, batch_size//c.minibatches)+x.shape[1:]) for x in batch)
            def minibatch(carry, b):
                params, opt_state = carry
                (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(params, b)
                changes, opt_state = optimizer.update(grads, opt_state, params)
                return (optax.apply_updates(params, changes), opt_state), metrics
            (params, opt_state), metrics = jax.lax.scan(minibatch, (params, opt_state), mini)
            return (params, opt_state, key), metrics
        (params, opt_state, key), metrics = jax.lax.scan(epoch, (params, opt_state, key), None, length=c.epochs)
        count = jnp.sum(done)
        summary = {"episodes": count, "return_sum": jnp.sum(ep_returns),
            "successes": jnp.sum(success), "collisions": jnp.sum(collision),
            "length_sum": jnp.sum(lengths), "mean_step_reward": jnp.mean(rewards),
            "losses": jnp.mean(metrics, axis=(0, 1))}
        return params, opt_state, env_state, observation, key, summary

    return optimizer, reset, update


def save_checkpoint(path, params, config, update, env_steps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    leaves = jax.tree.leaves(params)
    metadata = {"config": asdict(config), "update": update, "env_steps": env_steps,
                "format": "robotaxi-policy-v1", "jax_version": jax.__version__}
    # Write atomically so interruption cannot corrupt the previous snapshot.
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **{f"arr_{i}": np.asarray(x) for i, x in enumerate(leaves)},
                        metadata=np.array(json.dumps(metadata)))
    temporary.replace(path)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        c = PPOConfig(**metadata["config"])
        template = init_network(jax.random.PRNGKey(c.seed), hidden=c.hidden_size)
        leaves, treedef = jax.tree.flatten(template)
        restored = [jnp.asarray(archive[f"arr_{i}"]) for i in range(len(leaves))]
        if any(a.shape != b.shape for a, b in zip(leaves, restored)):
            raise ValueError("Checkpoint parameter shapes do not match its configuration")
    return jax.tree.unflatten(treedef, restored), c, metadata
