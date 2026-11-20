"""Measure the actual environment throughput after compilation, on this device."""
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from env.robotaxi import RobotaxiEnv


def main():
    n, steps = 32, 32
    key = jax.random.PRNGKey(1)
    p, initial = RobotaxiEnv.init_params(key, max_steps=steps+1)
    keys = jax.random.split(key, n)
    _, batch = jax.vmap(lambda k: RobotaxiEnv.reset(k, p, initial))(keys)
    action = jnp.array([0.2, 0.05])
    @jax.jit
    def vectorized(states):
        def one(states, _):
            _, states, _, _, _ = jax.vmap(lambda k, s: RobotaxiEnv.step(k, s, action, p))(keys, states)
            return states, None
        return jax.lax.scan(one, states, None, length=steps)[0].agent_pos
    # Compile both paths before measuring their steady-state execution.
    RobotaxiEnv.step(key, initial, action, p)[1].agent_pos.block_until_ready()
    vectorized(batch).block_until_ready()
    start = time.perf_counter()
    scalar_positions = []
    for i in range(n):
        s = jax.tree.map(lambda x: x[i], batch)
        for _ in range(steps):
            _, s, _, _, _ = RobotaxiEnv.step(keys[i], s, action, p)
        scalar_positions.append(s.agent_pos)
    scalar_positions = jnp.stack(scalar_positions)
    scalar_positions.block_until_ready()
    scalar_time = time.perf_counter()-start
    start = time.perf_counter()
    optimized_positions = vectorized(batch)
    optimized_positions.block_until_ready()
    vector_time = time.perf_counter()-start
    np.testing.assert_allclose(scalar_positions, optimized_positions, atol=2e-5)
    result = {"num_envs": n, "steps": steps, "devices": [str(d) for d in jax.devices()],
        "scalar_python_loop_s": scalar_time, "vmap_scan_s": vector_time,
        "scalar_steps_per_second": n*steps/scalar_time,
        "vmap_scan_steps_per_second": n*steps/vector_time,
        "speedup": scalar_time/vector_time,
        "note": "Both paths warmed up. Scalar path includes Python dispatch and state slicing. One timing sample."}
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
