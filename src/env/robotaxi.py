"""JAX bicycle environment. No changes to the supplied BaseEnv are necessary.

Coordinates are metres, headings radians, speeds m/s, accelerations m/s^2.
All physical objects retain the assignment's radius of one metre.
"""
from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from env.base import BaseEnv, BaseEnvState
from env.tasks import BaseTaskSampler
from utils.utils import convert_to_world_view


@dataclass(frozen=True)
class RobotaxiParams:
    map_height_width: tuple = (30, 60)
    max_steps_in_episode: int = 600
    fps: int = 10
    step_size: float = 0.1
    num_ray_sensors: int = 32
    perception_radius: float = 12.0
    agent_FOV: float = float(np.pi)
    wheelbase: float = 1.5
    max_speed: float = 4.0
    max_acceleration: float = 3.0
    max_steering: float = 0.65
    max_steering_rate: float = 1.5
    drag_linear: float = 0.15
    drag_quadratic: float = 0.025
    substeps: int = 4
    radius: float = 1.0
    goal_radius: float = 1.25
    map_scale: float = 3.0
    dynamic_speed: float = 0.7
    reset_noise: float = 0.15
    progress_reward: bool = False


class RobotaxiState(BaseEnvState):
    heading: jax.Array
    speed: jax.Array
    steering: jax.Array
    navigation_cost: jax.Array
    done: jax.Array
    episode_return: jax.Array
    success: jax.Array
    collision: jax.Array
    truncated: jax.Array


class RobotaxiObservation(eqx.Module):
    vector: jax.Array

    def as_vector(self):
        return self.vector


def wrap_angle(x):
    return jnp.arctan2(jnp.sin(x), jnp.cos(x))


class RobotaxiEnv(BaseEnv):
    @staticmethod
    def init_params(key, map_id=0, max_steps=600, path_length=128,
                    discretization_scale=1, perception_radius=12.0,
                    num_ray_sensors=32, fov=np.pi, fps=10, **kwargs):
        if discretization_scale != 1:
            raise ValueError("Navigation uses a one-metre grid.")
        task = BaseTaskSampler(map_id)
        scale = kwargs.pop("map_scale", 3.0)
        params = RobotaxiParams(
            map_height_width=(int(task.height * scale), int(task.width * scale)),
            max_steps_in_episode=max_steps, fps=fps, step_size=1.0/fps,
            perception_radius=perception_radius, num_ray_sensors=num_ray_sensors,
            agent_FOV=float(fov), map_scale=scale, **kwargs)
        world = lambda p: convert_to_world_view(p, (task.height, task.width)) * scale
        static = jax.vmap(world)(task.static_obstacles).astype(jnp.float32)
        moving = jax.vmap(world)(task.kinematic_obstacles).astype(jnp.float32)
        start, goal = world(task.agent_pos), world(task.goal_pos)
        nav = RobotaxiEnv._navigation_grid(goal, static, params)
        # The display path is extracted once. The policy uses the full cost field.
        path = RobotaxiEnv._trace_path(nav, start, goal, path_length)
        forward = path[min(3, len(path)-1)] - start
        heading = jnp.arctan2(forward[1], forward[0])
        state = RobotaxiState(
            time=jnp.int32(0), goal_pos=goal.astype(jnp.float32),
            agent_pos=start.astype(jnp.float32),
            agent_forward_dir=jnp.array([jnp.cos(heading), jnp.sin(heading)]),
            static_obstacles=static, kinematic_obstacles=moving,
            kinematic_obst_velocities=task.kinematic_obst_velocities.astype(jnp.float32)
                                      * params.dynamic_speed,
            path_array=jnp.asarray(path, jnp.float32), heading=heading,
            speed=jnp.float32(0), steering=jnp.float32(0), navigation_cost=nav,
            done=jnp.bool_(False), episode_return=jnp.float32(0),
            success=jnp.bool_(False), collision=jnp.bool_(False),
            truncated=jnp.bool_(False))
        return params, state

    @staticmethod
    @partial(jax.jit, static_argnames=("params",))
    def _navigation_grid(goal, obstacles, params):
        """Eight-connected Bellman relaxation on device; no toroidal wrapping.

        Inflate obstacles by both radii plus 0.1m, forbid corner cutting, and
        reserve the map boundary for the complete vehicle footprint.
        """
        h, w = params.map_height_width
        yy, xx = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
        grid = jnp.stack([xx, yy], -1)
        clearance = jnp.min(jnp.linalg.norm(grid[..., None, :] - obstacles, axis=-1), -1)
        free = ((clearance > 2*params.radius + 0.1) & (xx >= params.radius)
                & (xx <= w-1-params.radius) & (yy >= params.radius)
                & (yy <= h-1-params.radius))
        gx, gy = jnp.rint(goal).astype(jnp.int32)
        value = jnp.full((h, w), 1e4, dtype=jnp.float32).at[gy, gx].set(0.)
        shifts = ((0, 1), (0, -1), (1, 0), (-1, 0),
                  (1, 1), (1, -1), (-1, 1), (-1, -1))

        def relax(_, v):
            padded = jnp.pad(v, 1, constant_values=1e4)
            fp = jnp.pad(free, 1, constant_values=False)
            candidates = [v]
            for dy, dx in shifts:
                neighbor = padded[1+dy:1+dy+h, 1+dx:1+dx+w]
                allowed = fp[1+dy:1+dy+h, 1+dx:1+dx+w]
                if dy and dx:
                    allowed = allowed & fp[1+dy:1+dy+h, 1:1+w] & fp[1:1+h, 1+dx:1+dx+w]
                candidates.append(jnp.where(allowed, neighbor + np.hypot(dy, dx), 1e4))
            return jnp.where(free, jnp.min(jnp.stack(candidates), axis=0), 1e4)

        return jax.lax.fori_loop(0, h*w, relax, value)

    @staticmethod
    def _trace_path(cost, start, goal, length):
        # Host-only visualization helper; never called in an environment step.
        cost = np.asarray(cost)
        pos = np.rint(np.asarray(start)).astype(int)
        if cost[pos[1], pos[0]] >= 1e4:
            raise ValueError("Start cannot reach goal with a radius-one vehicle.")
        path = [pos.copy()]
        for _ in range(length-1):
            x, y = pos
            options = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, ny = x+dx, y+dy
                    if 0 <= ny < cost.shape[0] and 0 <= nx < cost.shape[1]:
                        if dx and dy and (cost[y, nx] >= 1e4 or cost[ny, x] >= 1e4):
                            continue
                        options.append((cost[ny, nx] + 0.01*np.hypot(dx, dy), nx, ny))
            _, nx, ny = min(options)
            pos = np.array([nx, ny])
            path.append(pos.copy())
        return np.asarray(path, dtype=np.float32)

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def reset(key, env_params, init_state):
        k1, k2 = jax.random.split(key)
        pos = init_state.agent_pos + jax.random.uniform(
            k1, (2,), minval=-env_params.reset_noise, maxval=env_params.reset_noise)
        heading = init_state.heading + jax.random.uniform(k2, (), minval=-0.15, maxval=0.15)
        state = eqx.tree_at(
            lambda s: (s.agent_pos, s.heading, s.agent_forward_dir), init_state,
            (pos, heading, jnp.array([jnp.cos(heading), jnp.sin(heading)])))
        return RobotaxiEnv.get_observation(state, env_params), state

    @staticmethod
    def _local_target(state, params):
        """Cost-field descent with a two-metre lookahead in world coordinates."""
        h, w = params.map_height_width
        xy = jnp.rint(state.agent_pos).astype(jnp.int32)
        offsets = jnp.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
                             [1, 1], [-1, 1], [1, -1], [-1, -1]], jnp.int32)
        def descend(_, point):
            candidates = point + offsets
            candidates = jnp.clip(candidates, jnp.array([0, 0]), jnp.array([w-1, h-1]))
            costs = state.navigation_cost[candidates[:, 1], candidates[:, 0]]
            # Same corner constraint as Bellman relaxation.
            safe = ((state.navigation_cost[point[1], candidates[:, 0]] < 1e4)
                    & (state.navigation_cost[candidates[:, 1], point[0]] < 1e4))
            costs = jnp.where(safe, costs + 0.001*jnp.linalg.norm(offsets, axis=-1), 1e4)
            return candidates[jnp.argmin(costs)]
        target = jax.lax.fori_loop(0, 3, descend, jnp.clip(xy, jnp.array([0, 0]), jnp.array([w-1, h-1])))
        return jnp.where(jnp.linalg.norm(state.goal_pos-state.agent_pos) < 4,
                         state.goal_pos, target.astype(jnp.float32))

    @staticmethod
    def _potential(state):
        """Bilinearly interpolate a finite navigation field (wall values capped).

        The cap avoids numerical spikes when approaching the inflation margin.
        """
        field = jnp.minimum(state.navigation_cost, 200.)
        h, w = field.shape
        p = jnp.clip(state.agent_pos, jnp.array([0., 0.]), jnp.array([w-1.001, h-1.001]))
        x, y = jnp.floor(p).astype(jnp.int32)
        a, b = p-jnp.floor(p)
        return -((1-a)*(1-b)*field[y, x] + a*(1-b)*field[y, x+1]
                 + (1-a)*b*field[y+1, x] + a*b*field[y+1, x+1]) / 10.

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def get_observation(env_state, env_params):
        s, p = env_state, env_params
        angles = s.heading + jnp.linspace(-p.agent_FOV, p.agent_FOV,
                                          p.num_ray_sensors, endpoint=False)
        dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], -1)
        obstacles = jnp.concatenate([s.static_obstacles, s.kinematic_obstacles])
        # Inflated circles give distance travelled by the vehicle centre until contact.
        m = s.agent_pos - obstacles
        b = dirs @ m.T
        c = jnp.sum(m*m, -1) - (2*p.radius)**2
        disc = b*b-c
        entry = -b-jnp.sqrt(jnp.maximum(disc, 0))
        hits = jnp.where((disc >= 0) & (entry >= 0), entry, jnp.inf)
        hits = jnp.where(c[None, :] <= 0, 0., hits)
        rays = jnp.min(hits, axis=-1)
        # Analytic distance to footprint-safe map boundary.
        bounds = jnp.array(p.map_height_width[::-1], jnp.float32)-1-p.radius
        t = jnp.where(dirs > 1e-6, (bounds-s.agent_pos)/jnp.where(abs(dirs)>1e-6, dirs, 1.),
                      jnp.where(dirs < -1e-6, (p.radius-s.agent_pos)/jnp.where(abs(dirs)>1e-6, dirs, 1.), jnp.inf))
        rays = jnp.clip(jnp.minimum(rays, jnp.min(t, -1)), 0., p.perception_radius)/p.perception_radius
        def body(vec):
            c, sn = jnp.cos(s.heading), jnp.sin(s.heading)
            return jnp.array([c*vec[0]+sn*vec[1], -sn*vec[0]+c*vec[1]])
        target = body(RobotaxiEnv._local_target(s, p)-s.agent_pos)/4.
        goal = body(s.goal_pos-s.agent_pos)/max(p.map_height_width)
        extras = jnp.concatenate([target, goal, jnp.array([
            s.speed/p.max_speed, s.steering/p.max_steering,
            s.time/p.max_steps_in_episode])])
        return RobotaxiObservation(jnp.concatenate([jnp.clip(extras, -1., 1.), rays]))

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def step(key, env_state, action, env_params):
        del key  # deterministic transitions, stochastic resets
        s, p = env_state, env_params
        action = jnp.clip(jnp.nan_to_num(action), -1., 1.)
        dt = p.step_size/p.substeps
        bounds = jnp.array(p.map_height_width[::-1], jnp.float32)-1-p.radius
        def integrate(_, carry):
            pos, theta, speed, steering, moving, vel, hit = carry
            steering += jnp.clip(action[1]*p.max_steering-steering,
                                 -p.max_steering_rate*dt, p.max_steering_rate*dt)
            force = action[0]*p.max_acceleration-p.drag_linear*speed-p.drag_quadratic*speed*jnp.abs(speed)
            speed = jnp.clip(speed+dt*force, 0., p.max_speed)
            omega = speed*jnp.tan(steering)/p.wheelbase
            direction = jnp.array([jnp.cos(theta+0.5*dt*omega), jnp.sin(theta+0.5*dt*omega)])
            next_pos = pos+dt*speed*direction
            next_moving = moving+dt*vel
            bounce = (next_moving < p.radius) | (next_moving > bounds)
            vel = jnp.where(bounce, -vel, vel)
            next_moving = jnp.clip(next_moving, p.radius, bounds)
            # Swept relative segments prevent tunnelling between substeps.
            before = jnp.concatenate([s.static_obstacles, moving])
            after = jnp.concatenate([s.static_obstacles, next_moving])
            rel = pos-before
            delta = (next_pos-pos)-(after-before)
            t = jnp.clip(-jnp.sum(rel*delta, -1)/(jnp.sum(delta*delta, -1)+1e-8), 0., 1.)
            collision = jnp.any(jnp.linalg.norm(rel+t[:, None]*delta, axis=-1) <= 2*p.radius)
            collision |= jnp.any((next_pos < p.radius) | (next_pos > bounds))
            hit |= collision
            return next_pos, wrap_angle(theta+dt*omega), speed, steering, next_moving, vel, hit

        pos, theta, speed, steering, moving, velocities, collision = jax.lax.fori_loop(
            0, p.substeps, integrate, (s.agent_pos, s.heading, s.speed, s.steering,
                s.kinematic_obstacles, s.kinematic_obst_velocities, jnp.bool_(False)))
        success = (jnp.linalg.norm(pos-s.goal_pos) <= p.goal_radius) & ~collision
        truncated = (s.time+1 >= p.max_steps_in_episode) & ~success & ~collision
        done = success | collision | truncated
        new = eqx.tree_at(lambda z: (z.agent_pos, z.heading, z.speed, z.steering,
            z.agent_forward_dir, z.kinematic_obstacles, z.kinematic_obst_velocities,
            z.time, z.done, z.success, z.collision, z.truncated), s,
            (pos, theta, speed, steering, jnp.array([jnp.cos(theta), jnp.sin(theta)]),
             moving, velocities, s.time+1, done, success, collision, truncated))
        # Potential shaping uses the same discount as PPO. Terminal potential is zero.
        phi_next = jnp.where(success | collision, 0., RobotaxiEnv._potential(new))
        shaping = 0.99*phi_next-RobotaxiEnv._potential(s)
        reward = (shaping-0.01-0.002*jnp.sum(action**2)
                  +20.*success-20.*collision)
        if p.progress_reward:
            target = RobotaxiEnv._local_target(s, p)
            shaping = jnp.linalg.norm(target-s.agent_pos)-jnp.linalg.norm(target-pos)
            reward = shaping - 0.015 - 0.002*jnp.sum(action**2) + 25.*success - 20.*collision
        new = eqx.tree_at(lambda z: z.episode_return, new, s.episode_return+reward)
        # Terminal states are absorbing. The trainer resets explicitly.
        new = jax.tree.map(lambda a, b: jnp.where(s.done, a, b), s, new)
        reward = jnp.where(s.done, 0., reward)
        obs = RobotaxiEnv.get_observation(new, p)
        return obs, new, reward, new.done, {
            "success": new.success, "collision": new.collision,
            "truncated": new.truncated, "terminated": new.success | new.collision,
            "episode_return": new.episode_return, "length": new.time,
            "distance_to_goal": jnp.linalg.norm(new.agent_pos-new.goal_pos),
            "shaping": jnp.where(s.done, 0., shaping)}


def heuristic_action(obs):
    """Explicit non-learning reference controller. Never mixed into PPO actions."""
    v = obs.as_vector()
    angle = jnp.arctan2(v[1], v[0])
    steer = jnp.clip(1.5*angle, -1., 1.)
    target_speed = 0.55*jnp.maximum(0.15, jnp.cos(angle))
    acceleration = jnp.clip(3*(target_speed-v[4]), -1., 1.)
    return jnp.array([acceleration, steer])
