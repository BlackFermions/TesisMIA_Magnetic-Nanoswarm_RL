"""Entorno Gymnasium: conteo por salida, recompensa r_t y condiciones de fin."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from nanoswarm.envs import dynamics
from nanoswarm.envs.dynamics import ACTIVE, AT_OTHER, AT_TARGET, PhysicsParams, YBifurcation
from nanoswarm.envs.observation import OBS_DIM, swarm_observation
from nanoswarm.envs.reward import RewardWeights, compute_reward, mean_distance


class VascularEnv(gym.Env):
    """Direccionar un enjambre de N partículas hacia la rama objetivo de una Y.

    Acción: a ∈ [-1, 1]², recortada a norma ≤ 1 (dirección e intensidad del imán).
    Fin: sale del dominio al menos `done_fraction` del enjambre (terminated) o
    se alcanza el horizonte (truncated).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        condition="R2",
        n_particles=200,
        horizon=800,
        done_fraction=0.95,
        physics=PhysicsParams(),
        reward_weights=RewardWeights(),
        geometry=None,
    ):
        self.condition = dynamics.CONDITIONS[condition]
        self.n_particles = n_particles
        self.horizon = horizon
        self.done_fraction = done_fraction
        self.physics = physics
        self.reward_weights = reward_weights
        self.geom = geometry or YBifurcation()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.geom.initial_positions(self.n_particles, self.np_random)
        self.status = np.full(self.n_particles, ACTIVE)
        self.time = 0.0
        self.steps = 0
        self.prev_action = np.zeros(2)
        self.dist = mean_distance(self.geom, self.pos)
        return self._obs(), self._info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        force = action / max(1.0, np.linalg.norm(action))

        prev_status, prev_dist = self.status, self.dist
        self.pos, self.status = dynamics.step(
            self.geom, self.condition, self.physics, self.pos, self.status,
            force, self.time, self.np_random,
        )
        self.time += self.physics.dt
        self.steps += 1
        self.dist = mean_distance(self.geom, self.pos)

        reward, terms = compute_reward(
            self.reward_weights, prev_dist, self.dist, prev_status, self.status, force
        )
        self.prev_action = force

        terminated = bool((self.status != ACTIVE).mean() >= self.done_fraction)
        truncated = self.steps >= self.horizon and not terminated
        info = self._info()
        info["reward_terms"] = terms
        return self._obs(), float(reward), terminated, truncated, info

    def _obs(self):
        return swarm_observation(
            self.geom, self.pos, self.status, self.time,
            self.physics.pulse_period, self.prev_action,
        )

    def _info(self):
        return {
            "eta": float((self.status == AT_TARGET).mean()),
            "eta_other": float((self.status == AT_OTHER).mean()),
            "frac_active": float((self.status == ACTIVE).mean()),
            "steps": self.steps,
        }
