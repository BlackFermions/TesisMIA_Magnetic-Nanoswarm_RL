"""Conjuntos de features (variantes de los datos que observa el agente).

El entorno siempre calcula las 11 features base (observation.py). Este envoltorio
quita o agrega columnas sin tocar la física ni la recompensa, para comparar
conjuntos de features con el mismo agente.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from nanoswarm.envs import dynamics
from nanoswarm.envs.observation import OBS_NAMES

PHASE = ["phase_sin", "phase_cos"]


def control_margin(env):
    """Margen de control F_max / (F_max + U(t)) ∈ (0, 1].

    1 sin flujo (la fuerza manda); cerca de 0 con flujo rápido (el flujo manda).
    Es la versión acotada de la razón fuerza/flujo F_max / U(t), que no está
    definida en R1 (U = 0).
    """
    u = dynamics.flow_speed(env.condition, env.physics, env.time)
    f = env.physics.v_mag
    return f / (f + u)


FEATURE_SETS = {
    "base": {"drop": [], "add": []},                 # las 11 features del entorno
    "margen": {"drop": [], "add": ["control_margin"]},  # + feature derivada de dominio
    "sin_fase": {"drop": PHASE, "add": []},          # − codificación cíclica de la fase
}
EXTRA = {"control_margin": control_margin}


def feature_names(name):
    fs = FEATURE_SETS[name]
    return [n for n in OBS_NAMES if n not in fs["drop"]] + fs["add"]


class FeatureSet(gym.ObservationWrapper):
    def __init__(self, env, name="base"):
        super().__init__(env)
        fs = FEATURE_SETS[name]
        self.keep = np.array([n not in fs["drop"] for n in OBS_NAMES])
        self.add = fs["add"]
        self.names = feature_names(name)
        dim = len(self.names)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)

    def observation(self, obs):
        extra = [EXTRA[k](self.env.unwrapped) for k in self.add]
        return np.concatenate([obs[self.keep], np.asarray(extra, dtype=np.float32)]).astype(np.float32)
