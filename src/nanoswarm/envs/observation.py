"""Estadísticas del enjambre que forman la observación o_t."""

import numpy as np

from nanoswarm.envs.dynamics import ACTIVE, AT_OTHER, AT_TARGET

OBS_NAMES = [
    "centroid_x", "centroid_y",  # relativo a la bifurcación
    "spread_x", "spread_y",  # desviación estándar
    "frac_target", "frac_other", "frac_active",
    "phase_sin", "phase_cos",
    "prev_action_x", "prev_action_y",
]
OBS_DIM = len(OBS_NAMES)


def swarm_observation(geom, pos, status, time, pulse_period, prev_action):
    active = status == ACTIVE
    if active.any():
        centroid = pos[active].mean(axis=0) - geom.junction
        spread = pos[active].std(axis=0)
    else:
        centroid = np.zeros(2)
        spread = np.zeros(2)
    phase = 2 * np.pi * time / pulse_period
    return np.concatenate([
        centroid,
        spread,
        [(status == AT_TARGET).mean(), (status == AT_OTHER).mean(), active.mean()],
        [np.sin(phase), np.cos(phase)],
        prev_action,
    ]).astype(np.float32)
