"""Recompensa basada en la distancia vascular sobre la línea central.

r_t = w_p Δd + w_e Δη_obj - w_m Δη_otras - w_u ||a_t||²

Δd es la reducción de la distancia vascular media del enjambre hacia la salida
objetivo, normalizada por el largo total del recorrido (madre + rama).
"""

from dataclasses import dataclass

import numpy as np

from nanoswarm.envs.dynamics import AT_OTHER, AT_TARGET


@dataclass(frozen=True)
class RewardWeights:
    progress: float = 1.0  # w_p
    target: float = 1.0  # w_e
    other: float = 1.0  # w_m
    effort: float = 0.0005  # w_u


def mean_distance(geom, pos):
    """Distancia vascular media de todas las partículas, normalizada."""
    seg, t, _ = geom.project(pos)
    path_length = geom.parent_length + geom.branch_length
    return geom.distance_to_target(seg, t).mean() / path_length


def compute_reward(weights, prev_dist, dist, prev_status, status, action):
    d_target = (status == AT_TARGET).mean() - (prev_status == AT_TARGET).mean()
    d_other = (status == AT_OTHER).mean() - (prev_status == AT_OTHER).mean()
    terms = {
        "progress": weights.progress * (prev_dist - dist),
        "target": weights.target * d_target,
        "other": -weights.other * d_other,
        "effort": -weights.effort * float(np.dot(action, action)),
    }
    return sum(terms.values()), terms
