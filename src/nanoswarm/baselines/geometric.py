"""Línea base geométrica: fuerza máxima perpendicular a la línea central.

Dos variantes, ambas sin aprendizaje y usando solo la observación y la geometría:

- "perpendicular" (la del README): cuando el centroide está a menos de
  `lookahead` de la bifurcación, empuja con fuerza máxima perpendicular al
  canal madre, hacia el lado de la rama objetivo; fuera de esa zona no actúa.
- "branch": empuja siempre en la dirección de la rama objetivo, es decir, hacia
  adelante y hacia su lado a la vez.
"""

import numpy as np

from nanoswarm.envs.dynamics import PARENT, TARGET


class GeometricHeuristic:
    def __init__(self, geom, mode="branch", lookahead=2.0):
        if mode not in ("perpendicular", "branch"):
            raise ValueError(f"modo desconocido: {mode}")
        self.mode = mode
        self.lookahead = lookahead
        self.side = geom.normals[PARENT] * np.sign(geom.tangents[TARGET] @ geom.normals[PARENT])
        self.branch_dir = geom.tangents[TARGET]

    def __call__(self, obs):
        if self.mode == "branch":
            return self.branch_dir.copy()
        centroid_x = obs[0]  # relativo a la bifurcación: negativo antes de llegar
        if -self.lookahead <= centroid_x <= 0.0:
            return self.side.copy()
        return np.zeros(2)
