"""Política aleatoria y política sin fuerza."""

import numpy as np


class RandomPolicy:
    """Acción uniforme en [-1, 1]², con su propio generador para ser reproducible."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs):
        return self.rng.uniform(-1.0, 1.0, size=2)


class ZeroPolicy:
    """Sin imán: el enjambre se reparte solo por el flujo."""

    def __call__(self, obs):
        return np.zeros(2)
