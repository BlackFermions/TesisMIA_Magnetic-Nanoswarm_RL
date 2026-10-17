"""Variantes del entorno del Sprint 2: un cambio por variante, acumulativas.

Cada variante parte de la anterior y cambia una sola cosa. Así, la diferencia
entre dos variantes consecutivas se atribuye a ese cambio (ablación gradual).
Justificación y resultados: notebooks/sprint2_arreglos_simulador.ipynb.
"""

import dataclasses
from dataclasses import dataclass

from nanoswarm.envs.dynamics import PhysicsParams
from nanoswarm.envs.reward import RewardWeights


@dataclass(frozen=True)
class Variant:
    name: str
    change: str  # el único cambio respecto de la variante anterior
    physics: PhysicsParams
    reward: RewardWeights
    horizon: int = 800  # pasos máximos por episodio


# Var1: w_p y w_m elegidos con las semillas de ajuste (notebook, sección 5);
# w_u se reescala por regla: la fuerza máxima durante todo el horizonte
# (800 pasos) cuesta 0.08, frente a 1.0 por entregar todo el enjambre.
REWARD_V1 = RewardWeights(progress=0.1, target=1.0, other=0.5, effort=0.0001)

_baseline = Variant("baseline", "Entorno v0.1 (Sprint 1)", PhysicsParams(), RewardWeights())
_var1 = Variant("var1", "Recompensa corregida", _baseline.physics, REWARD_V1)
_var2 = Variant("var2", "+ radio de partícula a = 0.05 W",
                dataclasses.replace(_var1.physics, particle_radius=0.05), REWARD_V1)
_var3 = Variant("var3", "+ pulso fuerte y lento (A = 0.9, T = 20)",
                dataclasses.replace(_var2.physics, pulse_amplitude=0.9, pulse_period=20.0), REWARD_V1)
VARIANTS = {v.name: v for v in (_baseline, _var1, _var2, _var3)}
