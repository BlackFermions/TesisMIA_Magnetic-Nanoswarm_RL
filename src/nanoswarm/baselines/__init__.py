"""Políticas de referencia."""

from nanoswarm.baselines.geometric import GeometricHeuristic
from nanoswarm.baselines.random_policy import RandomPolicy, ZeroPolicy


def make_policy(name, env, seed=None):
    """Construye una política de referencia por nombre."""
    if name == "zero":
        return ZeroPolicy()
    if name == "random":
        return RandomPolicy(seed)
    if name == "heur_perp":
        return GeometricHeuristic(env.geom, mode="perpendicular")
    if name == "heur_branch":
        return GeometricHeuristic(env.geom, mode="branch")
    raise ValueError(f"política desconocida: {name}")


POLICIES = ["zero", "random", "heur_perp", "heur_branch"]

__all__ = ["GeometricHeuristic", "RandomPolicy", "ZeroPolicy", "make_policy", "POLICIES"]
