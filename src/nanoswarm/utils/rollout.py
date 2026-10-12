"""Ejecución de episodios y metadatos de cada corrida."""

import datetime
import platform
import subprocess
from importlib.metadata import version

import numpy as np


def run_episode(env, policy, seed, record_every=0):
    """Corre un episodio completo y devuelve su resumen.

    Con record_every > 0 también devuelve las posiciones y estados del enjambre
    cada `record_every` pasos, y todas las observaciones y acciones.
    """
    obs, info = env.reset(seed=seed)
    terms_sum = {}
    episode_return = 0.0
    frames, observations, actions = [], [obs], []
    done = False
    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        for k, v in info["reward_terms"].items():
            terms_sum[k] = terms_sum.get(k, 0.0) + v
        if record_every:
            observations.append(obs)
            actions.append(np.asarray(action, dtype=float))
            if info["steps"] % record_every == 0:
                frames.append((env.pos.copy(), env.status.copy()))
        done = terminated or truncated

    summary = {
        "seed": seed,
        "eta": info["eta"],
        "eta_other": info["eta_other"],
        "frac_active": info["frac_active"],
        "steps": info["steps"],
        "terminated": terminated,
        "return": episode_return,
        **{f"r_{k}": v for k, v in terms_sum.items()},
    }
    if record_every:
        return summary, {"frames": frames, "obs": np.array(observations), "actions": np.array(actions)}
    return summary


def run_metadata():
    """Fecha, commit de git, plataforma y versiones, para los logs."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        commit += " (con cambios sin commit)" if dirty else ""
    except (OSError, subprocess.CalledProcessError):
        commit = "desconocido"
    return {
        "fecha": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        **{pkg: version(pkg) for pkg in ("numpy", "gymnasium", "pandas")},
    }
