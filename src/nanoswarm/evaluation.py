"""Evaluación de las políticas de referencia por variante, con semillas separadas.

Partición por semillas (no hay un dataset fijo en RL):
  - "ajuste":  se usan para decidir (pesos de la recompensa, parámetros físicos).
  - "reporte": nunca se usan para decidir; solo para los resultados finales.
"""

import dataclasses
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from nanoswarm.baselines import POLICIES, make_policy
from nanoswarm.envs import VascularEnv
from nanoswarm.envs.reward import RewardWeights
from nanoswarm.utils.rollout import run_episode, run_metadata

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ["R1", "R2", "R3", "R4"]
SPLITS = {"ajuste": 5_000, "reporte": 9_000}  # semilla base de cada partición
EPISODES = {"ajuste": 20, "reporte": 30}
UNIT_WEIGHTS = RewardWeights(progress=1.0, target=1.0, other=1.0, effort=1.0)


def episode_seeds(split, episodes=None):
    base = SPLITS[split]
    return list(range(base, base + (episodes or EPISODES[split])))


def make_env(variant, condition, reward=None):
    return VascularEnv(condition=condition, physics=variant.physics,
                       reward_weights=reward or variant.reward, horizon=variant.horizon)


def _run_cell(task):
    variant, reward, cond, policy_name, seeds = task
    env = make_env(variant, cond, reward)
    rows = []
    for seed in seeds:
        policy = make_policy(policy_name, env, seed=seed + 10_000)
        rows.append({"variant": variant.name, "condition": cond, "policy": policy_name,
                     **run_episode(env, policy, seed)})
    return rows


def evaluate(variant, split="reporte", episodes=None, policies=POLICIES, conditions=CONDITIONS,
             reward=None, n_jobs=None):
    """Corre cada política en cada condición con las mismas semillas (comparación pareada).

    `reward` reemplaza los pesos de la variante; con UNIT_WEIGHTS las columnas r_*
    quedan sin ponderar y se pueden reponderar después con `reweight`.
    """
    seeds = episode_seeds(split, episodes)
    reward = reward or variant.reward
    tasks = [(variant, reward, c, p, seeds) for c in conditions for p in policies]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        rows = [r for cell in pool.map(_run_cell, tasks) for r in cell]
    return pd.DataFrame(rows)


def reweight(df_unit, w):
    """Retorno con otros pesos, a partir de episodios corridos con UNIT_WEIGHTS."""
    return (w.progress * df_unit["r_progress"] + w.target * df_unit["r_target"]
            + w.other * df_unit["r_other"] + w.effort * df_unit["r_effort"])


def bootstrap_ci(x, n_boot=2000, seed=0):
    """IC 95 % de la media por bootstrap."""
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def summarize(df):
    """Métrica principal con IC 95 % y secundarias, por condición y política."""
    rows = []
    for (cond, pol), g in df.groupby(["condition", "policy"], sort=False):
        lo, hi = bootstrap_ci(g["eta"].to_numpy())
        rows.append({
            "condition": cond, "policy": pol,
            "eta": g["eta"].mean(), "eta_lo": lo, "eta_hi": hi,
            "perdida": g["eta_other"].mean(), "atascada": g["frac_active"].mean(),
            "transito": g["transit"].mean(), "esfuerzo": g["effort"].mean(),
            "retorno": g["return"].mean(),
        })
    return pd.DataFrame(rows)


def spearman(a, b):
    """Spearman = Pearson sobre los rangos (evita depender de scipy)."""
    return a.rank().corr(b.rank())


def reward_alignment(df, ret=None):
    """¿Ordena el retorno los episodios igual que η? Una correlación por condición."""
    ret = df["return"] if ret is None else ret
    return {cond: spearman(ret[g.index], g["eta"]) for cond, g in df.groupby("condition", sort=False)}


def write_log(tag, variant, split, df, elapsed, notes=""):
    """logs/metrics_<tag>.txt con metadatos, configuración y resultados."""
    summary = summarize(df)
    align = reward_alignment(df)
    config = {
        "variante": variant.name, "cambio": variant.change, "particion": split,
        "semillas": f"{df['seed'].min()}–{df['seed'].max()}", "episodios_por_celda": df["seed"].nunique(),
        "n_particulas": 200, "horizonte": variant.horizon,
        "fisica": dataclasses.asdict(variant.physics), "recompensa": dataclasses.asdict(variant.reward),
    }
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    path = ROOT / "logs" / f"metrics_{tag}.txt"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Métricas: {tag}\n\n")
        f.write("## Metadatos\n" + json.dumps(run_metadata(), indent=2, ensure_ascii=False) + "\n\n")
        f.write("## Configuración\n" + json.dumps(config, indent=2, ensure_ascii=False) + "\n\n")
        f.write("## Resultados (IC 95 % por bootstrap)\n")
        f.write("eta = fracción a la rama objetivo; perdida = a la otra rama; atascada = aún en el vaso; "
                "transito = tiempo medio de salida; esfuerzo = intensidad media de la fuerza de control (0–1)\n\n")
        f.write(summary.to_string(index=False, float_format=fmt) + "\n\n")
        f.write("## Alineación recompensa–η (Spearman por condición)\n")
        f.write(json.dumps({k: round(v, 3) for k, v in align.items()}, indent=2) + "\n\n")
        if notes:
            f.write(notes.rstrip() + "\n\n")
        f.write(f"Tiempo de cómputo: {elapsed:.1f} s\n")
    return path


def run_and_log(variant, split="reporte", tag=None, **kwargs):
    """Evalúa una variante, guarda el CSV por episodio y el log. Devuelve el DataFrame."""
    t0 = time.time()
    df = evaluate(variant, split, **kwargs)
    elapsed = time.time() - t0
    tag = tag or f"sprint2_{variant.name}"
    csv = ROOT / "results" / "tables" / f"episodes_{tag}.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    write_log(tag, variant, split, df, elapsed)
    df.attrs["elapsed"] = elapsed
    return df
