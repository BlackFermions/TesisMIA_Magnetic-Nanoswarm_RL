"""Importancia de variables por permutación para políticas congeladas.

En RL no hay un set de validación fijo: la permutación se hace en línea. En cada paso
del episodio, las columnas de un grupo de features se reemplazan por las de una
observación al azar del banco (observaciones reales de la propia política en su
nivel de entrenamiento). Se rompe la relación de la feature con el estado sin
sacarla de su rango normal.
"""

from concurrent.futures import ProcessPoolExecutor

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from nanoswarm.agents.ppo_sb3 import FrozenPolicy
from nanoswarm.envs.features import FeatureSet, feature_names
from nanoswarm.evaluation import make_env
from nanoswarm.variants import VARIANTS

# Nombre en el código del entorno → nombre en español para tablas y gráficos
NOMBRES = {
    "centroid_x": "centroide_x", "centroid_y": "centroide_y",
    "spread_x": "dispersion_x", "spread_y": "dispersion_y",
    "frac_target": "frac_entregada", "frac_other": "frac_perdida", "frac_active": "frac_activa",
    "phase_sin": "fase_sen", "phase_cos": "fase_cos",
    "prev_action_x": "accion_previa_x", "prev_action_y": "accion_previa_y",
    "control_margin": "margen_control",
}

# Las features que codifican una misma cantidad se permutan juntas
GRUPOS = {
    "centroide_x": ["centroid_x"],
    "centroide_y": ["centroid_y"],
    "dispersion_x": ["spread_x"],
    "dispersion_y": ["spread_y"],
    "fracciones": ["frac_target", "frac_other", "frac_active"],  # suman 1
    "fase": ["phase_sin", "phase_cos"],                           # un solo ángulo
    "accion_previa_x": ["prev_action_x"],
    "accion_previa_y": ["prev_action_y"],
    "margen_control": ["control_margin"],
}
SEMILLAS_BANCO = list(range(5_500, 5_510))  # disjuntas de las de evaluación (5000–5019) y de reporte


def grupos_de(feature_set):
    """Grupos presentes en un conjunto de features, con los índices de sus columnas."""
    nombres = feature_names(feature_set)
    return {g: [nombres.index(c) for c in cols] for g, cols in GRUPOS.items()
            if all(c in nombres for c in cols)}


def entorno(politica, condicion):
    meta = politica.meta
    return FeatureSet(make_env(VARIANTS[meta["config"]["env_variant"]], condicion), meta["feature_set"])


def recorrer(env, politica, semilla):
    """Un episodio: devuelve observaciones, acciones y η final."""
    obs, _ = env.reset(seed=semilla)
    observaciones, acciones = [], []
    done = False
    while not done:
        accion = politica(obs)
        observaciones.append(obs)
        acciones.append(accion)
        obs, _, term, trunc, info = env.step(accion)
        done = term or trunc
    return np.array(observaciones), np.array(acciones), info["eta"]


def banco(politica, condicion="R3", semillas=SEMILLAS_BANCO):
    """Observaciones y acciones de la política en su nivel, con el número de episodio de cada fila."""
    env = entorno(politica, condicion)
    obs, acc, ep = [], [], []
    for i, s in enumerate(semillas):
        o, a, _ = recorrer(env, politica, s)
        obs.append(o)
        acc.append(a)
        ep.append(np.full(len(o), i))
    return np.concatenate(obs), np.concatenate(acc), np.concatenate(ep)


class Permutar(gym.ObservationWrapper):
    """Reemplaza las columnas de un grupo por las de una fila al azar del banco."""

    def __init__(self, env, columnas, banco_obs):
        super().__init__(env)
        self.columnas = columnas
        self.banco_obs = banco_obs
        self.rng = np.random.default_rng(0)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed + 777)  # reemplazos reproducibles por episodio
        return super().reset(seed=seed, options=options)

    def observation(self, obs):
        obs = obs.copy()
        fila = self.banco_obs[self.rng.integers(len(self.banco_obs))]
        obs[self.columnas] = fila[self.columnas]
        return obs


def _tarea(tarea):
    """Una corrida en un nivel: η original y con cada grupo permutado, mismas semillas."""
    run, condicion, semillas = tarea
    torch.set_num_threads(1)
    politica = FrozenPolicy(run)
    meta = politica.meta
    banco_obs, _, _ = banco(politica, condicion=meta["condition"])
    filas = []
    for grupo, columnas in {"original": None, **grupos_de(meta["feature_set"])}.items():
        env = entorno(politica, condicion)
        if columnas is not None:
            env = Permutar(env, columnas, banco_obs)
        for s in semillas:
            filas.append({"feature_set": meta["feature_set"], "semilla_entren": meta["seed"],
                          "condicion": condicion, "grupo": grupo, "semilla_ep": s,
                          "eta": recorrer(env, politica, s)[2]})
    return filas


def importancia_permutacion(runs, condiciones, semillas, n_jobs=None):
    """η por episodio de cada corrida × nivel × grupo (incluye 'original', sin permutar)."""
    tareas = [(str(r), c, list(semillas)) for r in runs for c in condiciones]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return pd.DataFrame([f for filas in pool.map(_tarea, tareas) for f in filas])


def importancia(df):
    """Importancia = η original − η permutado, pareado por semilla de episodio."""
    orig = df[df["grupo"] == "original"].set_index(["feature_set", "semilla_entren", "condicion", "semilla_ep"])["eta"]
    perm = df[df["grupo"] != "original"].copy()
    claves = list(zip(perm["feature_set"], perm["semilla_entren"], perm["condicion"], perm["semilla_ep"]))
    perm["eta_original"] = orig.loc[claves].to_numpy()
    perm["importancia"] = perm["eta_original"] - perm["eta"]
    return (perm.groupby(["feature_set", "semilla_entren", "condicion", "grupo"], as_index=False)
            .agg(eta_original=("eta_original", "mean"), eta_permutado=("eta", "mean"),
                 importancia=("importancia", "mean")))
