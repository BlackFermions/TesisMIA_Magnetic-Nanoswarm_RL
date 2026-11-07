"""Búsqueda de hiperparámetros de PPO con Optuna (Sprint 5).

Un trial es una configuración de hiperparámetros: entrena una política por nivel
(R1–R4) en tramos de `evaluar_cada` pasos y, al final de cada tramo, mide η de
cada política en su propio nivel con semillas de ajuste. El valor del trial es la
eficiencia muestral (η medio de la curva), promediada en los niveles.

Los niveles se entrenan por tramos alternados para poder reportar a Optuna el
promedio de los cuatro en cada evaluación y podar el trial a mitad de camino.
Configuración: configs/train/search.yaml.
"""

import dataclasses
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import yaml

from nanoswarm.agents.ppo_sb3 import PPOConfig, make_model
from nanoswarm.envs.features import FeatureSet
from nanoswarm.evaluation import ROOT, episode_seeds, make_env
from nanoswarm.utils.rollout import run_episode
from nanoswarm.variants import VARIANTS

CONFIG = ROOT / "configs" / "train" / "search.yaml"
SALIDA = ROOT / "experiments" / "hpo"
optuna.logging.set_verbosity(optuna.logging.WARNING)


def cargar_config(path=CONFIG):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sugerir(trial, espacio):
    """Pide a Optuna un valor por hiperparámetro, según el espacio del yaml."""
    hp = {}
    for nombre, p in espacio.items():
        if p["tipo"] == "categorico":
            hp[nombre] = trial.suggest_categorical(nombre, p["valores"])
        else:
            hp[nombre] = trial.suggest_float(nombre, p["min"], p["max"], log=p["log"])
    return hp


def ppo_config(hp, cfg):
    fijo = cfg["fijo"]
    return dataclasses.replace(PPOConfig(), env_variant=fijo["entorno"], n_envs=fijo["n_envs"],
                               gae_lambda=fijo["gae_lambda"], **hp)


def evaluar(model, condition, feature_set, env_variant, seeds):
    """η medio de la política determinista en `condition`, con la normalización de entrenamiento congelada."""
    rms = model.get_env().obs_rms
    mean, std = rms.mean.copy(), np.sqrt(rms.var + 1e-8)

    def politica(obs):
        return model.predict(np.clip((obs - mean) / std, -10.0, 10.0), deterministic=True)[0]

    env = FeatureSet(make_env(VARIANTS[env_variant], condition), feature_set)
    return float(np.mean([run_episode(env, politica, s)["eta"] for s in seeds]))


def correr_trial(trial, hp, cfg, pasos_por_nivel, evaluar_cada, episodios, out_dir=None):
    """Entrena y evalúa una configuración en todos los niveles. Devuelve la curva por nivel.

    Con `trial` reporta la eficiencia muestral acumulada y lanza TrialPruned si Optuna lo poda.
    """
    torch.set_num_threads(1)
    fijo, niveles = cfg["fijo"], cfg["objetivo"]["niveles"]
    ppo = ppo_config(hp, cfg)
    seeds = episode_seeds("ajuste", episodios)
    modelos = {c: make_model(fijo["observacion"], c, fijo["semilla_entrenamiento"], ppo) for c in niveles}
    filas = []
    t0 = time.time()
    try:
        for k in range(1, pasos_por_nivel // evaluar_cada + 1):
            for c, m in modelos.items():
                m.learn(total_timesteps=evaluar_cada, reset_num_timesteps=False)
                # PPO completa rollouts enteros: el paso real puede pasar un poco de k × evaluar_cada
                filas.append({"nivel": c, "evaluacion": k, "paso": m.num_timesteps,
                              "eta": evaluar(m, c, fijo["observacion"], fijo["entorno"], seeds)})
            if trial is not None:
                parcial = pd.DataFrame(filas).groupby("nivel")["eta"].mean().mean()
                trial.report(parcial, k)
                if trial.should_prune():
                    raise optuna.TrialPruned()
    finally:
        curva = pd.DataFrame(filas)
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            curva.to_csv(out_dir / "curve.csv", index=False)
            meta = {"hiperparametros": hp, "ppo": dataclasses.asdict(ppo), "fijo": fijo,
                    "semillas_evaluacion": f"{seeds[0]}–{seeds[-1]}", "segundos": time.time() - t0}
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return curva


def resumir(curva):
    """Eficiencia muestral y η final por nivel, y sus promedios."""
    por_nivel = curva.groupby("nivel").agg(eficiencia=("eta", "mean"), eta_final=("eta", "last"))
    return {"eficiencia": por_nivel["eficiencia"].mean(), "eficiencia_std": por_nivel["eficiencia"].std(),
            "eta_final": por_nivel["eta_final"].mean(),
            **{f"eficiencia_{c}": v for c, v in por_nivel["eficiencia"].items()},
            **{f"eta_final_{c}": v for c, v in por_nivel["eta_final"].items()}}


def objetivo(cfg, estudio, pasos_por_nivel, evaluar_cada, episodios):
    def f(trial):
        hp = sugerir(trial, cfg["espacio"])
        t0 = time.time()
        curva = correr_trial(trial, hp, cfg, pasos_por_nivel, evaluar_cada, episodios,
                             SALIDA / estudio / f"trial_{trial.number:03d}")
        res = resumir(curva)
        for k, v in res.items():
            trial.set_user_attr(k, float(v))
        trial.set_user_attr("segundos", time.time() - t0)
        return res["eficiencia"]
    return f


def crear_sampler(nombre, cfg, trabajador=0):
    b = cfg["busqueda"]
    # Cada proceso usa su propia semilla; con la misma, los procesos propondrían las mismas configuraciones
    if nombre == "random":
        return optuna.samplers.RandomSampler(seed=b["random"]["semilla"] + trabajador)
    return optuna.samplers.TPESampler(seed=b["tpe"]["semilla"] + trabajador,
                                      n_startup_trials=b["tpe"]["trials_aleatorios_iniciales"])


def abrir_estudio(estudio, sampler, cfg, trabajador=0):
    """Estudio guardado en un journal en disco: lo comparten todos los procesos y se puede retomar."""
    (SALIDA / estudio).mkdir(parents=True, exist_ok=True)
    path = str(SALIDA / estudio / "journal.log")
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(
        path, lock_obj=optuna.storages.journal.JournalFileOpenLock(path)))
    poda = cfg["busqueda"]["poda"]
    pruner = optuna.pruners.MedianPruner(n_startup_trials=poda["trials_sin_poda"],
                                         n_warmup_steps=poda["evaluaciones_sin_poda"])
    return optuna.create_study(storage=storage, study_name=estudio, direction="maximize",
                               sampler=crear_sampler(sampler, cfg, trabajador), pruner=pruner, load_if_exists=True)


def _trabajador(args):
    estudio, sampler, n_trials, pasos, cada, episodios, trabajador = args
    cfg = cargar_config()
    study = abrir_estudio(estudio, sampler, cfg, trabajador)
    # Se detiene cuando el estudio llega a n_trials (contando los que están corriendo)
    study.optimize(objetivo(cfg, estudio, pasos, cada, episodios),
                   callbacks=[optuna.study.MaxTrialsCallback(n_trials, states=None)])


def buscar(estudio, sampler, n_trials=None, pasos_por_nivel=None, evaluar_cada=None, episodios=None,
           n_procesos=None):
    """Corre una búsqueda completa con varios procesos en paralelo y devuelve la tabla de trials."""
    cfg = cargar_config()
    b, o = cfg["busqueda"], cfg["objetivo"]
    n_trials = n_trials or b["trials_por_busqueda"]
    n_procesos = min(n_procesos or b["trials_en_paralelo"], n_trials)
    args = [(estudio, sampler, n_trials, pasos_por_nivel or o["pasos_por_nivel"], evaluar_cada or o["evaluar_cada"],
             episodios or o["episodios_evaluacion"], w) for w in range(n_procesos)]
    with ProcessPoolExecutor(max_workers=n_procesos) as pool:
        list(pool.map(_trabajador, args))
    tabla = tabla_trials(estudio, sampler)
    tabla.to_csv(SALIDA / estudio / "trials.csv", index=False)
    return tabla


def correr_control():
    """La configuración actual (Sprints 3 y 4), medida igual que un trial, sin Optuna."""
    cfg = cargar_config()
    o = cfg["objetivo"]
    t0 = time.time()
    curva = correr_trial(None, cfg["control"], cfg, o["pasos_por_nivel"], o["evaluar_cada"],
                         o["episodios_evaluacion"], SALIDA / "control")
    res = {k: float(v) for k, v in resumir(curva).items()}
    res["segundos"] = time.time() - t0
    (SALIDA / "control" / "resumen.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def tabla_trials(estudio, sampler):
    """Una fila por trial: hiperparámetros, métricas, estado, evaluaciones hechas y tiempo."""
    cfg = cargar_config()
    study = abrir_estudio(estudio, sampler, cfg)
    filas = []
    for t in study.trials:
        curva_path = SALIDA / estudio / f"trial_{t.number:03d}" / "curve.csv"
        curva = pd.read_csv(curva_path) if curva_path.exists() else pd.DataFrame({"evaluacion": [0], "paso": [0]})
        filas.append({"busqueda": sampler, "trial_id": t.number, "estado": t.state.name,
                      "semilla": cfg["fijo"]["semilla_entrenamiento"],
                      "evaluaciones": curva["evaluacion"].max(), "pasos_por_nivel": curva["paso"].max(),
                      **t.params, "valor": t.value, **t.user_attrs,
                      "segundos": t.user_attrs.get("segundos",
                                                   (t.datetime_complete - t.datetime_start).total_seconds()
                                                   if t.datetime_complete else np.nan)})
    return pd.DataFrame(filas)
