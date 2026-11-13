"""Agente PPO con Stable-Baselines3 (PyTorch).

La observación se estandariza con VecNormalize: μ y σ se actualizan solo durante
el entrenamiento y se congelan al evaluar (sin fuga de la condición de prueba).
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from nanoswarm.envs.features import FeatureSet
from nanoswarm.evaluation import make_env
from nanoswarm.variants import VARIANTS

TRAIN_SEED_BASE = 1_000  # semillas de entrenamiento: disjuntas de ajuste (5000) y reporte (9000)


@dataclass(frozen=True)
class PPOConfig:
    timesteps: int = 300_000
    learning_rate: float = 3e-4  # α: única y constante para todas las variantes
    n_envs: int = 4
    n_steps: int = 512           # por entorno → 2048 transiciones por actualización
    batch_size: int = 256
    gamma: float = 0.995
    # Valores por defecto de SB3 (los usados en los Sprints 3 y 4); el Sprint 5 los busca
    n_epochs: int = 10
    ent_coef: float = 0.0
    gae_lambda: float = 0.95
    env_variant: str = "var3"    # entorno resultante del Sprint 2


class CurveCallback(BaseCallback):
    """Curva de aprendizaje: η de cada episodio de entrenamiento frente a los pasos acumulados."""

    def __init__(self):
        super().__init__()
        self.rows = []

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.rows.append((self.num_timesteps, info["eta"]))
        return True


def make_vec(feature_set, condition, cfg):
    # Sin semilla aquí: PPO(seed=s) fija la del primer episodio de cada entorno en s + i (VecEnv.seed)
    variant = VARIANTS[cfg.env_variant]
    return DummyVecEnv([lambda: FeatureSet(make_env(variant, condition), feature_set) for _ in range(cfg.n_envs)])


def crear_modelo(feature_set, condition, semilla_ppo, cfg=PPOConfig()):
    """PPO sin entrenar con la semilla dada tal cual; los entornos empiezan con semilla_ppo + i."""
    venv = VecNormalize(make_vec(feature_set, condition, cfg), norm_obs=True, norm_reward=False, clip_obs=10.0)
    return PPO("MlpPolicy", venv, learning_rate=cfg.learning_rate, n_steps=cfg.n_steps,
               batch_size=cfg.batch_size, n_epochs=cfg.n_epochs, gamma=cfg.gamma,
               gae_lambda=cfg.gae_lambda, ent_coef=cfg.ent_coef, seed=semilla_ppo, verbose=0, device="cpu")


def semillas_entornos(model):
    """Semillas del primer episodio de cada entorno (antes de learn, que las consume)."""
    return list(model.get_env().venv._seeds)


def make_model(feature_set, condition, seed, cfg=PPOConfig()):
    """Como crear_modelo, con la semilla de los Sprints 3–4: PPO usa TRAIN_SEED_BASE + seed."""
    return crear_modelo(feature_set, condition, TRAIN_SEED_BASE + seed, cfg)


def train(feature_set, condition, seed, out_dir, cfg=PPOConfig()):
    """Entrena una política y guarda modelo, estadísticas de normalización y curva."""
    torch.set_num_threads(1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_seed = TRAIN_SEED_BASE + seed
    model = make_model(feature_set, condition, seed, cfg)
    venv = model.get_env()

    curve = CurveCallback()
    t0 = time.time()
    model.learn(total_timesteps=cfg.timesteps, callback=curve)
    elapsed = time.time() - t0

    model.save(out_dir / "model.zip")
    np.savez(out_dir / "obs_rms.npz", mean=venv.obs_rms.mean, var=venv.obs_rms.var)
    np.savetxt(out_dir / "curve.csv", np.array(curve.rows), delimiter=",", header="timestep,eta", comments="")
    meta = {"feature_set": feature_set, "condition": condition, "seed": seed, "train_seed": train_seed,
            "train_seconds": elapsed, "config": asdict(cfg)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class FrozenPolicy:
    """Política congelada: normaliza con las μ y σ del entrenamiento y actúa determinísticamente."""

    def __init__(self, run_dir):
        run_dir = Path(run_dir)
        self.model = PPO.load(run_dir / "model.zip", device="cpu")
        stats = np.load(run_dir / "obs_rms.npz")
        self.mean, self.std = stats["mean"], np.sqrt(stats["var"] + 1e-8)
        self.meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))

    def __call__(self, obs):
        z = np.clip((obs - self.mean) / self.std, -10.0, 10.0)
        action, _ = self.model.predict(z, deterministic=True)
        return action


def _eval_task(task):
    run, condition, seeds = task
    torch.set_num_threads(1)
    from nanoswarm.utils.rollout import run_episode  # import local: proceso hijo

    policy = FrozenPolicy(run)
    meta = policy.meta
    env = FeatureSet(make_env(VARIANTS[meta["config"]["env_variant"]], condition), meta["feature_set"])
    return [{"feature_set": meta["feature_set"], "train_condition": meta["condition"], "train_seed": meta["seed"],
             "condition": condition, **run_episode(env, policy, s)} for s in seeds]


def evaluate_runs(run_dirs, conditions, seeds, n_jobs=None):
    """Evalúa políticas congeladas en cada condición con las mismas semillas (comparación pareada)."""
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd

    tasks = [(str(r), c, list(seeds)) for r in run_dirs for c in conditions]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return pd.DataFrame([row for rows in pool.map(_eval_task, tasks) for row in rows])
