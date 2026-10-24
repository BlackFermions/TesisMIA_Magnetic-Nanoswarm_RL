"""Entrenamiento de PPO por conjunto de features, condición y semilla (en paralelo).

Cada corrida se guarda en experiments/pilot/<feature_set>_<condición>_s<semilla>/
con model.zip, obs_rms.npz (normalización de entrenamiento), curve.csv y meta.json.

Ejemplo:
  python scripts/train.py --feature-sets base margen sin_fase --condition R3 --seeds 0 1 2
"""

import argparse
import dataclasses
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from nanoswarm.agents.ppo_sb3 import PPOConfig, train

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "pilot"


def run_dir(feature_set, condition, seed):
    return OUT / f"{feature_set}_{condition}_s{seed}"


def _job(args):
    fs, cond, seed, timesteps = args
    cfg = dataclasses.replace(PPOConfig(), timesteps=timesteps)
    return train(fs, cond, seed, run_dir(fs, cond, seed), cfg)


def train_all(feature_sets, condition, seeds, timesteps, n_jobs=None):
    jobs = [(fs, condition, s, timesteps) for fs in feature_sets for s in seeds]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(_job, jobs))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature-sets", nargs="+", default=["base", "margen", "sin_fase"])
    p.add_argument("--condition", default="R3")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--timesteps", type=int, default=PPOConfig().timesteps)
    a = p.parse_args()
    t0 = time.time()
    for m in train_all(a.feature_sets, a.condition, a.seeds, a.timesteps):
        print(f"{m['feature_set']:>9} {m['condition']} s{m['seed']}: {m['train_seconds']:.0f} s")
    print(f"Total: {time.time() - t0:.0f} s")
