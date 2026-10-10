"""Evaluación cruzada de políticas congeladas.

Corre cada política en cada condición con las mismas semillas (comparación
pareada) y escribe:
  - results/tables/episodes_<tag>.csv  una fila por episodio
  - logs/metrics_<tag>.txt             configuración, metadatos y resumen

Ejemplo:
  python scripts/evaluate.py --tag baseline --episodes 30
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nanoswarm.baselines import POLICIES, make_policy
from nanoswarm.envs import VascularEnv
from nanoswarm.envs.dynamics import PhysicsParams
from nanoswarm.utils.rollout import run_episode, run_metadata

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="baseline")
    p.add_argument("--policies", nargs="+", default=POLICIES)
    p.add_argument("--conditions", nargs="+", default=["R1", "R2", "R3", "R4"])
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=0, help="semilla base; el episodio i usa seed + i")
    p.add_argument("--n-particles", type=int, default=200)
    p.add_argument("--horizon", type=int, default=800)
    p.add_argument("--v-mag", type=float, default=PhysicsParams.v_mag)
    return p.parse_args()


def bootstrap_ci(x, n_boot=2000, seed=0):
    """IC 95 % de la media por bootstrap."""
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    args = parse_args()
    physics = PhysicsParams(v_mag=args.v_mag)
    t0 = time.time()

    rows = []
    for cond in args.conditions:
        env = VascularEnv(condition=cond, n_particles=args.n_particles, horizon=args.horizon, physics=physics)
        for name in args.policies:
            for i in range(args.episodes):
                seed = args.seed + i
                policy = make_policy(name, env, seed=seed + 10_000)
                rows.append({"condition": cond, "policy": name, **run_episode(env, policy, seed)})
            print(f"{cond} {name:12s} listo")
    df = pd.DataFrame(rows)

    tables = ROOT / "results" / "tables"
    logs = ROOT / "logs"
    tables.mkdir(parents=True, exist_ok=True)
    logs.mkdir(exist_ok=True)
    csv_path = tables / f"episodes_{args.tag}.csv"
    df.to_csv(csv_path, index=False)

    summary = []
    for (cond, name), g in df.groupby(["condition", "policy"], sort=False):
        lo, hi = bootstrap_ci(g["eta"].to_numpy())
        summary.append({
            "condition": cond, "policy": name,
            "eta_mean": g["eta"].mean(), "eta_ci_lo": lo, "eta_ci_hi": hi,
            "eta_other": g["eta_other"].mean(), "stuck": g["frac_active"].mean(),
            "steps": g["steps"].mean(), "return": g["return"].mean(),
        })
    summary = pd.DataFrame(summary)

    config = {
        "tag": args.tag, "policies": args.policies, "conditions": args.conditions,
        "episodes": args.episodes, "seed_base": args.seed, "n_particles": args.n_particles,
        "horizon": args.horizon, "physics": dataclasses.asdict(physics),
        "reward_weights": dataclasses.asdict(env.reward_weights),
        "geometry": {"width": env.geom.width, "parent_length": env.geom.parent_length,
                     "branch_length": env.geom.branch_length,
                     "theta_deg": float(np.rad2deg(env.geom.theta))},
    }
    log_path = logs / f"metrics_{args.tag}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Métricas: {args.tag}\n\n")
        f.write("## Metadatos\n" + json.dumps(run_metadata(), indent=2, ensure_ascii=False) + "\n\n")
        f.write("## Configuración\n" + json.dumps(config, indent=2, ensure_ascii=False) + "\n\n")
        f.write(f"## Resultados ({args.episodes} episodios por celda, IC 95 % por bootstrap)\n")
        f.write("eta = fracción entregada a la rama objetivo (métrica central); "
                "stuck = fracción aún activa al final\n\n")
        f.write(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}") + "\n\n")
        f.write(f"Datos por episodio: {csv_path.relative_to(ROOT).as_posix()}\n")
        f.write(f"Tiempo total: {time.time() - t0:.1f} s\n")

    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nLog: {log_path}\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
