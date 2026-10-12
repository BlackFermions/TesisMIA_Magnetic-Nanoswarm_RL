"""EDA del entorno: datos generados por las políticas de referencia.

Requiere haber corrido antes:  python scripts/evaluate.py --tag baseline
Produce:
  results/figures/fig1_eta_por_politica.png   η por política y condición
  results/figures/fig2_enjambre_r2_vs_r4.png  heurística perpendicular en R2 y R4
  results/figures/fig3_barrido_vmag.png       η frente a la intensidad del imán
  results/figures/fig4_convergencia_N.png     variabilidad de η frente a N
  results/tables/eda_*.csv                    rangos de la observación, alineación de la recompensa
  logs/metrics_eda.txt                        configuración, metadatos y hallazgos
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nanoswarm.baselines import make_policy
from nanoswarm.envs import VascularEnv
from nanoswarm.envs.dynamics import AT_OTHER, AT_TARGET, PhysicsParams
from nanoswarm.envs.observation import OBS_NAMES
from nanoswarm.utils.rollout import run_episode, run_metadata

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"

# Paleta categórica de referencia, en orden fijo por política
COLORS = {"zero": "#2a78d6", "random": "#eb6834", "heur_perp": "#1baf7a", "heur_branch": "#eda100"}
LABELS = {"zero": "Sin fuerza", "random": "Aleatoria",
          "heur_perp": "Heurística perpendicular", "heur_branch": "Heurística rama"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STATUS_COLORS = {0: "#9a9993", AT_TARGET: "#2a78d6", AT_OTHER: "#eb6834"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "figure.dpi": 130, "savefig.bbox": "tight",
})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline-tag", default="baseline")
    p.add_argument("--episodes", type=int, default=10, help="episodios por punto de los barridos")
    p.add_argument("--seed", type=int, default=1000)
    return p.parse_args()


def mean_eta(cond, policy_name, episodes, seed, **env_kwargs):
    env = VascularEnv(condition=cond, **env_kwargs)
    etas = [run_episode(env, make_policy(policy_name, env, seed=seed + i + 10_000), seed + i)["eta"]
            for i in range(episodes)]
    return np.array(etas)


def fig_eta_by_policy(df):
    conds = list(dict.fromkeys(df["condition"]))
    policies = list(dict.fromkeys(df["policy"]))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    width = 0.8 / len(policies)
    rng = np.random.default_rng(0)
    for j, pol in enumerate(policies):
        for i, cond in enumerate(conds):
            eta = df[(df.condition == cond) & (df.policy == pol)]["eta"].to_numpy()
            x = i + (j - (len(policies) - 1) / 2) * width
            ax.scatter(x + rng.uniform(-width / 4, width / 4, len(eta)), eta, s=10,
                       color=COLORS[pol], alpha=0.55, linewidths=0)
            ax.hlines(eta.mean(), x - width * 0.4, x + width * 0.4, color=COLORS[pol], linewidth=2.5,
                      label=LABELS[pol] if i == 0 else None)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
    ax.text(len(conds) - 0.5, 0.51, "reparto 50/50", color=MUTED, fontsize=8, ha="right", va="bottom")
    ax.set_xticks(range(len(conds)), conds)
    ax.set_ylabel("η (fracción a la rama objetivo)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="x", visible=False)
    ax.set_title("η por política y condición de fidelidad (puntos = episodios, raya = media)",
                 loc="left", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    fig.savefig(FIG / "fig1_eta_por_politica.png")
    plt.close(fig)


def draw_geometry(ax, geom):
    for k in range(3):
        o, t, n, L = geom.origins[k], geom.tangents[k], geom.normals[k], geom.lengths[k]
        for side in (-1, 1):
            a = o + side * geom.half_width * n
            b = a + L * t
            ax.plot([a[0], b[0]], [a[1], b[1]], color=MUTED, linewidth=1)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig_swarm_snapshots(seed):
    """La misma política en R2 y R4: en R2 el enjambre se atasca en la pared."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, cond in zip(axes, ["R2", "R4"]):
        env = VascularEnv(condition=cond)
        summary, rec = run_episode(env, make_policy("heur_perp", env), seed, record_every=50)
        pos, status = rec["frames"][min(7, len(rec["frames"]) - 1)]  # ~350 pasos
        draw_geometry(ax, env.geom)
        for st, color in STATUS_COLORS.items():
            m = status == st
            ax.scatter(pos[m, 0], pos[m, 1], s=8, color=color, linewidths=0)
        ax.text(9.2, 2.2, "objetivo", color=MUTED, fontsize=8)
        ax.set_title(f"{cond}: paso 350 · η final = {summary['eta']:.2f}", loc="left", fontsize=10)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=5) for c in STATUS_COLORS.values()]
    fig.legend(handles, ["activa", "entregada al objetivo", "entregada a la otra rama"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Heurística perpendicular: atascada en la pared sin browniano (R2), funciona con browniano (R4)",
                 x=0.02, ha="left", fontsize=10)
    fig.savefig(FIG / "fig2_enjambre_r2_vs_r4.png")
    plt.close(fig)


def sweep_vmag(episodes, seed):
    rows = []
    for cond in ["R2", "R4"]:
        for v in [0.01, 0.03, 0.1, 0.3, 1.0]:
            for pol in ["zero", "heur_perp", "heur_branch"]:
                eta = mean_eta(cond, pol, episodes, seed, physics=PhysicsParams(v_mag=v))
                rows.append({"condition": cond, "v_mag": v, "policy": pol,
                             "eta_mean": eta.mean(), "eta_std": eta.std()})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "eda_barrido_vmag.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, cond in zip(axes, ["R2", "R4"]):
        for pol in ["zero", "heur_perp", "heur_branch"]:
            d = df[(df.condition == cond) & (df.policy == pol)]
            ax.plot(d.v_mag, d.eta_mean, color=COLORS[pol], linewidth=2, marker="o", markersize=5,
                    label=LABELS[pol])
        ax.set_xscale("log")
        ax.set_xlabel("v_mag (velocidad magnética / velocidad del flujo)")
        ax.set_title(cond, loc="left", fontsize=10)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("η media")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.17))
    fig.suptitle("Intensidad del imán: en R2 el control funciona por umbral (v_mag ≥ 0.3); en R4, de forma gradual",
                 x=0.02, ha="left", fontsize=10)
    fig.savefig(FIG / "fig3_barrido_vmag.png")
    plt.close(fig)
    return df


def sweep_n(episodes, seed):
    rows = []
    for n in [25, 50, 100, 200, 400]:
        for pol in ["zero", "heur_branch"]:
            eta = mean_eta("R4", pol, episodes * 2, seed, n_particles=n)
            rows.append({"n_particles": n, "policy": pol, "eta_mean": eta.mean(), "eta_std": eta.std()})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "eda_convergencia_N.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for pol in ["zero", "heur_branch"]:
        d = df[df.policy == pol]
        ax.plot(d.n_particles, d.eta_std, color=COLORS[pol], linewidth=2, marker="o", markersize=5,
                label=LABELS[pol])
    ax.set_xscale("log")
    ax.set_xticks([25, 50, 100, 200, 400], ["25", "50", "100", "200", "400"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("N (partículas por enjambre)")
    ax.set_ylabel("desviación estándar de η entre episodios")
    ax.set_title("R4: variabilidad de η frente al tamaño del enjambre", loc="left", fontsize=10)
    ax.legend(frameon=False)
    fig.savefig(FIG / "fig4_convergencia_N.png")
    plt.close(fig)
    return df


def observation_ranges(seed):
    """Calidad de la observación: rangos, NaN/inf y valores constantes."""
    obs_all = []
    for cond in ["R1", "R2", "R3", "R4"]:
        for pol in ["zero", "random", "heur_perp", "heur_branch"]:
            env = VascularEnv(condition=cond)
            for i in range(3):
                _, rec = run_episode(env, make_policy(pol, env, seed=seed + i), seed + i, record_every=1000)
                obs_all.append(rec["obs"])
    obs = np.concatenate(obs_all)
    df = pd.DataFrame({
        "variable": OBS_NAMES,
        "min": obs.min(axis=0), "max": obs.max(axis=0),
        "media": obs.mean(axis=0), "desv": obs.std(axis=0),
        "no_finitos": (~np.isfinite(obs)).sum(axis=0),
    })
    df.to_csv(TAB / "eda_rangos_observacion.csv", index=False)
    return df, len(obs)


def reward_alignment(df):
    """¿Ordena el retorno a las políticas igual que η?"""
    rows = []
    for cond, g in df.groupby("condition", sort=False):
        # Spearman = Pearson sobre los rangos (evita depender de scipy)
        rho = g["return"].rank().corr(g["eta"].rank()) if g["eta"].std() > 0 else np.nan
        rows.append({"condition": cond, "spearman_retorno_eta": rho})
    corr = pd.DataFrame(rows)
    means = df.groupby(["condition", "policy"], sort=False)[
        ["eta", "return", "r_progress", "r_target", "r_other", "r_effort"]].mean().reset_index()
    corr.to_csv(TAB / "eda_alineacion_recompensa.csv", index=False)
    means.to_csv(TAB / "eda_componentes_recompensa.csv", index=False)
    return corr, means


def main():
    args = parse_args()
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(TAB / f"episodes_{args.baseline_tag}.csv")
    fig_eta_by_policy(base)
    fig_swarm_snapshots(args.seed)
    print("figuras 1-2 listas")
    vmag = sweep_vmag(args.episodes, args.seed)
    print("barrido v_mag listo")
    nconv = sweep_n(args.episodes, args.seed)
    print("barrido N listo")
    ranges, n_obs = observation_ranges(args.seed)
    corr, means = reward_alignment(base)

    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    log = ROOT / "logs" / "metrics_eda.txt"
    with open(log, "w", encoding="utf-8") as f:
        f.write("# EDA del entorno\n\n")
        f.write("## Metadatos\n" + json.dumps(run_metadata(), indent=2, ensure_ascii=False) + "\n\n")
        f.write("## Configuración\n" + json.dumps({
            "baseline_tag": args.baseline_tag, "episodes_por_punto": args.episodes, "seed_base": args.seed,
            "physics_por_defecto": dataclasses.asdict(PhysicsParams())}, indent=2, ensure_ascii=False) + "\n\n")
        f.write(f"## Calidad de la observación ({n_obs} observaciones, 4 condiciones x 4 políticas)\n")
        f.write(ranges.to_string(index=False, float_format=fmt) + "\n\n")
        f.write("## Barrido de v_mag (η media)\n")
        f.write(vmag.pivot_table(index=["condition", "v_mag"], columns="policy", values="eta_mean")
                .to_string(float_format=fmt) + "\n\n")
        f.write("## Convergencia en N (R4, desviación estándar de η)\n")
        f.write(nconv.pivot_table(index="n_particles", columns="policy", values="eta_std")
                .to_string(float_format=fmt) + "\n\n")
        f.write("## Alineación recompensa-métrica (Spearman entre retorno y η, por condición)\n")
        f.write(corr.to_string(index=False, float_format=fmt) + "\n\n")
        f.write("## Componentes medios de la recompensa\n")
        f.write(means.to_string(index=False, float_format=fmt) + "\n\n")
        f.write(f"Tiempo total: {time.time() - t0:.1f} s\n")
    print(f"Log: {log}")


if __name__ == "__main__":
    main()
