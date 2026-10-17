"""Evaluación de las políticas de referencia en una variante del entorno.

Corre cada política en cada condición con las mismas semillas (comparación
pareada) y escribe:
  - results/tables/episodes_<tag>.csv  una fila por episodio
  - logs/metrics_<tag>.txt             configuración, metadatos y resumen

Ejemplos:
  python scripts/evaluate.py --variant var4                 # semillas de reporte
  python scripts/evaluate.py --variant baseline --split ajuste
  python scripts/evaluate.py --all                          # las 5 variantes del Sprint 2
"""

import argparse

from nanoswarm.baselines import POLICIES
from nanoswarm.evaluation import CONDITIONS, SPLITS, run_and_log, summarize
from nanoswarm.variants import VARIANTS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="baseline", choices=list(VARIANTS))
    p.add_argument("--all", action="store_true", help="evalúa todas las variantes")
    p.add_argument("--split", default="reporte", choices=list(SPLITS))
    p.add_argument("--episodes", type=int, default=None, help="por defecto: 20 en ajuste, 30 en reporte")
    p.add_argument("--policies", nargs="+", default=POLICIES)
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    p.add_argument("--tag", default=None, help="por defecto: sprint2_<variante>")
    return p.parse_args()


def main():
    args = parse_args()
    names = list(VARIANTS) if args.all else [args.variant]
    for name in names:
        df = run_and_log(VARIANTS[name], args.split, tag=None if args.all else args.tag,
                         episodes=args.episodes, policies=args.policies, conditions=args.conditions)
        print(f"\n== {name}: {VARIANTS[name].change} ({df.attrs['elapsed']:.0f} s)")
        print(summarize(df)[["condition", "policy", "eta", "eta_lo", "eta_hi"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
