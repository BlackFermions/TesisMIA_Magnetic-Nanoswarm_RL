"""Búsqueda de hiperparámetros de PPO (Sprint 5).

Cada búsqueda se guarda en experiments/hpo/<estudio>/: journal.log (estudio de
Optuna, permite retomar), trial_XXX/ (curva y configuración de cada trial) y
trials.csv (una fila por trial). Configuración: configs/train/search.yaml.

Ejemplos:
  python scripts/hpo.py --sampler random                # búsqueda aleatoria completa
  python scripts/hpo.py --sampler tpe                   # búsqueda bayesiana completa
  python scripts/hpo.py --control                       # configuración actual, medida igual que un trial
  python scripts/hpo.py --sampler random --estudio prueba --trials 2 --pasos 4000 --cada 2000 --episodios 2
"""

import argparse
import time

from nanoswarm.hpo import buscar, correr_control

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sampler", choices=["random", "tpe"])
    p.add_argument("--control", action="store_true",
                   help="mide la configuración actual como un trial más (experiments/hpo/control/)")
    p.add_argument("--estudio", default=None, help="por defecto: el nombre del sampler")
    p.add_argument("--trials", type=int, default=None, help="por defecto: el del yaml")
    p.add_argument("--pasos", type=int, default=None, help="pasos por nivel; por defecto: el del yaml")
    p.add_argument("--cada", type=int, default=None, help="pasos entre evaluaciones; por defecto: el del yaml")
    p.add_argument("--episodios", type=int, default=None, help="episodios por evaluación; por defecto: el del yaml")
    p.add_argument("--procesos", type=int, default=None, help="trials en paralelo; por defecto: el del yaml")
    a = p.parse_args()
    t0 = time.time()
    if a.control:
        print(correr_control())
        raise SystemExit
    if a.sampler is None:
        p.error("falta --sampler (o --control)")
    tabla = buscar(a.estudio or a.sampler, a.sampler, a.trials, a.pasos, a.cada, a.episodios, a.procesos)
    print(tabla[["trial_id", "estado", "evaluaciones", "valor", "segundos"]].to_string(index=False))
    print(f"Total: {(time.time() - t0) / 60:.1f} min")
