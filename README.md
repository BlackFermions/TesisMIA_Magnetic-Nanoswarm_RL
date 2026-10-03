# TesisMIA_Magnetic-Nanoswarm_RL
# Evaluación del impacto de la fidelidad física de entornos vasculares simulados en el direccionamiento autónomo de enjambres de nanopartículas magnéticas mediante aprendizaje por refuerzo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29-green.svg)](https://gymnasium.farama.org/)

Este repositorio contiene la implementación reproducible y el marco experimental del proyecto de tesis para la **Maestría en Ciencias con mención en Inteligencia Artificial** de la **Universidad Nacional de Ingeniería (UNI)**.

---

 Resumen del Proyecto

El proyecto evalúa la navegación autónoma y direccionamiento de un enjambre diluido de $N$ nanopartículas magnéticas no interactuantes en geometrías vasculares tridimensionales con bifurcaciones. Se analiza cómo afecta la fidelidad física del entorno simulado (*in silico*) a la capacidad de generalización y transferencia del agente de Aprendizaje por Refuerzo (RL).

### Aportes Metodológicos Principales
1. Entorno de fidelidad parametrizada:** Construcción de un entorno en Gymnasium con 4 condiciones anidadas:
   - R1: Sin flujo hemodinámico.
   - R2: Flujo estacionario.
   - R3: Flujo pulsátil.
   - R4: Flujo pulsátil + movimiento browniano.
2. **Evaluación cruzada y currículo:** Protocolo $R_k \times R_{test}$ comparando entrenamiento en condición fija vs. currículo de fidelidad ($R1 \rightarrow R4$).
3. omparativa de Algoritmos: Evaluación entre enfoques *Model-Free* (PPO) y *Model-Based* (DreamerV3).
4. Análisis Estadístico Avanzado: Reporte mediante Media Intercuartílica (IQM), intervalos por *bootstrap* estratificado y modelos binomiales de efectos mixtos.

---

 Flujo del Sistema

```text
+-------------------------------------------------------------------------+
| A. Construcción del Entorno Físico (Offline)                           |
| Geometrías VMR (SimVascular) -> svSolver (CFD) -> Campos u(x,t) y Malla |
+-------------------------------------------------------------------------+
                                    |
                                    v (Interpolación trilineal)
+-------------------------------------------------------------------------+
| B. Ciclo Agente-Entorno (Entrenamiento)                                 |
| Agente (PPO/DreamerV3) -> Acción F_t -> Dinámica Euler-Maruyama (N part)|
| -> Estado u(x,t) + Browniano -> Observación o_{t+1} & Recompensa r_t   |
+-------------------------------------------------------------------------+
                                    |
                                    v (Políticas Congeladas)
+-------------------------------------------------------------------------+
| C. Evaluación Cruzada sin Reentrenamiento                               |
| Evaluación R1-R4 x Geometrías de Prueba -> Métricas IQM & Ef. Mixtos  |
+-------------------------------------------------------------------------+
