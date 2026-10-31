# Matriz de consistencia

Tesis: Generalización de políticas de aprendizaje por refuerzo ante diferentes niveles de fidelidad de simulación

Alumno: Brayan Bruce Pérez Escobar

| Problema | Objetivos | Hipótesis | Variables | Indicadores |
|---|---|---|---|---|
| ¿Puede un currículo progresivo de fidelidad física mejorar la generalización de políticas de aprendizaje por refuerzo, frente al entrenamiento en un único nivel de fidelidad, en la navegación simulada de un enjambre de nanopartículas magnéticas? | Objetivo general: Evaluar si un currículo de fidelidad R1→R4 mejora la generalización de políticas PPO entre niveles de fidelidad, en comparación con el entrenamiento en un único nivel. | Hipótesis general: Las políticas entrenadas con currículo R1→R4 obtienen una brecha de fidelidad menor y un η medio mayor en la matriz de evaluación cruzada que las entrenadas en un solo nivel. | Var. independientes: régimen de entrenamiento (un nivel Ri o currículo R1→R4); nivel de evaluación Rj.<br><br>Var. dependiente: generalización de la política. | η (principal), FidelityGap, fracción perdida, tiempo de tránsito, esfuerzo de control, eficiencia muestral, estabilidad entre semillas. |
| | Objetivo específico 1: Verificar y fijar los cuatro niveles de fidelidad (R1 sin flujo, R2 flujo estacionario, R3 flujo pulsátil, R4 flujo pulsátil con movimiento browniano) como condiciones distinguibles para el control. | Hipótesis específica 1: Los cuatro niveles producen desempeños distintos en las políticas de referencia. | Var. independiente: nivel de fidelidad (R1 a R4).<br><br>Var. dependiente: desempeño de las políticas de referencia (aleatoria y heurística). | η con IC 95 % por nivel, correlación de Spearman entre retorno y η. |
| | Objetivo específico 2: Cuantificar cómo cambia el desempeño de una política PPO entrenada en un nivel Ri al evaluarla en cada nivel Rj. | Hipótesis específica 2: El desempeño cae cuando Ri ≠ Rj, y la caída depende de la dirección del cambio (es asimétrica). | Var. independientes: nivel de entrenamiento Ri; nivel de evaluación Rj.<br><br>Var. dependiente: desempeño fuera del nivel de entrenamiento. | η en la matriz 4×4, FidelityGap = η(Ri, Ri) − η(Ri, Rj). |
| | Objetivo específico 3: Comparar PPO con currículo R1→R4 y PPO entrenado en un único nivel, con el mismo presupuesto de interacciones. | Hipótesis específica 3: El currículo reduce la brecha de fidelidad media frente a cualquier entrenamiento en un solo nivel. | Var. independiente: régimen de entrenamiento (fijo o currículo).<br><br>Var. dependiente: generalización. | FidelityGap medio, η medio en la matriz, eficiencia muestral (área bajo la curva de aprendizaje), desviación entre semillas. |
| | Objetivo específico 4: Identificar qué variables de la observación determinan la transferencia entre niveles. | Hipótesis específica 4: Las variables cuyo significado depende del nivel (fase del pulso, margen de control) reducen la transferencia; una observación basada en el estado del enjambre transfiere mejor. | Var. independiente: conjunto de variables de la observación.<br><br>Var. dependiente: transferencia entre niveles. | Importancia por permutación (η original − η permutado), η en niveles no vistos, estabilidad del ranking entre semillas (Spearman). |
| | Objetivo específico 5 (extensión): Comparar PPO y DreamerV3 bajo el mismo protocolo. | Hipótesis específica 5: DreamerV3 presenta una brecha de fidelidad menor que PPO bajo el protocolo estudiado. | Var. independiente: algoritmo (PPO o DreamerV3).<br><br>Var. dependiente: robustez ante el cambio de fidelidad. | FidelityGap, η, eficiencia muestral, costo computacional (horas de CPU o GPU). |

## Definición de los indicadores

η es la fracción de partículas del enjambre que llegan a la rama objetivo al final del episodio. Es la métrica principal.

FidelityGap es la brecha de fidelidad: cuánto cae η cuando una política entrenada en el nivel Ri se evalúa en otro nivel Rj, FidelityGap = η(Ri, Ri) − η(Ri, Rj).

Las métricas secundarias son la fracción perdida en la otra rama, el tiempo de tránsito, el esfuerzo de control (magnitud acumulada de la fuerza aplicada), la eficiencia muestral (área bajo la curva de aprendizaje) y la estabilidad entre semillas (desviación estándar entre semillas de entrenamiento).
