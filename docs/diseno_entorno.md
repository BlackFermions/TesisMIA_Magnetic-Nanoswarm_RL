# Diseño del entorno

Especificación **vigente** del entorno simulado: qué es cada componente y por qué se eligió así. Cuando algo cambia, se actualiza este documento y se registra el cambio, con su motivo, en la [bitácora](bitacora.md).

**Versión:** v0.2 — entorno corregido en el Sprint 2 (2026-10-16): recompensa, radio de partícula, pulso. Configuración en `src/nanoswarm/variants.py` (variante `var3`); experimentos en `notebooks/sprint2_arreglos_simulador.ipynb`.

---

## 1. Geometría

**Vigente:** bifurcación en Y en 2D, sin unidades.

```
                       ╱ rama superior (OBJETIVO) → salida A
 entrada ═════════════<
 (canal madre)         ╲ rama inferior → salida B
```

| Parámetro | Símbolo | Valor inicial | Nota |
|---|---|---|---|
| Ancho de los canales | W | 1.0 | Escala de longitud |
| Largo del canal madre | L₀ | 5.0 | |
| Largo de cada rama | L₁ | 4.0 | |
| Ángulo de cada rama | θ | ±30° | |
| Rama objetivo | — | superior | |

- Cada tramo se describe por su línea central; una partícula pertenece al tramo cuya línea central tiene más cerca.
- **Paredes:** si el centro de una partícula queda a más de W/2 − a de su línea central (a = radio de la partícula), se refleja hacia el interior.
- **Salidas:** la partícula que cruza el extremo de una rama queda entregada en esa salida y deja de moverse.

**Por qué:**
- Es la geometría mínima que contiene la decisión de interés (elegir rama en una bifurcación).
- Coincide con el tipo de experimento de validación previsto en el README (direccionamiento en una bifurcación en Y).
- Al ser paramétrica, se pueden generar variantes (ángulos, anchos, ramas asimétricas) para probar la generalización.
- En 2D se ejecuta en segundos, lo que permite el EDA y el baseline en el plazo de la tarea.

**Alternativas consideradas:**
- *Anatomía real con SimVascular (3D):* postergada. Costo alto de CFD y no es necesaria para la pregunta de RL. Queda como prueba final opcional (ver [bitácora](bitacora.md), 2026-10-09).
- *Bifurcación en 3D con tubos cilíndricos:* posible siguiente versión.

## 2. Flujo

**Vigente:** perfil de Poiseuille en cada tramo, en la dirección de su línea central.

$$u(s) = U_{\max}\left(1 - \left(\tfrac{2s}{W}\right)^2\right)$$

donde $s$ es la distancia lateral a la línea central.

- $U_{\max} = 1$ (escala de velocidad).
- Ramas simétricas, por lo que el reparto natural del flujo es cercano a 50/50.
- Pulsátil (R3–R4): $U(t) = U_{\max}\,(1 + A\sin(2\pi t/T))$, de forma cuasiestacionaria, con **A = 0.9** y **T = 20** (en v0.1: A = 0.5 y T = 4). El flujo casi se detiene en cada pulso, y un latido dura aproximadamente lo mismo que el cruce del enjambre por la bifurcación.

**Por qué:** solución analítica exacta para flujo laminar en canal; es instantánea de evaluar y no requiere CFD.

**Limitaciones conocidas:** el perfil no es exacto cerca del vértice de la bifurcación. El pulsátil cuasiestacionario no es Womersley real (sin desfase ni aplanamiento del perfil); en v0.1, R3 resultó indistinguible de R2. En v0.2 se usa un pulso fuerte y lento (Sprint 2, Var3); con las heurísticas, R3 sigue igual a R2, y se verificará con PPO. Todos los episodios empiezan en la misma fase del pulso.

## 3. Dinámica de las partículas

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \left[\mathbf{u}(\mathbf{x}_t, t) + v_{mag}\,\mathbf{F}_t\right]\Delta t + \sqrt{2D\,\Delta t}\;\boldsymbol{\xi}, \qquad \boldsymbol{\xi} \sim \mathcal{N}(0, I)$$

Euler–Maruyama, vectorizado sobre las N partículas.

| Parámetro | Símbolo | Valor inicial | Nota |
|---|---|---|---|
| Número de partículas | N | 200 | Se ajusta en el EDA según la convergencia de η |
| Paso temporal | Δt | 0.05 | |
| Magnitud máxima de la fuerza de control (velocidad relativa a $U_{\max}$) | $F_{\max}$ (`v_mag` en el código) | 0.3 | Fuerza idealizada y fija: no se modela qué la produce |
| Radio de partícula | a | 0.05 W | El centro no se acerca a la pared a menos de a (Sprint 2, Var2) |
| Difusión | D | 0 o 0.01 | Solo en R4 |

**Por qué:** es la misma ecuación del README (sobreamortiguada, partículas no interactuantes), reducida a 2D.

**Radio finito (v0.2):** en v0.1 las partículas eran puntuales y podían quedar exactamente en la pared, donde el flujo es nulo. Eso exageraba el atasco al empujarlas contra la pared. Con a = 0.05 W el atasco desaparece (Sprint 2, Var2). El radio se aplica en los cuatro niveles: es una propiedad de la partícula, no del nivel de fidelidad.

## 4. Condiciones de fidelidad

| Condición | Flujo | Browniano |
|---|---|---|
| R1 | ninguno | no |
| R2 | estacionario | no |
| R3 | pulsátil | no |
| R4 | pulsátil | sí |

## 5. Interfaz de aprendizaje por refuerzo

- **Acción:** $a \in [-1,1]^2$, recortada a norma ≤ 1; $\mathbf{F}_t = a_t / \max(1, \lVert a_t \rVert)$.
- **Observación** (11 valores): centroide de las partículas activas (2), dispersión (2), fracción entregada al objetivo, fracción entregada a la otra salida, fracción activa, fase del pulso $(\sin\varphi, \cos\varphi)$ (2) y acción previa (2).
- **Recompensa:** $r_t = w_p\,\Delta \bar{d} + w_e\,\Delta \eta_{obj} - w_m\,\Delta \eta_{otras} - w_u \lVert a_t \rVert^2$, con **w_p = 0.1, w_e = 1, w_m = 0.5 y w_u = 0.0001** (en v0.1 eran 1, 1, 1 y 0.0005).
  - w_p y w_m se eligieron con las semillas de ajuste.
  - w_u se fijó por regla: la fuerza máxima durante todo el horizonte cuesta 0.08.
  - Resultado: Spearman retorno–η ≥ 0.98 en R2–R4 (Sprint 2, Var1).
- **Inicio:** partículas en la entrada, repartidas al azar a lo ancho del canal.
- **Fin:** cuando ha salido el 95 % del enjambre o se alcanza el horizonte H.
- **Métrica central:** $\eta = N_{obj}/N$.
- **Secundarias registradas por episodio:** fracción perdida en la otra salida, tiempo medio de tránsito de las partículas que salieron y esfuerzo (intensidad media $\lVert F \rVert^2$ de la fuerza de control, de 0 a 1).

## 6. Políticas de comparación

| Política | Descripción | Rol |
|---|---|---|
| Sin fuerza | $a = 0$ | Reparto natural por el flujo |
| Aleatoria | $a$ uniforme en $[-1,1]^2$ | Piso de control |
| Heurística perpendicular | Cerca de la bifurcación empuja perpendicular al canal, hacia el lado de la rama objetivo (la del README) | Modelo simple sin aprendizaje |
| Heurística rama | Empuja siempre en la dirección de la rama objetivo | Modelo simple sin aprendizaje; referencia principal |

PPO (y DreamerV3 como extensión) son los modelos principales y **no** forman parte del baseline.

---

## Historial de versiones

| Versión | Fecha | Cambio |
|---|---|---|
| v0.1 | 2026-10-09 | Entorno mínimo 2D en Y con flujo analítico, para la tarea de EDA y baseline |
| v0.2 | 2026-10-16 | Sprint 2: recompensa corregida, radio de partícula a = 0.05 W, pulso con A = 0.9 y T = 20; registro de tránsito y esfuerzo. Niveles R1–R4 congelados |
