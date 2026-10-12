# Diseño del entorno

Especificación **vigente** del entorno simulado: qué es cada componente y por qué se eligió así. Cuando algo cambia, se actualiza este documento y se registra el cambio, con su motivo, en la [bitácora](bitacora.md).

**Versión:** v0.1 — entorno mínimo 2D para la tarea de EDA y baseline (2026-10-09)

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
- **Paredes:** si una partícula queda a más de W/2 de su línea central, se refleja hacia el interior.
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
- Pulsátil (R3–R4): $U(t) = U_{\max}\,(1 + A\sin(2\pi t/T))$, de forma cuasiestacionaria.

**Por qué:** solución analítica exacta para flujo laminar en canal; es instantánea de evaluar y no requiere CFD.

**Limitaciones conocidas:** el perfil no es exacto cerca del vértice de la bifurcación. El pulsátil cuasiestacionario no es Womersley real (sin desfase ni aplanamiento del perfil); en el EDA del sprint 1, R3 resultó indistinguible de R2.

## 3. Dinámica de las partículas

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \left[\mathbf{u}(\mathbf{x}_t, t) + v_{mag}\,\mathbf{F}_t\right]\Delta t + \sqrt{2D\,\Delta t}\;\boldsymbol{\xi}, \qquad \boldsymbol{\xi} \sim \mathcal{N}(0, I)$$

Euler–Maruyama, vectorizado sobre las N partículas.

| Parámetro | Símbolo | Valor inicial | Nota |
|---|---|---|---|
| Número de partículas | N | 200 | Se ajusta en el EDA según la convergencia de η |
| Paso temporal | Δt | 0.05 | |
| Velocidad magnética relativa | $v_{mag}$ | 0.3 | Análogo de $ML/R$; se barre en el EDA |
| Difusión | D | 0 o 0.01 | Solo en R4 |

**Por qué:** es la misma ecuación del README (sobreamortiguada, partículas no interactuantes), reducida a 2D.

**Limitación conocida (EDA del sprint 1):** las partículas son puntuales y pueden quedar exactamente en la pared, donde el flujo es nulo. Esto exagera el atasco cuando se las empuja contra la pared. Pendiente: radio finito de partícula.

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
- **Recompensa:** avance medio, sobre la línea central, hacia la salida objetivo; más el incremento de la fracción entregada al objetivo; menos el incremento de la fracción entregada a la otra salida; menos el esfuerzo $\lVert a_t \rVert^2$. Los pesos se fijan en el código de configuración.
- **Inicio:** partículas en la entrada, repartidas al azar a lo ancho del canal.
- **Fin:** cuando ha salido el 95 % del enjambre o se alcanza el horizonte H.
- **Métrica central:** $\eta = N_{obj}/N$.

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
