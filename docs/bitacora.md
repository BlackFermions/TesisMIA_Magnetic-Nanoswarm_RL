# Bitácora de decisiones

Registro cronológico de cómo evoluciona la idea central de la tesis: observaciones del asesor, opciones consideradas, decisiones tomadas y su justificación. Las entradas no se borran; si una decisión cambia, se agrega una entrada nueva que la reemplaza.

El [README](../README.md) describe siempre el estado **actual** del proyecto.

**Estados:** `Pendiente` · `Decidido` · `Descartado` · `Reemplazado`

---

## 2026-10-09 — Observaciones del asesor sobre el enfoque

### 1. Tasa de aprendizaje (α)

**Observación del asesor:** responder sobre la tasa de aprendizaje y qué valores de α conviene analizar en los agentes de RL.

**Propuesta:**
- PPO: rejilla logarítmica {1·10⁻⁴, 3·10⁻⁴, 1·10⁻³} explorada en la prueba piloto, en la geometría de validación, con 2–3 semillas.
- Un único α por algoritmo, compartido por todos los regímenes de entrenamiento, para no favorecer a ninguno.
- α constante (sin decaimiento): un decaimiento lineal dejaría al currículo R1→R4 con α pequeño justo al llegar a R4 y confundiría el efecto del currículo con el del calendario de α.
- DreamerV3 con sus hiperparámetros por defecto, coherente con su diseño (por verificar en la configuración oficial).
- Reportar un análisis de sensibilidad de η frente a α para comprobar que las conclusiones no dependen de α.

**Estado:** `Pendiente` — falta validarlo con el asesor e incorporarlo al README y al preregistro.

### 2. "No parece una tesis de maestría en IA, sino de computación científica"

**Diagnóstico:** el README dedica la mayor parte del esfuerzo al simulador físico (CFD con SimVascular, verificación, números adimensionales), mientras que el RL aparece como herramienta aplicada (PPO y DreamerV3 sin modificar, observación y recompensa diseñadas a mano).

**Lo que ya es IA:** la generalización entre niveles de fidelidad (análogo de la brecha *sim-to-real*), el currículo, la comparación model-based frente a model-free y la evaluación estadística rigurosa.

**Propuesta de reorientación:**
- Cambiar el objeto de estudio de "un simulador vascular donde se entrena RL" a "un estudio sobre la generalización de políticas de RL entre fidelidades, con un banco de pruebas físico controlado".
- Formular preguntas de investigación de RL con hipótesis (brecha de transferencia, estrategia de entrenamiento, modelo del mundo, sensibilidad a hiperparámetros).
- Reducir el peso de la física a un instrumento verificado.

**Estado:** `Pendiente`

### 3. Componentes de IA candidatos

| Componente | Idea | Estado |
|---|---|---|
| Aleatorización de dominio | Régimen adicional: en cada episodio la condición física se elige al azar | `Pendiente` (recomendado) |
| Representación aprendida | Codificador sobre el conjunto de partículas frente a estadísticas diseñadas a mano | `Pendiente` |
| Memoria | Política recurrente sin la fase pulsátil frente a MLP con la fase | `Pendiente` |
| Análisis del modelo del mundo | Error de predicción de DreamerV3 por condición frente a brecha de transferencia | `Pendiente` |

**Criterio de selección acordado:** elegir según la orientación de la investigación actual y la utilidad de los resultados para la comunidad, no por preferencia personal. Pendiente revisar la literatura reciente para fundamentar la elección.

### 4. Geometrías: SimVascular frente a paramétricas

**Opción considerada:** reemplazar el CFD sobre anatomía real por bifurcaciones paramétricas con flujo analítico (Poiseuille para el estacionario, Womersley para el pulsátil), y dejar SimVascular como prueba final opcional en 1–2 anatomías reales.

**Ventajas:** mucho menos costo de cómputo y de trabajo, control total de las variables y posibilidad de generar muchas geometrías para evaluar la generalización.

**Desventaja:** menor realismo anatómico.

**Estado:** `Pendiente`

---

## 2026-10-09 — Paper base y entorno mínimo para la tarea de EDA y baseline

### 5. Paper base: Medany et al. (2025)

**Resumen:** control de un microrrobot impulsado por ultrasonido con DreamerV3, preentrenado en un simulador 2D (Pygame) y transferido a canales físicos. Políticas entrenadas sin flujo fallaron con flujo; lo resolvieron con ajustes ad hoc de la recompensa y un flujo modelado como fuerza constante.

**Vacíos que aborda la tesis:**
- La brecha de fidelidad física se observa de forma anecdótica, no se estudia de forma sistemática.
- Flujo sin base física (sin perfil, sin pulso, sin browniano).
- Generalización solo sobre la geometría, no sobre la física; sin comparar currículo frente a aleatorización.
- Un solo cúmulo tratado como punto, no un enjambre con métrica distributiva.
- Evaluación sin protocolo estadístico ni comparación equitativa entre PPO y DreamerV3.

**Estado:** `Decidido` — es el punto de partida de la tesis.

### 6. Baseline de la tarea

**Decisión:** el baseline no puede ser PPO, porque PPO es el modelo principal. Se usan la política sin fuerza y la aleatoria como baseline dummy, y la heurística geométrica como modelo simple.

**Estado:** `Decidido`

### 7. Entorno mínimo v0.1

**Decisión:** bifurcación en Y en 2D con flujo de Poiseuille analítico y cuatro condiciones (R1–R4) simplificadas. Especificación completa en [diseno_entorno.md](diseno_entorno.md).

**Motivo:** la tarea es para hoy; un entorno 2D analítico se ejecuta en segundos y contiene la decisión de interés (elegir rama). No cierra la decisión 4 (SimVascular frente a geometrías paramétricas): es el primer paso de cualquiera de las dos rutas.

**Estado:** `Decidido` (para la tarea)

### 8. Imágenes médicas como base de la geometría (EDA de anatomía real)

**Idea:** en RL no hay un dataset fijo; los datos del agente son las trayectorias que genera el entorno. Las imágenes médicas no las ve el agente, pero pueden **definir las geometrías**: medir en modelos del Vascular Model Repository (imágenes ya segmentadas, con líneas centrales y radios) las propiedades de las bifurcaciones reales.

**Variables a extraer por bifurcación:**
- ángulos entre ramas;
- razón de diámetros madre/hijas y cumplimiento de la ley de Murray;
- longitudes entre bifurcaciones;
- asimetría entre ramas (cambia el reparto natural del flujo, el equivalente al desbalance de clases).

**Por qué:** fija los rangos de la geometría paramétrica (θ, W, asimetría) con anatomía real en lugar de valores arbitrarios. Permite usar la anatomía sin hacer CFD sobre ella, lo que también responde a la decisión 4.

**Decisión accionable (siguiente sprint):** extraer ángulos, razones de diámetro y asimetría de N bifurcaciones del VMR. **Medible:** tabla y distribución de cada parámetro, con los rangos adoptados en [diseno_entorno.md](diseno_entorno.md).

**Por qué no ahora:** descargar y procesar modelos del VMR toma más que el plazo de la tarea (entrega 2026-10-09). Para la tarea, el EDA se hace sobre las trayectorias del entorno.

**Estado:** `Pendiente` (siguiente sprint)

### 9. Resultados del EDA y baseline (sprint 1)

Informe completo en [eda_baseline.md](eda_baseline.md).

**Hallazgos:**
- La fidelidad física cambia el ranking de las políticas: la heurística perpendicular del README entrega η = 0.00 en R2 y η = 0.88 en R4 (el browniano despega a las partículas de la pared).
- La heurística en la dirección de la rama logra η ≈ 0.95 en todas las condiciones; el reparto natural sin control es ≈ 0.49.
- La recompensa no está bien alineada con η (Spearman 0.66–0.72 en R2 y R3).
- R3 no se distingue de R2.

**Decisiones para el sprint 2:** ajustar la recompensa, rediseñar R3 y agregar un radio finito de partícula. Además: usar la heurística rama como referencia principal, ya que la perpendicular del README falla.

**Estado:** `Decidido`
