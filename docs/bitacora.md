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

**Estado:** `Decidido` — concretado en la entrada 10 (2026-10-16).

### 3. Componentes de IA candidatos

| Componente | Idea | Estado |
|---|---|---|
| Aleatorización de dominio | Régimen adicional: en cada episodio la condición física se elige al azar | `Pendiente` (recomendado) |
| Representación aprendida | Codificador sobre el conjunto de partículas frente a estadísticas diseñadas a mano | `Pendiente` |
| Memoria | Política recurrente sin la fase pulsátil frente a MLP con la fase | `Pendiente` |
| Análisis del modelo del mundo | Error de predicción de DreamerV3 por condición frente a brecha de transferencia | `Pendiente` |

**Criterio de selección acordado:** elegir según la orientación de la investigación actual y la utilidad de los resultados para la comunidad, no por preferencia personal. Pendiente revisar la literatura reciente para fundamentar la elección.

**Estado general:** `Reemplazado` — el componente de IA del núcleo es el currículo de fidelidad (entrada 11). Estas ideas quedan como trabajo futuro.

### 4. Geometrías: SimVascular frente a paramétricas

**Opción considerada:** reemplazar el CFD sobre anatomía real por bifurcaciones paramétricas con flujo analítico (Poiseuille para el estacionario, Womersley para el pulsátil), y dejar SimVascular como prueba final opcional en 1–2 anatomías reales.

**Ventajas:** mucho menos costo de cómputo y de trabajo, control total de las variables y posibilidad de generar muchas geometrías para evaluar la generalización.

**Desventaja:** menor realismo anatómico.

**Estado:** `Decidido` — ver entrada 13 (2026-10-16).

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

---

## 2026-10-16 — Replanteamiento con la asesora ([Proyecto.pdf](Paper/Proyecto.pdf))

**Contexto:** el profesor observó que los reportes no se entendían. Con la asesora se pulió el planteamiento en [Proyecto.pdf](Paper/Proyecto.pdf), que desde ahora es la referencia del proyecto: el README y el código se ajustan a él.

### 10. Nuevo título y foco

**Título:** *Generalización de políticas de aprendizaje por refuerzo ante diferentes niveles de fidelidad de simulación.*

**En simple:** ¿un controlador que aprendió en un simulador sigue funcionando si se lo prueba en un simulador más realista, o menos realista?

- **Problema de IA:** generalización de políticas de RL cuando el simulador de entrenamiento no coincide con el de evaluación (*simulator mismatch*).
- **Caso de estudio:** el direccionamiento de enjambres de nanopartículas magnéticas. La física es el laboratorio, no el objeto de estudio.

**Estado:** `Decidido`

### 11. Preguntas de investigación e hipótesis

1. **Núcleo:** ¿cómo varía el desempeño de una política cuando se entrena en un nivel de fidelidad y se evalúa en otro?
2. **Hipótesis principal:** un currículo progresivo de fidelidad R1 → R2 → R3 → R4 mejora la generalización frente al entrenamiento en un único nivel.
3. **Extensión (opcional):** ¿un modelo interno del entorno (DreamerV3, *model-based*) da más robustez que PPO (*model-free*)? Si DreamerV3 resulta demasiado costoso, la tesis queda completa con 1 y 2.

**Diseño del núcleo:** se entrenan πR1, πR2, πR3 y πR4, y cada una se evalúa en los cuatro niveles (matriz 4 × 4). La **brecha de fidelidad** mide cuánto cae el desempeño fuera del nivel de entrenamiento:

$$
\text{Gap}(i \to j) = \eta(\pi_{R_i} \text{ en } R_i) - \eta(\pi_{R_i} \text{ en } R_j)
$$

Preguntas que responde la matriz: ¿cuánto se degrada PPO?, ¿en qué dirección?, ¿entrenar en alta fidelidad generaliza a baja?, ¿y entrenar en baja generaliza a alta?

**Estado:** `Decidido`

### 12. Currículo: regla de avance y criterio de éxito

- **Regla de avance:** la política pasa de Rk a Rk+1 cuando su η en validación supera un umbral τ en 3 evaluaciones seguidas, o al agotar un tope de interacciones por nivel (para que nunca se estanque). τ y el tope se fijan en el piloto.
- **Comparación justa:** el currículo usa el mismo presupuesto total de interacciones que cada régimen fijo.
- **Criterio de éxito:** el currículo "gana" si su η promedio en R1–R4 es mayor que el de πR4, la política entrenada solo en el nivel final del currículo.

**Estado:** `Decidido` (la regla de avance es provisional; se revisa con el piloto)

### 13. Geometrías: Y simple primero, SimVascular como etapa 2

Resuelve la entrada 4.

- **Etapa 1, bifurcación en Y paramétrica:** todos los experimentos del núcleo (matriz, currículo, PPO). Es rápida y permite muchas repeticiones con semillas distintas.
- **Etapa 2, SimVascular:** se repiten los resultados clave en 1–2 anatomías reales del VMR para comprobar que las conclusiones se sostienen con geometría y flujo realistas. Sigue la ruta de Medany et al. (2025): simulador simple primero, entorno más real después.

**Por qué:** SimVascular aporta realismo pero no cambia la pregunta de investigación; hacerlo al final evita que bloquee el núcleo. Si el CFD toma más de lo previsto, la tesis queda completa con la etapa 1.

**Estado:** `Decidido`

### 14. Arreglos del simulador antes de entrenar (Sprint 2)

El EDA del Sprint 1 encontró tres fallas. Se corrigen antes de PPO como experimentos A/B, **un cambio por variante** (clase 04 del curso), midiendo con las heurísticas y las mismas semillas:

| Variante | Cambio | Cómo se mide si funcionó |
|---|---|---|
| Baseline | Entorno v0.1 (Sprint 1) | — |
| Var1 | Recompensa corregida | Spearman retorno–η ≥ 0.9 en R2 y R3; la heurística perpendicular en R2 obtiene menos retorno que "sin fuerza" |
| Var2 | + radio finito de partícula | Desaparece o se reduce el atasco en la pared en R2 (η de la heurística perpendicular) |
| Var3 | + pulso más fuerte en R3 y R4 | η en R3 distinta de R2 (IC 95 % sin solapar) |

**Por qué el radio no arregla R3:** en R2 y R3 el flujo recorre los mismos caminos y en R3 solo cambia la rapidez, así que una partícula sin control llega a la misma rama, solo antes o después. El radio corrige el contacto con la pared, no eso. Un pulso que casi detiene el flujo sí cambia el problema: el imán gana fuerza relativa en esos instantes y la política puede aprender a aprovecharlos.

**El radio se aplica en los cuatro niveles**, porque es una propiedad de la partícula, no del nivel de fidelidad.

**Estado:** `Decidido`

### 15. Niveles de fidelidad congelados

Tras los arreglos de la entrada 14, las definiciones no vuelven a cambiar:

| Nivel | Definición |
|:---:|---|
| R1 | Sin flujo |
| R2 | Flujo estacionario |
| R3 | Flujo pulsátil |
| R4 | Flujo pulsátil + movimiento browniano |

*Nota:* Proyecto.pdf repite "R3" en la última línea; corresponde a R4.

**Estado:** `Decidido`

### 16. Métricas

- **Principal:** eficiencia de direccionamiento η = N_obj / N.
- **Secundarias:** fracción perdida en otras ramas, tiempo de tránsito, esfuerzo de control, eficiencia muestral y estabilidad entre semillas.
- El retorno acumulado se usa solo como diagnóstico. Sale de la lista la dispersión final del enjambre.

**Estado:** `Decidido`

### 17. Plan de trabajo

1. **Sprint 2:** arreglos del simulador (entrada 14), congelar R1–R4 y reescribir el README según Proyecto.pdf.
2. **Piloto:** 3 semillas, una geometría, presupuesto reducido, PPO fijo en R1–R4 y currículo simple. No busca probar la hipótesis, sino verificar que PPO aprende, que los niveles son realmente distintos, que el currículo avanza, que la recompensa se correlaciona con η y que el costo de cómputo es viable. Aquí se fija también α (entrada 1).
3. **Experimento completo:** PPO fijo, luego PPO con currículo, y la matriz de evaluación cruzada.
4. **Extensión:** DreamerV3, si el costo medido en el piloto lo permite.
5. **Etapa 2:** comprobación en anatomía real con SimVascular (entrada 13).

**Estado:** `Decidido`

---

## 2026-10-16 — Resultados del Sprint 2: arreglos del simulador

Informe ejecutable: [notebooks/sprint2_arreglos_simulador.ipynb](../notebooks/sprint2_arreglos_simulador.ipynb). Logs: `logs/metrics_sprint2_*.txt`.

### 18. Partición por semillas

**Decisión:** semillas de **ajuste** (5000–5019) para todas las decisiones y de **reporte** (9000–9029) solo para resultados. Es el equivalente en RL de separar entrenamiento y prueba (clase 04, "divide primero, transforma después").

**Estado:** `Decidido`

### 19. Resultados de las variantes y una variante nueva (Var4)

| Variante | Resultado | Decisión |
|---|---|---|
| Var1: recompensa ($w_p = 0.1$, $w_m = 0.5$ por rejilla 3 × 3 en ajuste; $w_u = 0.0001$ por regla) | Spearman retorno–η en R2: de 0.72 a 0.98 | Se adopta |
| Var2: radio de partícula a = 0.05 W | Heurística perpendicular en R2: η de 0.00 a 0.95 | Se adopta |
| Var3: pulso A = 0.9, T = 20 | R3 sigue igual a R2. Con $v_{mag} = 0.3$ el imán domina en todo el latido | Se adopta, pero no basta |
| **Var4 (nueva):** $v_{mag} = 0.2$, con el horizonte escalado a 1200 pasos | R1 0.97 → R2 0.95 → R3 0.84 → R4 0.79 (heurística rama); todos los niveles se distinguen | Se adopta |

**Por qué se agregó Var4:** la entrada 14 suponía que un pulso fuerte bastaría para separar R3 de R2. El experimento mostró que no: el pulso solo importa cuando el imán no gana siempre. $v_{mag}$ se eligió como **el imán más fuerte con el que R3 se distingue de R2** en ambas heurísticas (barrido en semillas de ajuste). El horizonte crece en la misma proporción en que se debilita el imán (800 × 0.3/0.2), porque si no R1 se vuelve imposible.

**Entorno resultante:** v0.2 = Var4 ([diseno_entorno.md](diseno_entorno.md)). R1–R4 quedan congelados (entrada 15).

**Estado:** `Decidido`

### 20. Limitaciones registradas para el piloto

- **Criterio de la recompensa en R2 desde Var2:** ρ = 0.86–0.89. La causa es un empate en el techo (η ≈ 0.95 de las dos heurísticas), en el que la recompensa prefiere la que gasta menos imán. Un chequeo agregado después de ver el resultado confirma que ninguna política que entrega claramente menos recibe más puntos.
- **Techo de η en 0.95** porque el episodio termina al salir el 95 %. Cambio candidato: terminar al 100 %.
- **Fase inicial del pulso fija.** Cambio candidato: fase al azar.
- **$w_p = 0.1$ da una señal densa débil:** vigilar si PPO aprende en el piloto.

**Estado:** `Pendiente` (se revisan en el piloto)

### 21. Terminología: fuerza de control idealizada, sin asumir el mecanismo

**Decisión:** el proyecto no asume todavía qué produce la fuerza sobre el enjambre (magnética u otra). En los documentos y el notebook se habla de una **fuerza externa de control idealizada**, común a todas las partículas, con magnitud máxima $F_{\max}$, en lugar de "imán". En el código el parámetro conserva el nombre `v_mag`.

**Aclaración sobre la Var4:** $F_{\max}$ no es un factor de estudio. Es un parámetro fijo del entorno, igual en todos los niveles y para todas las políticas. La Var4 solo cambió su valor fijo de 0.3 a 0.2.

**Estado:** `Decidido`

### 22. Se retira la Var4

**Decisión:** $F_{\max}$ es un parámetro idealizado y fijo del proyecto ($F_{\max} = 0.3$), no algo a ajustar. Se retira la Var4 (entrada 19). El entorno resultante del Sprint 2 es la **Var3**: recompensa corregida + radio de partícula + pulso fuerte y lento. Con heurísticas, R3 sigue igual a R2; la verificación con un agente que observa la fase del pulso queda para el piloto con PPO. Informe: [notebooks/sem5_avance.ipynb](../notebooks/sem5_avance.ipynb).

**Estado:** `Decidido` (reemplaza la decisión sobre Var4 de la entrada 19)
