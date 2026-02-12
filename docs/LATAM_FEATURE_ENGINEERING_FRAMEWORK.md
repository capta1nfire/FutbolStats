# Ingeniería de Características Estratégica y Extracción de Señal en Mercados de Fútbol de Alta Entropía: Un Marco Diagnóstico y Metodológico para Ligas LATAM

## 1. Resumen Ejecutivo y Diagnóstico Estructural

La paradoja central que enfrenta el modelo XGBoost v1.0.1 —la incapacidad de extraer señal incremental de métricas avanzadas como los Goles Esperados (xG) y la degradación del rendimiento en ligas latinoamericanas— no constituye un fallo aislado de ingeniería de datos, sino un síntoma de un desajuste fundamental entre la arquitectura del modelo y la realidad estocástica del entorno que intenta predecir. El análisis forense de los 54,667 partidos procesados, contrastado con la literatura académica y la dinámica de los mercados de apuestas eficientes, revela que el enfoque actual de "predicción ab initio" (intentar predecir el resultado desde cero basándose únicamente en promedios históricos) es matemáticamente insostenible en entornos de alta entropía como la Primera División de Argentina.

La evidencia recopilada apunta a tres vectores de fallo críticos. Primero, la redundancia colineal: en su codificación actual de media móvil (rolling average), el xG actúa simplemente como un proxy del volumen de tiros, una variable que el modelo ya posee, lo que explica la neutralidad observada en los tests A/B. Segundo, la volatilidad estructural de LATAM: el uso de ventanas de tiempo largas (10 partidos) en ligas caracterizadas por una rotación extrema de plantillas ("roster turnover"), formatos de torneo cortos (Apertura/Clausura) y una paridad competitiva forzada, diluye la señal táctica reciente en un mar de ruido histórico irrelevante. Tercero, y más importante, la subutilización de la eficiencia del mercado: al ignorar las cuotas de cierre (closing lines) como priores bayesianos, el modelo intenta redescubrir información que el mercado ya ha descontado con mayor precisión, resultando en una desventaja sistemática de ~3-4% en el Brier Score frente a las casas de apuestas.

Este informe propone una reingeniería completa hacia la versión v1.0.2, pivotando de un modelo de "predicción de resultados" a un modelo de "predicción de ineficiencias de mercado" (Market Residual Modeling). Se establece que la única vía viable para superar el techo de rendimiento en Argentina es la implementación de métricas ajustadas por oponente mediante Regresión Ridge, la incorporación de variables de contexto físico (altitud, fatiga de viaje), y el uso de objetivos de entrenamiento probabilísticos (soft labels) derivados de simulaciones de xG para mitigar la varianza inherente a los resultados binarios en deportes de baja anotación.

## 2. Anatomía de la Falla de Señal en Entornos de Alta Entropía

### 2.1 La Falacia de los Promedios Móviles y la Dilución Contextual

El diagnóstico del usuario identifica correctamente que "un promedio rolling de los últimos N partidos pierde información posicional". Sin embargo, el problema es más profundo: se trata de una dilución contextual irreversible. Una media móvil simple asume implícitamente que la muestra de partidos pasados proviene de una distribución estacionaria y homogénea, una asunción que se viola flagrantemente en el fútbol, y más aún en Sudamérica.

Considere un promedio de xG de 1.75 generado por un equipo argentino en sus últimos 5 partidos. Este número comprime y oculta realidades dispares:

- Un partido dominante (3.0 xG) contra un equipo descendido.
- Un partido conservador (0.2 xG) jugando con 10 hombres en la altura de La Paz o Quito.
- Un partido "muerto" (0.5 xG) donde el equipo ya estaba clasificado a la siguiente fase de la Copa Libertadores y rotó su plantilla.

Al aplicar una media móvil (mean()) sobre estos eventos heterogéneos, el modelo v1.0.1 destruye la señal específica de cada contexto. En Europa, donde la jerarquía de los equipos es estable y los factores externos (altitud, viajes extremos) son mínimos, el promedio conserva cierta validez predictiva. En LATAM, el promedio es un artefacto estadístico que no representa la "forma real" del equipo ante el próximo rival. La literatura sugiere que modelos que no incorporan ajustes por la fuerza del oponente (opponent-adjusted metrics) fallan sistemáticamente en predecir márgenes de victoria fuera de entornos controlados.

### 2.2 El Problema de la Redundancia Informacional (xG vs. Tiros)

Los resultados del Test A/B, donde la adición de xG al modelo base (que ya contiene tiros) resulta en un impacto NEUTRAL o NOISE, confirman la hipótesis de colinealidad. En términos de teoría de la información, el xG agregado en ventanas largas tiende a converger con el volumen de tiros.

El valor del xG reside en su capacidad para distinguir la calidad de una oportunidad individual. Sin embargo, cuando se agrega a nivel de partido y luego se promedia a lo largo de la temporada, la varianza de la "calidad de tiro" se suaviza, y la métrica se convierte en un proxy lineal del volumen ofensivo. Dado que el modelo XGBoost ya está capturando la "presión ofensiva" a través de shots_avg y corners_avg, el xG no aporta nueva entropía informativa. Para que el xG añada señal incremental, debe ser transformado para capturar dimensiones que los tiros no ven: eficiencia de conversión sobre lo esperado (finishing skill) o generación de peligro ajustada por la densidad defensiva del rival, algo que un simple rolling mean no logra.

Además, existe un problema de ruido de codificación. Los modelos de xG (sea Opta, Understat o propietarios) tienen sus propios sesgos y errores de calibración. Al sumar estos errores sobre 10 partidos, se introduce una capa de incertidumbre epistémica que puede superar la ganancia de señal, especialmente en ligas con datos de tracking menos fiables o estandarizados como las de segunda línea en LATAM.

### 2.3 La Singularidad Argentina: Entropía y Estructura de Mercado

La brecha de rendimiento observada en Argentina (Brier 0.6535) frente a la Primeira Liga portuguesa (0.5805) no debe interpretarse únicamente como una deficiencia del modelo, sino como una característica intrínseca del sistema modelado. Argentina presenta un perfil de "alta entropía" definido por factores estructurales que aumentan la aleatoriedad irreducible del juego.

#### 2.3.1 Paridad Competitiva y la Prevalencia del Empate

La Liga Profesional Argentina opera bajo un régimen de paridad forzada, exacerbada por formatos de torneo masivos (28 equipos) y promedios de descenso que incentivan el conservadurismo táctico. Una liga con una alta tasa de empates y baja varianza de goles es intrínsecamente más difícil de predecir (mayor entropía de Shannon) que una liga estratificada como la portuguesa, donde los "Tres Grandes" (Benfica, Porto, Sporting) garantizan resultados asimétricos predecibles. En Argentina, la diferencia de calidad entre el equipo clasificado 5º y el 20º es marginal, lo que reduce la eficacia de las características basadas en la "diferencia de fuerza" (goal_diff_avg).

#### 2.3.2 Inestabilidad Temporal: El Ciclo de Exportación

El fenómeno de la "rotación de plantilla" (roster turnover) es un destructor de señal crítico en Sudamérica. Los equipos funcionan como economías exportadoras de talento. Un equipo puede tener métricas excelentes en el Torneo Apertura, vender a sus dos mejores delanteros a la MLS o Europa en la ventana de transferencias, y comenzar el Clausura con un perfil estadístico totalmente diferente.

El modelo actual, al utilizar una ventana de 10 partidos con time decay, sigue "recordando" el rendimiento de los jugadores vendidos. Esto genera un sesgo de supervivencia negativo: el modelo sobreestima a los equipos que acaban de vender talento (porque sus métricas pasadas son buenas) y subestima a los que se han reforzado pero aún no han generado datos. En Europa, la estabilidad de las plantillas durante la temporada permite que los promedios móviles funcionen; en Argentina, la inestabilidad de la plantilla hace que los datos históricos de hace 2 meses sean, a menudo, obsoletos.

#### 2.3.3 La Eficiencia del Mercado Local

El hecho de que el mercado supere al modelo por un 4.2% en Argentina indica que las casas de apuestas no están ciegas ante estas realidades. Los oddsmakers y los sindicatos de apuestas incorporan información cualitativa (ventas de jugadores, crisis institucionales, deudas salariales, priorización de Copa Libertadores) que un modelo puramente técnico ignora. El mercado está "fijando precios" basándose en la realidad actual del vestuario, mientras que el modelo v1.0.1 está "fijando precios" basándose en fantasmas estadísticos de hace diez jornadas.

## 3. Reingeniería de Características: Capturando el Contexto Latente

Para la versión v1.0.2, la ingeniería de características debe abandonar la agregación descriptiva y adoptar la contextualización relativa. El objetivo es purificar la señal eliminando el ruido introducido por la calidad del oponente y las condiciones ambientales.

### 3.1 Métricas Ajustadas por Oponente (Opponent-Adjusted Metrics)

La solución matemática estándar para corregir la distorsión del calendario es el ajuste por oponente. No todos los goles o xG son iguales; su valor depende de la calidad defensiva del rival que los permitió.

#### 3.1.1 Implementación vía Regresión Ridge (RAPM)

La técnica más robusta, importada de la analítica avanzada de la NBA (Regularized Adjusted Plus-Minus) y aplicada al fútbol, es la Regresión Ridge. Este método permite aislar la contribución ofensiva y defensiva intrínseca de cada equipo resolviendo un sistema de ecuaciones lineales simultáneas para toda la liga.

**Formulación Matemática:**

Se construye una matriz de diseño dispersa $X$ donde cada fila representa un partido y cada columna representa un equipo (duplicado para ataque y defensa).

$$Y = X\beta + \epsilon$$

Donde:
- $Y$ es el vector de diferenciales de xG (xG For - xG Against) por partido.
- $X$ contiene $+1$ para el equipo local y $-1$ para el visitante en sus respectivas columnas.
- $\beta$ es el vector de coeficientes que representa el "Rating Ajustado" de cada equipo.

Al aplicar regularización Ridge (penalización L2), controlamos el sobreajuste en las primeras jornadas cuando el tamaño de la muestra es pequeño.

$$\hat{\beta} = (X^T X + \lambda I)^{-1} X^T Y$$

Los coeficientes resultantes $\beta$ proporcionan una medida de fuerza del equipo que es independiente del calendario. Si un equipo tiene un $\beta_{att} = +0.5$, significa que genera 0.5 xG más que el promedio de la liga contra una defensa promedio. Este feature es infinitamente más predictivo que un promedio simple, especialmente en ligas desequilibradas o con calendarios asimétricos.

#### 3.1.2 Implementación Dinámica vía Elo-xG

Dado que la Regresión Ridge es estática (o requiere reentrenamiento constante), un sistema Elo modificado para xG ofrece una alternativa dinámica que se actualiza partido a partido. En lugar de actualizar el rating basado en ganar/perder, se actualiza basado en el rendimiento de xG relativo a lo esperado.

**Algoritmo de Actualización:**

1. **Expectativa:** Calcular el xG esperado ($xG_{exp}$) basado en la diferencia de ratings actuales entre el Equipo A (Ataque) y el Equipo B (Defensa).
2. **Residuo:** Calcular el delta de rendimiento: $\Delta = xG_{real} - xG_{exp}$.
3. **Actualización:** $R_{nuevo} = R_{viejo} + K \cdot \Delta$.

Este enfoque captura la "forma" reciente de manera más orgánica que una media móvil, ya que penaliza a un equipo que genera poco xG contra una defensa débil y recompensa a uno que genera mucho xG contra una defensa de élite, independientemente del resultado del partido.

### 3.2 Condicionamiento por Estado del Juego (Game State Adjustment)

Los datos de xG están contaminados por el "Efecto del Marcador" (Game State Effect). Los equipos que van ganando tienden a ceder la posesión y reducir su producción ofensiva, mientras que los que van perdiendo inflan sus números de tiros y xG debido a la urgencia, no necesariamente a la superioridad.

**Metodología de Corrección:**

Para v1.0.2, se recomienda descomponer las métricas ofensivas en función del estado del marcador.

- **xG Neutral:** Calcular el ritmo de generación de xG (xG por 90 min) considerando solo los minutos en los que el partido está empatado o la diferencia es de un solo gol ($|GD| \le 1$). Esto elimina el ruido de los "minutos basura" o de las tácticas de conservación de ventaja.

- **Ajuste Lineal:** Si la segmentación reduce demasiado la muestra, se puede aplicar una regresión para ajustar el xG total:

$$xG_{adj} = xG_{total} - \gamma \cdot (\text{Minutos Ganando}) + \delta \cdot (\text{Minutos Perdiendo})$$

Donde $\gamma$ y $\delta$ son coeficientes aprendidos que representan la tendencia media de la liga a "sacar el pie del acelerador" o "volcarse al ataque".

### 3.3 Ingeniería de Características Ambientales para LATAM

Para abordar la debilidad específica en las ligas andinas y argentinas, es imperativo modelar las restricciones físicas que no existen en Europa.

#### 3.3.1 Factor de Altitud y Fatiga de Viaje

En Sudamérica, la ventaja de local no es una constante psicológica; es una variable fisiológica. Jugar en La Paz (3600m), Quito (2850m) o Bogotá (2640m) induce hipoxia y reduce el VO2max de los jugadores no aclimatados en un ~15-25%.

**Nuevas Features Propuestas:**

- `altitude_diff`: Diferencia de altitud entre el estadio local y el estadio de origen del visitante.
- `travel_strain`: Producto de la distancia recorrida y la diferencia de altitud, ponderada por los días de descanso. Un viaje largo al nivel del mar es manejable; un viaje corto con un cambio de altitud de 3000m es devastador.
- **Interacción:** `altitude_diff * team_origin_sea_level`. El impacto es asimétrico; afecta mucho más a equipos del llano (Buenos Aires, Rosario) que visitan la altura, que viceversa (aunque el descenso también tiene efectos fisiológicos, son menos agudos en el corto plazo del partido).

#### 3.3.2 Índice de Continuidad de Plantilla (Roster Continuity)

Para combatir el ruido generado por el mercado de pases, se debe introducir una variable que cuantifique la validez de los datos históricos.

- **Feature:** `squad_continuity_score`. Porcentaje de los minutos jugados en los últimos 5 partidos que corresponden a jugadores que estuvieron presentes en el ciclo anterior (o semestre anterior).

- **Uso en XGBoost:** Esta variable actúa como una "compuerta". Si la continuidad es baja, el modelo (especialmente un árbol de decisión) puede aprender a reducir la importancia de las rolling features y aumentar el peso de las cuotas del mercado o los ratings base del club.

## 4. Reingeniería del Objetivo: Modelado de Residuos de Mercado

El hallazgo de que el mercado supera al modelo por un margen significativo (Brier 0.6348 vs 0.6625) dicta un cambio de paradigma. Intentar predecir el resultado del partido independientemente del mercado ("Beat the bookie from scratch") es una estrategia ineficiente cuando el mercado es fuerte. La estrategia óptima es predecir el error del mercado.

### 4.1 Definición del Target Residual

En lugar de entrenar el modelo para predecir $Y \in \{0, 1\}$ (Gana/No Gana), se entrena para predecir la discrepancia entre el resultado y la probabilidad implícita del mercado.

**Nuevo Target ($R$):**

$$R = Y_{outcome} - P_{market\_closing\_devigged}$$

- Si el equipo local gana ($Y=1$) y el mercado le daba un 60% ($P=0.6$), el target es $+0.4$.
- Si pierde ($Y=0$), el target es $-0.6$.

Al entrenar XGBoost sobre $R$, el modelo aprende exclusivamente los patrones que el mercado no ha descontado. Si el mercado ya ha incorporado el xG, las lesiones y la localía, el modelo no encontrará señal en esas variables para predecir $R$ y su salida será cercana a 0 (lo que implica "No Bet"). Sin embargo, si el modelo detecta que en situaciones de alta altitud o viajes largos el mercado sistemáticamente sobreestima al favorito, aprenderá a predecir un residuo positivo o negativo, señalando una oportunidad de valor (Alpha).

### 4.2 El Valor de la Línea de Cierre (CLV) como Proxy de la Verdad

Dado que los resultados de fútbol son eventos binarios de alta varianza (un tiro al poste cambia el resultado pero no la probabilidad subyacente), utilizar el resultado real como única fuente de verdad es ruidoso. Una estrategia avanzada utilizada por sindicatos es entrenar modelos "Sombra" (Shadow Models) que intentan predecir la Línea de Cierre (Closing Line) basándose en la información disponible en la apertura (Opening Line).

La Línea de Cierre de Pinnacle se considera la estimación más eficiente y precisa de la probabilidad verdadera del evento. Entrenar un modelo para predecir hacia dónde se moverá la línea (steam chasing) puede ser más rentable en mercados líquidos que intentar predecir el partido en sí. Si el modelo v1.0.2 puede predecir que la cuota de River Plate bajará de 2.10 a 1.90, existe una ventaja matemática clara independientemente del resultado final del partido.

## 5. Arquitectura de Modelado: Etiquetas Suaves y Probabilidad

La naturaleza binaria del target actual (0 o 1) desperdicia la riqueza de la información de xG. Un equipo que empata 1-1 habiendo generado 3.5 xG vs 0.2 xG recibe el mismo "label" (empate) que uno que empata 1-1 en un partido parejo. Esto confunde al modelo, enseñándole que ambos procesos son iguales.

### 5.1 Entrenamiento con Soft Labels (Probabilistic Targets)

Se recomienda adoptar un esquema de Soft Classification o Regresión Logística sobre Probabilidades. En lugar de usar el resultado final como target, se utiliza la "Probabilidad de Victoria basada en xG" (xG-based Win Probability) calculada mediante simulación de Monte Carlo de los tiros del partido.

**Implementación en XGBoost:**

Dado que XGBoost requiere instancias discretas para clasificación, se puede utilizar la técnica de duplicación de filas con ponderación (sample weights):

Para un partido donde la simulación de xG dice que el Equipo A debió ganar el 70% de las veces, empatar el 20% y perder el 10%:

- Se crean 3 filas para el mismo partido con los mismos features.
- Fila 1: Target = Win, Weight = 0.70
- Fila 2: Target = Draw, Weight = 0.20
- Fila 3: Target = Loss, Weight = 0.10

Esto obliga al modelo a aprender la distribución de probabilidad subyacente a la calidad del juego, filtrando la aleatoriedad del resultado final. Esto es crucial en ligas de baja anotación como la argentina, donde un solo error arbitral o un rebote fortuito puede invertir el resultado binario pero no la "verdad" estadística del rendimiento.

## 6. Hoja de Ruta Estratégica para v1.0.2

Basado en este diagnóstico, se estructura la siguiente hoja de ruta priorizada para la evolución del modelo.

### Fase 1: Pivote "Market-First" (Prioridad Crítica)

**Acción:** Reorientar el target del modelo para predecir Residuos de Mercado ($Y - P_{market}$) en lugar de resultados brutos.

**Justificación:** El mercado es la "feature" más potente disponible en el entorno argentino. Utilizarlo como ancla ($\alpha=1.0$ en el backtest) y modelar solo la corrección reduce drásticamente la varianza y el error Brier.

**Técnica:** XGBoost con objetivo de regresión (reg:squarederror) sobre el residuo, usando las cuotas de apertura y las métricas fundamentales como inputs.

### Fase 2: Purificación de Señal (Ajuste por Oponente)

**Acción:** Sustituir las medias móviles crudas por Ratings Ridge o Elo-xG.

**Justificación:** Elimina el sesgo de calendario, crucial en torneos cortos. Permite comparar el rendimiento de un equipo contra rivales de distinta jerarquía de forma estandarizada.

**Técnica:** Implementar un pipeline de pre-procesamiento que calcule los coeficientes Ridge semanalmente (sin data leakage) y los inyecte como features de estado del equipo.

### Fase 3: Contextualización Física y Estructural

**Acción:** Integrar variables de Altitud, Distancia de Viaje y Continuidad de Plantilla.

**Justificación:** Captura determinantes físicos y estructurales de las ligas LATAM que los modelos genéricos europeos ignoran.

**Técnica:** Cruzar datos de estadios con bases de datos de elevación y calcular distancias geodésicas. Crear métrica de retención de minutos de jugadores para ponderar la relevancia histórica.

### Fase 4: Entrenamiento Probabilístico (Soft Targets)

**Acción:** Entrenar un sub-modelo experimental utilizando probabilidades derivadas de xG (simuladas) como target.

**Justificación:** Reduce el ruido aleatorio de los resultados, permitiendo que el modelo converja más rápido hacia la "calidad de juego" real.

## 7. Apéndice Matemático y Recetas de Implementación

### 7.1 Cálculo de Ratings Ajustados por Oponente (Ridge)

Para implementar el ajuste descrito en la sección 3.1:

**Construcción de la Matriz Dispersa:**

Sea $M$ el número de partidos y $T$ el número de equipos. Construir una matriz $X$ de dimensiones $M \times T$.

Para cada partido $i$ entre el Local $A$ y el Visitante $B$:

$$X_{i,A} = 1, \quad X_{i,B} = -1, \quad X_{i,k} = 0 \text{ para } k \neq A,B$$

**Vector Objetivo:**

$y_i = xG\_Pro_i - xG\_Contra_i$ (Diferencial de xG del partido $i$).

**Resolución Ridge:**

En Python (sklearn):

```python
from sklearn.linear_model import Ridge

# alpha es el parámetro de regularización lambda
clf = Ridge(alpha=10.0, fit_intercept=True) 
clf.fit(X_train, y_train)
team_ratings = clf.coef_
home_advantage = clf.intercept_
```

Los coeficientes resultantes son los ratings ajustados ofensivos/defensivos netos de cada equipo, limpios de la influencia del oponente.

### 7.2 Función Objetivo del Residuo de Mercado

Para el modelo "Shadow" de la sección 4.1:

**Conversión de Cuotas:**

$P_{implied} = \frac{1}{Odds_{cierre}}$

Normalizar para eliminar el margen (vig): $P_{real} = \frac{P_{implied}}{\sum P_{implied}}$.

**Cálculo del Residuo:**

- Si Gana Local: $R = 1 - P_{real\_local}$
- Si Empata/Pierde: $R = 0 - P_{real\_local}$

**Entrenamiento:**

Entrenar XGBoost para minimizar el MSE de $R$. La predicción del modelo $\hat{R}$ se suma a la probabilidad del mercado para obtener la probabilidad final ajustada:

$$P_{final} = P_{mercado} + \hat{R}$$

Si $\hat{R} \approx 0$, no hay apuesta. Si $\hat{R}$ es significativamente positivo, indica valor sobre el mercado.

## 8. Conclusión

La "dificultad para encontrar señal" reportada no es un callejón sin salida, sino una señal de que el modelo ha alcanzado el límite de la aproximación descriptiva en un entorno de alta eficiencia y ruido. El xG no está roto; está mal contextualizado. La liga argentina no es impredecible; es estructuralmente diferente.

La estrategia ganadora para v1.0.2 no consiste en construir un mejor simulador de fútbol desde cero, sino en construir un corrector de mercado sofisticado. Al aceptar la eficiencia del mercado como punto de partida y utilizar técnicas de ingeniería de características avanzadas (ajuste por oponente, soft labels, factores ambientales) para identificar las brechas específicas donde el mercado falla, el sistema puede transitar de un modelo que "pierde consistentemente" a uno que opera quirúrgicamente en los márgenes de rentabilidad. Este enfoque de "Residuos y Contexto" es el estándar de oro para la modelización en mercados maduros y volátiles.
