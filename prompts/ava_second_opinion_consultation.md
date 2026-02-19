# Consulta: Segunda Opinión — Proyecto de Videoanálisis de Fútbol con IA

## Contexto del solicitante

Soy el dueño de **Bon Jogo**, un sistema de predicciones de resultados de fútbol (1X2) con ML (XGBoost) y narrativas LLM, en producción con ~50+ features que incluyen: promedios rodantes de goles/tiros/corners, xG de Understat, ratings de jugadores de Sofascore, weather/bio-adaptability, talent delta, y más. El sistema lleva 6 semanas de desarrollo activo y se acerca al rendimiento del mercado (bookmakers).

Estoy diseñando un **proyecto complementario**: un sistema de videoanálisis de fútbol asistido por IA que procese partidos grabados de transmisiones de TV (broadcast) y extraiga métricas tácticas que mis proveedores actuales (API-Football, Sofascore) no proporcionan.

**Este proyecto está en fase de ideación y estructuración. No hay código aún.** Necesito una revisión crítica de lo que hemos definido hasta ahora.

---

## Descripción del proyecto

### Objetivo
Desarrollar un sistema de visión por computadora que procese video de partidos de fútbol (fuente: transmisiones IPTV broadcast) y extraiga métricas tácticas granulares para alimentar el modelo de predicciones de Bon Jogo.

### Alcance
- **Input**: Video broadcast de IPTV (cámaras de TV que siguen al balón, ~30-50% del campo visible en cada frame, con cortes de cámara, replays y close-ups).
- **Output**: Métricas tácticas estructuradas que se almacenan en base de datos PostgreSQL (Railway) y se integran como features del modelo ML de Bon Jogo.
- **Usuario final**: El propio dueño del proyecto (no es producto para reventa, al menos inicialmente).
- **Procesamiento**: Local (hardware propio con GPU), no en la nube. Procesamiento post-partido (no en vivo), más rápido que tiempo real, con capacidad de paralelismo multi-partido.
- **Cobertura objetivo**: ~26 ligas de fútbol (~9,900 partidos/temporada).

### Infraestructura disponible
- Conexión simétrica de 10 Gbps
- Presupuesto para hardware de cómputo pesado (GPU local)
- Presupuesto para almacenamiento de video
- Base de datos PostgreSQL en Railway (ya en producción)

### Limitación aceptada
El video broadcast NO muestra el campo completo. Se acepta trabajar con lo visible en cámara. La premisa operativa es: "la cámara sigue la acción, y la acción es lo relevante para el análisis" — cuando el local ataca, lo que importa es el ataque local vs la defensa visitante, y eso es lo que está en cámara.

---

## Mapa de fases propuesto

### FASE 0 — Cimientos
| Componente | Descripción |
|---|---|
| Homografía del campo | Detectar líneas/marcaciones del terreno, calcular matriz de transformación pixel → coordenadas reales (metros). Sin esto no hay distancias ni posición espacial. |
| Detección de jugadores | Bounding boxes de cada jugador visible en el frame. |
| Clasificación por equipo | Distinguir Home / Away / Árbitro por color de camiseta. |
| Detección y tracking del balón | Localizar el balón frame a frame, manejar oclusiones. |
| Identificación por dorsal | Leer número de camiseta para identificar jugadores individuales. |
| Pipeline de video | Lectura de video, extracción de frames, almacenamiento de resultados. |

### FASE 1 — Métricas tácticas core
| Métrica | Descripción |
|---|---|
| Posesión territorial | Porcentaje de posesión por tercios del campo (defensivo, medio, ofensivo). |
| Pressing: altura e intensidad | Altura promedio de la línea de presión, número de jugadores involucrados, duración de secuencias de pressing. |
| Compactación defensiva | Distancia entre líneas defensivas (defensa-mediocampo, mediocampo-delantera). |
| Velocidad de transición | Tiempo (segundos) desde recuperación de balón hasta llegada al área rival. |
| Patrones de build-up | Clasificación de salida: corta vs larga, por banda vs por el centro, frecuencia de progresión por zonas. |
| Distancias entre entidades | Distancias en metros entre jugadores, entre jugador y balón, entre líneas. |

### FASE 2 — Métricas avanzadas + integración
| Métrica | Descripción |
|---|---|
| xG propio | Expected goals calculado con contexto visual: presión defensiva al momento del tiro, ángulo corporal, equilibrio del tirador. |
| Duelos con contexto espacial | Resultado de 1v1 con ubicación en el campo, identidad de jugadores, contexto táctico. |
| Estructura en set pieces | Organización en corners/tiros libres: marcaje zonal vs hombre, posiciones. |
| Offside probabilístico | Flag de "probable fuera de juego" basado en posiciones relativas (no exacto al centímetro). |
| Pipeline completo | Video → Extracción → Métricas → PostgreSQL (Railway) → Features ML → Predicciones Bon Jogo. |
| Procesamiento paralelo | Múltiples partidos procesándose simultáneamente. |

### FASE 3 — Experimental
| Feature | Descripción |
|---|---|
| Foul vs simulación | Probabilidad de falta real vs diving basado en velocidad, contacto y mecánica de caída. |
| Análisis de audio | Presión de la hinchada medida por volumen/patrones de audio. |
| Inferencia off-camera | Estimar posiciones de jugadores no visibles en el frame. |

---

## Métricas que Bon Jogo ya tiene (para evitar duplicación)

**De API-Football**: Goles, tiros (al arco, fuera, bloqueados, dentro/fuera del área), posesión %, corners, faltas, tarjetas, offsides, pases (total, precisos, %), atajadas, xG (parcial).

**De Sofascore**: Ratings de jugadores (pre y post-match), formaciones, lineups, big chances, pass accuracy. (Nota: las stats de partido de Sofascore se capturan pero aún no se integran al modelo ML).

**De Understat**: xG rolling, xPTS, justice regression.

**Calculadas internamente**: Rolling averages con time-decay, goal diff, rest days, strength gap, weather/bio (thermal shock, circadian disruption), talent delta, xi_weighted/percentiles/weaklink.

---

## Lo que necesito de ti

Actúa como un **arquitecto senior de sistemas de visión por computadora especializado en sports analytics**. Revisa críticamente todo lo anterior y respóndeme:

1. **¿Qué estamos omitiendo que sea crítico?** — Componentes técnicos, decisiones de arquitectura, o consideraciones fundamentales que no hemos mencionado y que podrían hacer fracasar el proyecto si no se abordan desde el inicio.

2. **¿La Fase 0 está completa?** — ¿Faltan componentes en los cimientos que deberían estar ahí antes de construir métricas tácticas?

3. **¿El orden de las fases tiene sentido?** — ¿Hay dependencias que no estamos viendo? ¿Algo de Fase 1 debería estar en Fase 0, o viceversa?

4. **¿Cuáles son los riesgos técnicos más altos?** — De todo lo listado, ¿qué tiene mayor probabilidad de fracasar o de requerir significativamente más esfuerzo del esperado?

5. **¿Qué stack tecnológico recomiendas?** — Modelos base (YOLO versión, pose estimation, OCR para dorsales), frameworks, pipelines de video. Sé específico con versiones y alternativas.

6. **¿El hardware local es realista?** — Para procesar ~9,900 partidos/temporada con el pipeline descrito, ¿qué specs mínimas de GPU/CPU/RAM/almacenamiento recomiendas?

7. **¿Hay datasets públicos o benchmarks** que deberíamos conocer para entrenar/evaluar el sistema? (ej: SoccerNet, StatsBomb open data, etc.)

8. **¿Qué métricas de calidad** deberíamos definir para validar que el sistema funciona? (ej: mAP para detección, error de homografía en metros, accuracy de clasificación de equipo, etc.)

9. **¿Hay algo en nuestra premisa que sea fundamentalmente incorrecto?** — Si crees que algún supuesto es erróneo, dilo directamente.

10. **¿Qué nos recomendarías como MVP mínimo absoluto** — el subset más pequeño que demuestre valor y justifique continuar invirtiendo?

---

## Restricciones para tu respuesta

- Sé directo y crítico. Preferimos archivar el proyecto a construir sobre bases débiles.
- No asumas que tenemos experiencia previa en computer vision. Si algo es más complejo de lo que parece, dilo.
- Cuando menciones tecnologías, da nombres específicos con versiones (no "usa YOLO", sino "YOLOv8x o YOLO11 de Ultralytics por X razón").
- Si hay trade-offs importantes (velocidad vs precisión, generalización vs especialización), explícalos.
- Responde en español.
