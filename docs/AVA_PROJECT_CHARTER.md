# AVA — Auditor de Video Análisis
## Project Charter v1.0

**Proyecto**: Sistema de Videoanálisis de Fútbol Asistido por Inteligencia Artificial
**Codename**: AVA (Auditor de Video Análisis)
**Relación**: Herramienta complementaria de Bon Jogo
**Estado**: Fase de diseño — sin implementación
**Fecha**: 2026-02-15
**Owner**: David

---

## 1. Resumen Ejecutivo

AVA es un sistema de visión por computadora que procesa video de partidos de fútbol desde transmisiones broadcast (IPTV) y extrae métricas tácticas granulares que los proveedores actuales de Bon Jogo (API-Football, Sofascore, Understat) no proporcionan.

El objetivo no es reemplazar proveedores de datos desde el día uno, sino generar features tácticas propias — pressing, transiciones, posesión territorial, estructura defensiva — que enriquezcan el modelo de predicciones XGBoost de Bon Jogo y generen ventaja predictiva (alpha) sobre los bookmakers.

**Veredicto de viabilidad**: Viable con scope disciplinado. Requiere validación mediante MVP antes de inversión significativa en hardware.

**Criterio de decisión**: Si el MVP demuestra mejora medible en log-loss/Brier del modelo de Bon Jogo en backtest temporal, el proyecto escala. Si no, se archiva y se evalúa comprar datos de tracking a proveedores especializados.

---

## 2. Visión y Objetivo

### 2.1 Visión

Construir un sistema capaz de observar, comprender y estructurar la dinámica de un partido de fútbol a partir de video broadcast, actuando como un analista táctico automatizado que extrae información no disponible en ninguna API pública.

### 2.2 Objetivo medible

Generar al menos una feature táctica derivada de video que:
1. No exista en API-Football, Sofascore ni Understat
2. Pase validación PIT (Point-In-Time) — solo datos pre-kickoff del partido a predecir
3. Demuestre lift estadísticamente significativo en el modelo XGBoost de Bon Jogo (log-loss o Brier score) en backtest temporal out-of-sample

### 2.3 Lo que AVA NO es

- No es un producto para reventa (al menos inicialmente)
- No es procesamiento en tiempo real (post-partido, batch)
- No pretende tracking omnisciente del campo completo (trabaja con lo visible en cámara)
- No es un reemplazo de proveedores actuales en Fase 1 — es un complemento
- No es un sistema de VAR ni de arbitraje automatizado

---

## 3. Contexto: Integración con Bon Jogo

### 3.1 Estado actual de Bon Jogo

Bon Jogo es un sistema de predicciones 1X2 (Victoria Local, Empate, Victoria Visitante) con:
- **Modelo ML**: XGBoost multi-class classifier con ~50+ features
- **Proveedores de datos**: API-Football (stats, eventos, lineups), Sofascore (ratings, formaciones), Understat (xG, xPTS)
- **Features actuales**: Rolling averages (goles, tiros, corners), xG, weather/bio-adaptability, talent delta, xi_weighted (ratings por posición), justice regression
- **Gap identificado**: Las estadísticas de partido de Sofascore (posesión, big chances, pass accuracy) se capturan pero NO se consumen en el modelo ML

### 3.2 Qué aporta AVA que hoy no existe

| Categoría | Métrica de AVA | Equivalente actual | Gap |
|---|---|---|---|
| Pressing | Altura e intensidad de presión | No existe | Total |
| Transiciones | Velocidad recuperación → área rival | No existe | Total |
| Posesión cualitativa | Posesión por tercios del campo | Solo % global (Sofascore) | Parcial |
| Estructura defensiva | Densidad defensiva en zona del balón | No existe | Total |
| Identidad táctica | Field Tilt (centro de gravedad del equipo) | No existe | Total |
| Build-up | Patrones de salida (corta/larga, banda/centro) | No existe | Total |

### 3.3 Flujo de integración objetivo

```
Video IPTV → AVA (local) → Métricas tácticas → PostgreSQL local → Sync → Railway DB
                                                                            ↓
                                        Bon Jogo ML pipeline ← Features rolling pre-match (PIT-safe)
```

Las métricas de AVA se calculan post-partido y se agregan como rolling averages pre-match para el modelo de predicciones, respetando PIT compliance (solo datos de partidos anteriores al que se predice).

---

## 4. Constraints y Limitaciones Aceptadas

### 4.1 Input: Video broadcast (IPTV)

**Fuente**: Suscripción IPTV con cobertura de ~26 ligas.
**Resolución típica**: 1080p (variable por canal).
**Conexión**: 10 Gbps simétricos.

**Limitaciones inherentes del broadcast**:
- La cámara sigue al balón: solo ~30-50% del campo visible en cada frame
- Cortes de cámara: replays, close-ups, tomas de hinchada, grafismos
- Perspectiva angular variable (pan, tilt, zoom constante)
- Compresión de video (H.264/H.265) introduce artefactos

### 4.2 Camera-Operator Bias

La cámara no es un observador neutral. El director de TV decide qué mostrar, introduciendo un sesgo sistemático:
- Métricas macro-estructurales (compactación total, distancia entre líneas completas) **no son confiables** desde broadcast
- Métricas de densidad relativa (jugadores visibles en radio del balón, field tilt del bloque visible) **sí son viables**
- El sesgo es relativamente consistente dentro de una misma liga (misma productora), lo que permite al modelo ML aprender a ajustar implícitamente con volumen suficiente

**Decisión de diseño**: AVA priorizará métricas de densidad relativa y métricas de la zona de acción sobre métricas que requieran visión del campo completo.

### 4.3 Procesamiento

- **Post-partido** (no en vivo)
- **Local** (GPU propia, no cloud para producción — cloud solo para MVP)
- **Más rápido que tiempo real** (objetivo: 3-5x real-time)
- **Paralelizable** (múltiples partidos simultáneos)

### 4.4 Uso y legalidad

- Uso personal, no comercial (inicialmente)
- Responsabilidad legal del owner
- Si se escala a producto comercial, se requiere revisión legal de licencias de contenido broadcast

---

## 5. Arquitectura de Fases

### Fase -1: Data, Evaluación e Infraestructura

**Objetivo**: Crear las bases de datos, etiquetado y evaluación antes de escribir una línea de modelo.

| Componente | Descripción | Entregable |
|---|---|---|
| **Estrategia de etiquetado** | Definir guideline de anotación: qué se anota, cómo, criterios de calidad | Documento de guideline |
| **Setup de anotación** | Instalar y configurar CVAT (Computer Vision Annotation Tool) | Instancia CVAT funcional |
| **Ground truth dataset** | Anotar manualmente un subset (10-20 partidos): bounding boxes de jugadores, balón, líneas de campo, equipo, eventos | Dataset anotado en formato COCO/YOLO |
| **Métricas de calidad** | Definir targets por componente (ver Sección 8) | Tabla de KPIs |
| **Infraestructura MLOps** | DVC (versionado de datasets), MLflow (experimentos), MinIO o NAS (almacenamiento local) | Pipeline de tracking de experimentos |
| **Benchmark baseline** | Descargar SoccerNet, ejecutar baselines pre-entrenados, medir performance en tu video | Reporte de baseline |

**Criterio de salida**: Ground truth anotado, métricas definidas, infraestructura de experimentos operativa.

---

### Fase 0: Percepción Robusta

**Objetivo**: Construir la capa de percepción que convierte video crudo en datos estructurados confiables.

#### 0.1 Pipeline de Video

| Componente | Detalle |
|---|---|
| **Decodificación** | Decord (GPU-acelerada) o NVIDIA DeepStream SDK (NVDEC). NO usar `cv2.VideoCapture` — es un cuello de botella de I/O que asfixiará el CPU. |
| **Formato** | Input: H.264/H.265 desde IPTV. Frames: tensores GPU (evitar copias CPU↔GPU). |
| **Almacenamiento intermedio** | MinIO (S3-compatible) o NAS local para videos y resultados intermedios. PostgreSQL local para métricas, con sync periódico a Railway. |

#### 0.2 Segmentación de Transmisión

| Componente | Detalle |
|---|---|
| **Camera Switch Detection** | TransNetV2 — detecta hard cuts entre cámaras. Es el estándar de la industria para shot boundary detection. |
| **Shot Classification** | Clasificador ligero (fine-tuned sobre SoccerNet) que categoriza cada segmento: Game Cam (plano táctico amplio) / Close-up / Replay / Grafismo / Crowd. |
| **Política de procesamiento** | Solo procesar frames clasificados como "Game Cam". Descartar replays, close-ups y grafismos. Esto evita contar eventos repetidos y contaminar métricas. |

#### 0.3 Clock OCR / Sincronización Temporal

| Componente | Detalle |
|---|---|
| **OCR del marcador** | Detección y lectura del reloj gráfico de la transmisión (ej: "43:12") usando PaddleOCR PP-OCRv4 o EasyOCR. |
| **Mapeo temporal** | Construir una función `video_timestamp → match_minute` para sincronizar métricas visuales con eventos de API-Football. |
| **Manejo de gaps** | Detectar medio tiempo, pausas, interrupciones. El reloj no siempre es visible o legible. |

#### 0.4 Detección de Entidades

| Entidad | Modelo | Notas |
|---|---|---|
| **Jugadores** | YOLO11m exportado a TensorRT FP16 | Balance velocidad/precisión. Fine-tune con ground truth propio + SoccerNet. |
| **Árbitros** | Mismo modelo, clase separada | Se detectan igual que jugadores pero se clasifican aparte. |
| **Balón** | YOLO11n fine-tuned (detector separado) | Dedicado porque el balón es pequeño, rápido y frecuentemente ocluido. No mezclar con el detector de jugadores. |

#### 0.5 Tracking Multi-Objeto (MOT)

| Componente | Detalle |
|---|---|
| **Tracker primario** | BoT-SORT (nativo en Ultralytics) — superior a ByteTrack cuando la cámara se mueve rápido (broadcast). |
| **Asociación** | Hungarian algorithm para matching detecciones → tracks. |
| **Suavizado temporal** | Filtros de Kalman para suavizar posiciones frame a frame. Sin esto, las velocidades y distancias tendrán varianza enorme. |
| **Identidad** | Tracking anónimo en Fase 0: "ID_45 del Equipo A". La identidad real del jugador se resuelve en Fase 1.5. |

#### 0.6 Clasificación por Equipo

| Componente | Detalle |
|---|---|
| **Método primario** | Extraer bounding box del torso, convertir a espacio de color HSV, clustering con K-Means o HDBSCAN (K=3: Local, Visitante, Árbitro/Porteros). |
| **Método alternativo** | SigLIP (Google) para embeddings ligeros de apariencia + clustering. |
| **Validación** | Comparar con lineup de Sofascore (número de jugadores por equipo). |
| **Reto conocido** | Partidos con kits de color similar. El clasificador debe emitir confidence score y flagear partidos problemáticos. |

#### 0.7 Homografía Dinámica

| Componente | Detalle |
|---|---|
| **Detección de líneas** | Segmentación semántica de líneas del campo: SegFormer o UNet fine-tuned en SoccerNet Camera Calibration. |
| **Cálculo de homografía** | Ajuste robusto con RANSAC + OpenCV. Modelos basados en redes neuronales del SoccerNet Calibration Challenge (ej: TVCalib). |
| **Homografía dinámica** | Recalcular en cada frame (o cada N frames) porque la cámara hace pan/tilt/zoom constantemente. |
| **Suavizado temporal** | Filtros de Kalman sobre los parámetros de la homografía para evitar jitter (un jugador quieto no debe parecer que se mueve a 5 m/s por inestabilidad de la matriz). |
| **Score de confianza** | Cada frame emite un score de calidad de la homografía. Si no hay suficientes líneas visibles (ocluidas por jugadores, césped desgastado, sombras), el frame se marca como "homografía no confiable" y se descarta para métricas espaciales. |
| **Output** | Función `pixel(x,y) → campo(metros_x, metros_y)` por frame válido. |

#### 0.8 Confidence Scores (transversal)

Cada componente del pipeline debe emitir un score de confianza:

| Componente | Score | Uso |
|---|---|---|
| Detección | Confidence del bounding box | Descartar detecciones < threshold |
| Tracking | Estabilidad del track (frames consecutivos) | Filtrar tracks cortos/ruidosos |
| Clasificación equipo | Probabilidad de la asignación | Flagear asignaciones ambiguas |
| Homografía | Reprojection error del frame | Descartar frames con homografía inestable |
| Clock OCR | Confidence de lectura | Interpolar minutos cuando OCR falla |

**Sin scores de confianza, Bon Jogo ingerirá ruido puro (GIGO).** Esta es la diferencia entre un sistema útil y uno que contamina el modelo.

**Criterio de salida Fase 0**: mAP jugadores >0.85, homografía válida en >50% de frames, clasificación de equipo >95% accuracy, Clock OCR funcional.

---

### Fase 0.5: Action Spotting

**Objetivo**: Detectar cuándo ocurren eventos (pases, tiros, recuperaciones) para habilitar métricas temporales.

| Componente | Detalle |
|---|---|
| **Modelo temporal** | Action spotting sobre SoccerNet (temporal action detection). Detectar el instante exacto de: pase, tiro, recuperación, falta, corner. |
| **Anclaje cruzado** | Cruzar eventos detectados visualmente con eventos de API-Football para validación y para cubrir gaps donde el CV falla. |
| **Política de fallback** | Si el action spotting no detecta un evento pero API-Football sí lo reporta, anclar al timestamp del Clock OCR. |

**Dependencia**: Sin Action Spotting, las métricas de Fase 1 (velocidad de transición, posesión temporal) son imposibles de calcular.

**Criterio de salida**: mAP >0.70 para pases y tiros en el subset de validación.

---

### Fase 1: Métricas Tácticas (sin identidad individual)

**Objetivo**: Generar las primeras features tácticas útiles para Bon Jogo, sin depender de identidad de jugadores.

| Métrica | Definición | Cómo se calcula | Feature para Bon Jogo |
|---|---|---|---|
| **Field Tilt** | Centro de gravedad longitudinal del bloque visible de cada equipo | Promedio de coordenadas X (eje largo del campo) de todos los jugadores detectados del equipo, por ventanas de 5 minutos | Rolling average de field tilt por equipo (últimos N partidos) |
| **Posesión territorial** | Distribución de posesión por tercios del campo | Clasificar zona del balón (tercio defensivo / medio / ofensivo) en cada frame válido, acumular tiempo | % posesión en tercio ofensivo como rolling average |
| **Pressing height proxy** | Altura promedio del bloque del equipo que no tiene el balón | Promedio de coordenadas X de los jugadores del equipo sin posesión, solo en frames donde la posesión está en su mitad defensiva | Rolling average de pressing height |
| **Densidad defensiva** | Número de defensores en radio del balón | Contar jugadores del equipo defensor dentro de un radio de 15m del balón en frames de ataque rival | Promedio de densidad defensiva por equipo |
| **Velocidad de transición proxy** | Segundos entre recuperación y llegada a zona de ataque | Medir tiempo entre Action Spot "recuperación" y primer frame donde el centro de gravedad del equipo cruza al tercio ofensivo | Promedio de velocidad de transición por equipo |

**Integración con Bon Jogo**:
1. Métricas calculadas post-partido y almacenadas en PostgreSQL local
2. Sync a Railway DB
3. Features agregadas como rolling averages pre-match con time-decay, respetando PIT compliance
4. Inyectadas al modelo XGBoost junto con las features existentes

**GATE 1 — Prueba ácida (Go/No-Go)**:
- Ejecutar backtest temporal out-of-sample con las nuevas features
- Medir delta en log-loss y Brier score vs baseline sin features de AVA
- Evaluar SHAP feature importance
- **Si hay mejora consistente**: Go — escalar hardware, expandir ligas
- **Si no hay mejora**: No-Go — evaluar si el problema es la calidad del pipeline o si las métricas realmente no aportan. Si tras iteración sigue sin mejorar, archivar y considerar comprar datos de tracking

---

### Fase 1.5: Identidad de Jugadores

**Objetivo**: Asignar identidad real a los tracks anónimos de Fase 0.

| Componente | Detalle |
|---|---|
| **Re-ID por apariencia** | OSNet (torchreid) — genera embeddings de apariencia para cada track. Permite re-identificar jugadores tras cortes de cámara sin depender del dorsal. |
| **OCR de dorsales** | PaddleOCR PP-OCRv4 con voto temporal por track. No decidir por frame individual — acumular lecturas a lo largo del track y votar. |
| **Inicialización con lineups** | Usar lineup confirmada de Sofascore para restringir las identidades posibles (11 jugadores + suplentes). |
| **Fusión** | Combinar Re-ID + OCR + lineup para asignar identidad con score de confianza. |

**Métricas habilitadas**: Stats individuales por jugador (duelos ganados, pases intentados, distancia recorrida parcial).

**Criterio de salida**: Re-ID Rank-1 accuracy >0.90. OCR consolidada por track >0.80.

---

### Fase 2: Métricas Avanzadas

**Prerequisito**: Fase 1 validada (Gate 1 pasado) + Fase 1.5 operativa.

| Métrica | Descripción |
|---|---|
| **Duelos con contexto** | Resultado de 1v1 con ubicación en el campo, identidad de jugadores, contexto táctico (ataque/defensa/transición) |
| **Estructura en set pieces** | Organización en corners/tiros libres: marcaje zonal vs hombre, posiciones relativas |
| **Patrones de build-up** | Clasificación de secuencias de salida: corta/larga, banda/centro, progresión por zonas |
| **Offside probabilístico** | Flag de "probable fuera de juego" basado en posiciones relativas del penúltimo defensor vs atacante en el frame del pase. No exacto al centímetro — indicador probabilístico |
| **Procesamiento paralelo** | Pipeline optimizado para N partidos simultáneos en múltiples GPUs |

---

### Fase 3: Experimental

**Prerequisito**: Fases 0-2 estables y validadas.

| Feature | Descripción | Viabilidad |
|---|---|---|
| **xG propio visual** | Expected goals con contexto visual: presión defensiva, ángulo corporal, equilibrio del tirador. Requiere pose estimation 3D. | Baja desde broadcast. Requiere RTMPose/MMPose + homografía perfecta. |
| **Foul vs simulación** | Probabilidad de falta real vs diving basado en velocidad pre-contacto, mecánica de caída | Frontera de investigación. Precisión esperada baja. |
| **Análisis de audio** | Presión de la hinchada medida por volumen, patrones de audio, cánticos | Requiere separación de fuentes de audio. Investigación activa. |
| **Inferencia off-camera** | Estimar posiciones de jugadores no visibles usando patrones tácticos aprendidos | Investigación avanzada. Similar a lo que hace SkillCorner con años de R&D. |

---

## 6. Stack Tecnológico

### 6.1 Core

| Capa | Tecnología | Versión | Justificación |
|---|---|---|---|
| **Lenguaje** | Python | 3.12+ | Consistencia con Bon Jogo |
| **Framework ML** | PyTorch | 2.5+ | Ecosistema de CV dominante |
| **CUDA** | NVIDIA CUDA | 12.4+ | Requerido para TensorRT |
| **Inferencia** | TensorRT | FP16 | Obligatorio para throughput. Sin esto, los tiempos de inferencia arruinan la escalabilidad. |

### 6.2 Percepción

| Componente | Tecnología | Alternativa | Notas |
|---|---|---|---|
| **Decodificación video** | Decord (DMLC) | NVIDIA DeepStream (NVDEC) | No usar cv2.VideoCapture |
| **Detección jugadores** | YOLO11m (Ultralytics) | YOLO11l (más preciso, más lento) | Exportar a TensorRT FP16 |
| **Detección balón** | YOLO11n fine-tuned | — | Modelo dedicado, separado de jugadores |
| **Tracking MOT** | BoT-SORT (Ultralytics) | ByteTrack + ReID | Nativo en Ultralytics, robusto con cámara móvil |
| **Camera cuts** | TransNetV2 | — | Estándar de la industria |
| **Clasificación equipo** | HSV clustering + HDBSCAN | SigLIP embeddings | K=3: Local, Visitante, Árbitro |
| **Homografía** | SoccerNet Calibration (TVCalib) | SegFormer + RANSAC | Suavizado temporal con Kalman |
| **Clock OCR** | PaddleOCR PP-OCRv4 | EasyOCR | Lectura del marcador gráfico |
| **Re-ID** | OSNet (torchreid) | FastReID | Para Fase 1.5 |
| **OCR dorsales** | PaddleOCR + voto temporal | TrOCR | Para Fase 1.5 |
| **Pose estimation** | RTMPose (MMPose) | YOLO-Pose | Solo para Fase 3 (xG visual) |
| **Action spotting** | SoccerNet baselines | — | Para Fase 0.5 |

### 6.3 Infraestructura / MLOps

| Componente | Tecnología | Propósito |
|---|---|---|
| **Anotación** | CVAT | Ground truth labeling |
| **Versionado datasets** | DVC | Tracking de datasets versionados |
| **Experimentos** | MLflow | Tracking de runs, métricas, modelos |
| **Almacenamiento objetos** | MinIO | S3-compatible local para videos y resultados |
| **Base de datos local** | PostgreSQL | Métricas intermedias antes de sync a Railway |
| **Orquestación** | Prefect | Scheduling y pipeline management |
| **Pre-procesamiento video** | FFmpeg 7.x | Corte, conversión, extracción |

---

## 7. Hardware

### 7.1 Cálculo de throughput

**Datos base**:
- ~9,900 partidos/temporada (26 ligas)
- ~90 minutos efectivos por partido
- 5 FPS de inferencia (suficiente para métricas tácticas — no necesitas 25 FPS)
- Cada frame pasa por: decodificación → detección → tracking → homografía → clasificación

**A 5 FPS con pipeline optimizado (TensorRT FP16)**:

| GPU | FPS efectivo estimado | Tiempo por partido | Partidos/día (24h) |
|---|---|---|---|
| 1x RTX 4090 (24GB) | ~40-60 FPS (pipeline) | ~20-30 min | ~48-72 |
| 2x RTX 4090 | ~80-120 FPS | — | ~96-144 |
| 4x RTX 4090 | ~160-240 FPS | — | ~192-288 |

**Nota**: A 5 FPS de sampling, un partido de 90 min tiene 27,000 frames. A 50 FPS de procesamiento, eso es ~9 minutos por partido. Con 2 GPUs procesando en paralelo: ~144 partidos/día. Para 9,900 partidos: **~69 días**. Factible dentro de una temporada.

### 7.2 Especificaciones recomendadas

#### MVP (Cloud — RunPod/Vast.ai)
- 1x RTX 4090 o A6000 alquilada
- Costo estimado: ~$0.40-0.80/hora
- Para validar 50 partidos del MVP

#### Producción (Local)

| Componente | Mínimo | Recomendado |
|---|---|---|
| **GPU** | 2x RTX 4090 (24GB cada una) | 4x RTX 4090 o 2x RTX 6000 Ada (48GB) |
| **CPU** | AMD Threadripper Pro (32 cores, 64 lanes PCIe) | AMD Threadripper Pro (64 cores) |
| **RAM** | 128 GB DDR5 | 256 GB DDR5 |
| **NVMe (scratch)** | 4 TB Gen4 RAID 0 (lectura caliente) | 8-16 TB Gen5 RAID 0 |
| **NAS/HDD (archivo)** | 80 TB | 150-250 TB |
| **Red** | 10 GbE (ya disponible) | — |

#### Almacenamiento de video estimado
- 1080p H.264: ~3-5 GB por partido
- ~9,900 partidos: **30-50 TB por temporada** (solo video comprimido)
- Resultados intermedios (frames, features): +20-40 TB
- **Total: ~60-120 TB por temporada**

---

## 8. Métricas de Calidad

### 8.1 Por componente (definir desde Fase -1)

| Componente | Métrica | Target mínimo | Target ideal |
|---|---|---|---|
| **Detección jugadores** | mAP@0.5:0.95 | >0.85 | >0.90 |
| **Detección balón** | mAP@0.5:0.95 | >0.70 | >0.80 |
| **Clasificación equipo** | Accuracy | >0.95 | >0.98 |
| **Tracking** | HOTA | >0.75 | >0.85 |
| **Tracking** | IDF1 | >0.80 | >0.90 |
| **Tracking** | ID switches/minuto | <2 | <0.5 |
| **Homografía** | Reprojection error (RMSE, metros) | <2.0m | <1.0m |
| **Clock OCR** | Accuracy de lectura | >0.90 | >0.95 |
| **Camera shot classifier** | F1 (Game Cam vs otros) | >0.90 | >0.95 |
| **Action spotting** | mAP (pases, tiros) | >0.65 | >0.75 |
| **Re-ID** (Fase 1.5) | Rank-1 accuracy | >0.85 | >0.92 |
| **OCR dorsales** (Fase 1.5) | Accuracy consolidada por track | >0.75 | >0.85 |

### 8.2 Métricas de pipeline

| Métrica | Target |
|---|---|
| % frames con homografía válida por partido | >50% (mínimo), >70% (ideal) |
| Tasa de éxito de pipeline (partidos sin crash) | >95% |
| Throughput (partidos/día con hardware mínimo) | >48 |

### 8.3 Métricas de negocio (impacto en Bon Jogo)

| Métrica | Cómo medir | Criterio de éxito |
|---|---|---|
| **Log-loss delta** | Backtest temporal out-of-sample: baseline vs baseline+AVA features | Mejora >0 consistente |
| **Brier score delta** | Mismo backtest | Mejora >0 consistente |
| **SHAP feature importance** | Importancia relativa de features AVA en el modelo | Al menos 1 feature de AVA en top-20 |
| **Correlación con proveedores** | Posesión AVA vs posesión API-Football | r >0.80 (validación sanity check) |

---

## 9. MVP: Validación Go/No-Go

### 9.1 Scope

| Parámetro | Valor |
|---|---|
| **Partidos** | 50 partidos de 1 sola liga |
| **Liga** | Elegir una con cámaras consistentes (ej: Premier League, La Liga) |
| **Hardware** | 1x GPU alquilada en RunPod/Vast.ai |
| **Duración estimada** | 4-6 semanas |
| **Inversión estimada** | <$500 (cloud GPU + tiempo) |

### 9.2 Pipeline del MVP

```
TransNetV2 (filtrar replays)
    → YOLO11m (detección jugadores, NO balón)
    → BoT-SORT (tracking básico)
    → K-Means HSV (clasificación Local/Visitante)
    → Homografía pre-entrenada SoccerNet (básica)
    → Field Tilt por ventanas de 5 minutos
    → PostgreSQL → Railway → Feature en XGBoost
```

**Qué NO incluye el MVP**: Balón, dorsales, pose estimation, action spotting, Re-ID, Clock OCR completo, procesamiento paralelo.

### 9.3 Métrica del MVP

**Field Tilt**: Centro de gravedad longitudinal del bloque visible de jugadores por equipo, promediado en ventanas de 5 minutos.

Es la métrica más simple que demuestra que el pipeline visual genera información útil que no existe en ningún proveedor.

### 9.4 Prueba ácida

1. Calcular field tilt para 50 partidos
2. Generar rolling average de field tilt por equipo (últimos N partidos)
3. Inyectar como feature pre-match al modelo XGBoost de Bon Jogo
4. Ejecutar backtest temporal out-of-sample
5. Medir:
   - ¿Mejora el log-loss? ¿Mejora el Brier score?
   - ¿El feature aparece con importancia significativa en SHAP?
   - ¿La mejora es consistente o es ruido?

### 9.5 Decisión

| Resultado | Acción |
|---|---|
| Mejora consistente en métricas | **GO** — Comprar hardware, expandir a más ligas y métricas |
| Mejora marginal/inconsistente | Iterar: probar otras métricas (pressing height, posesión territorial). Si tras 2-3 iteraciones no hay lift → No-Go |
| Sin mejora | **NO-GO** — Archivar. Evaluar comprar datos de tracking de proveedor especializado (SkillCorner, Stats Perform) |

---

## 10. Riesgos y Mitigaciones

### 10.1 Riesgos técnicos

| # | Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|---|
| R1 | **Homografía inestable** (PTZ jitter, líneas ocluidas) | Alta | Crítico | Kalman temporal, score de confianza por frame, descartar frames malos, modelos SoccerNet pre-entrenados |
| R2 | **Tracking de balón** inconsistente (pequeño, rápido, ocluido) | Alta | Alto | Detector dedicado (YOLO11n fine-tuned), fallback a eventos API-Football, inferir posesión por proximidad |
| R3 | **Re-ID tras cortes de cámara** (tracker pierde IDs) | Alta | Alto | OSNet embeddings, lineup Sofascore como constraint, aceptar pérdida parcial en Fase 0 |
| R4 | **Generalización entre ligas** (calidad video, césped, kits, iluminación, overlays) | Media | Alto | Fine-tuning por cluster de ligas, métricas de drift por liga, empezar con 1 liga |
| R5 | **I/O bottleneck** (decodificación de miles de videos) | Media | Medio | Decord/NVDEC, NVMe RAID para scratch, pipeline por etapas |
| R6 | **OCR de dorsales** no funciona en broadcast | Alta | Medio | Pospuesto a Fase 1.5, no es bloqueante para métricas por equipo |

### 10.2 Riesgos de producto

| # | Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|---|
| R7 | **Features de CV no mejoran predicciones** (más granularidad ≠ mejor 1X2) | Media | Crítico | MVP como gate. No invertir antes de validar lift real. |
| R8 | **Sobreajuste** por agregar muchas features ruidosas | Media | Alto | Ablation studies, feature selection riguroso, backtest temporal estricto |
| R9 | **Escala no alcanzable** con hardware local | Baja | Alto | Empezar con subset, cloud para picos, optimizar pipeline agresivamente |

### 10.3 Riesgos operacionales

| # | Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|---|
| R10 | **Legal/copyright** de video broadcast | Baja (uso personal) | Alto (si comercializa) | Uso personal. Revisión legal antes de cualquier comercialización. |
| R11 | **IPTV inestable** (cortes, calidad variable) | Baja | Bajo | Grabación redundante, validación de calidad post-captura |

---

## 11. Datasets y Benchmarks

### 11.1 Datasets esenciales

| Dataset | Contenido | Uso en AVA |
|---|---|---|
| **SoccerNet** (soccernet.info) | 500+ partidos completos, anotaciones ricas (acciones, calibración, tracking, Re-ID) | Benchmark principal. Training de homografía, action spotting, baseline tracking. |
| **SoccerNet Camera Calibration** | Ground truth de calibración de cámaras broadcast | Entrenar/evaluar homografía |
| **SportsMOT** | Tracking multi-objeto en deportes | Benchmark de tracking |
| **StatsBomb 360** (open data) | Datos de tracking profesional (limitado) | Validación de métricas tácticas vs ground truth |
| **Ground truth propio** | 10-20 partidos anotados manualmente en CVAT | Fine-tuning y evaluación en tu dominio específico (IPTV) |

### 11.2 Papers de referencia

| Paper | Tema | Relevancia |
|---|---|---|
| SoccerNet: A Scalable Dataset for Action Spotting | Event detection | Fase 0.5 |
| Camera Calibration and Player Localization in SoccerNet | Homografía broadcast | Fase 0 |
| Deep Learning for Sports Analytics: A Survey (2023) | Panorama completo | Contexto general |
| Top-view Trajectory Estimation (CVPR 2023) | Transformación de coordenadas | Fase 0 |

---

## 12. Infraestructura MLOps

### 12.1 Versionado y trazabilidad

```
Video (MinIO/NAS)
    ↓
Dataset etiquetado (DVC - versionado)
    ↓
Experimento (MLflow - tracking de métricas, parámetros, artefactos)
    ↓
Modelo entrenado (MLflow model registry)
    ↓
Métricas extraídas (PostgreSQL local)
    ↓
Sync → Railway DB → Bon Jogo ML pipeline
```

### 12.2 Monitoreo de drift

| Tipo de drift | Qué monitorear | Acción |
|---|---|---|
| **Drift de cámara** (nueva liga, nuevo broadcaster) | Métricas de confianza por liga, reprojection error | Re-calibrar homografía, fine-tune detector |
| **Drift de modelo** (degradación de detección) | mAP sobre subset de validación periódico | Re-entrenar con nuevos datos |
| **Drift de feature** (feature de AVA pierde importancia en XGBoost) | SHAP monitoring | Investigar causa, ajustar o eliminar feature |

---

## 13. Timeline Estimado

| Fase | Duración estimada | Prerequisito |
|---|---|---|
| **Fase -1**: Data & infraestructura | 2-3 semanas | Ninguno |
| **MVP**: 50 partidos, 1 liga, Field Tilt | 4-6 semanas | Fase -1 |
| **Gate Go/No-Go** | 1 semana | MVP completado |
| **Fase 0**: Percepción robusta (completa) | 6-8 semanas | Gate aprobado |
| **Fase 0.5**: Action Spotting | 3-4 semanas | Fase 0 |
| **Fase 1**: Métricas tácticas | 4-6 semanas | Fase 0.5 |
| **Gate 1**: Validación lift en Bon Jogo | 1-2 semanas | Fase 1 |
| **Fase 1.5**: Identidad de jugadores | 4-6 semanas | Gate 1 |
| **Fase 2**: Métricas avanzadas | 8-12 semanas | Fase 1.5 |
| **Fase 3**: Experimental | Indefinido | Fase 2 estable |

**Total hasta Gate Go/No-Go**: ~7-10 semanas
**Total hasta Fase 1 validada**: ~5-6 meses (si todo va bien)

---

## 14. Decision Gates

```
Fase -1 → MVP
         ↓
    ¿Field Tilt mejora XGBoost?
         ├── SÍ → Comprar hardware, Fase 0 completa
         └── NO → Iterar 2-3 métricas más
                    ├── Mejora → Go
                    └── No mejora → ARCHIVAR
                                    └── Evaluar comprar datos de SkillCorner/Stats Perform

Fase 0 → Fase 0.5 → Fase 1
         ↓
    GATE 1: ¿Features tácticas mejoran Bon Jogo?
         ├── SÍ → Fase 1.5 + Fase 2
         └── NO → Quedarse en features por equipo, no escalar

Fase 1.5
    ¿Re-ID funciona (>0.90 accuracy)?
         ├── SÍ → Fase 2 con métricas individuales
         └── NO → Quedarse en métricas por equipo
```

---

## 15. Estrategia de salida

Si en cualquier gate el proyecto no demuestra valor:

1. **Archivar** el desarrollo de CV propio
2. **Evaluar** compra de datos de tracking a proveedores: SkillCorner (broadcast tracking), Stats Perform, Second Spectrum
3. **Reutilizar** la infraestructura de features de Bon Jogo — las features se diseñan igual (rolling averages PIT-safe), solo cambia la fuente de datos
4. **Preservar** el conocimiento adquirido sobre CV deportivo como activo intelectual

---

## Anexo A: Glosario

| Término | Definición |
|---|---|
| **Field Tilt** | Centro de gravedad longitudinal del bloque de jugadores de un equipo. Indica dominio territorial. |
| **Homografía** | Transformación matemática que mapea coordenadas de pixel (video) a coordenadas reales (metros en el campo). |
| **MOT** | Multi-Object Tracking — seguimiento de múltiples objetos (jugadores) a través de frames de video. |
| **Re-ID** | Re-Identification — re-identificar un jugador tras oclusión o corte de cámara. |
| **PIT** | Point-In-Time — principio que garantiza que solo se usan datos disponibles antes del evento a predecir. |
| **GIGO** | Garbage In, Garbage Out — datos de mala calidad producen resultados de mala calidad. |
| **PTZ** | Pan-Tilt-Zoom — movimientos de la cámara broadcast. |
| **Camera-Operator Bias** | Sesgo introducido por las decisiones del director de TV sobre qué mostrar. |
| **Action Spotting** | Detección temporal del instante exacto en que ocurre un evento deportivo. |
| **SHAP** | SHapley Additive exPlanations — método para explicar la importancia de cada feature en el modelo. |

---

## Anexo B: Referencias

1. SoccerNet — https://www.soccer-net.org/
2. Ultralytics YOLO11 — https://docs.ultralytics.com/
3. TransNetV2 — https://github.com/soCzech/TransNetV2
4. BoT-SORT — https://github.com/NirAharon/BoT-SORT
5. OSNet (torchreid) — https://github.com/KaiyangZhou/deep-person-reid
6. PaddleOCR — https://github.com/PaddlePaddle/PaddleOCR
7. RTMPose (MMPose) — https://github.com/open-mmlab/mmpose
8. CVAT — https://github.com/cvat-ai/cvat
9. DVC — https://dvc.org/
10. MLflow — https://mlflow.org/
11. Decord — https://github.com/dmlc/decord
12. MinIO — https://min.io/

---

*Documento generado por AVA (Auditor de Video Análisis) — v1.0 — 2026-02-15*
*Basado en: visión del owner, análisis AVA, y 3 revisiones técnicas independientes.*
