# FutbolStats — SCRAPING_ARCHITECTURE (Resilient & Observed)

Este documento especifica el **Sistema de Scraping Resiliente y Observado** (satélite) que alimenta directamente la arquitectura SOTA definida en `docs/ARCHITECTURE_SOTA.md` y el diccionario `docs/FEATURE_DICTIONARY_SOTA.md`.

**Objetivo**: extraer datos (principalmente Sofascore: XI/ratings/formación; opcionalmente otras fuentes web) de forma **anti-bloqueo**, **bandwidth-efficient** (IPRoyal 2GB) y **point-in-time correct** (captured_at < kickoff), entregando payloads estables para el backend.

**Filosofía**: *Resiliencia sobre velocidad*. El scraper prefiere:
- reintentos con rotación controlada
- degradación (imputación + flags) sobre crash
- trazabilidad end-to-end (observabilidad + auditoría)

---

## 0) Guardrails SOTA (no negociables)

1. **Point-in-time**: todo dato que se use en features pre-partido debe estar capturado con `captured_at < matches.date` (kickoff).
2. **Schema contract**: el scraper NO “inventa” campos. Produce un payload versionado y validado. Si cambia el esquema upstream → **Hard Fail** (alerta + skip).
3. **Bandwidth budget**: bloquear recursos y minimizar navegación. El patrón preferido es **Network Interception** para capturar JSONs internos (evitar parsear HTML).
4. **Idempotencia**: el servicio debe poder re-ejecutarse sin duplicar filas (unique keys + upsert).
5. **Seguridad**: credenciales de proxy/sesiones solo por env vars. Nunca loggear secrets (proxy user/pass, cookies).

---

## 1) MOTOR DE NAVEGACIÓN (The Stealth Engine)

### 1.1 Playwright: modo y perfil anti-detección

**Stack base**: Python + Playwright + `playwright-stealth` (y parches propios).

**Decisión de diseño**:
- Preferir **Chromium** estable, en **headless=false** si el objetivo lo requiere (algunos antibots lo detectan menos); si headless es necesario, usar “new headless” con mitigaciones.
- Ejecutar por defecto con un **perfil efímero** (context nuevo por sesión) para aislar fingerprints.

#### Config recomendada (conceptual)
- **User-Agent**: rotación desde un set curado (Chrome estable, desktop). Evitar UAs exóticos.
- **Locale/Timezone/Geo** coherentes con la IP (geo-targeting).
- **Viewport** realista (1366×768, 1920×1080) y deviceScaleFactor normal.
- **Navigator flags**:
  - `navigator.webdriver = undefined`
  - spoof de `plugins`, `languages`, `platform`
- **WebGL/Canvas**: usar stealth; no forzar valores “raros”.
- **TLS fingerprinting**: Playwright no permite control total del JA3; mitigación práctica:
  - usar residential proxies (IPRoyal) y mantener configuración estable
  - evitar patrones de request masivos
  - minimizar navegaciones completas

> Nota: para targets con detección agresiva, considerar **Playwright + CDP** (ver sección 5.3) para mayor control y coherencia de sesión.

### 1.2 Optimización de costos (IPRoyal): bloqueo de recursos

La regla: **solo** permitir el mínimo necesario para que la app cargue y dispare llamadas XHR/fetch que contienen el JSON objetivo.

#### Estrategia de routing
Bloquear por tipo y por extensión:
- **Abortar siempre**: `image`, `media`, `font`
- **Abortar normalmente**: `stylesheet` (CSS), salvo que el sitio no dispare XHR sin CSS (raro)
- **Permitir**:
  - `document` (solo la primera carga)
  - `script` (necesario para disparar XHR)
  - `xhr`, `fetch` (objetivo principal)

Bloqueo adicional por URL:
- trackers/ads (`doubleclick`, `google-analytics`, etc.)
- mapas/tiles, videos, CDNs de imágenes

**Bandwidth heuristics**:
- si el target tiene “single-page app”, cargar **una sola vez** el documento y luego navegar por rutas internas (si posible) sin recargar.
- reutilizar **cookies/session storage** durante una “sesión sticky” corta para extraer varios matches seguidos.

### 1.3 Estrategia de extracción: Network Interception (primaria)

#### Principio
No parsear HTML. En su lugar:
1. abrir página del evento/partido
2. interceptar respuestas de red y capturar JSONs de endpoints internos (p.ej. `api/v1/event/...`, `.../lineups`, `.../players`, `.../incidents`)
3. validar schema + persistir

#### Patrón técnico
Registrar listeners:
- `page.on("response", handler)` para capturar `response.url` + `status` + `json()`
- Filtrar por:
  - host permitido
  - paths esperados (whitelist)
  - content-type JSON

**Whitelist estricta**:
- reduce riesgo de capturar basura
- facilita observabilidad (cuántos endpoints se capturaron)

#### Anti-corruption layer (ACL)
El scraper no expone el JSON upstream “tal cual”. Normaliza a un schema interno versionado:
- `schema_version`: `"sofascore.event.v1"`
- `source`: `"sofascore"`
- `source_event_id`
- `captured_at` (UTC)
- `payload` (JSON bruto opcional, comprimido si se almacena)
- `normalized` (solo campos que el backend necesita para features SOTA)

---

## 2) SISTEMA INMUNE (Resiliencia y Contingencia)

### 2.1 Taxonomía de errores (y qué hacer)

#### Soft Fail (reintentar)
Condiciones típicas:
- timeout / navegación lenta
- 403/429 temporal
- `net::ERR_*` (proxy falló, DNS, handshake)
- “empty response” o JSON incompleto pero sin cambio de schema (p.ej. endpoint no disparó)

Acción:
- Retry con backoff + jitter
- rotar proxy/session (según tipo)
- si el fallo es por endpoint faltante, intentar ruta alternativa (si existe)

#### Hard Fail (no reintentar ciegamente)
Condiciones:
- **Cambio de schema** (faltan campos críticos o tipo distinto)
- 404 persistente del endpoint interno esperado (posible refactor upstream)
- challenge antibot persistente (captcha) que no se resuelve con rotación

Acción:
- registrar evento `schema_break` o `anti_bot_challenge`
- enviar alerta (Sentry)
- marcar match como `skipped` con razón, NO quemar proxies

### 2.2 Máquina de estados (state machine)

Estados por “scrape attempt”:
- `INIT` → `BROWSER_READY` → `NAVIGATING` → `INTERCEPTING`
- `CAPTURED` → `VALIDATED` → `PERSISTED` → `DONE`

Estados de error:
- `SOFT_FAIL` (retryable)
- `HARD_FAIL` (non-retry)
- `CIRCUIT_OPEN` (bloqueo global temporal)

Transiciones clave:
- `NAVIGATING` timeout → `SOFT_FAIL` → retry
- `INTERCEPTING` no capturó endpoints en T segundos → `SOFT_FAIL` (a veces el site no disparó XHR)
- `VALIDATED` schema mismatch → `HARD_FAIL`

### 2.3 Retry policy (resiliente, bandwidth-aware)

Recomendación:
- `max_attempts_per_match = 4`
- Backoff exponencial con jitter:
  - `sleep = base * 2^k + random(0, jitter)`
  - base=2s, jitter=1s
- Rotación:
  - en 403/429: rotar IP (nueva sesión/proxy)
  - en timeout: 1 retry con misma IP (puede ser latencia), luego rotar

### 2.4 Circuit Breaker (protege proxies)

Objetivo: evitar “quemar” tráfico en incidentes upstream.

Definición:
- Ventana móvil `W = 10 minutos`
- si `error_rate_soft + error_rate_hard >= X%` con mínimo `N` intentos, abrir circuito

Parámetros recomendados:
- `N_min = 30` intentos
- `X = 35%` (ajustar por target)
- `cooldown = 15 minutos`
- `half_open_probe = 3` intentos (solo a targets “canary”) para decidir si cerrar

Algoritmo:
1. calcular métricas en ventana W
2. si supera umbral: `OPEN` por cooldown
3. tras cooldown: `HALF_OPEN` → permitir pocos probes
4. si probes OK: `CLOSED`, si fallan: `OPEN` de nuevo

Eventos:
- `circuit_opened`, `circuit_half_open`, `circuit_closed` (telemetría)

---

## 3) GESTIÓN DE PROXIES (IPRoyal Integration)

### 3.1 Principios de rotación (sticky vs rotating)

**Recomendación base (resiliente + bandwidth)**:
- Usar **Sticky sessions** por “batch corto” (p.ej. 5–20 partidos) para reutilizar cookies y reducir overhead de handshakes.
- Rotar IP cuando:
  - 403/429
  - `anti_bot_challenge` detectado
  - >2 timeouts seguidos
  - consumo anómalo de KB por request (posible challenge heavy)

**Cuándo usar Rotating agresivo**:
- durante backfills grandes si el target rate-limitea por IP fuerte
- cuando se detecta patrón de bloqueos por stickiness

### 3.2 Geo-targeting
Configurar `country`/`region` según:
- target del sitio (algunos solo sirven bien en ciertas regiones)
- coherencia con `timezone/locale` del contexto del browser

Regla práctica:
- para Sofascore, normalmente EU/US funciona; preferir el geo con menor fricción (medir).

### 3.3 Configuración operativa (env vars)
No commitear credenciales. Variables sugeridas:
- `IPROYAL_HOST`
- `IPROYAL_PORT`
- `IPROYAL_USERNAME`
- `IPROYAL_PASSWORD`
- `IPROYAL_COUNTRY` (opcional)
- `SCRAPER_PROXY_MODE=sticky|rotating`
- `SCRAPER_STICKY_TTL_SECONDS=600`

### 3.4 Pool y asignación de sesiones

**ProxySessionManager** (concepto):
- mantiene N sesiones vivas (contexts) por destino
- cada sesión tiene:
  - `session_id`
  - `proxy_id` (hash)
  - `created_at`, `last_used_at`
  - counters: successes/fails, bytes_estimated

Política:
- LRU para reutilizar sesiones “buenas”
- cuarentena para sesiones con fallos repetidos

---

## 4) OBSERVABILIDAD Y TELEMETRÍA (The Nervous System)

### 4.1 Logging estructurado (JSON logs)

Formato obligatorio por evento (línea JSON):
- `ts` (UTC ISO)
- `service` (ej: `sofascore_scraper`)
- `run_id` (UUID por ejecución)
- `match_id` (interno FutbolStats, si se conoce)
- `source_event_id` (id proveedor)
- `attempt` (int)
- `state` (INIT/NAVIGATING/INTERCEPTING/VALIDATED/...)
- `proxy` (redactado): `proxy_pool`, `geo`, `session_id`, `ip_hash`
- `network`:
  - `bytes_in_est` (estimado)
  - `requests_total`, `requests_blocked`
  - `xhr_captured_count`
- `result`:
  - `status`: success/soft_fail/hard_fail/skipped
  - `error_type` (timeout/403/schema_break/...)
  - `http_status` (si aplica)
  - `duration_ms`
- `integrity`:
  - `schema_version`
  - `null_field_rate` (0..1)
  - `missing_required_fields` (lista corta)

**IMPORTANTE**: no loggear cookies, headers sensibles, proxy user/pass.

### 4.2 Métricas para Grafana (Prometheus o logs→Loki)

#### Success Rate
- `scraper_attempts_total{source}`
- `scraper_success_total{source}`
- `scraper_fail_total{source,type=soft|hard,reason}`
Derived:
- `success_rate = success/attempts`

#### Data Integrity Score
Para cada payload normalizado:
- `integrity_null_rate` (por match)
- `integrity_required_missing_total`

KPI recomendado:
- `data_integrity_score = 1 - null_field_rate`
Segmentar por:
- `endpoint_family` (lineups/ratings/formation)
- `league_id` si disponible

#### Proxy Consumption (KB)
Playwright no siempre expone bytes exactos por respuesta; aproximación robusta:
- preferir `Content-Length` de responses cuando exista
- fallback: `len(response.body())` para XHR capturados (solo para whitelisted JSONs)
- sumar bytes de requests permitidas (aprox)

Métricas:
- `scraper_bytes_in_est_total{source}`
- `scraper_requests_blocked_total{type}`
- `scraper_requests_total{type}`

### 4.3 Alertas (Sentry + Grafana)

Sentry (eventos):
- `schema_break` (HARD)
- `anti_bot_challenge` persistente (HARD)
- `circuit_opened` (WARN)
- `data_integrity_degraded` (WARN) si null_rate > umbral

Grafana (ejemplos de umbral):
- `success_rate < 0.75` por 30 min
- `circuit_opened` > 0 en 1h
- `avg(null_field_rate) > 0.20` en 1h para endpoints críticos
- `bytes_in_est_total` anómalo (challenge / heavy payload)

---

## 5) INTERFAZ DE SERVICIO (Integración con FutbolStats)

### 5.1 Firma de servicio (contrato Python)

El scraper se expone como servicio invocable por scheduler/ETL del backend. Debe operar **con match_id canónico** y devolver un payload normalizado.

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass
class ScrapeResult:
    source: str
    schema_version: str
    captured_at: datetime  # UTC
    source_event_id: str
    normalized: dict[str, Any]        # campos para features
    raw_payload: Optional[dict[str, Any]] = None  # opcional (puede omitirse para ahorrar storage)
    integrity: dict[str, Any] = None  # null_rate, missing_fields, etc.


class SofascoreScraperService:
    async def scrape_match(
        self,
        *,
        match_id: int,
        kickoff_utc: datetime,
        home_team: str,
        away_team: str,
        league_id: Optional[int] = None,
        season: Optional[int] = None,
        priority: str = "normal",  # normal|high (ej: close to kickoff)
    ) -> ScrapeResult:
        """Captura XI/ratings/formación via network interception. Debe garantizar captured_at < kickoff_utc."""
        ...
```

**Garantías**:
- si `datetime.utcnow() >= kickoff_utc`: el método debe rechazar o degradar (para evitar usar datos post-kickoff en pre-match).
- si faltan endpoints, debe incluir `xi_missing=1` y/o flags.

### 5.2 Integración con el backend (storage + feature groups)

La integración debe alinearse con los feature groups del diccionario SOTA:
- `xi/*` y `xi_dist/*` (Sofascore)

Propuesta de flujo:
1. Scheduler detecta partidos próximos (`status="NS"`) y dentro de ventana (ej: 0–72h).
2. Para partidos con lineup disponible (o cerca de kickoff), invoca `SofascoreScraperService.scrape_match`.
3. Persistir en tablas:
   - `match_external_refs` (link a `source_event_id`)
   - `match_sofascore_player` (XI starters + rating + position)
   - opcional `match_sofascore_lineup` (formation)
4. FeatureEngineer (o un `FeatureBuilder` nuevo) consume esas tablas para producir:
   - `xi_weighted_*`, percentiles, etc., con `captured_at` gating.

**Idempotencia**:
- unique keys sugeridas:
  - `(match_id, team_side, player_id_ext, captured_at_rounded)` o `(match_id, team_side, player_id_ext, snapshot_type)`
  - `(match_id, team_side, snapshot_type)` en lineup/formation

### 5.3 Mejora: CDP (Chrome DevTools Protocol) y cookies controladas

Cuándo preferir CDP:
- targets con fingerprinting más agresivo
- necesidad de exportar/importar cookies/session storage para stickiness
- necesidad de inspeccionar network en más detalle

Estrategia híbrida recomendada:
- Playwright estándar para la mayoría
- modo CDP activable por feature flag:
  - `SCRAPER_USE_CDP=true`

Cookies:
- persistir cookies por sesión sticky (solo memoria) para varios matchs en el batch
- si se detecta antibot, invalidar cookie jar y rotar IP

Justificación:
- mejora coherencia de sesión y reduce desafíos repetidos (menos consumo de GB).

---

## 6) Seguridad, compliance y minimización de riesgo

- No persistir HTML completo ni screenshots salvo en modo debug (y con redacción).
- Redactar URLs si contienen query params sensibles.
- Respetar robots/ToS según política interna (este documento es técnico; la decisión legal es externa).

---

## 7) Implementación recomendada (módulos)

Propuesta de estructura (orientativa):
```
app/scraper/
  __init__.py
  config.py                 # env vars + defaults
  stealth.py                # playwright-stealth + patches
  routing.py                # bloqueo de recursos + whitelist
  interceptors.py           # capturas de responses JSON
  schemas/                  # pydantic models (normalized payload) + versioning
  proxy.py                  # IPRoyal session manager
  circuit_breaker.py        # breaker global por source
  service_sofascore.py      # SofascoreScraperService
  telemetry.py              # logs + counters
```

**Validación de schema**:
- usar modelos Pydantic versionados por `schema_version`
- required fields definidos por endpoint family

---

## 8) Checklist operativo (antes de producción)

1. `success_rate` sostenido > 0.85 en 24h.
2. `null_field_rate` p50 < 0.10 para XI/ratings.
3. Circuit breaker funcionando (simular caída de target).
4. Proxy bandwidth estable y sin picos (challenge detection).
5. Point-in-time test: `captured_at < kickoff` garantizado para un set de matches.

