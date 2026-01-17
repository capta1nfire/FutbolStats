# Propuesta: Optimización de Live Score Updates

**Fecha**: 2026-01-17
**Autor**: Master (Claude)
**Estado**: Pendiente revisión Auditor
**Prioridad**: P1 (UX crítico para partidos en vivo)

---

## 1. Problema Actual

### Síntomas
- La app puede tardar **hasta 60 segundos** en reflejar un gol
- Apple Sports muestra goles en **<5 segundos**
- Usuarios perciben la app como "lenta" durante partidos en vivo

### Arquitectura Actual
```
iOS List View (PredictionsListView)
    └── Polling cada 60s → /predictions (endpoint pesado)
                               └── Query DB completa
                               └── ~2-5KB por partido
                               └── Incluye datos que no cambian (equipos, odds, narrativas)

iOS Detail View (MatchDetailView)
    └── Polling cada 30s → /predictions/{id}
    └── MatchCache overlay → List View (TTL 30s)
```

**Problema**: El polling de 60s en la lista es el cuello de botella. El usuario ve datos stale mientras navega la parrilla.

---

## 2. Solución Propuesta: Smart Short-Polling

### Arquitectura Nueva
```
iOS App
    └── LiveScoreManager (singleton)
            └── Polling cada 15s → /live-summary (nuevo endpoint)
                                       └── Solo partidos LIVE
                                       └── ~50 bytes por partido
                                       └── RAM cache (0ms DB)
            └── Actualiza MatchCache
            └── scenePhase-aware (pausa en background)

Backend
    └── GET /live-summary
            └── RAM dict actualizado por sync job
            └── Response: {matchId: {status, elapsed, elapsed_extra, home, away}}
            └── Latencia target: <50ms
```

### ¿Por qué NO WebSockets?
| Criterio | WebSockets | Short-Polling |
|----------|------------|---------------|
| Complejidad backend | Alta (conexiones persistentes) | Baja (stateless) |
| Escalabilidad Railway | Problemática (RAM por conexión) | Excelente |
| Manejo de reconexión | Complejo | Trivial (cada request es independiente) |
| Debugging | Difícil | Fácil (logs HTTP estándar) |
| Battery iOS | Similar con 15s interval | Similar |

**Conclusión**: Short-polling con endpoint optimizado es la mejor relación costo/beneficio.

---

## 3. Especificación Técnica

### 3.1 Backend: Endpoint `/live-summary`

```python
# Nuevo endpoint en app/routes/predictions.py

@router.get("/live-summary")
async def get_live_summary() -> dict:
    """
    Endpoint ultra-ligero para polling frecuente.
    Solo devuelve partidos en estado LIVE con campos mínimos.

    Response ~50 bytes/partido, latencia <50ms.
    """
    return live_score_cache.get_summary()
```

**Response Schema**:
```json
{
  "ts": 1705500000,
  "matches": {
    "12345": {"s": "2H", "e": 67, "ex": 0, "h": 2, "a": 1},
    "12346": {"s": "HT", "e": 45, "ex": 2, "h": 0, "a": 0}
  }
}
```

Campos compactos:
- `s`: status (1H, HT, 2H, ET, FT, etc.)
- `e`: elapsed minutes
- `ex`: elapsed_extra (injury time)
- `h`: home goals
- `a`: away goals

**Tamaño estimado**:
- 10 partidos live = ~500 bytes
- 50 partidos live = ~2.5KB

### 3.2 Backend: RAM Cache

```python
# Nuevo módulo app/services/live_cache.py

class LiveScoreCache:
    """
    RAM cache para /live-summary.
    Actualizado por el sync job cada 30s.
    """

    def __init__(self):
        self._data: dict[int, dict] = {}
        self._updated_at: float = 0

    def update(self, matches: list[Match]) -> None:
        """Llamado por sync job con partidos LIVE."""
        live_statuses = {"1H", "HT", "2H", "ET", "BT", "P", "LIVE"}
        self._data = {
            m.id: {
                "s": m.status,
                "e": m.elapsed,
                "ex": m.elapsed_extra or 0,
                "h": m.home_goals,
                "a": m.away_goals
            }
            for m in matches
            if m.status in live_statuses
        }
        self._updated_at = time.time()

    def get_summary(self) -> dict:
        return {
            "ts": int(self._updated_at),
            "matches": self._data
        }

# Singleton
live_score_cache = LiveScoreCache()
```

### 3.3 iOS: LiveScoreManager

```swift
// Nuevo archivo: Services/LiveScoreManager.swift

@MainActor
final class LiveScoreManager: ObservableObject {
    static let shared = LiveScoreManager()

    private var pollTask: Task<Void, Never>?
    private let pollInterval: TimeInterval = 15.0

    @Published private(set) var isPolling = false

    // MARK: - Lifecycle (scenePhase aware)

    func startPolling() {
        guard pollTask == nil else { return }
        isPolling = true

        pollTask = Task {
            while !Task.isCancelled {
                await fetchLiveSummary()
                try? await Task.sleep(for: .seconds(pollInterval))
            }
        }
    }

    func stopPolling() {
        pollTask?.cancel()
        pollTask = nil
        isPolling = false
    }

    // MARK: - Fetch

    private func fetchLiveSummary() async {
        guard let url = URL(string: "\(baseURL)/live-summary") else { return }

        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            let summary = try JSONDecoder().decode(LiveSummary.self, from: data)

            // Update MatchCache with fresh data
            for (matchId, score) in summary.matches {
                MatchCache.shared.update(
                    matchId: matchId,
                    status: score.status,
                    elapsed: score.elapsed,
                    elapsedExtra: score.elapsedExtra,
                    homeGoals: score.homeGoals,
                    awayGoals: score.awayGoals
                )
            }
        } catch {
            print("[LiveScoreManager] Error: \(error)")
        }
    }
}

// MARK: - Models

struct LiveSummary: Codable {
    let ts: Int
    let matches: [Int: LiveScore]

    struct LiveScore: Codable {
        let s: String      // status
        let e: Int         // elapsed
        let ex: Int        // elapsed_extra
        let h: Int         // home goals
        let a: Int         // away goals

        var status: String { s }
        var elapsed: Int { e }
        var elapsedExtra: Int { ex }
        var homeGoals: Int { h }
        var awayGoals: Int { a }
    }
}
```

### 3.4 iOS: Integración con scenePhase

```swift
// En FutbolStatsApp.swift o ContentView.swift

@main
struct FutbolStatsApp: App {
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ContentView()
                .onChange(of: scenePhase) { _, newPhase in
                    switch newPhase {
                    case .active:
                        LiveScoreManager.shared.startPolling()
                    case .inactive, .background:
                        LiveScoreManager.shared.stopPolling()
                    @unknown default:
                        break
                    }
                }
        }
    }
}
```

---

## 4. Fases de Implementación

### Fase 1: Backend Endpoint (Esfuerzo: Bajo)
- [ ] Crear `app/services/live_cache.py`
- [ ] Agregar endpoint `GET /live-summary`
- [ ] Actualizar sync job para poblar cache
- [ ] Tests unitarios
- [ ] Deploy a Railway

**Entregable**: Endpoint funcionando, verificable con curl.

### Fase 2: iOS LiveScoreManager (Esfuerzo: Medio)
- [ ] Crear `LiveScoreManager.swift`
- [ ] Crear modelos `LiveSummary`
- [ ] Integrar con `MatchCache`
- [ ] Agregar scenePhase handling
- [ ] Logging para debugging

**Entregable**: App polling cada 15s, visible en logs.

### Fase 3: Métricas y Observabilidad (Esfuerzo: Bajo)
- [ ] Prometheus metrics para `/live-summary`
  - `live_summary_requests_total`
  - `live_summary_latency_ms`
  - `live_summary_matches_count`
- [ ] Bloque en `ops.json` para monitoreo

**Entregable**: Dashboard con métricas de uso.

### Fase 4 (Futuro): Live Activities
- [ ] Investigar Apple Live Activities API
- [ ] Evaluar Push Notifications vs polling híbrido
- [ ] Diseñar UX de Dynamic Island

**Entregable**: Propuesta técnica separada.

---

## 5. Impacto Esperado

| Métrica | Actual | Target |
|---------|--------|--------|
| Latencia goles (list view) | ~60s | <15s |
| Latencia goles (detail view) | ~30s | <15s |
| Payload por poll | ~2-5KB | ~500 bytes |
| Requests/min (10 usuarios live) | 10 | 40 |
| Carga DB por request | Query completa | 0 (RAM cache) |

### Budget de Requests
- 15s interval = 4 requests/min/usuario
- 100 usuarios simultáneos en partido live = 400 req/min = 6.6 req/s
- Railway puede manejar esto sin problema (endpoint <50ms)

---

## 6. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| RAM cache stale | Baja | Medio | Sync job cada 30s, TTL en cache |
| Polling excesivo | Media | Bajo | scenePhase + rate limiting |
| Endpoint lento | Baja | Alto | No DB queries, solo RAM read |
| Inconsistencia cache/DB | Baja | Medio | Cache es overlay, no source of truth |

---

## 7. Preguntas para Auditor

1. **Intervalo de polling**: ¿15s es aceptable o preferimos 10s/20s?

2. **Autenticación**: ¿`/live-summary` requiere API key o puede ser público (es solo scores)?

3. **Redis vs RAM**: ¿Invertimos en Redis ahora o RAM dict es suficiente para MVP?

4. **Límite de partidos**: ¿Cap en número de partidos live en response (ej. top 50)?

5. **Prioridad Fase 4**: ¿Live Activities es P1 o P2 para roadmap?

---

## 8. Implementación Completada (Post-Auditor Review)

### Ajustes aplicados según feedback del Auditor:

1. **Auth**: `/live-summary` requiere `X-API-Key` header (no es público)
2. **Rate limiting**: 60 req/min por IP
3. **Cache**: Query DB ultra-ligera con cache L1 de 5s (no RAM dict puro)
4. **Gating iOS**: Solo pollea cuando hay live matches o usuario viendo live
5. **Backoff**: 60s cuando 0 live matches, 15s cuando hay live
6. **Match IDs**: Usa `match_id` interno (no external_id)
7. **Cap**: Máximo 50 partidos live en response

### Ejemplos curl autorizados

**Con 0 partidos live:**
```bash
curl -s -H "X-API-Key: efb85a4a291f917578dd9c625b91b87ace5846c1d92bf250552955cd50e6e1e3" \
  "https://web-production-f2de9.up.railway.app/live-summary" | jq '.'

# Expected response:
{
  "ts": 1737151200,
  "matches": {}
}
```

**Con partidos live:**
```bash
# Expected response (example):
{
  "ts": 1737151200,
  "matches": {
    "73456": {"s": "2H", "e": 67, "ex": 0, "h": 2, "a": 1},
    "73457": {"s": "HT", "e": 45, "ex": 0, "h": 0, "a": 0},
    "73458": {"s": "2H", "e": 92, "ex": 2, "h": 1, "a": 1}
  }
}
```

### Archivos modificados

**Backend:**
- [main.py:2423-2538](app/main.py#L2423-L2538) - Endpoint `/live-summary` con cache L1
- [telemetry/metrics.py:1284-1332](app/telemetry/metrics.py#L1284-L1332) - Métricas Prometheus
- [main.py:7998-8006](app/main.py#L7998-L8006) - Bloque en ops.json

**iOS:**
- [LiveScoreManager.swift](ios/FutbolStats/FutbolStats/Services/LiveScoreManager.swift) - Manager con gating
- [AppConfiguration.swift](ios/FutbolStats/FutbolStats/Services/AppConfiguration.swift) - API key config
- [FutbolStatsApp.swift:17-29](ios/FutbolStats/FutbolStats/FutbolStatsApp.swift#L17-L29) - scenePhase integration
- [PredictionsViewModel.swift:218-230](ios/FutbolStats/FutbolStats/ViewModels/PredictionsViewModel.swift#L218-L230) - Gating update

### Métricas Prometheus

```
live_summary_requests_total{status="ok|error"}
live_summary_latency_ms (histogram, buckets: 1,5,10,25,50,100,250,500)
live_summary_matches_count (gauge)
```

### Próximos pasos

1. **Deploy a Railway** - Push cambios y verificar endpoint
2. **Configurar API_KEY en iOS Info.plist** - Agregar key para producción
3. **Monitorear métricas** - Verificar latencia <50ms y requests/min

---

*Implementación completada 2026-01-17. Pendiente deploy y verificación en producción.*
