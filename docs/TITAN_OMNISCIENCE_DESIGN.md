# TITAN OMNISCIENCE - Technical Design Document

**Sistema Enterprise de Scraping Masivo para Prediccion Deportiva**

| Metadata | Valor |
|----------|-------|
| Version | 2.0 DRAFT |
| Fecha | 2026-01-25 |
| Autor | Claude Code (para revision ABE) |
| Estado | FASE 1 COMPLETADA (implementación) - Pendiente deploy Railway + aprobación Owner |

---

## 1. Resumen Ejecutivo

TITAN OMNISCIENCE es un **robot de scraping masivo** disenado para recolectar datos deportivos de **cientos de fuentes** (300+) y alimentar el modelo XGBoost de FutbolStats con 615 variables.

### Vision
> "Un bot que visite TODO internet deportivo: casas de apuestas, blogs, periodicos, redes sociales, APIs publicas, foros, y cualquier pagina con datos relevantes."

### Objetivos
1. **Scraping masivo**: 300+ fuentes heterogeneas (web, APIs, feeds)
2. **Backfill historico**: 2015/16 - 2026 (10 temporadas)
3. **Operacion continua**: Crawling 24/7 post-backfill
4. **Flexibilidad total**: Agregar fuentes sin cambiar codigo core
5. **Auto-descubrimiento**: Detectar nuevas ligas/equipos/jugadores automaticamente

### Numeros Clave
- **300+ fuentes** de datos (escalable a N)
- **41 tablas** normalizadas (3NF)
- **615 variables** con prioridad XGBoost (1-615)
- **8 tiers** de poder predictivo
- **6 categorias**: Core, Match Data, In-Game Events, Performance, Auxiliary, Psychology

---

## 2. Arquitectura de Scraping Masivo

### 2.1 Tipos de Fuentes (300+)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UNIVERSO DE FUENTES (300+)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ TIER A: APIs ESTRUCTURADAS (~20 fuentes)                            │   │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │ │API-Foot │ │Understat│ │SofaScore│ │Open-Mete│ │Transferm│        │   │
│  │ │ ball    │ │  (xG)   │ │(ratings)│ │o(weather│ │arkt     │        │   │
│  │ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘        │   │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │ │FBref    │ │Opta     │ │StatsBomb│ │WhoScored│ │FotMob   │        │   │
│  │ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ TIER B: CASAS DE APUESTAS (~50 fuentes)                             │   │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │ │Bet365   │ │Pinnacle │ │Betfair  │ │William  │ │1xBet    │        │   │
│  │ │         │ │(sharp)  │ │Exchange │ │Hill     │ │         │        │   │
│  │ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘        │   │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │ │Unibet   │ │Bwin     │ │Betway   │ │888sport │ │Marathonb│        │   │
│  │ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘        │   │
│  │ ... + 40 mas regionales (Codere, Caliente, etc.)                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ TIER C: PERIODICOS Y BLOGS DEPORTIVOS (~100 fuentes)                │   │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │ │Marca    │ │AS       │ │ESPN     │ │Sky Sport│ │Goal.com │        │   │
│  │ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘        │   │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │ │Gazzetta │ │L'Equipe │ │Kicker   │ │O Globo  │ │Ole      │        │   │
│  │ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘        │   │
│  │ ... + blogs especializados (Spielverlagerung, StatsBomb blog, etc.) │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ TIER D: REDES SOCIALES Y FEEDS (~50 fuentes)                        │   │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │ │Twitter/X│ │Instagram│ │Reddit   │ │Telegram │ │Discord  │        │   │
│  │ │(cuentas)│ │(clubes) │ │(r/soccer│ │(canales)│ │(servers)│        │   │
│  │ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘        │   │
│  │ + RSS feeds de periodicos + Google News alerts                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ TIER E: SITIOS ESPECIALIZADOS (~80 fuentes)                         │   │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │ │Injury   │ │Transfer │ │Referee  │ │Weather  │ │Stadium  │        │   │
│  │ │trackers │ │rumours  │ │stats    │ │services │ │capacity │        │   │
│  │ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘        │   │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │ │Physio   │ │Youth    │ │Women's  │ │Historical│ │Fan forums│       │   │
│  │ │reports  │ │leagues  │ │football │ │archives │ │          │        │   │
│  │ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Arquitectura del Crawler

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TITAN OMNISCIENCE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     SOURCE REGISTRY (300+ fuentes)                     │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │ sources.yaml - Configuracion declarativa de TODAS las fuentes   │  │ │
│  │  │                                                                  │  │ │
│  │  │ - url_patterns    - Donde buscar                                 │  │ │
│  │  │ - selectors       - Como extraer (CSS, XPath, regex, JSON path)  │  │ │
│  │  │ - rate_limits     - Cuanto esperar                               │  │ │
│  │  │ - auth            - Credenciales (si aplica)                     │  │ │
│  │  │ - schedule        - Cuando ejecutar                              │  │ │
│  │  │ - priority        - Tier de importancia                          │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         CRAWLER ENGINE                                 │ │
│  │                                                                        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │ │
│  │  │   Scrapy    │  │  Playwright │  │   Requests  │  │  API Client │  │ │
│  │  │  (crawling) │  │ (JS render) │  │  (simple)   │  │  (REST/GQL) │  │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │ │
│  │                                                                        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │ │
│  │  │ Proxy Pool  │  │  UA Rotator │  │ Rate Limit  │  │  Retry/DLQ  │  │ │
│  │  │ (anti-block)│  │ (stealth)   │  │ (per domain)│  │  (resilienc)│  │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    EXTRACTION PIPELINE                                 │ │
│  │                                                                        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │ │
│  │  │  HTML Parse │  │  JSON Parse │  │  NLP/Regex  │  │  LLM Extract│  │ │
│  │  │ (BeautifulS)│  │  (jq/path)  │  │  (patterns) │  │  (Gemini)   │  │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │ │
│  │                                                                        │ │
│  │  Para datos no estructurados (noticias, tweets, blogs):               │ │
│  │  LLM extrae entidades -> match_id, player_id, evento, fecha           │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                      ENTITY RESOLUTION                                 │ │
│  │                                                                        │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │ "Real Madrid" = "Real Madrid CF" = "R. Madrid" = "Los Blancos"  │  │ │
│  │  │ "Mbappe" = "Kylian Mbappe" = "K. Mbappe" = "Mbappe Lottin"      │  │ │
│  │  │                                                                  │  │ │
│  │  │ - Fuzzy matching (Levenshtein, Jaro-Winkler)                     │  │ │
│  │  │ - Alias tables (team_aliases, player_aliases)                    │  │ │
│  │  │ - Context clues (liga, fecha, rival)                             │  │ │
│  │  │ - Manual review queue para ambiguos                              │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         STORAGE LAYER                                  │ │
│  │  ┌─────────────────────────┐    ┌─────────────────────────────────┐  │ │
│  │  │   PostgreSQL (Railway)  │    │      Cloudflare R2 (Blobs)      │  │ │
│  │  │   - 41 tablas 3NF       │    │   - HTML snapshots (evidence)   │  │ │
│  │  │   - raw_extractions     │    │   - JSON raw responses          │  │ │
│  │  │   - entity_aliases      │    │   - Screenshots (JS pages)      │  │ │
│  │  │   - source_registry     │    │   - Backups comprimidos         │  │ │
│  │  └─────────────────────────┘    └─────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Source Registry (sources.yaml)

```yaml
# Ejemplo de configuracion declarativa para agregar fuentes SIN codigo

sources:
  # ═══════════════════════════════════════════════════════════════
  # TIER A: APIs estructuradas
  # ═══════════════════════════════════════════════════════════════
  api_football:
    type: api
    base_url: "https://v3.football.api-sports.io"
    auth:
      type: header
      key: "x-apisports-key"
      value: "${API_FOOTBALL_KEY}"
    rate_limit:
      requests_per_minute: 450
      daily_budget: 75000
    endpoints:
      fixtures: "/fixtures?date={date}"
      stats: "/fixtures/statistics?fixture={id}"
      odds: "/odds?fixture={id}"
    schedule: "*/1 * * * *"  # cada minuto
    priority: 1

  understat:
    type: scrape_api
    base_url: "https://understat.com"
    headers:
      X-Requested-With: "XMLHttpRequest"
    rate_limit:
      requests_per_second: 1.0
    endpoints:
      league: "/getLeagueData/{league}/{season}"
      match: "/getMatchData/{match_id}"
    extract:
      xg_home: "$.xG.h"
      xg_away: "$.xG.a"
    schedule: "*/30 * * * *"
    priority: 1

  # ═══════════════════════════════════════════════════════════════
  # TIER B: Casas de apuestas
  # ═══════════════════════════════════════════════════════════════
  bet365:
    type: scrape_js
    engine: playwright
    base_url: "https://www.bet365.com"
    requires_proxy: true
    rate_limit:
      requests_per_minute: 10
    selectors:
      odds_home: "div.odds-home span.price"
      odds_draw: "div.odds-draw span.price"
      odds_away: "div.odds-away span.price"
    anti_bot:
      wait_for: "div.odds-container"
      human_delay: [2, 5]  # segundos random
    schedule: "0 */6 * * *"  # cada 6 horas
    priority: 1

  pinnacle:
    type: api
    base_url: "https://pinnacle.com/api"
    auth:
      type: basic
      username: "${PINNACLE_USER}"
      password: "${PINNACLE_PASS}"
    rate_limit:
      requests_per_minute: 60
    endpoints:
      odds: "/odds/straight?sportId=29"
    schedule: "*/15 * * * *"
    priority: 1

  betfair_exchange:
    type: api
    base_url: "https://api.betfair.com/exchange"
    auth:
      type: oauth
      app_key: "${BETFAIR_APP_KEY}"
    rate_limit:
      requests_per_second: 5
    extract:
      back_odds: "$.runners[*].ex.availableToBack[0].price"
      lay_odds: "$.runners[*].ex.availableToLay[0].price"
      volume: "$.runners[*].totalMatched"
    priority: 1

  # ... 47 casas de apuestas mas ...

  # ═══════════════════════════════════════════════════════════════
  # TIER C: Periodicos deportivos
  # ═══════════════════════════════════════════════════════════════
  marca:
    type: scrape_html
    base_url: "https://www.marca.com/futbol"
    rate_limit:
      requests_per_minute: 30
    url_patterns:
      - "/futbol/primera-division/*.html"
      - "/futbol/champions-league/*.html"
    selectors:
      title: "h1.article-title"
      body: "div.article-body p"
      date: "time.article-date"
      entities: "a[href*='/futbol/']"  # links a equipos/jugadores
    extract_with_llm: true  # Usar Gemini para extraer entidades
    entity_types:
      - player_injury
      - transfer_rumor
      - lineup_news
      - coach_statement
    schedule: "*/10 * * * *"
    priority: 3

  espn:
    type: scrape_html
    base_url: "https://www.espn.com/soccer"
    rate_limit:
      requests_per_minute: 30
    url_patterns:
      - "/story/_/id/*"
      - "/team/_/id/*"
    selectors:
      title: "h1.headline"
      body: "div.article-body"
    extract_with_llm: true
    priority: 3

  # ... 98 periodicos/blogs mas ...

  # ═══════════════════════════════════════════════════════════════
  # TIER D: Redes sociales
  # ═══════════════════════════════════════════════════════════════
  twitter_accounts:
    type: social
    platform: twitter
    accounts:
      - "@FabrizioRomano"   # transfers
      - "@OptaJoe"          # stats
      - "@bet365"           # odds
      - "@reaborjjaa"       # La Liga injuries
      # ... 500+ cuentas relevantes
    rate_limit:
      requests_per_minute: 15  # Twitter API limits
    extract_with_llm: true
    entity_types:
      - transfer_confirmed
      - injury_update
      - lineup_confirmed
    priority: 2

  reddit_soccer:
    type: social
    platform: reddit
    subreddits:
      - "r/soccer"
      - "r/LaLiga"
      - "r/PremierLeague"
    rate_limit:
      requests_per_minute: 60
    extract_with_llm: true
    priority: 4

  # ═══════════════════════════════════════════════════════════════
  # TIER E: Fuentes especializadas
  # ═══════════════════════════════════════════════════════════════
  transfermarkt:
    type: scrape_html
    base_url: "https://www.transfermarkt.com"
    rate_limit:
      requests_per_minute: 20
    url_patterns:
      - "/{team}/startseite/verein/{id}"
      - "/{player}/profil/spieler/{id}"
    selectors:
      market_value: "div.market-value"
      injury_status: "span.injury-label"
      transfer_history: "table.transferhistorie tr"
    priority: 2

  physioroom:  # Injury tracker
    type: scrape_html
    base_url: "https://www.physioroom.com"
    selectors:
      player: "td.player-name"
      injury: "td.injury-type"
      return_date: "td.expected-return"
    priority: 2

  # ═══════════════════════════════════════════════════════════════
  # TEMPLATE: Agregar nueva fuente
  # ═══════════════════════════════════════════════════════════════
  _template:
    type: scrape_html | scrape_js | api | social | rss
    base_url: "https://..."
    auth: null | {type: header|basic|oauth, ...}
    requires_proxy: false
    rate_limit:
      requests_per_minute: 30
      requests_per_second: null
      daily_budget: null
    selectors: {}
    extract_with_llm: false
    entity_types: []
    schedule: "*/30 * * * *"
    priority: 1-5
    enabled: true
```

### 2.4 Flujo ELT Principal

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ CRAWL   │────▶│  RAW    │────▶│ RESOLVE │────▶│TRANSFORM│────▶│  SERVE  │
│ (300+)  │     │ STORE   │     │ ENTITIES│     │ (3NF)   │     │ (Model) │
└─────────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘
     │               │               │               │               │
     ▼               ▼               ▼               ▼               ▼
  Scrape         R2 blobs        Match to       41 tables       feature_
  Parse          raw_*           canonical      PIT valid       matrix
  Extract        tables          IDs            615 vars        XGBoost
```

### 2.5 Job State Machine

```
                    ┌──────────────┐
                    │   PENDING    │
                    └──────┬───────┘
                           │ schedule
                           ▼
                    ┌──────────────┐
             ┌──────│   RUNNING    │──────┐
             │      └──────────────┘      │
             │ error                      │ success
             ▼                            ▼
      ┌──────────────┐            ┌──────────────┐
      │    FAILED    │            │  COMPLETED   │
      └──────┬───────┘            └──────────────┘
             │ retry (max 3)
             ▼
      ┌──────────────┐
      │     DLQ      │──────▶ Manual review
      └──────────────┘
```

---

## 3. Modelo Operacional (Scraping Masivo)

### 3.1 Capacidad del Crawler

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CAPACIDAD ESTIMADA 24/7                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Fuentes activas:        300+                                                │
│  Requests/hora:          ~50,000 (distribuidos entre fuentes)               │
│  Requests/dia:           ~1,200,000                                          │
│  Paginas parseadas/dia:  ~500,000                                            │
│  Entidades extraidas/dia: ~100,000                                           │
│  Storage R2/mes:         ~50-100 GB (HTML snapshots + JSON)                  │
│                                                                              │
│  Distribucion por tier:                                                      │
│  ├── TIER A (APIs):       ~40% requests (alta frecuencia, bajo volumen)     │
│  ├── TIER B (Apuestas):   ~25% requests (odds cambian constantemente)       │
│  ├── TIER C (Periodicos): ~20% requests (noticias cada 10-30 min)           │
│  ├── TIER D (Social):     ~10% requests (rate limits estrictos)             │
│  └── TIER E (Especial):   ~5% requests (datos menos volatiles)              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Fases de Operacion

| Fase | Descripcion | Fuentes Activas |
|------|-------------|-----------------|
| **BOOTSTRAP** | Configurar primeras 20 fuentes core | 20 |
| **SCALE** | Agregar 50 fuentes/semana hasta 300+ | 20 → 300 |
| **BACKFILL** | Carga historica 2015-2026 (paralelo) | 300+ |
| **STEADY** | Operacion normal 24/7 | 300+ |
| **EXPANSION** | Agregar fuentes on-demand | N |

### 3.3 Rate Limiting Inteligente

```python
class RateLimiter:
    """Rate limiting adaptativo por dominio."""

    def __init__(self, source_config: dict):
        self.config = source_config
        self.request_history = deque(maxlen=1000)
        self.error_history = deque(maxlen=100)
        self.blocked_until = None

    async def acquire(self) -> bool:
        """Obtener permiso para hacer request."""

        # 1. Check si estamos bloqueados
        if self.blocked_until and datetime.utcnow() < self.blocked_until:
            return False

        # 2. Check rate limits configurados
        rpm = self.config.get("rate_limit", {}).get("requests_per_minute", 30)
        recent = [r for r in self.request_history
                  if r > datetime.utcnow() - timedelta(minutes=1)]
        if len(recent) >= rpm:
            return False

        # 3. Backoff adaptativo si hay muchos errores
        error_rate = len(self.error_history) / 100
        if error_rate > 0.3:  # >30% errores
            await asyncio.sleep(error_rate * 10)  # delay proporcional

        self.request_history.append(datetime.utcnow())
        return True

    def on_response(self, status_code: int):
        """Ajustar comportamiento segun respuesta."""
        if status_code == 429:  # Too Many Requests
            self.blocked_until = datetime.utcnow() + timedelta(minutes=5)
            logger.warning(f"Rate limited, backing off 5 min")
        elif status_code == 403:  # Forbidden (possible block)
            self.blocked_until = datetime.utcnow() + timedelta(hours=1)
            logger.error(f"Possible IP block, backing off 1 hour")
        elif status_code >= 500:
            self.error_history.append(datetime.utcnow())
```

### 3.4 Anti-Bloqueo y Stealth

```python
class StealthCrawler:
    """Tecnicas para evitar deteccion y bloqueo."""

    # Pool de User Agents (rotar por request)
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit...",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36...",
        # ... 50+ user agents reales
    ]

    # Pool de proxies (rotar por dominio)
    PROXY_POOL = [
        "http://proxy1.provider.com:8080",
        "http://proxy2.provider.com:8080",
        # ... N proxies residenciales
    ]

    async def fetch(self, url: str, source_config: dict) -> Response:
        """Fetch con stealth."""

        headers = {
            "User-Agent": random.choice(self.USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        # Usar proxy si la fuente lo requiere
        proxy = None
        if source_config.get("requires_proxy"):
            proxy = random.choice(self.PROXY_POOL)

        # Delay humano (2-5 segundos random)
        if source_config.get("anti_bot", {}).get("human_delay"):
            delay = random.uniform(*source_config["anti_bot"]["human_delay"])
            await asyncio.sleep(delay)

        # Para sitios JS-heavy, usar Playwright
        if source_config.get("type") == "scrape_js":
            return await self._fetch_with_playwright(url, headers, proxy)

        # Para sitios simples, usar httpx
        async with httpx.AsyncClient(proxies=proxy) as client:
            return await client.get(url, headers=headers, timeout=30)
```

### 3.5 Deteccion Automatica de Nuevas Entidades

```python
async def on_new_entity_detected(entity_type: str, raw_data: dict):
    """
    Cuando el crawler detecta una entidad nueva (liga, equipo, jugador):
    1. Crear registro en DB
    2. Buscar en TODAS las fuentes activas
    3. Queue backfill historico
    """

    if entity_type == "competition":
        # Nueva liga detectada (ej: Liga colombiana)
        competition = await create_competition(raw_data)

        # Buscar en todas las fuentes que soporten esa region
        sources = await get_sources_for_region(raw_data.get("country"))
        for source in sources:
            await queue_discovery_job(source, competition.id)

        # Buscar equipos de esa liga
        await queue_team_discovery(competition.id)

    elif entity_type == "team":
        # Nuevo equipo detectado
        team = await create_team(raw_data)

        # Crear aliases conocidos
        await create_aliases(team, raw_data.get("alternate_names", []))

        # Buscar jugadores del equipo
        await queue_player_discovery(team.id)

        # Backfill stats historicas
        await queue_team_backfill(team.id, start_season="2015")

    elif entity_type == "player":
        # Nuevo jugador detectado
        player = await create_player(raw_data)

        # Crear aliases
        await create_aliases(player, [
            raw_data.get("short_name"),
            raw_data.get("full_name"),
            # ... variaciones del nombre
        ])

        # Backfill stats
        await queue_player_backfill(player.id)
```

### 3.6 LLM para Extraccion de Datos No Estructurados

```python
class LLMExtractor:
    """Usar Gemini para extraer entidades de texto libre."""

    SYSTEM_PROMPT = """
    Eres un extractor de datos deportivos. Dado un articulo de periodico o tweet,
    extrae las siguientes entidades si estan presentes:

    - player_injury: {player_name, injury_type, severity, expected_return}
    - transfer: {player_name, from_team, to_team, fee, loan}
    - lineup: {match, team, formation, starting_xi}
    - suspension: {player_name, reason, matches}
    - quote: {person, statement, context}

    Responde SOLO en JSON. Si no hay datos relevantes, responde {}.
    """

    async def extract(self, text: str, source_url: str) -> list[dict]:
        """Extraer entidades de texto usando LLM."""

        response = await self.gemini.generate(
            system=self.SYSTEM_PROMPT,
            user=f"URL: {source_url}\n\nTexto:\n{text[:4000]}"  # limit context
        )

        try:
            entities = json.loads(response)
            return self._validate_and_resolve(entities)
        except json.JSONDecodeError:
            logger.warning(f"LLM returned invalid JSON for {source_url}")
            return []

    async def _validate_and_resolve(self, entities: list[dict]) -> list[dict]:
        """Validar y resolver entidades a IDs canonicos."""
        resolved = []
        for entity in entities:
            # Resolver nombres a IDs
            if "player_name" in entity:
                player_id = await self.resolve_player(entity["player_name"])
                if player_id:
                    entity["player_id"] = player_id
                    resolved.append(entity)
                else:
                    # Queue para revision manual
                    await self.queue_manual_review(entity)
        return resolved
```

---

## 4. Data Contracts

### 4.1 Schema de Tablas Principales

Las 41 tablas se organizan en 6 categorias:

#### CORE (8 tablas, 80 vars)
```sql
-- Entidades maestras
matches          -- PK: match_id (central)
teams            -- PK: team_id, team_type = CLUB | NATIONAL_TEAM
players          -- PK: player_id
stadiums         -- PK: stadium_id (altitud, coordenadas)
referees         -- PK: referee_id
managers         -- PK: manager_id
competitions     -- PK: competition_id
seasons          -- PK: season_id (is_world_cup_year flag)
```

#### MATCH DATA (6 tablas, 112 vars)
```sql
-- Contexto pre-partido
match_weather    -- FK: match_id (temperatura, viento, humedad)
match_context    -- FK: match_id (posiciones, zonas descenso/titulo)
match_lineups    -- FK: match_id, player_id (XI, formacion)
match_odds       -- FK: match_id (cuotas apertura/cierre)
match_travel     -- FK: match_id, team_id (fatiga, jet lag)
match_h2h        -- FK: match_id (historial head-to-head)
```

#### IN-GAME EVENTS (8 tablas, 114 vars)
```sql
-- Eventos atomicos con coordenadas (x,y)
match_events     -- PK: event_id (minuto, tipo, jugador)
goals            -- FK: event_id (goleador, xG, asistencia)
shots            -- FK: event_id (big_chance, on_target)
cards            -- FK: event_id (amarilla, roja, motivo)
substitutions    -- FK: event_id (tactica, lesion)
fouls            -- FK: event_id (severidad, zona)
set_pieces       -- FK: event_id (corner, falta, saque)
var_reviews      -- FK: event_id (duracion, decision)
```

#### PERFORMANCE AGGREGATES (4 tablas, 162 vars)
```sql
-- Stats acumuladas
team_season_stats    -- FK: team_id, season_id (xG, forma, rachas)
player_season_stats  -- FK: player_id, season_id (goles, ratings)
player_match_stats   -- FK: player_id, match_id (rendimiento individual)
referee_season_stats -- FK: referee_id, season_id (tarjetas, sesgo)
```

#### AUXILIARY (4 tablas, 34 vars)
```sql
-- Historial y metadata
cities               -- PK: city_id (coordenadas, timezone)
player_injuries      -- FK: player_id (tipo, duracion, recurrente)
player_transfers     -- FK: player_id (from/to team, fee)
manager_tenures      -- FK: manager_id, team_id (periodos)
```

#### PSYCHOLOGY (11 tablas, 113 vars)
```sql
-- Factores psicologicos y contextuales
player_personal_events       -- FK: player_id (duelos, nacimientos)
match_geopolitical_context   -- FK: match_id (conflictos, tensiones)
player_tournament_context    -- FK: player_id (convocatorias, torneos)
player_media_incidents       -- FK: player_id (errores virales)
player_discipline_history    -- FK: player_id (sanciones, multas)
match_attendance             -- FK: match_id (aforo, sentimiento)
match_stoppage_times         -- FK: match_id (tiempo anadido)
referee_match_history        -- FK: referee_id, team_id (historial)
manager_match_context        -- FK: match_id (tenure, H2H tecnicos)
match_emotional_connections  -- FK: match_id (ley del ex, familiares)
cross_correlation_features   -- FK: match_id (variables derivadas)
```

### 4.2 Tiers de Prioridad XGBoost

| Tier | Rango | Descripcion | NULL Esperado |
|------|-------|-------------|---------------|
| **TIER 1** | 1-50 | Cuotas cierre, prob. implicitas, xG, forma | RARE |
| **TIER 2** | 51-100 | Forma ultimos 5, rachas, xG reciente | RARE |
| **TIER 3** | 101-150 | H2H historico, contexto competitivo | SOMETIMES |
| **TIER 4** | 151-250 | Alineacion, clima, estadio, altitud | SOMETIMES |
| **TIER 5** | 251-350 | Arbitro, fatiga, viajes, descanso | SOMETIMES |
| **TIER 6** | 351-450 | Stats individuales, entrenador | SOMETIMES |
| **TIER 7** | 451-550 | Asistencia, ambiente, emocionales | OFTEN |
| **TIER 8** | 551-615 | Psicologia avanzada, geopolitica | OFTEN |

### 4.3 Ejemplo: Feature Row para XGBoost

```python
# Una fila = un partido con 615 columnas
feature_row = {
    # TIER 1 - Cuotas (max predictivo)
    "odds_home_close": 1.85,
    "odds_draw_close": 3.40,
    "odds_away_close": 4.50,
    "implied_prob_home": 0.54,
    "implied_prob_draw": 0.29,
    "implied_prob_away": 0.17,

    # TIER 2 - Forma
    "xg_last5": 1.82,
    "form_last5": "WWDLW",  # encoded
    "goals_scored_last5": 8,

    # TIER 3 - H2H
    "h2h_total_matches": 42,
    "h2h_home_wins": 18,
    "h2h_avg_goals_last5": 2.6,

    # ... 600+ columnas mas

    # Target
    "outcome": "HOME_WIN"  # o DRAW, AWAY_WIN
}
```

---

## 5. Estrategia de Calidad de Datos

### 5.1 Validaciones por Capa

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA QUALITY PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  EXTRACT          TRANSFORM           LOAD            SERVE      │
│     │                 │                 │                │       │
│     ▼                 ▼                 ▼                ▼       │
│  ┌──────┐         ┌──────┐         ┌──────┐         ┌──────┐   │
│  │Schema│         │ PIT  │         │ Ref  │         │Drift │   │
│  │Valid │         │Check │         │Integ │         │Detect│   │
│  └──────┘         └──────┘         └──────┘         └──────┘   │
│     │                 │                 │                │       │
│     ▼                 ▼                 ▼                ▼       │
│  - JSON schema    - t < t0          - FK exists      - Stats    │
│  - Type coerce    - captured_at     - No orphans     - Dist     │
│  - Range check    - no future data  - Cascade OK     - Anomaly  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Point-in-Time (PIT) Enforcement

```python
class PITValidator:
    """Garantiza que ningun dato futuro contamina features."""

    async def validate(self, match_id: int, feature_data: dict) -> bool:
        match = await get_match(match_id)
        t0 = match.kickoff_utc

        errors = []

        # Check 1: captured_at < kickoff
        if feature_data.get("captured_at") >= t0:
            errors.append(f"captured_at {feature_data['captured_at']} >= kickoff {t0}")

        # Check 2: No stats del partido actual
        if feature_data.get("source_match_id") == match_id:
            errors.append("Feature uses same match data (leakage)")

        # Check 3: Rolling windows solo con partidos anteriores
        if feature_data.get("rolling_window_end") >= t0:
            errors.append("Rolling window includes future matches")

        if errors:
            await self.quarantine(match_id, errors)
            return False

        return True
```

### 5.3 Imputacion por Tier

| Tier | Estrategia Imputacion | Flag |
|------|----------------------|------|
| 1-2 | Media liga + `*_missing=1` | CRITICAL |
| 3-4 | Media historica equipo | WARNING |
| 5-6 | Default conservador | INFO |
| 7-8 | NULL permitido (excluir fila) | SKIP |

### 5.4 Metricas de Calidad

```python
QUALITY_METRICS = {
    "completeness": "% campos no-NULL por tier",
    "freshness": "Horas desde ultima actualizacion",
    "consistency": "% registros que pasan FK checks",
    "accuracy": "% matches con outcome verificable",
    "timeliness": "Latencia promedio ingesta"
}

# Thresholds
QUALITY_THRESHOLDS = {
    "completeness_tier1": 0.95,  # 95% minimo
    "completeness_tier2": 0.90,
    "completeness_tier3": 0.80,
    "freshness_hours": 24,
    "consistency": 0.99
}
```

---

## 6. Seguridad y Compliance

### 6.1 Manejo de Credenciales

```yaml
# Secrets Management
secrets:
  storage: Railway Environment Variables
  rotation: Manual (documentar en runbook)
  access: Solo app container + CI/CD

  required:
    - DATABASE_URL
    - API_FOOTBALL_KEY
    - CLOUDFLARE_R2_ACCESS_KEY
    - CLOUDFLARE_R2_SECRET_KEY

  optional:
    - SENTRY_DSN
    - PROMETHEUS_PUSH_GATEWAY
```

### 6.2 Rate Limit Compliance

```python
# Respetar ToS de cada fuente
RATE_LIMITS = {
    "api_football": {
        "requests_per_minute": 450,
        "requests_per_day": 75000,
        "respect_429": True,
        "backoff_base": 2.0
    },
    "understat": {
        "requests_per_second": 1.0,
        "courtesy_delay": True,  # No hay ToS oficial
        "user_agent": "FutbolStats/1.0 (research)"
    },
    "sofascore": {
        "requests_per_second": 0.5,
        "rotate_user_agent": True,
        "respect_robots_txt": True
    }
}
```

### 6.3 Logging y Auditoria

```python
# Audit trail obligatorio
@audit_log
async def ingest_job(job_id: str, source: str, params: dict):
    """Todo job loggea: quien, que, cuando, resultado."""
    log_entry = {
        "job_id": job_id,
        "source": source,
        "started_at": datetime.utcnow(),
        "params": params,  # Sin secretos
        "status": "RUNNING"
    }
    await db.insert("job_runs", log_entry)
```

---

## 7. Estructura de Codigo Propuesta

### 7.1 Arbol de Directorios

```
titan/
├── __init__.py
├── config.py                 # Settings + rate limits
├── orchestrator/
│   ├── __init__.py
│   ├── scheduler.py          # APScheduler jobs
│   ├── job_manager.py        # Idempotency + DLQ
│   └── triggers.py           # Event handlers
├── extractors/
│   ├── __init__.py
│   ├── base.py               # Abstract extractor
│   ├── api_football.py       # Existente (refactorizar)
│   ├── understat.py          # Existente (refactorizar)
│   ├── sofascore.py          # Existente (refactorizar)
│   ├── open_meteo.py         # Existente (refactorizar)
│   └── plugins/              # Nuevas fuentes (futuro)
│       └── __init__.py
├── transformers/
│   ├── __init__.py
│   ├── normalizer.py         # Raw -> 3NF
│   ├── validator.py          # Schema checks
│   ├── pit_checker.py        # Anti-leakage
│   └── feature_builder.py    # 615 vars
├── loaders/
│   ├── __init__.py
│   ├── postgres.py           # Bulk upserts
│   ├── r2.py                 # Blob storage
│   └── staging.py            # Tablas raw_*
├── models/
│   ├── __init__.py
│   ├── core.py               # 8 tablas core
│   ├── match_data.py         # 6 tablas match_*
│   ├── events.py             # 8 tablas eventos
│   ├── performance.py        # 4 tablas stats
│   ├── auxiliary.py          # 4 tablas aux
│   └── psychology.py         # 11 tablas psych
├── quality/
│   ├── __init__.py
│   ├── metrics.py            # Quality scores
│   ├── alerts.py             # Threshold violations
│   └── reports.py            # Daily summaries
└── cli/
    ├── __init__.py
    ├── backfill.py           # Manual backfill
    ├── status.py             # Job status
    └── validate.py           # Data quality checks
```

**Nota (Implementación FASE 1)**:
- La FASE 1 se implementó como módulo **`app/titan/`** dentro del backend existente (FastAPI), para minimizar fricción y permitir deploy incremental.
- La estructura `titan/` standalone queda como refactor futuro (Fase 2+) si se decide extraer el módulo.

### 7.2 Integracion con FutbolStats Existente

```python
# app/etl/ -> titan/extractors/ (refactor gradual)
# app/features/engineering.py -> titan/transformers/feature_builder.py

# Coexistencia durante migracion
from titan.extractors import APIFootballExtractor
from titan.transformers import FeatureBuilder

# O mantener backwards compatibility
from app.etl.api_football import APIFootballProvider  # legacy
from titan.extractors.api_football import Extractor   # new
```

---

## 8. Roadmap por Fases (Scraping Masivo)

### FASE 1: Vertical Slice Mínimo (Infra + 1 extractor end-to-end) ✅ COMPLETADA (2026-01-25)
- [x] Crear módulo `app/titan/` (extractor, job manager, materializer, dashboard)
- [x] Migraciones `titan_001..004` (schema `titan`, `raw_extractions`, `job_dlq`, `feature_matrix`)
- [x] `idempotency_key` determinístico `CHAR(32)` + constraint UNIQUE
- [x] DLQ con retry info + backoff + timestamps `TIMESTAMPTZ` (UTC tz-aware)
- [x] `feature_matrix` MVP (Tier 1 = odds requerido; Tier 2/3 opcional) + constraint PIT
- [x] Endpoint protegido `GET /dashboard/titan.json` (auth `X-Dashboard-Token`)
- [x] Runner end-to-end `app/titan/runner.py` (CLI `python -m app.titan.runner`)
- [x] Queries PIT-safe a `public.matches` (explícitas)
- [x] Tests unitarios (idempotencia, PIT, insertion policy) passing

**NO incluido en FASE 1 (diferido)**:
- R2 storage / `aioboto3`
- `sources.yaml` config parser
- Proxies / Playwright / casas de apuestas / redes sociales
- Completar 41 tablas (solo las mínimas del slice)

### FASE 2: Crawler Engine (20 fuentes iniciales)
- [ ] Implementar base crawler (Scrapy + Playwright)
- [ ] Rate limiter adaptativo por dominio
- [ ] User-Agent rotator + stealth mode
- [ ] Migrar 4 providers existentes (API-Football, Understat, SofaScore, Open-Meteo)
- [ ] Agregar 16 fuentes nuevas de casas de apuestas
- [ ] Tests de integracion por fuente

### FASE 3: Entity Resolution + LLM Extraction
- [ ] Implementar fuzzy matching (equipos, jugadores)
- [ ] Crear tablas de aliases (team_aliases, player_aliases)
- [ ] Integrar Gemini para extraccion de texto no estructurado
- [ ] Cola de revision manual para entidades ambiguas
- [ ] Dashboard de entidades no resueltas

### FASE 4: Scale to 100+ fuentes
- [ ] Agregar 50 casas de apuestas regionales
- [ ] Agregar 30 periodicos deportivos principales
- [ ] Configurar Twitter/Reddit scrapers
- [ ] Optimizar paralelismo (async pool sizing)
- [ ] Monitoreo de health por fuente

### FASE 5: Scale to 300+ fuentes
- [ ] Agregar 100+ blogs y fuentes especializadas
- [ ] Agregar injury trackers, transfer rumours
- [ ] Implementar auto-discovery de nuevas fuentes
- [ ] Alertas de fuentes rotas/bloqueadas
- [ ] Self-healing (detectar cambios HTML, ajustar selectores)

### FASE 6: Backfill Historico (Paralelo)
- [ ] Script de backfill por liga/temporada/fuente
- [ ] Queue de jobs con prioridad por tier
- [ ] Monitoreo de progreso (% completado por tabla)
- [ ] Validacion de completeness por tier
- [ ] Reconciliacion entre fuentes (datos conflictivos)

### FASE 7: Operacion Continua 24/7
- [ ] Activar scheduler para sync continuo
- [ ] Dashboard de calidad de datos
- [ ] Alertas de degradacion
- [ ] Auto-scaling basado en carga
- [ ] Documentacion operacional

---

## 9. Riesgos y Mitigaciones (Scraping Masivo)

| Riesgo | Probabilidad | Impacto | Mitigacion |
|--------|--------------|---------|------------|
| IP bloqueado por fuente | ALTA | MEDIO | Proxy pool residencial + rotacion |
| Cambio estructura HTML | ALTA | MEDIO | Snapshots R2 + alertas + self-healing |
| Rate limit excedido | MEDIA | ALTO | Budget por fuente + backoff adaptativo |
| Deteccion como bot | MEDIA | ALTO | Stealth mode + human delays + fingerprint |
| LLM extrae datos incorrectos | MEDIA | MEDIO | Validacion cruzada + review queue |
| Costo proxies excesivo | MEDIA | MEDIO | Tier fuentes por necesidad de proxy |
| PostgreSQL storage limit | BAJA | ALTO | Archivado a R2 + particionado |
| Fuente legal issues | BAJA | ALTO | robots.txt compliance + ToS review |
| Datos conflictivos entre fuentes | MEDIA | MEDIO | Scoring de confiabilidad por fuente |

---

## 10. Supervivencia Operativa (Políticas Críticas)

> **NOTA ABE**: Estas políticas son **OBLIGATORIAS** para garantizar viabilidad operacional.
> Sin ellas, el sistema colapsa en semanas por deuda técnica acumulada.

### 10.A Estrategia de Mantenimiento "Pareto" (80/20)

**Principio**: El 20% de las fuentes (Tier A) provee el 80% del valor predictivo.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRIORIZACIÓN DE MANTENIMIENTO                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  GOLDEN SOURCES (20 fuentes) - Prioridad ABSOLUTA                           │
│  ├── API-Football (fixtures, stats, odds)                                   │
│  ├── Understat (xG, xPTS)                                                   │
│  ├── SofaScore (lineups, ratings)                                           │
│  ├── Open-Meteo (weather)                                                   │
│  ├── Pinnacle (sharp odds)                                                  │
│  ├── Betfair Exchange (volume, liquidity)                                   │
│  ├── Bet365 (reference odds)                                                │
│  └── ... 13 fuentes más Tier A/B críticas                                   │
│                                                                              │
│  Si una Golden Source falla:                                                 │
│  → Alerta inmediata (Slack/PagerDuty)                                       │
│  → Reparación en <24 horas                                                  │
│  → Escalar a Owner si no se resuelve                                        │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SECONDARY SOURCES (280 fuentes) - Best-effort                              │
│  ├── Tier C: Periódicos (Marca, ESPN, etc.)                                 │
│  ├── Tier D: Redes sociales (Twitter, Reddit)                               │
│  └── Tier E: Blogs y fuentes especializadas                                 │
│                                                                              │
│  Si una Secondary Source falla:                                              │
│  → Desactivar automáticamente (enabled: false)                              │
│  → Agregar a cola de reparación (FIFO, baja prioridad)                      │
│  → NO bloquea pipeline principal                                            │
│  → Reparación cuando haya bandwidth (semanal)                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementación**:

```python
class SourceHealthManager:
    """Gestión de salud de fuentes con priorización Pareto."""

    GOLDEN_SOURCES = {
        "api_football", "understat", "sofascore", "open_meteo",
        "pinnacle", "betfair_exchange", "bet365", "william_hill",
        "unibet", "bwin", "betway", "888sport", "marathonbet",
        "transfermarkt", "fbref", "whoscored", "fotmob",
        "opta", "statsbomb", "physioroom"
    }

    async def on_source_failure(self, source_id: str, error: Exception):
        """Manejar fallo de fuente según prioridad."""

        if source_id in self.GOLDEN_SOURCES:
            # CRÍTICO: Alerta inmediata
            await self.alert_critical(
                f"🚨 GOLDEN SOURCE FAILED: {source_id}",
                error=str(error),
                channel="alerts-critical"
            )
            # Reintentar agresivamente
            await self.schedule_immediate_retry(source_id, max_attempts=10)

        else:
            # SECUNDARIO: Desactivar y encolar
            await self.disable_source(source_id)
            await self.queue_for_repair(source_id, priority="low")
            logger.info(f"Secondary source {source_id} disabled, queued for repair")

    async def get_repair_queue(self) -> list[str]:
        """Obtener cola de reparación ordenada por prioridad."""
        return await db.fetch_all("""
            SELECT source_id FROM source_repair_queue
            ORDER BY
                CASE WHEN source_id = ANY($1) THEN 0 ELSE 1 END,  -- Golden first
                failed_at ASC  -- FIFO dentro de cada tier
        """, [list(self.GOLDEN_SOURCES)])
```

**Métricas de salud**:

| Métrica | Golden Sources | Secondary Sources |
|---------|----------------|-------------------|
| Uptime target | 99.5% | 80% |
| Max downtime | 4 horas | 7 días |
| Alerta | Inmediata | Semanal digest |
| SLA reparación | 24 horas | Best-effort |

---

### 10.B Protocolo de Ahorro de Ancho de Banda (Bandwidth-Miser)

**Principio**: Reducir payload de 2MB a <100KB por página scrapeada.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     INTERCEPTORES DE RED (Playwright)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  BLOQUEAR (abort request):                                                   │
│  ├── Imágenes: *.png, *.jpg, *.jpeg, *.gif, *.webp, *.svg, *.ico           │
│  ├── Fuentes: *.woff, *.woff2, *.ttf, *.otf, *.eot                         │
│  ├── Media: *.mp4, *.webm, *.mp3, *.wav, *.ogg                             │
│  ├── Analytics conocidos:                                                    │
│  │   ├── google-analytics.com/*                                             │
│  │   ├── googletagmanager.com/*                                             │
│  │   ├── facebook.net/tr/*                                                  │
│  │   ├── doubleclick.net/*                                                  │
│  │   ├── hotjar.com/*                                                       │
│  │   ├── segment.io/*                                                       │
│  │   └── ... 50+ dominios de tracking                                       │
│  ├── Ads: googlesyndication.com/*, amazon-adsystem.com/*                    │
│  └── CSS no crítico (si no afecta extracción)                               │
│                                                                              │
│  PERMITIR:                                                                   │
│  ├── HTML principal (document)                                              │
│  ├── JSON/XHR responses (datos)                                             │
│  ├── JS mínimo necesario para render                                        │
│  └── CSS crítico (si selectores dependen de él)                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementación**:

```python
class BandwidthMiserInterceptor:
    """Interceptor de red para minimizar ancho de banda."""

    # Extensiones a bloquear
    BLOCKED_EXTENSIONS = {
        # Imágenes
        '.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.ico', '.bmp',
        # Fuentes
        '.woff', '.woff2', '.ttf', '.otf', '.eot',
        # Media
        '.mp4', '.webm', '.mp3', '.wav', '.ogg', '.avi', '.mov',
        # Otros
        '.pdf', '.zip', '.exe',
    }

    # Dominios de analytics/tracking a bloquear
    BLOCKED_DOMAINS = {
        'google-analytics.com', 'googletagmanager.com', 'doubleclick.net',
        'facebook.net', 'facebook.com/tr', 'connect.facebook.net',
        'hotjar.com', 'hotjar.io', 'segment.io', 'segment.com',
        'mixpanel.com', 'amplitude.com', 'heap.io', 'fullstory.com',
        'newrelic.com', 'nr-data.net', 'sentry.io',
        'googlesyndication.com', 'amazon-adsystem.com', 'ads.twitter.com',
        'bat.bing.com', 'clarity.ms', 'cloudflareinsights.com',
        # ... agregar más según se descubran
    }

    async def setup_interceptor(self, page: Page):
        """Configurar interceptor en página Playwright."""

        async def handle_route(route: Route):
            url = route.request.url.lower()
            resource_type = route.request.resource_type

            # 1. Bloquear por tipo de recurso
            if resource_type in ('image', 'media', 'font'):
                await route.abort()
                return

            # 2. Bloquear por extensión
            if any(url.endswith(ext) for ext in self.BLOCKED_EXTENSIONS):
                await route.abort()
                return

            # 3. Bloquear dominios de tracking
            if any(domain in url for domain in self.BLOCKED_DOMAINS):
                await route.abort()
                return

            # 4. Permitir el resto
            await route.continue_()

        await page.route("**/*", handle_route)

    def estimate_savings(self, original_bytes: int, final_bytes: int) -> dict:
        """Calcular ahorro de ancho de banda."""
        saved = original_bytes - final_bytes
        return {
            "original_kb": original_bytes / 1024,
            "final_kb": final_bytes / 1024,
            "saved_kb": saved / 1024,
            "reduction_pct": (saved / original_bytes * 100) if original_bytes > 0 else 0
        }
```

**Estrategia de proxies híbrida**:

```yaml
# Optimización de costos de proxy
proxy_strategy:
  # Tier A/B (APIs y casas de apuestas): Proxy residencial
  # Razón: Anti-bot agresivo, necesitan parecer usuarios reales
  golden_sources:
    proxy_type: residential
    cost_per_gb: $15
    use_when: requires_proxy: true

  # Tier C/D/E (periódicos, blogs): Datacenter o sin proxy
  # Razón: Protección mínima, aceptan bots "educados"
  secondary_sources:
    proxy_type: datacenter  # o ninguno
    cost_per_gb: $0.50
    use_when: requires_proxy: false

  # Fallback: Si datacenter falla, intentar residencial
  fallback:
    enabled: true
    escalate_after_failures: 3
```

**Ahorro estimado**:

| Escenario | Sin optimización | Con Bandwidth-Miser |
|-----------|-----------------|---------------------|
| Payload promedio/página | 2 MB | 80 KB |
| GB/día (500K páginas) | 1,000 GB | 40 GB |
| Costo proxy/mes (residencial) | $15,000 | $600 |
| Costo proxy/mes (híbrido) | $15,000 | $200-400 |

---

### 10.C Política de "Fail-Open" en Entidades (Entity Resolution)

**Principio**: No bloquear pipeline por ambigüedad. Auto-resolver con >70% confianza.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENTITY RESOLUTION - FAIL-OPEN                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: "K. Mbappe" (de Understat)                                          │
│                                                                              │
│  Candidatos encontrados:                                                     │
│  ├── player_id=12345 "Kylian Mbappé" (PSG/Real Madrid) - Score: 0.95       │
│  ├── player_id=67890 "Kevin Mbappe" (Ligue 2) - Score: 0.45                │
│  └── player_id=11111 "Karl Mbappe" (Amateur) - Score: 0.30                 │
│                                                                              │
│  Decisión automática:                                                        │
│  ├── Score > 0.90: AUTO-MATCH (confianza alta)                              │
│  ├── Score 0.70-0.90: AUTO-MATCH + FLAG (revisar después)                   │
│  ├── Score < 0.70: SKIP (no asociar, loggear para análisis)                 │
│  └── Sin candidatos: CREATE NEW + FLAG (entidad nueva?)                     │
│                                                                              │
│  Output: player_id=12345, auto_resolved=true, confidence=0.95               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementación**:

```python
class FailOpenEntityResolver:
    """Entity resolution que nunca bloquea el pipeline."""

    # Umbrales de confianza
    CONFIDENCE_AUTO_MATCH = 0.90      # Match directo, sin flag
    CONFIDENCE_AUTO_MATCH_FLAG = 0.70 # Match con flag para revisión
    CONFIDENCE_SKIP = 0.70            # Por debajo: no asociar

    async def resolve(
        self,
        entity_type: str,  # "team", "player", "competition"
        raw_name: str,
        context: dict = None  # liga, fecha, equipo rival, etc.
    ) -> ResolvedEntity:
        """Resolver entidad con política fail-open."""

        # 1. Buscar candidatos
        candidates = await self._find_candidates(entity_type, raw_name, context)

        if not candidates:
            # Sin candidatos: crear placeholder o skip
            return ResolvedEntity(
                entity_id=None,
                confidence=0.0,
                auto_resolved=False,
                action="SKIP_NO_CANDIDATES",
                needs_review=True,
                raw_input=raw_name
            )

        # 2. Obtener mejor candidato
        best = max(candidates, key=lambda c: c.score)

        # 3. Decidir según confianza
        if best.score >= self.CONFIDENCE_AUTO_MATCH:
            # Alta confianza: match directo
            return ResolvedEntity(
                entity_id=best.entity_id,
                confidence=best.score,
                auto_resolved=True,
                action="AUTO_MATCH_HIGH",
                needs_review=False,
                raw_input=raw_name
            )

        elif best.score >= self.CONFIDENCE_AUTO_MATCH_FLAG:
            # Media confianza: match + flag
            return ResolvedEntity(
                entity_id=best.entity_id,
                confidence=best.score,
                auto_resolved=True,
                action="AUTO_MATCH_FLAGGED",
                needs_review=True,  # Revisar en batch semanal
                raw_input=raw_name
            )

        else:
            # Baja confianza: no asociar
            return ResolvedEntity(
                entity_id=None,
                confidence=best.score,
                auto_resolved=False,
                action="SKIP_LOW_CONFIDENCE",
                needs_review=True,
                raw_input=raw_name,
                best_candidate=best  # Para análisis
            )

    async def _find_candidates(
        self,
        entity_type: str,
        raw_name: str,
        context: dict
    ) -> list[Candidate]:
        """Buscar candidatos usando múltiples estrategias."""

        candidates = []

        # 1. Exact match en aliases
        exact = await db.fetch_one("""
            SELECT entity_id, 1.0 as score
            FROM entity_aliases
            WHERE entity_type = $1 AND LOWER(alias) = LOWER($2)
        """, [entity_type, raw_name])
        if exact:
            candidates.append(Candidate(**exact, method="exact_alias"))

        # 2. Fuzzy match (Jaro-Winkler)
        fuzzy = await db.fetch_all("""
            SELECT entity_id, name,
                   similarity(LOWER(name), LOWER($2)) as score
            FROM entities
            WHERE entity_type = $1
              AND similarity(LOWER(name), LOWER($2)) > 0.5
            ORDER BY score DESC
            LIMIT 5
        """, [entity_type, raw_name])
        for f in fuzzy:
            candidates.append(Candidate(**f, method="fuzzy"))

        # 3. Context boost (si tenemos liga/equipo)
        if context and candidates:
            candidates = self._apply_context_boost(candidates, context)

        return candidates

    def _apply_context_boost(
        self,
        candidates: list[Candidate],
        context: dict
    ) -> list[Candidate]:
        """Aumentar score si el contexto coincide."""

        for c in candidates:
            boost = 0.0
            # Si el jugador está en el equipo mencionado
            if context.get("team_id") and c.current_team_id == context["team_id"]:
                boost += 0.15
            # Si la liga coincide
            if context.get("competition_id") and c.competition_id == context["competition_id"]:
                boost += 0.10

            c.score = min(1.0, c.score + boost)

        return sorted(candidates, key=lambda c: c.score, reverse=True)
```

**Tabla de flags para revisión**:

```sql
CREATE TABLE titan.entity_resolution_flags (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL,
    raw_input TEXT NOT NULL,
    resolved_entity_id INTEGER,  -- NULL si SKIP
    confidence FLOAT NOT NULL,
    action VARCHAR(50) NOT NULL,
    context JSONB,
    source_id VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ,
    reviewed_by VARCHAR(100),
    review_action VARCHAR(50)  -- CONFIRM, REJECT, REMAP
);

-- Índice para revisiones pendientes
CREATE INDEX idx_flags_pending ON entity_resolution_flags(reviewed_at)
WHERE reviewed_at IS NULL;
```

**Dashboard de revisión semanal**:

```python
async def get_weekly_review_summary() -> dict:
    """Resumen semanal de entidades para revisar."""

    stats = await db.fetch_one("""
        SELECT
            COUNT(*) as total_flagged,
            COUNT(*) FILTER (WHERE action = 'AUTO_MATCH_FLAGGED') as auto_matched,
            COUNT(*) FILTER (WHERE action = 'SKIP_LOW_CONFIDENCE') as skipped,
            COUNT(*) FILTER (WHERE action = 'SKIP_NO_CANDIDATES') as no_candidates,
            AVG(confidence) as avg_confidence
        FROM entity_resolution_flags
        WHERE created_at > NOW() - INTERVAL '7 days'
          AND reviewed_at IS NULL
    """)

    return {
        "total_pending": stats["total_flagged"],
        "breakdown": {
            "auto_matched_needs_confirm": stats["auto_matched"],
            "skipped_low_confidence": stats["skipped"],
            "new_entities_maybe": stats["no_candidates"]
        },
        "avg_confidence": stats["avg_confidence"],
        "estimated_review_time_minutes": stats["total_flagged"] * 0.5  # 30s por item
    }
```

**Métricas de éxito**:

| Métrica | Target | Acción si falla |
|---------|--------|-----------------|
| Auto-resolve rate | >85% | Mejorar aliases/fuzzy |
| False positive rate | <5% | Subir umbral confianza |
| Pipeline blocked | 0% | Siempre fail-open |
| Review backlog | <500/semana | Automatizar más |

---

## 10.D Modo Bootstrapping (Estrategia de Arranque Conservador)

> **DECISIÓN OWNER**: Presupuesto disponible $1,500-2,000/mes, pero arrancar en modo conservador.
> Escalar solo cuando el modelo demuestre valor y el presupuesto se vuelva limitante.

**Principio**: No quemar $1,500/mes en proxies durante fase de pruebas. Demostrar valor primero.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FASES DE ESCALAMIENTO ECONÓMICO                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FASE 0: BOOTSTRAPPING (~$150-300/mes)                                      │
│  ════════════════════════════════════════                                   │
│  ├── Tier A: APIs Oficiales (API-Football, Understat, etc.)     ✅ ACTIVO  │
│  ├── Tier C: Prensa (Marca, ESPN, etc.) - SIN proxies agresivos ✅ ACTIVO  │
│  ├── Tier B: Casas de Apuestas (Playwright + proxies)           ❌ APAGADO │
│  ├── Tier D: Redes Sociales                                     ❌ APAGADO │
│  └── Tier E: Fuentes Especializadas                             ⏸️ MÍNIMO  │
│                                                                              │
│  Presupuesto:                                                                │
│  ├── API-Football: $99/mes (ya existente)                                   │
│  ├── R2 Storage: ~$5-10/mes                                                 │
│  ├── Gemini LLM: ~$20-50/mes (extracción limitada)                         │
│  ├── Proxies: $0 (no necesarios para Tier A/C)                             │
│  └── TOTAL: ~$150-300/mes                                                   │
│                                                                              │
│  Criterios de éxito para escalar:                                           │
│  ├── ✓ Pipeline estable 7 días consecutivos                                │
│  ├── ✓ >90% completeness en Tier 1-2 features                              │
│  ├── ✓ Modelo XGBoost entrenado con datos TITAN                            │
│  ├── ✓ Accuracy >= modelo actual (baseline)                                │
│  └── ✓ Owner aprueba escalamiento                                          │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FASE 1: EXPANSIÓN MODERADA (~$500-800/mes)                                 │
│  ════════════════════════════════════════════                               │
│  ├── Tier A: APIs Oficiales                                     ✅ ACTIVO  │
│  ├── Tier C: Prensa (ampliada)                                  ✅ ACTIVO  │
│  ├── Tier B: TOP 5 Casas de Apuestas (Pinnacle, Betfair, etc.)  ✅ ACTIVO  │
│  ├── Tier D: Twitter cuentas clave (@FabrizioRomano, @OptaJoe)  ✅ ACTIVO  │
│  └── Tier E: Transfermarkt, Physioroom                          ✅ ACTIVO  │
│                                                                              │
│  Presupuesto adicional:                                                      │
│  ├── Proxies residenciales: ~$200-400/mes (solo para Tier B)               │
│  ├── Twitter API: ~$100/mes (si es necesario)                              │
│  └── TOTAL: ~$500-800/mes                                                   │
│                                                                              │
│  Criterios de éxito para escalar:                                           │
│  ├── ✓ Odds data mejora accuracy del modelo                                │
│  ├── ✓ ROI positivo en predicciones (si aplica)                            │
│  └── ✓ Bandwidth de proxies se vuelve limitante                            │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FASE 2: OPERACIÓN COMPLETA (~$1,500-2,000/mes)                             │
│  ═══════════════════════════════════════════════                            │
│  ├── Todos los Tiers activos                                    ✅ ACTIVO  │
│  ├── 300+ fuentes                                               ✅ ACTIVO  │
│  ├── Backfill histórico completo                                ✅ ACTIVO  │
│  └── Operación 24/7                                             ✅ ACTIVO  │
│                                                                              │
│  Solo llegar aquí cuando:                                                    │
│  ├── ✓ El modelo demuestra valor real                                      │
│  ├── ✓ El presupuesto actual es insuficiente para las necesidades         │
│  └── ✓ Owner autoriza "abrir la llave"                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementación de Gates de Escalamiento**:

```python
class BudgetGate:
    """Control de escalamiento basado en métricas y presupuesto."""

    PHASES = {
        "bootstrap": {
            "budget_max": 300,
            "tiers_enabled": ["A", "C"],
            "proxy_budget": 0,
            "sources_max": 30
        },
        "expansion": {
            "budget_max": 800,
            "tiers_enabled": ["A", "B_limited", "C", "D_limited", "E_limited"],
            "proxy_budget": 400,
            "sources_max": 80
        },
        "full": {
            "budget_max": 2000,
            "tiers_enabled": ["A", "B", "C", "D", "E"],
            "proxy_budget": 1500,
            "sources_max": 300
        }
    }

    async def check_scale_readiness(self, current_phase: str) -> dict:
        """Verificar si estamos listos para escalar a la siguiente fase."""

        if current_phase == "bootstrap":
            return {
                "ready": await self._check_bootstrap_criteria(),
                "next_phase": "expansion",
                "blockers": await self._get_blockers("bootstrap")
            }
        elif current_phase == "expansion":
            return {
                "ready": await self._check_expansion_criteria(),
                "next_phase": "full",
                "blockers": await self._get_blockers("expansion")
            }
        return {"ready": False, "next_phase": None, "blockers": ["Already at max phase"]}

    async def _check_bootstrap_criteria(self) -> bool:
        """Criterios para salir de bootstrap."""
        checks = {
            "pipeline_stable_7d": await self._pipeline_stable_days() >= 7,
            "tier1_completeness": await self._tier_completeness(1) >= 0.90,
            "tier2_completeness": await self._tier_completeness(2) >= 0.85,
            "model_trained": await self._model_trained_with_titan(),
            "accuracy_baseline": await self._accuracy_meets_baseline()
        }
        return all(checks.values())

    async def _check_expansion_criteria(self) -> bool:
        """Criterios para ir a operación completa."""
        checks = {
            "odds_improves_accuracy": await self._odds_feature_value() > 0.02,
            "proxy_budget_limiting": await self._proxy_utilization() > 0.80,
            "owner_approved": await self._owner_approval_flag()
        }
        return all(checks.values())
```

**Configuración sources.yaml para Bootstrapping**:

```yaml
# sources_bootstrap.yaml - Configuración inicial conservadora

_meta:
  phase: bootstrap
  budget_limit: 300
  description: "Solo Tier A (APIs) y Tier C (Prensa sin proxies)"

sources:
  # ═══════════════════════════════════════════════════════════════
  # TIER A: APIs - ACTIVAS (sin costo de proxy)
  # ═══════════════════════════════════════════════════════════════
  api_football:
    enabled: true
    priority: 1
    # ... config existente

  understat:
    enabled: true
    priority: 1
    # ... config existente

  sofascore:
    enabled: true
    priority: 1
    requires_proxy: false  # Intentar sin proxy primero

  open_meteo:
    enabled: true
    priority: 1

  fbref:
    enabled: true
    priority: 2
    requires_proxy: false

  # ═══════════════════════════════════════════════════════════════
  # TIER B: Casas de Apuestas - DESACTIVADAS en bootstrap
  # ═══════════════════════════════════════════════════════════════
  bet365:
    enabled: false  # ❌ Requiere proxy residencial
    _note: "Activar en fase expansion"

  pinnacle:
    enabled: false  # ❌ Requiere proxy
    _note: "Activar en fase expansion"

  betfair_exchange:
    enabled: false  # ❌ Requiere proxy
    _note: "Activar en fase expansion"

  # ═══════════════════════════════════════════════════════════════
  # TIER C: Prensa - ACTIVAS (sin proxies agresivos)
  # ═══════════════════════════════════════════════════════════════
  marca:
    enabled: true
    priority: 3
    requires_proxy: false
    rate_limit:
      requests_per_minute: 10  # Conservador

  espn:
    enabled: true
    priority: 3
    requires_proxy: false
    rate_limit:
      requests_per_minute: 10

  # ═══════════════════════════════════════════════════════════════
  # TIER D/E: DESACTIVADAS en bootstrap
  # ═══════════════════════════════════════════════════════════════
  twitter_accounts:
    enabled: false
    _note: "Activar en fase expansion"

  transfermarkt:
    enabled: false  # Tiene anti-bot agresivo
    _note: "Activar en fase expansion con proxy"
```

**Dashboard de Estado de Fase**:

```python
async def get_phase_status() -> dict:
    """Estado actual del sistema y progreso hacia siguiente fase."""

    current = await db.fetch_one("SELECT value FROM config WHERE key = 'titan_phase'")
    phase = current["value"] if current else "bootstrap"

    return {
        "current_phase": phase,
        "budget": {
            "limit": BudgetGate.PHASES[phase]["budget_max"],
            "current_month": await get_current_month_spend(),
            "utilization_pct": await get_budget_utilization()
        },
        "sources": {
            "enabled": await count_enabled_sources(),
            "limit": BudgetGate.PHASES[phase]["sources_max"],
            "by_tier": await count_sources_by_tier()
        },
        "scale_readiness": await BudgetGate().check_scale_readiness(phase),
        "metrics": {
            "pipeline_stable_days": await pipeline_stable_days(),
            "tier1_completeness": await tier_completeness(1),
            "tier2_completeness": await tier_completeness(2)
        }
    }
```

**Ejemplo de output**:

```json
{
  "current_phase": "bootstrap",
  "budget": {
    "limit": 300,
    "current_month": 187.50,
    "utilization_pct": 62.5
  },
  "sources": {
    "enabled": 12,
    "limit": 30,
    "by_tier": {"A": 5, "C": 7, "B": 0, "D": 0, "E": 0}
  },
  "scale_readiness": {
    "ready": false,
    "next_phase": "expansion",
    "blockers": [
      "pipeline_stable_7d: 4/7 days",
      "model_trained: false"
    ]
  },
  "metrics": {
    "pipeline_stable_days": 4,
    "tier1_completeness": 0.92,
    "tier2_completeness": 0.87
  }
}
```

---

### 10.E MVP Definition (FASE 0 - Non-Goals Explícitos)

> **CRÍTICO**: Esta sección define qué está EN SCOPE y qué está FUERA para la primera versión funcional.

**Principio**: Entregar valor demostrable en 4-6 semanas con el mínimo absoluto viable.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MVP SCOPE (FASE 0)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ✅ IN SCOPE (MVP)                        ❌ NON-GOALS (Fase 1+)            │
│  ═══════════════════                      ══════════════════════             │
│                                                                              │
│  FUENTES:                                 FUENTES:                           │
│  ├── 5-12 Golden Sources                  ├── Casas de apuestas (Tier B)    │
│  │   ├── API-Football (fixtures/stats)    ├── Redes sociales (Tier D)       │
│  │   ├── Understat (xG)                   ├── Scraping con Playwright       │
│  │   ├── SofaScore (lineups/ratings)      ├── Proxies residenciales         │
│  │   ├── Open-Meteo (weather)             └── >12 fuentes simultáneas       │
│  │   ├── FBref (stats avanzadas)                                            │
│  │   └── + 2-7 opcionales estables                                          │
│  ├── Solo APIs/HTML estables                                                │
│  └── Sin proxies (requests directos)                                        │
│                                                                              │
│  FEATURES:                                FEATURES:                          │
│  ├── Tier 1: Cuotas/probabilidades        ├── Tier 5-8 (psicología, etc.)   │
│  │   (de API-Football, no scraping)       ├── Features derivadas complejas  │
│  ├── Tier 2: Forma/rachas/xG              ├── Cross-correlation features    │
│  ├── Tier 3: H2H/contexto competitivo     └── Entity resolution avanzado    │
│  └── Tier 4 parcial: clima/estadio                                          │
│      (solo si Open-Meteo lo provee)                                         │
│                                                                              │
│  BACKFILL:                                BACKFILL:                          │
│  ├── 3 ligas: La Liga, Premier, Serie A   ├── >5 ligas simultáneas          │
│  ├── 3 temporadas: 2022/23 - 2024/25      ├── >5 temporadas                 │
│  └── ~3,000-4,500 partidos                └── Copas/torneos internacionales │
│                                                                              │
│  STORAGE:                                 STORAGE:                           │
│  ├── PostgreSQL: tablas transformadas     ├── Particionado por fecha        │
│  ├── R2: raw JSON/HTML (evidencia)        ├── Archivado automático          │
│  └── Sin particionado inicial             └── Replicación multi-región      │
│                                                                              │
│  OPERACIÓN:                               OPERACIÓN:                         │
│  ├── Sync manual o cron básico            ├── Scheduler distribuido         │
│  ├── Alertas por log/Sentry               ├── Auto-scaling                  │
│  ├── Dashboard JSON endpoint              ├── Grafana dashboards            │
│  └── Retry simple (max 3)                 └── Circuit breakers avanzados    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Golden Sources MVP (5-12 fuentes)**:

| # | Fuente | Tipo | Datos | Proxy | Prioridad |
|---|--------|------|-------|-------|-----------|
| 1 | API-Football | API REST | fixtures, stats, odds, lineups | NO | CRÍTICO |
| 2 | Understat | JSON interno | xG, xGA, xPTS | NO | CRÍTICO |
| 3 | SofaScore | JSON interno | ratings, lineups, events | NO | ALTO |
| 4 | Open-Meteo | API REST | weather histórico y forecast | NO | ALTO |
| 5 | FBref | HTML estable | stats avanzadas (scraping educado) | NO | MEDIO |
| 6 | Transfermarkt* | HTML | market values, transfers | NO* | OPCIONAL |
| 7 | Football-Data.co.uk | CSV/API | odds históricas | NO | OPCIONAL |

*Transfermarkt tiene anti-bot moderado; intentar sin proxy, desactivar si falla.

**Ligas MVP (3 ligas × 3 temporadas = ~4,000 partidos)**:

| Liga | ID API-Football | Partidos/Temp | Total |
|------|-----------------|---------------|-------|
| La Liga | 140 | ~380 | ~1,140 |
| Premier League | 39 | ~380 | ~1,140 |
| Serie A | 135 | ~380 | ~1,140 |
| **TOTAL** | - | - | **~3,420** |

**Features Target MVP (Tier 1-3 + Tier 4 parcial)**:

| Tier | Variables | % del Total | Cobertura Esperada |
|------|-----------|-------------|-------------------|
| Tier 1 | 1-50 | 8% | >95% |
| Tier 2 | 51-100 | 8% | >90% |
| Tier 3 | 101-150 | 8% | >80% |
| Tier 4 (parcial) | 151-200 | 8% | >60% |
| **MVP TOTAL** | ~200 | 32% de 615 | - |

**Criterios de Éxito MVP**:

```python
MVP_SUCCESS_CRITERIA = {
    # Data quality
    "tier1_completeness": 0.95,      # 95% de features Tier 1 no-NULL
    "tier2_completeness": 0.90,      # 90% de features Tier 2 no-NULL
    "tier3_completeness": 0.80,      # 80% de features Tier 3 no-NULL

    # Coverage
    "matches_backfilled": 3000,       # Mínimo 3,000 partidos
    "leagues_active": 3,              # 3 ligas operativas
    "sources_healthy": 5,             # 5 fuentes funcionando

    # Reliability
    "pipeline_uptime_7d": 0.95,       # 95% uptime en 7 días
    "pit_violations": 0,              # Cero violaciones PIT

    # Model
    "model_trainable": True,          # XGBoost entrena sin errores
    "accuracy_vs_baseline": 0.0       # >= modelo actual (no peor)
}
```

**Backfill Histórico Paralelo (2015/16 → 2024/25)**:

> **ESTRATEGIA HÍBRIDA**: MVP pequeño para validar rápido + backfill histórico en background.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ESTRATEGIA DE BACKFILL                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FASE 1: MVP (semanas 1-2)           PARALELO: Backfill histórico           │
│  ════════════════════════            ════════════════════════════           │
│                                                                              │
│  ┌─────────────────────┐             ┌─────────────────────┐                │
│  │ 3 temporadas        │             │ 10 temporadas       │                │
│  │ 2022/23 - 2024/25   │             │ 2015/16 - 2024/25   │                │
│  │ ~3,420 partidos     │             │ ~11,400 partidos    │                │
│  │ PRIORIDAD: ALTA     │             │ PRIORIDAD: BAJA     │                │
│  └─────────────────────┘             └─────────────────────┘                │
│           │                                    │                            │
│           ▼                                    ▼                            │
│  ✅ Predicciones activas             ⏳ Corre en background                 │
│  ✅ Pipeline validado                ⏳ Pausable si hay carga               │
│  ✅ Modelo entrenado                 ⏳ Checkpoint por (liga, temp)         │
│                                      ⏳ Rate limits conservadores           │
│                                                                              │
│  MERGE cuando backfill completo + pasa quality gates                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Job de Backfill Histórico**:

```python
# Job separado, prioridad baja, pausable
HISTORICAL_BACKFILL_CONFIG = {
    "job_name": "historical_backfill",
    "priority": "LOW",           # Cede recursos al pipeline live
    "pausable": True,            # Se pausa si load > threshold

    # Scope
    "seasons": [
        "2015/16", "2016/17", "2017/18", "2018/19",
        "2019/20", "2020/21", "2021/22"  # MVP ya tiene 2022/23+
    ],
    "leagues": [140, 39, 135],   # La Liga, Premier, Serie A

    # Rate limits conservadores (no competir con live)
    "rate_limits": {
        "requests_per_minute": 30,    # vs 60 del pipeline normal
        "concurrent_leagues": 1,       # Una liga a la vez
        "pause_if_live_matches": True  # Parar si hay partidos en vivo
    },

    # Checkpoint para resumir sin duplicar
    "checkpoint": {
        "granularity": "season",       # Guarda progreso por temporada
        "key": "backfill:{league_id}:{season}",
        "resume_on_restart": True
    },

    # Quality gates antes de merge
    "merge_gates": {
        "tier1_completeness": 0.90,    # Puede ser menor que MVP
        "pit_violations": 0,
        "min_matches_per_season": 350  # ~92% de partidos
    }
}
```

**Cuándo habilitar features históricas**:

| Feature | Requiere | Habilitado |
|---------|----------|------------|
| Predicciones live | MVP (3 temp) | Inmediato |
| H2H últimos 5 años | Backfill 2019+ | Cuando pase gates |
| Tendencias 10 años | Backfill completo | Cuando pase gates |
| Modelo con >10k partidos | Backfill completo | Cuando pase gates |

---

### 10.F DDL Estándar y Convenciones de Timestamps

> **OBLIGATORIO**: Todas las tablas TITAN siguen este estándar para auditoría y PIT compliance.

**Política de Timezone**:
- **SIEMPRE UTC**: Todos los timestamps se almacenan y comparan en UTC
- **TIMESTAMPTZ**: Usar `TIMESTAMPTZ` (no `TIMESTAMP`) para evitar ambigüedad
- **PIT Critical**: `captured_at` y `kickoff_utc` DEBEN ser TIMESTAMPTZ para validaciones PIT correctas

**Columnas Estándar (tablas de ingesta/raw)**:

```sql
-- Columnas de auditoría obligatorias (SIEMPRE UTC)
created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- Cuándo se insertó el registro
updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- Última modificación
source_id       VARCHAR(50) NOT NULL,                -- Fuente origen (api_football, understat, etc.)
captured_at     TIMESTAMPTZ NOT NULL,                -- Cuándo se capturó el dato en origen (PIT)

-- Trigger para updated_at automático
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';
```

**Excepción: Tablas derivadas (ej: `feature_matrix`)**:
- Las tablas derivadas/materializadas NO tienen `source_id` único (agregan múltiples fuentes)
- En su lugar usan: `*_captured_at` por fuente (ej: `odds_captured_at`, `xg_captured_at`)
- Y un `pit_max_captured_at = GREATEST(todos los *_captured_at)` para el constraint PIT
- Ver DDL de `titan.feature_matrix` más adelante en esta sección

**Ejemplo DDL - Tabla `titan.matches` (ingesta directa)**:

```sql
CREATE TABLE titan.matches (
    -- PK
    match_id        BIGINT PRIMARY KEY,

    -- Core fields
    home_team_id    INT NOT NULL REFERENCES titan.teams(team_id),
    away_team_id    INT NOT NULL REFERENCES titan.teams(team_id),
    competition_id  INT NOT NULL REFERENCES titan.competitions(competition_id),
    season_id       INT NOT NULL REFERENCES titan.seasons(season_id),
    stadium_id      INT REFERENCES titan.stadiums(stadium_id),
    referee_id      INT REFERENCES titan.referees(referee_id),

    -- Match data (TIMESTAMPTZ para PIT compliance)
    kickoff_utc     TIMESTAMPTZ NOT NULL,
    status          VARCHAR(20) NOT NULL,  -- SCHEDULED, LIVE, FT, POSTPONED
    home_score      SMALLINT,
    away_score      SMALLINT,

    -- Auditoría estándar (OBLIGATORIO - siempre UTC)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_id       VARCHAR(50) NOT NULL,
    captured_at     TIMESTAMPTZ NOT NULL,

    -- Constraints
    CONSTRAINT valid_kickoff CHECK (kickoff_utc > '2000-01-01'::TIMESTAMPTZ),
    CONSTRAINT valid_scores CHECK (home_score >= 0 AND away_score >= 0)
);

-- Índices mínimos obligatorios
CREATE INDEX idx_matches_kickoff ON titan.matches(kickoff_utc);
CREATE INDEX idx_matches_competition_season ON titan.matches(competition_id, season_id);
CREATE INDEX idx_matches_teams ON titan.matches(home_team_id, away_team_id);
CREATE INDEX idx_matches_captured ON titan.matches(captured_at);  -- Para PIT queries

-- Trigger updated_at
CREATE TRIGGER update_matches_updated_at
    BEFORE UPDATE ON titan.matches
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

**Ejemplo DDL - Tabla `titan.raw_extractions` (staging)**:

```sql
CREATE TABLE titan.raw_extractions (
    -- PK compuesto
    extraction_id   BIGSERIAL PRIMARY KEY,

    -- Identificación del job
    source_id       VARCHAR(50) NOT NULL,
    job_id          UUID NOT NULL,
    url             TEXT NOT NULL,

    -- Contenido raw
    response_type   VARCHAR(20) NOT NULL,  -- json, html, csv
    response_body   JSONB,                 -- NULL si guardamos en R2
    r2_path         TEXT,                  -- Path en R2 si es muy grande
    http_status     SMALLINT NOT NULL,

    -- Metadata extracción (TIMESTAMPTZ - siempre UTC)
    captured_at     TIMESTAMPTZ NOT NULL,    -- Cuándo se hizo el request
    response_time_ms INT,

    -- Auditoría
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Para idempotency (32 hex chars from SHA256[:32])
    idempotency_key CHAR(32) NOT NULL,

    -- Constraints
    CONSTRAINT unique_extraction UNIQUE (idempotency_key)
);

-- Índice para lookups
CREATE INDEX idx_raw_source_captured ON titan.raw_extractions(source_id, captured_at);
```

**Política de Storage**:

```yaml
storage_policy:
  postgresql:
    # Tablas transformadas (3NF) - permanente
    titan.matches: permanent
    titan.teams: permanent
    titan.players: permanent
    titan.match_odds: permanent
    titan.team_season_stats: permanent
    # ... todas las 41 tablas

    # Staging tables - retención limitada
    titan.raw_extractions:
      retention_days: 30
      archive_to: r2

  r2:
    # Blobs raw - retención larga para evidencia
    bucket: futbolstats-titan-raw
    paths:
      - /extractions/{source_id}/{date}/{job_id}.json
      - /snapshots/{source_id}/{date}/{url_hash}.html
    retention_days: 365
    lifecycle_rule: INTELLIGENT_TIERING
```

**Timestamps Críticos para PIT**:

| Timestamp | Significado | Uso |
|-----------|-------------|-----|
| `kickoff_utc` | Hora del partido | Referencia temporal del evento |
| `captured_at` | Cuándo se capturó el dato | **CRÍTICO para PIT**: features con `captured_at < kickoff_utc` |
| `created_at` | Cuándo entró a la DB | Auditoría |
| `updated_at` | Última modificación | Auditoría |

**DDL - Tabla `titan.feature_matrix` (derivada para ML)**:

```sql
-- Tabla materializada con features listos para XGBoost
-- NOTA: Esta es una tabla DERIVADA. Los captured_at son MAX() de fuentes origen.

CREATE TABLE titan.feature_matrix (
    -- PK
    match_id            BIGINT PRIMARY KEY REFERENCES titan.matches(match_id),

    -- Referencia temporal (del partido)
    kickoff_utc         TIMESTAMPTZ NOT NULL,
    competition_id      INT NOT NULL,
    season_id           INT NOT NULL,

    -- ═══════════════════════════════════════════════════════════════
    -- TIER 1: ODDS (max predictivo) + sus captured_at
    -- ═══════════════════════════════════════════════════════════════
    odds_home_close     DECIMAL(6,3),
    odds_draw_close     DECIMAL(6,3),
    odds_away_close     DECIMAL(6,3),
    implied_prob_home   DECIMAL(5,4),
    implied_prob_draw   DECIMAL(5,4),
    implied_prob_away   DECIMAL(5,4),
    odds_captured_at    TIMESTAMPTZ,  -- MAX(captured_at) de match_odds válido PIT

    -- ═══════════════════════════════════════════════════════════════
    -- TIER 1-2: xG (Understat)
    -- ═══════════════════════════════════════════════════════════════
    xg_home_season      DECIMAL(5,2),  -- xG acumulado temporada
    xg_away_season      DECIMAL(5,2),
    xg_home_last5       DECIMAL(5,2),  -- xG últimos 5 partidos
    xg_away_last5       DECIMAL(5,2),
    xg_captured_at      TIMESTAMPTZ,   -- MAX(captured_at) de team_season_stats

    -- ═══════════════════════════════════════════════════════════════
    -- TIER 2: FORMA
    -- ═══════════════════════════════════════════════════════════════
    form_home_last5     VARCHAR(5),    -- "WWDLW"
    form_away_last5     VARCHAR(5),
    goals_home_last5    SMALLINT,
    goals_away_last5    SMALLINT,
    conceded_home_last5 SMALLINT,
    conceded_away_last5 SMALLINT,
    form_captured_at    TIMESTAMPTZ,   -- MAX(captured_at) de matches previos

    -- ═══════════════════════════════════════════════════════════════
    -- TIER 3: H2H
    -- ═══════════════════════════════════════════════════════════════
    h2h_total_matches   SMALLINT,
    h2h_home_wins       SMALLINT,
    h2h_draws           SMALLINT,
    h2h_away_wins       SMALLINT,
    h2h_avg_goals       DECIMAL(4,2),
    h2h_captured_at     TIMESTAMPTZ,   -- MAX(captured_at) de match_h2h

    -- ... (features adicionales Tier 4-8 se agregan en fases posteriores)

    -- ═══════════════════════════════════════════════════════════════
    -- PIT COMPLIANCE: timestamp más reciente de TODAS las fuentes
    -- ═══════════════════════════════════════════════════════════════
    pit_max_captured_at TIMESTAMPTZ NOT NULL,  -- GREATEST(odds_captured_at, xg_captured_at, ...)

    -- Target (solo para partidos FT)
    outcome             VARCHAR(10),   -- HOME_WIN, DRAW, AWAY_WIN (NULL si no FT)

    -- Flags de calidad
    tier1_complete      BOOLEAN NOT NULL DEFAULT FALSE,  -- odds + xG disponibles
    tier2_complete      BOOLEAN NOT NULL DEFAULT FALSE,  -- forma completa
    tier3_complete      BOOLEAN NOT NULL DEFAULT FALSE,  -- H2H disponible

    -- Auditoría
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- PIT Constraint: CRÍTICO - ningún dato puede ser posterior al kickoff
    CONSTRAINT pit_valid CHECK (pit_max_captured_at < kickoff_utc)
);

-- Índices para queries comunes
CREATE INDEX idx_fm_kickoff ON titan.feature_matrix(kickoff_utc);
CREATE INDEX idx_fm_competition_season ON titan.feature_matrix(competition_id, season_id);
CREATE INDEX idx_fm_pit ON titan.feature_matrix(pit_max_captured_at);
CREATE INDEX idx_fm_tier1 ON titan.feature_matrix(tier1_complete) WHERE tier1_complete = TRUE;

-- Trigger updated_at
CREATE TRIGGER update_feature_matrix_updated_at
    BEFORE UPDATE ON titan.feature_matrix
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

**Cálculo de `captured_at` en tabla derivada**:

```python
def build_feature_row(match_id: int) -> dict:
    """
    Para feature_matrix (tabla derivada), captured_at se calcula como:
    MAX(source_captured_at) de cada fuente, SIEMPRE < kickoff_utc
    """
    match = get_match(match_id)
    t0 = match.kickoff_utc

    # Obtener datos de cada fuente CON su captured_at
    odds = get_latest_odds(match_id, before=t0)       # captured_at incluido
    xg = get_latest_xg(match.home_team_id, before=t0) # captured_at incluido
    form = get_latest_form(match_id, before=t0)       # captured_at incluido
    h2h = get_h2h(match.home_team_id, match.away_team_id, before=t0)

    return {
        "match_id": match_id,
        "kickoff_utc": t0,

        # Features
        "odds_home_close": odds.home if odds else None,
        "xg_home_season": xg.season_xg if xg else None,
        # ...

        # Captured_at por fuente (para trazabilidad)
        "odds_captured_at": odds.captured_at if odds else None,
        "xg_captured_at": xg.captured_at if xg else None,
        "form_captured_at": form.captured_at if form else None,
        "h2h_captured_at": h2h.captured_at if h2h else None,

        # PIT global = max de todos (para constraint)
        "pit_max_captured_at": max(filter(None, [
            odds.captured_at if odds else None,
            xg.captured_at if xg else None,
            form.captured_at if form else None,
            h2h.captured_at if h2h else None,
        ])),

        # Flags
        "tier1_complete": odds is not None and xg is not None,
        "tier2_complete": form is not None,
        "tier3_complete": h2h is not None,
    }
```

---

### 10.G Data Lineage - Top 50 Features (Tier 1)

> **TRAZABILIDAD**: Para cada feature crítica, documentar origen, cálculo y timestamp que manda.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATA LINEAGE - TIER 1 (Features 1-50)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Feature → Source → Extraction → Transform → Timestamp                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

| # | Feature | Source | Endpoint/Selector | Transform | PIT Timestamp |
|---|---------|--------|-------------------|-----------|---------------|
| 1 | `odds_home_close` | api_football | `/odds?fixture={id}` | `bookmakers[pinnacle].bets[1x2].home` | `captured_at` del último sync antes de `kickoff_utc` |
| 2 | `ah_home_odds_close` | api_football | `/odds?fixture={id}` | `bookmakers[pinnacle].bets[ah].home` | ídem |
| 3 | `odds_away_close` | api_football | `/odds?fixture={id}` | `bookmakers[pinnacle].bets[1x2].away` | ídem |
| 4 | `ah_away_odds_close` | api_football | `/odds?fixture={id}` | `bookmakers[pinnacle].bets[ah].away` | ídem |
| 5 | `odds_draw_close` | api_football | `/odds?fixture={id}` | `bookmakers[pinnacle].bets[1x2].draw` | ídem |
| 6-11 | `odds_over25_close`, etc. | api_football | `/odds?fixture={id}` | Extraer de array `bets` | ídem |
| 12-14 | `implied_prob_*` | **CALCULADO** | - | `1 / odds_*_close` normalizado | Hereda de odds |
| 15-17 | `odds_movement_*` | **CALCULADO** | - | `odds_*_close - odds_*_open` | Hereda de odds |
| 18-22 | `ah_line_*`, `sharp_*` | api_football / **CALC** | `/odds` | Línea AH + detección smart money | ídem |
| 23-24 | `odds_vs_form_discrepancy_*` | **CALCULADO** | - | `implied_prob - form_based_prob` | Max de ambos |
| 25-26 | `xg_last5`, `xga_last5` | understat | `/team/{id}/{season}` | Rolling 5 partidos, `sum(xG)/5` | `captured_at` de Understat |
| 27 | `h2h_avg_goals_last5` | api_football | `/fixtures/headtohead` | `avg(home_goals + away_goals)` últimos 5 | `captured_at` H2H |
| 28-29 | `goals_scored_last5`, `goals_conceded_last5` | api_football | `/fixtures?team={id}` | Sum últimos 5 partidos FT | `captured_at` fixtures |
| 30 | `goals_last5` (player) | api_football | `/players?id={id}` | Sum goles últimos 5 | `captured_at` player stats |
| 31-33 | `form_last5*` | api_football | `/teams?id={id}` | String "WWDLW" o puntos | ídem |
| 34-40 | `h2h_*` | api_football | `/fixtures/headtohead` | Agregados de H2H | `captured_at` H2H |
| 41-44 | `*_streak` | **CALCULADO** | - | Conteo secuencial desde fixtures | Hereda de fixtures |
| 45 | `player_xg_match` | understat | `/match/{id}` | Player xG del partido **anterior** | `captured_at` < `kickoff_utc` |
| 46-52 | `xg_*` stats | understat | `/team/{id}` | Agregados xG temporada/rolling | `captured_at` Understat |

**Regla de Oro PIT**:

```python
def get_feature_timestamp(feature_name: str, match: Match) -> datetime:
    """
    Para cualquier feature, el timestamp PIT es el MÁS RECIENTE
    de todos los captured_at de las fuentes involucradas,
    SIEMPRE que sea < match.kickoff_utc.
    """
    sources = FEATURE_LINEAGE[feature_name]["sources"]
    timestamps = [
        get_latest_captured_at(source, before=match.kickoff_utc)
        for source in sources
    ]
    return max(timestamps)  # El más reciente que sigue siendo válido PIT
```

**Validación PIT (query de auditoría)**:

```sql
-- Encontrar features con posible data leakage
SELECT
    fm.match_id,
    fm.kickoff_utc,
    fm.odds_captured_at,
    fm.xg_captured_at,
    CASE
        WHEN fm.odds_captured_at >= fm.kickoff_utc THEN 'LEAKAGE: odds'
        WHEN fm.xg_captured_at >= fm.kickoff_utc THEN 'LEAKAGE: xg'
        ELSE 'OK'
    END as pit_status
FROM titan.feature_matrix fm
WHERE fm.odds_captured_at >= fm.kickoff_utc
   OR fm.xg_captured_at >= fm.kickoff_utc;
-- Esperado: 0 filas
```

---

### 10.H Job Contract (Idempotency, Retry, DLQ)

> **CONTRATO**: Especificación exacta de cómo se comportan los jobs de extracción.

**Idempotency Key**:

```python
def compute_idempotency_key(
    source_id: str,
    endpoint: str,
    params: dict,
    date_bucket: str  # YYYY-MM-DD o YYYY-MM-DD-HH según granularidad
) -> str:
    """
    Genera key única para evitar duplicados.
    Si el job ya existe con esta key, se skipea.
    """
    normalized_params = json.dumps(params, sort_keys=True)
    raw = f"{source_id}|{endpoint}|{normalized_params}|{date_bucket}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]

# Ejemplos:
# api_football|/fixtures|{"league":140,"season":2024}|2026-01-25
# understat|/team/87/2024|{}|2026-01-25
```

**Date Bucket por Tipo de Datos**:

| Tipo de Dato | Granularidad | Ejemplo Key |
|--------------|--------------|-------------|
| Fixtures (programados) | DIARIA | `2026-01-25` |
| Odds (pre-match) | CADA 6 HORAS | `2026-01-25-12` |
| Live scores | CADA HORA | `2026-01-25-14` |
| Stats post-partido | DIARIA | `2026-01-25` |
| xG (Understat) | DIARIA | `2026-01-25` |

**Retry Policy**:

```python
class RetryPolicy:
    """Política de reintentos por tipo de error."""

    RETRY_CONFIG = {
        # Errores transitorios: reintentar
        "timeout": {"max_attempts": 3, "backoff": "exponential", "base_delay": 5},
        "502": {"max_attempts": 3, "backoff": "exponential", "base_delay": 10},
        "503": {"max_attempts": 3, "backoff": "exponential", "base_delay": 30},
        "connection_error": {"max_attempts": 3, "backoff": "exponential", "base_delay": 5},

        # Rate limit: reintentar con backoff largo
        "429": {"max_attempts": 5, "backoff": "exponential", "base_delay": 60},

        # Errores permanentes: NO reintentar, ir a DLQ
        "400": {"max_attempts": 1, "send_to_dlq": True},
        "401": {"max_attempts": 1, "send_to_dlq": True, "alert": True},
        "403": {"max_attempts": 1, "send_to_dlq": True, "alert": True},
        "404": {"max_attempts": 1, "send_to_dlq": False},  # Recurso no existe, skip

        # Errores de parsing: NO reintentar
        "parse_error": {"max_attempts": 1, "send_to_dlq": True},
    }

    async def execute_with_retry(self, job: Job) -> JobResult:
        """Ejecutar job con política de retry."""
        attempts = 0
        last_error = None

        while attempts < self.get_max_attempts(job):
            attempts += 1
            try:
                result = await job.execute()
                return JobResult(status="SUCCESS", data=result)
            except Exception as e:
                last_error = e
                error_type = self.classify_error(e)
                config = self.RETRY_CONFIG.get(error_type, {"max_attempts": 1})

                if attempts >= config["max_attempts"]:
                    break

                delay = self.calculate_backoff(config, attempts)
                await asyncio.sleep(delay)

        # Agotados los reintentos
        return await self.handle_failure(job, last_error)
```

**Dead Letter Queue (DLQ)**:

```sql
CREATE TABLE titan.job_dlq (
    dlq_id          BIGSERIAL PRIMARY KEY,
    job_id          UUID NOT NULL,
    source_id       VARCHAR(50) NOT NULL,
    idempotency_key CHAR(32) NOT NULL,  -- SHA256[:32] hex

    -- Detalles del fallo
    error_type      VARCHAR(50) NOT NULL,
    error_message   TEXT,
    http_status     SMALLINT,
    attempts        SMALLINT NOT NULL,

    -- Request info para replay
    endpoint        TEXT NOT NULL,
    params          JSONB,

    -- Timestamps (TIMESTAMPTZ - siempre UTC)
    first_attempt   TIMESTAMPTZ NOT NULL,
    last_attempt    TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Resolución
    resolved_at     TIMESTAMPTZ,
    resolution      VARCHAR(50),  -- RETRIED, SKIPPED, MANUAL_FIX
    resolved_by     VARCHAR(100)
);

CREATE INDEX idx_dlq_pending ON titan.job_dlq(source_id, created_at)
WHERE resolved_at IS NULL;
```

**Job States**:

```
                    ┌──────────────┐
                    │   PENDING    │
                    └──────┬───────┘
                           │ scheduler picks up
                           ▼
                    ┌──────────────┐
             ┌──────│   RUNNING    │──────┐
             │      └──────────────┘      │
             │ transient error            │ success
             │ (retry)                    │
             ▼                            ▼
      ┌──────────────┐            ┌──────────────┐
      │   RETRYING   │            │  COMPLETED   │
      └──────┬───────┘            └──────────────┘
             │ max retries
             ▼
      ┌──────────────┐
      │    FAILED    │
      └──────┬───────┘
             │ permanent error
             ▼
      ┌──────────────┐
      │     DLQ      │──────▶ Manual review / auto-retry later
      └──────────────┘
```

---

### 10.I Checklist Operacional (1 Página)

> **RUNBOOK MÍNIMO**: Alertas y acciones para operación diaria de TITAN.

**Nota sobre Dashboards (alineado con 10.E)**:
- **MVP**: Dashboard = endpoint `/dashboard/titan.json` (JSON puro, consumible por cualquier cliente)
- **Fase 1+**: Grafana dashboards opcionales cuando haya métricas Prometheus integradas
- El "dashboard" en este checklist se refiere al endpoint JSON, NO a Grafana

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TITAN OPERATIONAL CHECKLIST (MVP)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ALERTAS CRÍTICAS (Acción inmediata <1h)                                    │
│  ═══════════════════════════════════════                                    │
│                                                                              │
│  🚨 [CRITICAL] Golden Source Down                                           │
│     Condición: source_id IN (api_football, understat, sofascore, open_meteo)│
│                AND consecutive_failures >= 3                                 │
│     Acción: Revisar logs, verificar API status, contactar Owner si >4h      │
│                                                                              │
│  🚨 [CRITICAL] Tier 1 Coverage <90%                                         │
│     Query: SELECT AVG(CASE WHEN odds_home_close IS NOT NULL THEN 1 ELSE 0   │
│            END) FROM titan.feature_matrix WHERE kickoff_utc > NOW()-'7d'    │
│     Acción: Identificar fuente fallando, revisar DLQ                        │
│                                                                              │
│  🚨 [CRITICAL] PIT Violation Detected                                       │
│     Query: SELECT COUNT(*) FROM titan.feature_matrix                        │
│            WHERE pit_max_captured_at >= kickoff_utc                         │
│     Acción: STOP pipeline, investigar fuente de leakage, purgar datos       │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ALERTAS WARNING (Revisar en <24h)                                          │
│  ═════════════════════════════════                                          │
│                                                                              │
│  ⚠️ [WARNING] Tier 2 Coverage <85%                                          │
│     Similar a Tier 1 pero umbral 85%                                        │
│                                                                              │
│  ⚠️ [WARNING] Rate Limit Spike (>10% requests 429)                          │
│     Query: SELECT source_id, COUNT(*) FILTER (WHERE http_status = 429)      │
│            / COUNT(*)::float as rate_429 FROM titan.raw_extractions         │
│            WHERE created_at > NOW() - '1h' GROUP BY source_id               │
│     Acción: Reducir rate limit config, revisar si IP bloqueada              │
│                                                                              │
│  ⚠️ [WARNING] DLQ Backlog >50                                               │
│     Query: SELECT COUNT(*) FROM titan.job_dlq WHERE resolved_at IS NULL     │
│     Acción: Revisar errores, resolver manualmente o auto-retry              │
│                                                                              │
│  ⚠️ [WARNING] Entity Resolution Backlog >200/día                            │
│     Query: SELECT COUNT(*) FROM titan.entity_resolution_flags               │
│            WHERE created_at > NOW() - '1d' AND reviewed_at IS NULL          │
│     Acción: Revisar aliases, mejorar fuzzy matching                         │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  MÉTRICAS DASHBOARD (Revisar diario)                                        │
│  ═══════════════════════════════════                                        │
│                                                                              │
│  📊 Completeness por Tier                                                   │
│     tier1_pct: >95% ✅  |  90-95% ⚠️  |  <90% 🚨                            │
│     tier2_pct: >90% ✅  |  85-90% ⚠️  |  <85% 🚨                            │
│     tier3_pct: >80% ✅  |  70-80% ⚠️  |  <70% 🚨                            │
│                                                                              │
│  📊 Freshness (última actualización)                                        │
│     api_football: <1h ✅  |  1-6h ⚠️  |  >6h 🚨                             │
│     understat: <6h ✅  |  6-24h ⚠️  |  >24h 🚨                              │
│     sofascore: <6h ✅  |  6-24h ⚠️  |  >24h 🚨                              │
│                                                                              │
│  📊 Error Rates por Dominio                                                 │
│     rate_429: <5% ✅  |  5-10% ⚠️  |  >10% 🚨                               │
│     rate_403: <1% ✅  |  1-5% ⚠️  |  >5% 🚨 (posible bloqueo)               │
│     rate_5xx: <2% ✅  |  2-5% ⚠️  |  >5% 🚨                                 │
│                                                                              │
│  📊 Costo Proxy (Fase 1+)                                                   │
│     monthly_gb: tracking only en MVP (esperado: $0)                         │
│     Cuando activo: <budget ✅  |  80-100% budget ⚠️  |  >100% 🚨            │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  QUERIES DE DIAGNÓSTICO RÁPIDO                                              │
│  ═════════════════════════════════                                          │
│                                                                              │
│  -- Estado general                                                          │
│  SELECT source_id,                                                          │
│         COUNT(*) as total_24h,                                              │
│         COUNT(*) FILTER (WHERE http_status = 200) as success,               │
│         COUNT(*) FILTER (WHERE http_status = 429) as rate_limited,          │
│         MAX(captured_at) as last_success                                    │
│  FROM titan.raw_extractions                                                 │
│  WHERE created_at > NOW() - '24h'                                           │
│  GROUP BY source_id;                                                        │
│                                                                              │
│  -- DLQ pendiente                                                           │
│  SELECT source_id, error_type, COUNT(*)                                     │
│  FROM titan.job_dlq WHERE resolved_at IS NULL                               │
│  GROUP BY source_id, error_type ORDER BY COUNT(*) DESC;                     │
│                                                                              │
│  -- Coverage por liga (últimos 7 días)                                      │
│  SELECT c.name,                                                             │
│         COUNT(*) as matches,                                                │
│         AVG(CASE WHEN fm.odds_home_close IS NOT NULL THEN 1 ELSE 0 END)    │
│           as tier1_coverage                                                 │
│  FROM titan.feature_matrix fm                                               │
│  JOIN titan.matches m ON fm.match_id = m.match_id                           │
│  JOIN titan.competitions c ON m.competition_id = c.competition_id           │
│  WHERE m.kickoff_utc > NOW() - '7d'                                         │
│  GROUP BY c.name;                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Endpoint Dashboard MVP**:

```python
@app.get("/dashboard/titan.json")
async def titan_dashboard():
    """Dashboard operacional TITAN."""
    return {
        "phase": "bootstrap",
        "sources": await get_source_health(),
        "coverage": {
            "tier1": await tier_coverage(1),
            "tier2": await tier_coverage(2),
            "tier3": await tier_coverage(3)
        },
        "freshness": await get_freshness_by_source(),
        "error_rates": await get_error_rates_24h(),
        "dlq": {
            "pending": await count_dlq_pending(),
            "by_source": await dlq_by_source()
        },
        "entity_flags": await count_entity_flags_pending(),
        "last_updated": datetime.utcnow().isoformat()
    }
```

---

## 11. Tareas para Owner (OWNER_TASKS)

### Acciones Requeridas del Owner

1. **Crear cuenta Cloudflare R2**
   - Bucket name sugerido: `futbolstats-titan-raw`
   - Region: Auto (o EU si hay preferencia GDPR)
   - Obtener: Access Key ID + Secret Access Key

2. **Proxies residenciales (DIFERIDO - Fase Expansion)**
   - **NO contratar en Fase Bootstrap**
   - Opciones para cuando sea necesario:
     - Bright Data (ex-Luminati): ~$15/GB
     - Oxylabs: ~$12/GB
     - Smartproxy: ~$8/GB
   - Contratar solo cuando: criterios de bootstrap cumplidos + Owner aprueba

3. **Verificar limites Railway PostgreSQL**
   - Ejecutar: `railway status` o revisar dashboard
   - Necesitamos: Storage disponible, max connections
   - Considerar upgrade si <50GB disponible

4. **Presupuesto mensual estimado TITAN (por fase)**

   **FASE BOOTSTRAP (inicial):**
   - API-Football: $99/mes (ya existente)
   - R2: ~$5-10/mes
   - LLM (Gemini): ~$20-50/mes
   - Proxies: $0
   - **TOTAL Bootstrap**: ~$150-300/mes

   **FASE EXPANSION (cuando se apruebe):**
   - Todo lo anterior +
   - Proxies residenciales: ~$200-400/mes
   - Twitter API: ~$100/mes (opcional)
   - **TOTAL Expansion**: ~$500-800/mes

   **FASE FULL (máximo autorizado):**
   - Proxies: ~$800-1500/mes
   - **TOTAL Full**: ~$1,500-2,000/mes (tope autorizado por Owner)

5. **Decisiones de negocio (RESUELTAS - ver Sección 10.E MVP Definition)**
   - ✅ Prioridad fase inicial: Tier A (APIs) + Tier C (Prensa) - SIN casas de apuestas
   - ✅ Casas de apuestas: Activar en Fase Expansion cuando el modelo demuestre valor
   - ✅ Ligas MVP: La Liga, Premier League, Serie A (3 ligas × ~380 partidos/temporada)
   - ✅ Temporadas MVP: 2022/23, 2023/24, 2024/25 (3 temporadas = ~3,420 partidos)
   - ✅ Backfill histórico: 2015/16 - 2024/25 (10 temporadas, ~11,400 partidos) - paralelo, prioridad baja
   - ✅ Budget máximo: $1,500-2,000/mes (escalable cuando se justifique)

6. **Lista inicial de fuentes a scrapear**
   - Necesito que me pases una lista de:
     - Casas de apuestas que usas/conoces
     - Periodicos deportivos que consultas
     - Cuentas de Twitter que sigues
     - Cualquier otra fuente de datos que consideres valiosa

---

## 12. Verificacion Post-Implementacion

### Checklist de Validacion

```bash
# 0. Ejecutar una corrida pequeña (evidencia end-to-end)
python -m app.titan.runner --date 2026-01-26 --league 140 --limit 5

# 1. Verificar tablas creadas (FASE 1)
psql $DATABASE_URL -c "\dt titan.*"
# Esperado FASE 1: 3 tablas (raw_extractions, job_dlq, feature_matrix)

# 2. Verificar coverage tier 1-2
SELECT
  COUNT(*) as total_matches,
  AVG(CASE WHEN odds_home_close IS NOT NULL THEN 1 ELSE 0 END) as tier1_coverage
FROM titan.feature_matrix;
# Esperado: tier1_coverage > 0.95

# 3. Verificar PIT compliance
SELECT COUNT(*) FROM titan.feature_matrix
WHERE pit_max_captured_at >= kickoff_utc;
# Esperado: 0 (cero violaciones - constraint PIT debería prevenirlo)

# 4. Verificar dashboard TITAN (protegido)
curl -s -H "X-Dashboard-Token: $TOKEN" \
  "$API_URL/dashboard/titan.json" | jq '.pit_compliance, .dlq, .extractions, .feature_matrix'
# Esperado: pit_compliance.compliant = true, y métricas presentes
```

---

## Aprobacion

| Rol | Nombre | Fecha | Estado |
|-----|--------|-------|--------|
| Auditor Backend | ABE | 2026-01-25 | CONDICIONES INCORPORADAS |
| Auditor 2 | - | 2026-01-25 | APROBADO (FASE 1 COMPLETADA) |
| Owner | David | - | PENDIENTE |

### Condiciones ABE (incorporadas en Sección 10):
- ✅ 10.A: Estrategia de Mantenimiento Pareto (80/20)
- ✅ 10.B: Protocolo de Ahorro de Ancho de Banda (Bandwidth-Miser)
- ✅ 10.C: Política de Fail-Open en Entidades

### Condiciones Auditor 2 - Ronda 1 (Sección 10.D):
- ✅ 10.D: Modo Bootstrapping (arranque conservador ~$150-300/mes)
- ✅ Tier B (Casas de Apuestas) desactivado hasta demostrar valor
- ✅ Escalamiento condicionado a métricas + aprobación Owner
- ✅ Presupuesto máximo autorizado: $1,500-2,000/mes

### Condiciones Auditor 2 - Ronda 2 (Secciones 10.E-10.I):
- ✅ 10.E: MVP Definition (Non-Goals explícitos, 5-12 Golden Sources, Tier 1-3 features)
- ✅ 10.F: DDL Estándar y Timestamps (created_at, updated_at, source_id, captured_at)
- ✅ 10.G: Data Lineage (Top 50 Features con origen, cálculo, timestamp PIT)
- ✅ 10.H: Job Contract (Idempotency key, retry policy, DLQ)
- ✅ 10.I: Checklist Operacional (alertas Critical/Warning, queries diagnóstico)

### Pendiente:
- [ ] Deploy migraciones en Railway + corrida piloto (evidencia en DB)
- [ ] Aprobación Owner

---

**Siguiente paso**: Deploy de migraciones en Railway, corrida piloto con `app/titan/runner.py`, y luego plan detallado para FASE 2+ con tickets.
