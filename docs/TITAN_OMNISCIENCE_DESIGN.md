# TITAN OMNISCIENCE - Technical Design Document

**Sistema Enterprise de Scraping Masivo para Prediccion Deportiva**

| Metadata | Valor |
|----------|-------|
| Version | 2.0 DRAFT |
| Fecha | 2026-01-25 |
| Autor | Claude Code (para revision ABE) |
| Estado | PENDIENTE APROBACION |

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

### FASE 1: Infraestructura Base
- [ ] Crear estructura `titan/` en repositorio
- [ ] Definir modelos SQLAlchemy para 41 tablas
- [ ] Setup Cloudflare R2 bucket para raw storage
- [ ] Implementar `sources.yaml` config parser
- [ ] Setup proxy pool (residencial) para anti-bloqueo
- [ ] CI/CD para nuevos modulos

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

## 10. Tareas para Owner (OWNER_TASK.md)

### Acciones Requeridas del Owner

1. **Crear cuenta Cloudflare R2**
   - Bucket name sugerido: `futbolstats-titan-raw`
   - Region: Auto (o EU si hay preferencia GDPR)
   - Obtener: Access Key ID + Secret Access Key

2. **Contratar servicio de proxies residenciales**
   - Opciones recomendadas:
     - Bright Data (ex-Luminati): ~$15/GB
     - Oxylabs: ~$12/GB
     - Smartproxy: ~$8/GB
   - Necesitamos: ~100GB/mes inicialmente
   - Presupuesto estimado: $800-1500/mes

3. **Verificar limites Railway PostgreSQL**
   - Ejecutar: `railway status` o revisar dashboard
   - Necesitamos: Storage disponible, max connections
   - Considerar upgrade si <50GB disponible

4. **Presupuesto mensual estimado TITAN**
   - R2: ~$5-15/mes (50-100GB stored)
   - Proxies: ~$800-1500/mes
   - Railway: Plan actual + posible upgrade
   - API-Football: $99/mes (ya existente)
   - LLM (Gemini): ~$50-100/mes (extraccion texto)
   - **TOTAL**: ~$1000-1800/mes

5. **Decisiones de negocio**
   - Prioridad de fuentes para fase inicial (casas de apuestas?)
   - Ligas prioritarias para backfill (Top 5 primero?)
   - Temporada minima historica (2015? 2010?)
   - Tolerancia a datos faltantes por tier
   - Budget maximo mensual para proxies

6. **Lista inicial de fuentes a scrapear**
   - Necesito que me pases una lista de:
     - Casas de apuestas que usas/conoces
     - Periodicos deportivos que consultas
     - Cuentas de Twitter que sigues
     - Cualquier otra fuente de datos que consideres valiosa

---

## 11. Verificacion Post-Implementacion

### Checklist de Validacion

```bash
# 1. Verificar tablas creadas
psql $DATABASE_URL -c "\dt titan.*"
# Esperado: 41 tablas

# 2. Verificar coverage tier 1-2
SELECT
  COUNT(*) as total_matches,
  AVG(CASE WHEN odds_home_close IS NOT NULL THEN 1 ELSE 0 END) as tier1_coverage
FROM titan.feature_matrix;
# Esperado: tier1_coverage > 0.95

# 3. Verificar PIT compliance
SELECT COUNT(*) FROM titan.feature_matrix
WHERE captured_at >= kickoff_utc;
# Esperado: 0 (cero violaciones)

# 4. Verificar jobs running
curl -s -H "X-Dashboard-Token: $TOKEN" \
  "$API_URL/dashboard/ops.json" | jq '.titan.jobs'
# Esperado: jobs activos sin errores
```

---

## Aprobacion

| Rol | Nombre | Fecha | Estado |
|-----|--------|-------|--------|
| Auditor Backend | ABE | - | PENDIENTE |
| Owner | David | - | PENDIENTE |

---

**Siguiente paso**: Una vez aprobado este documento por ABE, se procedera a crear el plan de accion detallado por cada fase con tickets especificos.
