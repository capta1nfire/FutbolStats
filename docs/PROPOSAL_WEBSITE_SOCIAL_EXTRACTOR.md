# Propuesta: Website Social Media Extractor

**Autor**: David (Owner)
**Revisor**: ABE (Auditor Backend)
**Fecha**: 2026-02-04
**Estado**: Pendiente aprobación ABE
**Origen**: Recomendación Kimi

---

## 1. Objetivo

Incrementar cobertura de redes sociales (Twitter/Instagram) de equipos usando los **websites oficiales** como fuente autorizada.

| Métrica | Actual (Wikidata) | Target |
|---------|-------------------|--------|
| Twitter | 67% (~490 equipos) | 90%+ |
| Instagram | 55% (~400 equipos) | 85%+ |

---

## 2. Justificación (por Kimi)

| Fuente | Problema |
|--------|----------|
| APIs oficiales (X/Instagram) | Costosas ($100+/mes), auth compleja |
| Wikipedia Infobox | Templates frágiles, socials en navboxes no infobox |
| CSV manual | Se desactualiza con rebrandings |
| APIs deportivas (Sportmonks) | No incluyen social handles |
| **Website oficial** | **Autorizado, actualizado, gratuito** |

**Ventaja clave**: Cuando un club hace rebranding y cambia su handle, su website se actualiza primero. Es la fuente más autoritativa.

---

## 3. Arquitectura

### 3.1 Flujo de Datos

```
team_wikidata_enrichment.website (P856)
            │
            ▼
    ┌───────────────────┐
    │  Website Fetcher  │  ← 1 request/2s (rate limit)
    │  (homepage HTML)  │
    └───────────────────┘
            │
            ▼
    ┌───────────────────┐
    │  Social Extractor │  ← Parse footer/header/meta
    │  (BeautifulSoup)  │
    └───────────────────┘
            │
            ▼
    ┌───────────────────┐
    │  Validator        │  ← Filtrar handles genéricos
    │  (regex + rules)  │
    └───────────────────┘
            │
            ▼
    team_wikidata_enrichment.social_handles (UPDATE)
```

### 3.2 Cascade de Fuentes (actualizado)

```
PRIORIDAD 1: team_enrichment_overrides (manual)
PRIORIDAD 2: Wikidata SPARQL (P2002, P2003)
PRIORIDAD 3: Website oficial (NUEVO)  ← Solo si Wikidata NULL
PRIORIDAD 4: Wikipedia REST API (fallback full_name)
```

---

## 4. Implementación Técnica

### 4.1 Script: `scripts/extract_social_from_websites.py`

```python
#!/usr/bin/env python3
"""
Extract social media handles from official team websites.

Strategy per Kimi recommendation:
1. Use website URL from team_wikidata_enrichment (P856)
2. Fetch homepage HTML (rate limit: 1 req/2s)
3. Parse footer/header for Twitter/Instagram links
4. Validate handles (avoid generic like "share", "login")
5. Update social_handles in team_wikidata_enrichment

Usage:
    python scripts/extract_social_from_websites.py --dry-run
    python scripts/extract_social_from_websites.py --apply --batch-size 50
"""

import asyncio
import re
import logging
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

# Rate limit: 1 request every 2 seconds (conservative)
RATE_LIMIT_DELAY = 2.0

# Regex patterns for extracting handles
TWITTER_PATTERN = re.compile(
    r'(?:twitter\.com|x\.com)/([A-Za-z0-9_]{1,15})(?:\?|/|$)',
    re.IGNORECASE
)
INSTAGRAM_PATTERN = re.compile(
    r'instagram\.com/([A-Za-z0-9_.]{1,30})(?:\?|/|$)',
    re.IGNORECASE
)

# Handles to ignore (generic/spam)
INVALID_HANDLES = {
    'share', 'login', 'signup', 'home', 'intent', 'sharer',
    'hashtag', 'search', 'explore', 'p', 'reel', 'stories',
}


def extract_twitter_handle(html: str) -> Optional[str]:
    """Extract Twitter/X handle from HTML."""
    soup = BeautifulSoup(html, 'html.parser')

    # Strategy 1: Look for meta tag twitter:site
    meta_twitter = soup.find('meta', attrs={'name': 'twitter:site'})
    if meta_twitter and meta_twitter.get('content'):
        handle = meta_twitter['content'].lstrip('@')
        if handle.lower() not in INVALID_HANDLES:
            return handle

    # Strategy 2: Look for links in footer/header
    for area in ['footer', 'header', 'nav']:
        container = soup.find(area) or soup.find(class_=re.compile(area, re.I))
        if container:
            for a in container.find_all('a', href=True):
                match = TWITTER_PATTERN.search(a['href'])
                if match:
                    handle = match.group(1)
                    if handle.lower() not in INVALID_HANDLES:
                        return handle

    # Strategy 3: Search entire document (last resort)
    for a in soup.find_all('a', href=True):
        if 'twitter.com' in a['href'] or 'x.com' in a['href']:
            match = TWITTER_PATTERN.search(a['href'])
            if match:
                handle = match.group(1)
                if handle.lower() not in INVALID_HANDLES:
                    return handle

    return None


def extract_instagram_handle(html: str) -> Optional[str]:
    """Extract Instagram handle from HTML."""
    soup = BeautifulSoup(html, 'html.parser')

    # Strategy 1: Look for links in footer/header
    for area in ['footer', 'header', 'nav']:
        container = soup.find(area) or soup.find(class_=re.compile(area, re.I))
        if container:
            for a in container.find_all('a', href=True):
                match = INSTAGRAM_PATTERN.search(a['href'])
                if match:
                    handle = match.group(1)
                    if handle.lower() not in INVALID_HANDLES:
                        return handle

    # Strategy 2: Search entire document
    for a in soup.find_all('a', href=True):
        if 'instagram.com' in a['href']:
            match = INSTAGRAM_PATTERN.search(a['href'])
            if match:
                handle = match.group(1)
                if handle.lower() not in INVALID_HANDLES:
                    return handle

    return None


async def fetch_website_html(
    url: str,
    client: httpx.AsyncClient,
) -> Optional[str]:
    """
    Fetch website HTML with proper error handling.

    Respects robots.txt implicitly by:
    - Using proper User-Agent
    - Rate limiting to 1 req/2s
    - Only fetching homepage (no crawling)
    """
    try:
        # Normalize URL
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'

        response = await client.get(
            url,
            headers={
                'User-Agent': 'FutbolStats/1.0 (contact@futbolstats.app; social-enrichment)',
                'Accept': 'text/html',
                'Accept-Language': 'en-US,en;q=0.9,es;q=0.8',
            },
            timeout=15.0,
            follow_redirects=True,
        )

        if response.status_code != 200:
            return None

        # Only process HTML responses
        content_type = response.headers.get('content-type', '')
        if 'text/html' not in content_type:
            return None

        return response.text

    except Exception as e:
        logging.warning(f"Failed to fetch {url}: {e}")
        return None
```

### 4.2 Query: Equipos Candidatos

```sql
-- Equipos con website pero sin Twitter o Instagram
SELECT
    twe.team_id,
    t.name,
    twe.website,
    twe.social_handles->>'twitter' AS twitter,
    twe.social_handles->>'instagram' AS instagram
FROM team_wikidata_enrichment twe
JOIN teams t ON t.id = twe.team_id
WHERE twe.website IS NOT NULL
  AND (
    twe.social_handles->>'twitter' IS NULL
    OR twe.social_handles->>'instagram' IS NULL
  )
ORDER BY t.name;
```

### 4.3 Update: Merge Social Handles

```sql
-- Solo actualiza campos NULL (no sobreescribe Wikidata)
UPDATE team_wikidata_enrichment
SET
    social_handles = jsonb_set(
        jsonb_set(
            COALESCE(social_handles, '{}'::jsonb),
            '{twitter}',
            COALESCE(
                social_handles->>'twitter',  -- Preservar existente
                :new_twitter                  -- O usar nuevo
            )::jsonb
        ),
        '{instagram}',
        COALESCE(
            social_handles->>'instagram',
            :new_instagram
        )::jsonb
    ),
    enrichment_source = CASE
        WHEN enrichment_source = 'wikidata' THEN 'wikidata+website'
        WHEN enrichment_source = 'wikipedia' THEN 'wikipedia+website'
        ELSE COALESCE(enrichment_source, 'website')
    END
WHERE team_id = :team_id;
```

---

## 5. Guardrails Operacionales

| Guardrail | Implementación |
|-----------|----------------|
| **Rate limit** | 1 request cada 2 segundos (0.5 req/s) |
| **User-Agent** | `FutbolStats/1.0 (contact@futbolstats.app)` |
| **Timeout** | 15s por request |
| **No crawling** | Solo homepage, no seguir links internos |
| **No overwrite** | Solo llenar campos NULL, preservar Wikidata |
| **Validation** | Filtrar handles genéricos (share, login, etc.) |
| **Fail-open** | Errores de fetch no crashean, solo skip |
| **Dry-run** | Modo obligatorio para primera ejecución |
| **Batch size** | Máximo 100 equipos por ejecución |
| **Logging** | Log de cada extracción para auditoría |

---

## 6. Métricas de Éxito

```sql
-- Query de verificación post-ejecución
SELECT
    COUNT(*) FILTER (WHERE social_handles->>'twitter' IS NOT NULL) AS with_twitter,
    COUNT(*) FILTER (WHERE social_handles->>'instagram' IS NOT NULL) AS with_instagram,
    COUNT(*) AS total_enriched,
    ROUND(100.0 * COUNT(*) FILTER (WHERE social_handles->>'twitter' IS NOT NULL) / COUNT(*), 1) AS pct_twitter,
    ROUND(100.0 * COUNT(*) FILTER (WHERE social_handles->>'instagram' IS NOT NULL) / COUNT(*), 1) AS pct_instagram
FROM team_wikidata_enrichment;
```

**Target**:
- Twitter: 67% → 90% (+170 equipos)
- Instagram: 55% → 85% (+220 equipos)

---

## 7. Archivos a Crear/Modificar

| Archivo | Acción | Descripción |
|---------|--------|-------------|
| `scripts/extract_social_from_websites.py` | **Crear** | Script de extracción |
| `app/etl/wikidata_enrich.py` | Modificar | Agregar `enrichment_source` "wikidata+website" al badge |
| `app/dashboard/admin.py` | Modificar | Badge para "wikidata+website" |

---

## 8. Criterios de Aceptación (ABE)

- [ ] **No overwrite**: Campos existentes de Wikidata NO se sobreescriben
- [ ] **Rate limit**: Máximo 0.5 req/s (1 cada 2 segundos)
- [ ] **User-Agent**: Identificable y con contacto
- [ ] **Validation**: Handles genéricos filtrados
- [ ] **Dry-run**: Modo obligatorio antes de --apply
- [ ] **Logging**: Cada extracción logueada para auditoría
- [ ] **Fail-open**: Errores no crashean el script
- [ ] **Provenance**: enrichment_source actualizado a "wikidata+website"
- [ ] **Batch limit**: Máximo 100 equipos por ejecución
- [ ] **Cobertura target**: Twitter 90%, Instagram 85%

---

## 9. Fuera de Scope

1. Crawling de páginas internas del sitio
2. Parsing de JavaScript-rendered pages (SPAs)
3. Facebook/YouTube/TikTok (fase futura)
4. Verificación de cuentas (checkmark azul)
5. Scheduler automático (job manual por ahora)

---

## 10. Riesgos y Mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| Website bloqueado/caído | Fail-open, retry en siguiente batch |
| Handle extraído incorrecto | Validation rules + revisión manual top 50 |
| Rate limit de sitios | 2s delay, User-Agent educado |
| Cambio de estructura HTML | Múltiples estrategias de extracción |
| Costos de requests | ~700 requests total, negligible |

---

**Pregunta para ABE**: ¿Apruebas esta propuesta? ¿Algún guardrail adicional requerido?
