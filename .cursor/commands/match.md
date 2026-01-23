Busca un partido por equipo (o texto libre) y devuelve el estado/predicción más relevante.

## Inputs
- El texto después de `/match` es el query del usuario (equipo, liga, etc). Ej: `/match america` o `/match real madrid`.
- Si no hay query, pregunta qué equipo/partido buscar.

## Preferencia de fuente
1) **API (preferida)**: `GET /predictions/upcoming` (rápido, no requiere saber schema SQL).
2) **DB (fallback)**: usar `railway-postgres` para buscar fixtures por `ILIKE`.

## Config
- `API_BASE`: `FUTBOLSTATS_API_BASE` (o pregunta).
- `X-API-Key`: `FUTBOLSTATS_API_KEY` (o pide al usuario). No imprimir completa.

## API flow (Bash)
1) Llama upcoming:
   - `curl -s -H "X-API-Key: $API_KEY" "$API_BASE/predictions/upcoming" | jq`
2) Filtra por el query (case-insensitive) sobre strings (home/away/league):
   - Si el shape es desconocido, primero imprime `keys` y detecta dónde están los nombres.
3) Devuelve el “mejor match” (o top 5) con:
   - id, kickoff UTC/local si existe, home vs away, liga, predicción (1/X/2), probas si existen, status.

## DB flow (si existe MCP postgres)
- Hazlo **read-only**:
  1) Descubre tablas/campos (information_schema) si no conoces schema.
  2) Busca fixtures/matches y teams con `ILIKE '%query%'`.
  3) Devuelve top 10 más cercanos (orden por kickoff más próximo).

## Output
- Un bloque “**Resultados**” (top 5).
- Un bloque “**Recomendación**” indicando cuál abrirías (id) y por qué.
