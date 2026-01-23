Escanea el repo por posibles **secretos hardcodeados** y devuelve hallazgos accionables.

Nota: existe también el Skill nativo `secrets-scan` en `.cursor/skills/`. Este command se mantiene con nombre distinto para **evitar colisión** en el menú `/`.

## Guardrails (obligatorio)
- **No imprimas** valores completos que parezcan secretos. Muestra solo:
  - archivo + línea
  - tipo de secreto sospechado
  - un **snippet redactado** con este formato fijo: **primeros 3 chars + "..." + últimos 3 chars** (ej: `"rpa...7R3"`)
    - Si el string tiene **< 7 chars**, devuelve `"***"` (no intentes reconstruir nada).
- **No uses Bash/Shell ni edites/escribas archivos** para el escaneo. Solo lectura con `Read`, `Grep`, `Glob`.
- Excluir explícitamente:
  - `CLAUDE.local.md`
  - `.env` y cualquier `.env*`
  - `.claude/` (si existe)
  - `node_modules/`, `venv/`, `.venv/`
  - `ios/**/Secrets.xcconfig`

## Patrones a buscar (Grep)
1) Tokens comunes:
   - `rpa_`
   - `ghp_`
   - `Bearer `
   - `AIza`
   - `sk-`
2) URLs con credenciales:
   - `postgresql://` con `user:pass@`
   - `https://` con `user:pass@`
3) Keys genéricas:
   - `api_key`, `apikey`, `secret`, `token`, `password`, `dsn`
4) Strings largos sospechosos:
   - secuencias alfanuméricas >= 32 chars (solo en `.py`, `.ts`, `.tsx`, `.swift`, `.md`)

## Output
- Tabla con columnas: `file`, `line`, `kind`, `snippet_redacted`, `severity (high/med/low)`, `next_action`.
- Cierra con una recomendación: “mover a env var”, “rotar token”, “agregar a .gitignore”, etc.
