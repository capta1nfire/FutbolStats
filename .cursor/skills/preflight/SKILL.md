# Preflight Environment Check

Ejecuta una verificación completa del entorno antes de iniciar trabajo. Reporta cada check como PASS o FAIL.

## Checks a ejecutar

### 1. Database (MCP)
```sql
SELECT 1 AS mcp_connected;
```
Si falla: reportar que MCP railway-postgres no está conectado.

### 2. Variables de entorno
Verificar que `.env` existe en la raíz del proyecto y contiene las variables críticas:
- `DATABASE_URL`
- `API_FOOTBALL_KEY`
- `FUTBOLSTATS_API_KEY`
- `DASHBOARD_TOKEN`
- `GEMINI_API_KEY`

NO imprimir los valores completos. Solo reportar presencia (set/unset) y primeros 8 caracteres.

### 3. Git state
```bash
git status --short
git branch --show-current
git remote get-url origin
```

### 4. API Health
```bash
curl -s --max-time 5 "https://web-production-f2de9.up.railway.app/health"
```

### 5. Dashboard connectivity
Verificar el Next.js dev server si está corriendo:
```bash
curl -s --max-time 3 "http://localhost:3000" > /dev/null 2>&1 && echo "Dashboard dev server: RUNNING" || echo "Dashboard dev server: NOT RUNNING"
```

### 6. Skills directory
Verificar que `.claude/skills` existe y es un symlink a `.cursor/skills`:
```bash
ls -la .claude/skills
```

## Output format

Generar una tabla markdown con el resultado:

| Check | Status | Details |
|-------|--------|---------|
| MCP PostgreSQL | PASS/FAIL | ... |
| .env file | PASS/FAIL | ... |
| Git state | PASS/FAIL | branch, clean/dirty |
| API Health | PASS/FAIL | response time |
| Dashboard dev | PASS/FAIL | running/not running |
| Skills dir | PASS/FAIL | symlink status |

Si algún check crítico falla (MCP, .env, API Health), sugerir el fix específico y NO proceder con trabajo hasta que el usuario confirme.
