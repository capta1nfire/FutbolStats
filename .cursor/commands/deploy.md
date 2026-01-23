Deploy a producción (Railway) siguiendo el protocolo del proyecto.

## Guardrails (obligatorio)
- **Manual-only**: no ejecutes ningún comando destructivo sin confirmación explícita del usuario.
- Si el repo no es git o no tiene remoto configurado, **detente** y explica qué falta.
- Nunca hagas `push --force`.

## Protocolo
1) Verifica branch:
   - `git branch --show-current`
   - Debe ser `main` (o el branch de deploy definido por el proyecto).

2) Verifica estado limpio:
   - `git status --porcelain` debe estar vacío

3) Verifica remoto:
   - `git remote -v`
   - `git fetch origin`
   - `git status` (up-to-date)

4) Preflight:
   - corre el smoke mínimo (ejecuta `/verify` después del deploy)

5) Antes del deploy:
   - muestra `git log origin/main..HEAD --oneline`
   - **pide confirmación explícita**: “¿Confirmas `git push origin main`?”

6) Deploy:
   - `git push origin main`

7) Post-deploy:
   - espera ~60s
   - corre `/verify` (o `curl $API_BASE/health`)
   - si falla, trae `/logs error`
