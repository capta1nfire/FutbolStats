Muestra logs recientes de Railway de forma rápida y utilizable.

## Inputs
- Argumentos después de `/logs` son un filtro opcional (ej: `/logs error` o `/logs FASTPATH`).

## Pasos
1) Si hay filtro:
   - `railway logs -n 80 --filter "<FILTRO>"`
2) Si no hay filtro:
   - `railway logs -n 80`

## Heurística de triage (al final)
- Extrae (en texto) los **3 errores** más relevantes y agrúpalos por causa probable.
- Si ves señales de:
  - **FASTPATH / narrativas**: menciona `app/llm/` y el job correspondiente.
  - **scheduler drift / jobs**: menciona `app/scheduler.py` y `job_runs`.
  - **DB**: menciona SQLAlchemy/timeout y sugiere revisar pool/conexiones.

## Output
- Primero imprime los logs (o un snippet si son muy largos).
- Luego: “**Resumen**” + “**Siguiente paso recomendado**”.
