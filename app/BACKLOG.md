# 🚀 Bitácora de Desarrollo & Ideas

> "Documento vivo para capturar ideas sin interrumpir el flujo actual."

## 🧠 Brain Dump (Zona de Aterrizaje)
*Escribe aquí cualquier idea rápida, bug o mejora tal como llegue a tu mente.*

## ⏳ Pendiente
- [x] Dashboard: agregar links **visibles** a JSON por sección (Ops/PIT/History/Logs) y aclarar qué JSON se está viendo.

- [x] PIT report: agregar bloque `interpretation` (fix escala `skill_vs_market`, guard-rail `low_n_predictions` con umbral `<30` explícito). Commits: baf71e3, 81a403d

- [x] PIT: Brier vs market vs uniform + ROI/EV con CI (cuando N lo permita) expuesto para auditoría (daily + weekly).

- [x] Dashboard: "Progreso hacia re-test/Alpha" con snapshots históricos en DB (AlphaProgressSnapshot + endpoints + job diario 09:10 UTC) + baseline #0 capturado. Commits: 69683f9, 4498d73

- [x] Auditoría con "Composer" (P0 ImportError weekly_pit_report + P1 anti-leakage `<` + hardening `_save_pit_report_to_db`) aplicada. Commit: 7014f29

- [x] Actualizar el codigo fuente en la App de IOS para que refleje los nuevos cambios en el codigo.
  - ValueBet fields hechos opcionales para manejar datos incompletos del backend
  - APIClient: agregado caso `emptyResponse` para respuestas vacías
  - Token hardcodeado removido de AppConfiguration.swift
  - scheduler.py normalizado para generar value_bets en formato consistente con engine.py
  - Legacy value_bets (6) permanecen; UI muestra "—" en campos faltantes; no rompe
  - Script de migración creado (scripts/migrate_legacy_value_bets.py) - opcional, requiere acceso a producción

- [ ] Investigar Predictions frozen para los partidos del 9 de enero y probablemente 10 de enro (y los futuros) Probable bug?

- [ ] Por que las estadisticas post-partido no estan en nuestra database?

- [ ] Monitorear narrativa y comparar con un review de un humano (analista deportivo o periodista) para validar/reforzar prescicion de nuestra narrativa.

- [ ] Revisar que esta en commit sin pushear.

- [ ] Revisar que todas las ligas esten en el scheduler (todas reciban data).

- [ ] Auditar sesion de trabajo de la tarde noche de ayer.

- [ ] crear un checklist y mapa (ayer datos iban a pasar por fuera de "ML Engine")

- [ ] verificar que cada equipo de cada liga esta recibiendo predicciones, odds, estadisticas, etc.
- [ ] Reestructura xgboots y features basado en analisis de Google (pineado)

- [ ] Por que desaparecen los ppartidos de la App? El sabado 10 de enero inicialmente aparecieron 35, luego aparecieron 34 y hoy solo aparecen 8? Se estan filtrando o eliminando?


- [ ] Pandas vs Polars

- [ ] Mejorar narrativa, qwen intercambia numeros (1) con palabras (Uno). Que diga el resultado 1-1, 2-2, 2-1, etc es irrelevante ya el usuario sabe que el equipo/gano/empato/perdio y lo que realmente quiere entender es el porque, haciendonos desperdiciar palabras (tokens) diciendo el marcador cuando podemos usarlo para explicar mas el contexto del match. En la seccion inferior titulada "los factores clave" se repite la misma informacion de la narrativa, estamos duplicando la info. Evitar mencionar tanto El nombre del equipo, si ya se menciona el Inter, al nombrarlo nuevamente usar "Los Azurri" o "los locales". Implementar el uso de "sobrenombres" a los equipos para que en la narrativa se enriquezaca y dinamicamente/aleatoriamente sean utilizados, por ejemplo para referirse al "Real Madrid" use "Los Merengues", Atletico de Madrid "Los Colchoneros", Colombia "Los Cafeteros" deben ser "apodos" no peyorativos o despectivos. Tambien considero que debajo de la narrativa debe ir una seccion "Tabla" con las estadisticas del partido (como en el entretiempo de un partido), con eso podemos ajustar la narrativa para marcar puntualmente donde acertamos o perdimos la prediccion, dejando que la tabla de estadisticas que vamos a poner (mas adelante) explique por si sola los numeros.

- [x] Para el cierre formal (operacional) solo faltan los 2 pendientes y ya:
Screenshot de 1 alerta "Firing" (prueba controlada bajando temporalmente el umbral y luego revertir).
Confirmación de estabilidad 24h (screenshot del panel de series/requests mostrando que no hubo spikes).
**CLOSED 2026-01-14**: Informe 24h generado en docs/MONITORING_REPORT_20260113.md - 0 alertas, sistema estable.

- [x] Supervision de telemetria de las ultimas 24 horas
**CLOSED 2026-01-14**: Ver docs/MONITORING_REPORT_20260113.md - LLM 6/6 ok, 0 rejected, DQ sin anomalias.

- [ ] Revisar los payloads de las narrativas que ya estan persistiendo: "A partir de ahora, cuando una narrativa “desmejore”, el flujo correcto es:
abrir /dashboard/ops/llm_audit/{match_id}.json?token=...,
revisar llm_prompt_input_json + llm_validation_errors,
y ahí sabremos en 1 minuto si fue data incompleta o salida no soportada del modelo."

- [x] Perfecto. Con esos dos checks, ya no tengo observaciones: está bien instrumentado y auditable.
Para cerrar el caso, el criterio es exactamente el que pusiste:
llm_narrative_status=ok para audit 99,
llm_prompt_input_json.stats_summary poblado correctamente,
y si hubo normalización, queda en llm_validation_errors.
Cuando veas el próximo tick y esos campos, lo marcamos CLOSED. (esto es respecto a las narrativas LLMs).
**CLOSED 2026-01-14**: Audits 98-103 verificados con llm_narrative_status=ok. Ver docs/MONITORING_REPORT_20260113.md

- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]

---

