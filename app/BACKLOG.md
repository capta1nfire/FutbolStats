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

- [ ] Crear "sobrenombres a los equipos para que en la narrativa dinamicamente/aleatoriamente sean utilizados, por ejemplo para referirse al "Real Madrid" use "Los Merengues", Atletico de Madrid "Los Colchoneros", deben ser "apodos" no peyorativos o despectivos.


- [ ] Pandas vs Polars

- [ ] Mejorar narrativa, qwen intercambia numeros (1) con palabras (Uno). Que diga el resultado 1-1, 2-2, 2-1, etc es irrelevante ya el usuario sabe que el equipo/gano/empato/perdio y lo que realmente quiere entender es el porque, perdemos palabras (tokens) diciendo el marcador cuando podemos usarlo para explicar mas el contexto del match. En los factores clave se repite la misma informacion de la narrativa, estamos duplicando info.
- [ ] _
- [ ] _
- [ ] _

---

