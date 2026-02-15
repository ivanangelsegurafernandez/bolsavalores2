# Veredicto CORE-13 (actualizado)

Generado: 2026-02-15T10:45:46.539993+00:00
Muestra cerrada analizada: **31197** trades (winrate base **49.55%**).

## Invariantes / sanidad del dataset

- Filas cerradas detectadas: **31197**.
- Filas cerradas sin `result_bin` válido descartadas: **0**.
- Filas descartadas por error al derivar features: **0**.
- Filas con `hora_bucket` en fallback neutro (0.5) por falta total de timestamp: **0**.
- Chequeo de consistencia contable: **OK** (`rows_loaded = rows_closed - invalid_result_bin - feature_error`).

## Señal por variable

| feature | corr | auc_uni | uniq | dominancia | veredicto |
|---|---:|---:|---:|---:|---|
| rsi_9 | +0.0033 | 0.5008 | 6777 | 0.003 | BAJA |
| rsi_14 | -0.0065 | 0.4960 | 5831 | 0.001 | BAJA |
| sma_5 | +0.0072 | 0.5036 | 28713 | 0.000 | BAJA |
| sma_20 | +0.0072 | 0.5037 | 29127 | 0.000 | BAJA |
| cruce_sma | +0.0007 | 0.5001 | 2 | 0.987 | CASI_CONSTANTE |
| breakout | +0.0037 | 0.5008 | 2 | 0.946 | CASI_CONSTANTE |
| rsi_reversion | +0.0050 | 0.5015 | 2 | 0.901 | CASI_CONSTANTE |
| racha_actual | +0.2112 | 0.6291 | 28 | 0.250 | UTIL |
| payout | +0.0050 | 0.5028 | 11 | 0.550 | BAJA |
| puntaje_estrategia | +0.0137 | 0.5024 | 2 | 0.968 | CASI_CONSTANTE |
| volatilidad | +0.0028 | 0.5009 | 29198 | 0.000 | BAJA |
| es_rebote | +0.0171 | 0.5041 | 2 | 0.938 | CASI_CONSTANTE |
| hora_bucket | -0.0019 | 0.4990 | 24 | 0.075 | BAJA |

## Diagnóstico de eventos raros (objetivo: bajar dominancia)

| feature | pct_no_cero | dominancia | uniq |
|---|---:|---:|---:|
| cruce_sma | 98.70% | 0.987 | 2 |
| breakout | 94.62% | 0.946 | 2 |
| rsi_reversion | 9.88% | 0.901 | 2 |
| puntaje_estrategia | 100.00% | 0.968 | 2 |
| es_rebote | 6.18% | 0.938 | 2 |

## Segmentación (AUC univariado de `racha_actual`) 

| segmento | n | winrate | auc_racha_actual |
|---|---:|---:|---:|
| payout:medio | 4736 | 49.32% | 0.7015 |
| payout:alto | 9299 | 50.17% | 0.6710 |
| payout:bajo | 17162 | 49.27% | 0.6626 |
| activo:1HZ25V | 7489 | 49.97% | 0.6438 |
| hora:h12-17 | 5148 | 50.12% | 0.6417 |
| vol:bajo | 10295 | 49.35% | 0.6392 |
| hora:h06-11 | 9727 | 49.10% | 0.6388 |
| activo:1HZ50V | 5695 | 50.38% | 0.6306 |

## Estabilidad temporal (walk-forward simple, AUC de `racha_actual`) 

| fold_inicio | n | auc_racha_actual |
|---|---:|---:|
| 2026-01-20 | 6239 | 0.8481 |
| 2026-01-25 | 6239 | 0.5853 |
| 2026-01-28 | 6239 | 0.5729 |
| 2026-02-03 | 6239 | 0.5728 |
| 2026-02-08 | 6239 | 0.5657 |

## Ablation automático (Objetivo 3)

- AUC walk-forward agregado (`solo racha_actual`): **0.5744**.
- AUC walk-forward agregado (`racha_actual + rediseñadas`): **0.5772**.
- Delta (`combo - solo`): **+0.0028**.

| fold_inicio | n_test | auc_solo_racha | auc_combo_rediseñado |
|---|---:|---:|---:|
| 2026-01-24 | 5199 | 0.5763 | 0.5801 |
| 2026-01-27 | 5199 | 0.5815 | 0.5875 |
| 2026-01-30 | 5199 | 0.5761 | 0.5766 |
| 2026-02-05 | 5199 | 0.5763 | 0.5807 |
| 2026-02-08 | 5199 | 0.5565 | 0.5584 |

## Veredicto

- Variables con aporte real hoy: `racha_actual`.
- Variables con aporte bajo o nulo hoy: `rsi_9`, `rsi_14`, `sma_5`, `sma_20`, `cruce_sma`, `breakout`, `rsi_reversion`, `payout`, `puntaje_estrategia`, `volatilidad`, `es_rebote`, `hora_bucket`.
- Sí: **aún existen variables que prácticamente no ponderan** en el estado actual.
- Recomendación: mantener `racha_actual` como baseline; rediseñar eventos binarios a señales continuas de intensidad/proximidad a umbral; comparar siempre con ablation walk-forward (solo racha vs combo) para evitar maquillaje; validar por segmentación (activo/payout/volatilidad/hora) y por tiempo (walk-forward); `hora_bucket` ya no está fijo, pero su aporte predictivo actual sigue bajo.
