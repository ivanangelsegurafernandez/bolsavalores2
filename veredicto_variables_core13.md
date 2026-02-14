# Veredicto CORE-13 (actualizado)

Generado: 2026-02-14T04:58:24.120065+00:00
Muestra cerrada analizada: **30698** trades (winrate base **49.55%**).

## Invariantes / sanidad del dataset

- Filas cerradas detectadas: **30698**.
- Filas cerradas sin `result_bin` válido descartadas: **0**.
- Filas descartadas por error al derivar features: **0**.
- Filas con `hora_bucket` en fallback neutro (0.5) por falta total de timestamp: **0**.
- Chequeo de consistencia contable: **OK** (`rows_loaded = rows_closed - invalid_result_bin - feature_error`).

## Señal por variable

| feature | corr | auc_uni | uniq | dominancia | veredicto |
|---|---:|---:|---:|---:|---|
| rsi_9 | +0.0022 | 0.5004 | 6746 | 0.003 | BAJA |
| rsi_14 | -0.0073 | 0.4956 | 5811 | 0.001 | BAJA |
| sma_5 | +0.0064 | 0.5029 | 28217 | 0.000 | BAJA |
| sma_20 | +0.0064 | 0.5029 | 28628 | 0.000 | BAJA |
| cruce_sma | +0.0007 | 0.5001 | 2 | 0.987 | CASI_CONSTANTE |
| breakout | +0.0040 | 0.5009 | 2 | 0.946 | CASI_CONSTANTE |
| rsi_reversion | +0.0047 | 0.5014 | 2 | 0.901 | CASI_CONSTANTE |
| racha_actual | +0.2148 | 0.6313 | 27 | 0.250 | UTIL |
| payout | +0.0052 | 0.5028 | 11 | 0.550 | BAJA |
| puntaje_estrategia | +0.0138 | 0.5024 | 2 | 0.969 | CASI_CONSTANTE |
| volatilidad | +0.0031 | 0.5012 | 28699 | 0.000 | BAJA |
| es_rebote | +0.0187 | 0.5045 | 2 | 0.938 | CASI_CONSTANTE |
| hora_bucket | -0.0021 | 0.4991 | 24 | 0.075 | BAJA |

## Diagnóstico de eventos raros (objetivo: bajar dominancia)

| feature | pct_no_cero | dominancia | uniq |
|---|---:|---:|---:|
| cruce_sma | 98.68% | 0.987 | 2 |
| breakout | 94.57% | 0.946 | 2 |
| rsi_reversion | 9.88% | 0.901 | 2 |
| puntaje_estrategia | 100.00% | 0.969 | 2 |
| es_rebote | 6.17% | 0.938 | 2 |

## Segmentación (AUC univariado de `racha_actual`) 

| segmento | n | winrate | auc_racha_actual |
|---|---:|---:|---:|
| payout:medio | 4676 | 49.25% | 0.7033 |
| payout:alto | 9140 | 50.21% | 0.6734 |
| payout:bajo | 16882 | 49.27% | 0.6655 |
| activo:1HZ25V | 7396 | 49.89% | 0.6468 |
| hora:h12-17 | 5044 | 50.18% | 0.6436 |
| vol:bajo | 10131 | 49.30% | 0.6411 |
| hora:h06-11 | 9611 | 49.24% | 0.6405 |
| activo:1HZ50V | 5598 | 50.41% | 0.6328 |

## Estabilidad temporal (walk-forward simple, AUC de `racha_actual`) 

| fold_inicio | n | auc_racha_actual |
|---|---:|---:|
| 2026-01-20 | 6139 | 0.8523 |
| 2026-01-25 | 6139 | 0.5865 |
| 2026-01-28 | 6139 | 0.5740 |
| 2026-02-03 | 6139 | 0.5733 |
| 2026-02-07 | 6139 | 0.5698 |

## Ablation automático (Objetivo 3)

- AUC walk-forward agregado (`solo racha_actual`): **0.5765**.
- AUC walk-forward agregado (`racha_actual + rediseñadas`): **0.5789**.
- Delta (`combo - solo`): **+0.0024**.

| fold_inicio | n_test | auc_solo_racha | auc_combo_rediseñado |
|---|---:|---:|---:|
| 2026-01-24 | 5116 | 0.5735 | 0.5773 |
| 2026-01-27 | 5116 | 0.5856 | 0.5907 |
| 2026-01-30 | 5116 | 0.5759 | 0.5778 |
| 2026-02-05 | 5116 | 0.5731 | 0.5774 |
| 2026-02-08 | 5116 | 0.5664 | 0.5685 |

## Veredicto

- Variables con aporte real hoy: `racha_actual`.
- Variables con aporte bajo o nulo hoy: `rsi_9`, `rsi_14`, `sma_5`, `sma_20`, `cruce_sma`, `breakout`, `rsi_reversion`, `payout`, `puntaje_estrategia`, `volatilidad`, `es_rebote`, `hora_bucket`.
- Sí: **aún existen variables que prácticamente no ponderan** en el estado actual.
- Recomendación: mantener `racha_actual` como baseline; rediseñar eventos binarios a señales continuas de intensidad/proximidad a umbral; comparar siempre con ablation walk-forward (solo racha vs combo) para evitar maquillaje; validar por segmentación (activo/payout/volatilidad/hora) y por tiempo (walk-forward); `hora_bucket` ya no está fijo, pero su aporte predictivo actual sigue bajo.
