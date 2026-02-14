# Veredicto CORE-13 (actualizado)

Generado: 2026-02-14T04:40:19.864129+00:00
Muestra cerrada analizada: **30698** trades (winrate base **49.55%**).

## Invariantes / sanidad del dataset

- Filas cerradas detectadas: **30698**.
- Filas cerradas sin `result_bin` válido descartadas: **0**.
- Filas descartadas por error al derivar features: **0**.
- Filas con `hora_bucket` en fallback neutro (0.5) por falta total de timestamp: **0**.

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

## Veredicto

- Variables con aporte real hoy: `racha_actual`.
- Variables con aporte bajo o nulo hoy: `rsi_9`, `rsi_14`, `sma_5`, `sma_20`, `cruce_sma`, `breakout`, `rsi_reversion`, `payout`, `puntaje_estrategia`, `volatilidad`, `es_rebote`, `hora_bucket`.
- Sí: **aún existen variables que prácticamente no ponderan** en el estado actual.
- Recomendación: mantener `racha_actual`, revisar/rediseñar features casi constantes (`cruce_sma`, `breakout`, `rsi_reversion`, `es_rebote`, `puntaje_estrategia`) y corregir `hora_bucket` para que no quede fijo.
