# Diagnóstico unificado IA (bots 45-50 + incremental + señales)

- Generado (UTC): 2026-02-15T20:37:55.567246+00:00

## Modelo actual
- reliable: False
- auc: 0.5666666666666667
- brier: 0.2586050223219748
- features activas: ['racha_actual']
- chequeo orientación AUC: ok | auc=0.566667 | auc_invertida=0.433333
- nota AUC: Orientación AUC aparentemente correcta.

## Incremental
- filas: 154
- duplicados exactos: 29
- volatilidad: uniq=1 dom=1.0 top=0.9999546000702375
- hora_bucket: uniq=1 dom=1.0 top=0.0
- payout: uniq=8 dom=0.597403 top=0.95
- racha_actual: uniq=11 dom=0.253247 top=1.0

## Bots (cierres y WR)
| Bot | Rows | Cerrados | Win | Loss | WR | Last TS |
|---|---:|---:|---:|---:|---:|---|
| fulll45 | 12012 | 5868 | 2883 | 2985 | 49.13% | 2026-02-15T04:00:22.938710+00:00 |
| fulll46 | 10329 | 5065 | 2541 | 2524 | 50.17% | 2026-02-10T22:24:02.583320+00:00 |
| fulll47 | 10330 | 5066 | 2537 | 2529 | 50.08% | 2026-02-10T22:24:01.490602+00:00 |
| fulll48 | 10324 | 5058 | 2514 | 2544 | 49.70% | 2026-02-10T22:23:48.786682+00:00 |
| fulll49 | 10348 | 5064 | 2536 | 2528 | 50.08% | 2026-02-10T22:24:02.156623+00:00 |
| fulll50 | 10352 | 5076 | 2446 | 2630 | 48.19% | 2026-02-10T22:23:48.797948+00:00 |

## Señales IA cerradas
- total cerradas: 82 (de 82 filas de log)
- >= 65%: n=82, hits=37, wr=45.12%
- >= 70%: n=82, hits=37, wr=45.12%
- >= 75%: n=69, hits=30, wr=43.48%
- >= 80%: n=38, hits=19, wr=50.00%

## Bins altos (overconfidence)
| Bin | n | avg_pred | avg_real | gap |
|---|---:|---:|---:|---:|
| [0.70,0.80) | 44 | 0.7529 | 0.4091 | +0.3438 |
| [0.80,0.90) | 29 | 0.8227 | 0.4828 | +0.3399 |
| [0.90,1.00) | 9 | 1.0000 | 0.5556 | +0.4444 |

## Acciones sugeridas
1. Subir muestra incremental a >=400 filas cerradas antes de cambios estructurales.
2. Corregir generación de volatilidad en origen bot: actualmente está plana en incremental.
3. Corregir generación de hora_bucket en origen bot: actualmente está plano en incremental.
4. Mantener gate conservador (LB + evidencia). No forzar gatillo duro 70% todavía.
5. Priorizar recalibración + shrink dinámico hasta reducir gap de bins altos a <10pp.
6. Rediseñar features binarias dominantes a intensidades continuas: cruce_sma, breakout, rsi_reversion, puntaje_estrategia, es_rebote, volatilidad, hora_bucket.
