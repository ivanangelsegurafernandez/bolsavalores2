# Diagnóstico unificado IA (bots 45-50 + incremental + señales)

- Generado (UTC): 2026-02-15T20:59:20.684883+00:00

## Modelo actual
- reliable: False
- auc: 0.34375
- brier: 0.3075384375731013
- features activas: ['racha_actual', 'rsi_14', 'rsi_reversion', 'sma_5']
- chequeo orientación AUC: possible_inversion | auc=0.34375 | auc_invertida=0.65625
- nota AUC: AUC<0.50: revisar posible inversión de orientación (p -> 1-p).

## Orientación operativa (señales cerradas)
- estado: ok | auc=0.550855 | auc_invertida=0.449145
- nota: Orientación de señales cerradas aparentemente correcta.

## Incremental
- filas: 372
- duplicados exactos: 35
- ratio duplicados exactos: 9.41%
- volatilidad: uniq=1 dom=1.0 top=0.9999546000702375
- hora_bucket: uniq=1 dom=1.0 top=0.0
- payout: uniq=10 dom=0.572581 top=0.95
- racha_actual: uniq=14 dom=0.282258 top=1.0

## Bots (cierres y WR)
| Bot | Rows | Cerrados | Win | Loss | WR | Last TS |
|---|---:|---:|---:|---:|---:|---|
| fulll45 | 12572 | 6147 | 3016 | 3131 | 49.06% | 2026-02-15T20:58:59.883982+00:00 |
| fulll46 | 14178 | 6945 | 3489 | 3456 | 50.24% | 2026-02-15T20:58:57.468905+00:00 |
| fulll47 | 14165 | 6943 | 3474 | 3469 | 50.04% | 2026-02-15T20:58:49.048027+00:00 |
| fulll48 | 14140 | 6926 | 3443 | 3483 | 49.71% | 2026-02-15T20:58:48.571898+00:00 |
| fulll49 | 14190 | 6947 | 3472 | 3475 | 49.98% | 2026-02-15T20:59:01.163335+00:00 |
| fulll50 | 14209 | 6959 | 3404 | 3555 | 48.92% | 2026-02-15T20:57:19.891314+00:00 |

## Señales IA cerradas
- total cerradas: 117 (de 117 filas de log)
- >= 65%: n=117, hits=53, wr=45.30%
- >= 70%: n=117, hits=53, wr=45.30%
- >= 75%: n=104, hits=46, wr=44.23%
- >= 80%: n=50, hits=25, wr=50.00%

## Bins altos (overconfidence)
| Bin | n | avg_pred | avg_real | gap |
|---|---:|---:|---:|---:|
| [0.70,0.80) | 67 | 0.7567 | 0.4179 | +0.3387 |
| [0.80,0.90) | 39 | 0.8225 | 0.4872 | +0.3353 |
| [0.90,1.00) | 11 | 0.9835 | 0.5455 | +0.4380 |
- max |gap| bins altos: 43.80%

## Checklist objetivo (sin autoengaño)
| Métrica | Estado | Detalle |
|---|:---:|---|
| orientación modelo/señal consistente (AUC>0.50) | ❌ | model=possible_inversion, signals=ok |
| volatilidad y hora_bucket dejan de ser ROTA | ❌ | volatilidad_uniq=1, hora_bucket_uniq=1 |
| duplicados exactos incremental <2% | ❌ | duplicates_ratio=9.41% |
| gap bins altos <10pp | ❌ | max_gap_abs_high_bins=43.80% |
| señales cerradas >=70% con evidencia >=200 | ❌ | n_ge70=117 |
| hit real en >=70% >60% sostenido | ❌ | wr_ge70=45.30% con n=117 |

## Acciones sugeridas
1. Subir muestra incremental a >=400 filas cerradas antes de cambios estructurales.
2. Eliminar duplicados exactos en incremental: ratio actual=9.41% (> 2%).
3. Revisar orientación del modelo: AUC=0.34375 (<0.50), AUC invertida estimada=0.65625.
4. Corregir generación de volatilidad en origen bot: actualmente está plana en incremental.
5. Corregir generación de hora_bucket en origen bot: actualmente está plano en incremental.
6. Mantener gate conservador (LB + evidencia). No forzar gatillo duro 70% todavía.
7. Priorizar recalibración + shrink dinámico hasta reducir gap de bins altos a <10pp.
8. Aumentar evidencia en señales >=75% hasta n>=200 antes de usar 75% como gatillo duro.
9. Rediseñar features binarias dominantes a intensidades continuas: breakout, cruce_sma, es_rebote, hora_bucket, puntaje_estrategia, rsi_reversion, volatilidad.

## Fases recomendadas (orden de impacto)
1. Fase A - Honestidad de señal: corregir orientación y mantener gate conservador por LB+N.
2. Fase B - Sanidad de datos: revivir volatilidad/hora_bucket en origen y deduplicar incremental <2%.
3. Fase C - Evidencia: acumular n>=200 en el umbral objetivo antes de endurecer gatillos.
