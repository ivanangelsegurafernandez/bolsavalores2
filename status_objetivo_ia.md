# Estado IA y avance al objetivo

- Actualizado (UTC): 2026-02-11T09:43:44.851996+00:00
- Objetivo principal (Prob IA real): 70%
- Efectividad real global de cierres (bots 45-50): 49.51% (14759/29809)
- Brecha vs objetivo: -20.49%

## Señales IA cerradas (log)
- Total señales registradas: 20
- Total señales cerradas: 20
- Señales cerradas con prob >=70%: 20
- Acierto real en señales >=70%: 45.00% (9/20) | IC95%=[25.82%,65.79%]
- Estado semáforo objetivo 70%: 🔴 Aún no

## Recomendaciones priorizadas para subir Prob IA real
1. Brecha principal: estás en 49.51% global vs objetivo 70%. En corto plazo, prioriza reducir exposición REAL y subir filtro de calidad antes de aumentar volumen.
2. Muestra IA >=70% insuficiente (n=20). No tomes decisiones estructurales hasta llegar al menos a n>=200 cierres IA >=70%.
3. Hay sobreconfianza en bins de probabilidad (gap pred-real >10 pts). Aplicar shrinkage recomendado: p_ajustada = 0.6*p_calibrada + 0.4*tasa_base_rolling.
4. Meta de modelo actual: reliable=False, auc=0.636187033295001, brier=0.2372369500567637. Monitorear semanalmente ECE/Brier y recalibrar más frecuente que reentrenar base.

## Resumen por bot (cierres)
| Bot | Cerrados | Ganancias | Pérdidas | % Éxito |
|---|---:|---:|---:|---:|
| fulll45 | 4480 | 2185 | 2295 | 48.77% |
| fulll46 | 5065 | 2541 | 2524 | 50.17% |
| fulll47 | 5066 | 2537 | 2529 | 50.08% |
| fulll48 | 5058 | 2514 | 2544 | 49.70% |
| fulll49 | 5064 | 2536 | 2528 | 50.08% |
| fulll50 | 5076 | 2446 | 2630 | 48.19% |

## Sensibilidad por umbral (señales IA cerradas)
| Umbral | n | hit rate | IC95% | Muestra suficiente |
|---:|---:|---:|---:|:---:|
| 55% | 20 | 45.00% | [25.82%,65.79%] | ⚠️ |
| 60% | 20 | 45.00% | [25.82%,65.79%] | ⚠️ |
| 65% | 20 | 45.00% | [25.82%,65.79%] | ⚠️ |
| 70% | 20 | 45.00% | [25.82%,65.79%] | ⚠️ |
| 75% | 7 | 28.57% | [8.22%,64.11%] | ⚠️ |
| 80% | 4 | 25.00% | [4.56%,69.94%] | ⚠️ |
