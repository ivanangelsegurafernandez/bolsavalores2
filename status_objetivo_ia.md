# Estado IA y avance al objetivo

- Actualizado (UTC): 2026-02-13T15:42:58.434596+00:00
- Objetivo principal (Prob IA real): 70%
- Efectividad real global de cierres (bots 45-50): 49.56% (15194/30655)
- Brecha vs objetivo: -20.44%

## Señales IA cerradas (log)
- Total señales registradas: 82
- Total señales cerradas: 82
- Señales cerradas con prob >=70%: 82
- Acierto real en señales >=70%: 45.12% (37/82) | IC95%=[34.81%,55.87%]
- Estado semáforo objetivo 70%: 🔴 Aún no

## Recomendaciones priorizadas para subir Prob IA real
1. Brecha principal: estás en 49.56% global vs objetivo 70%. En corto plazo, prioriza reducir exposición REAL y subir filtro de calidad antes de aumentar volumen.
2. Muestra IA >=70% insuficiente (n=82). No tomes decisiones estructurales hasta llegar al menos a n>=200 cierres IA >=70%.
3. Umbral operativo sugerido temporal: >= 80% (hit=50.00%, IC95%=[34.85%,65.15%], n=38).
4. Hay sobreconfianza en bins de probabilidad (gap pred-real >10 pts). Aplicar shrinkage recomendado: p_ajustada = 0.6*p_calibrada + 0.4*tasa_base_rolling.
5. Bot más inflado del log IA: fulll46 (inflación=+54.7%, n=11). Aplicar penalización por bot (beta_bot) y bajar stake hasta salir de ALERTA/CRITICO.
6. Meta de modelo actual: reliable=True, auc=0.7703148347760467, brier=0.20568506652585608. Monitorear semanalmente ECE/Brier y recalibrar más frecuente que reentrenar base.

## Riesgo de calibración por bot (log IA)
| Bot | n | %Real | %Pred media | Inflación | beta_bot sugerido | IC95% real | Semáforo |
|---|---:|---:|---:|---:|---:|---:|:---:|
| fulll47 | 9 | 66.67% | 83.04% | +16.37% | 11.37% | [35.42%,87.94%] | CRITICO |
| fulll48 | 8 | 50.00% | 76.65% | +26.65% | 21.65% | [21.52%,78.48%] | CRITICO |
| fulll50 | 17 | 52.94% | 79.62% | +26.68% | 21.68% | [30.96%,73.84%] | CRITICO |
| fulll45 | 17 | 52.94% | 80.27% | +27.33% | 22.33% | [30.96%,73.84%] | CRITICO |
| fulll49 | 20 | 30.00% | 80.90% | +50.90% | 45.90% | [14.55%,51.90%] | CRITICO |
| fulll46 | 11 | 27.27% | 82.01% | +54.74% | 49.74% | [9.75%,56.57%] | CRITICO |

## Resumen por bot (cierres)
| Bot | Cerrados | Ganancias | Pérdidas | % Éxito |
|---|---:|---:|---:|---:|
| fulll45 | 5326 | 2620 | 2706 | 49.19% |
| fulll46 | 5065 | 2541 | 2524 | 50.17% |
| fulll47 | 5066 | 2537 | 2529 | 50.08% |
| fulll48 | 5058 | 2514 | 2544 | 49.70% |
| fulll49 | 5064 | 2536 | 2528 | 50.08% |
| fulll50 | 5076 | 2446 | 2630 | 48.19% |

## Sensibilidad por umbral (señales IA cerradas)
| Umbral | n | hit rate | IC95% | Muestra suficiente |
|---:|---:|---:|---:|:---:|
| 55% | 82 | 45.12% | [34.81%,55.87%] | ✅ |
| 60% | 82 | 45.12% | [34.81%,55.87%] | ✅ |
| 65% | 82 | 45.12% | [34.81%,55.87%] | ✅ |
| 70% | 82 | 45.12% | [34.81%,55.87%] | ✅ |
| 75% | 69 | 43.48% | [32.43%,55.21%] | ✅ |
| 80% | 38 | 50.00% | [34.85%,65.15%] | ⚠️ |
