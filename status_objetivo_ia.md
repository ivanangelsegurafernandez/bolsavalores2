# Estado IA y avance al objetivo

- Actualizado (UTC): 2026-02-13T13:10:17.500839+00:00
- Objetivo principal (Prob IA real): 70%
- Efectividad real global de cierres (bots 45-50): 49.71% (17854/35914)
- Brecha vs objetivo: -20.29%

## Señales IA cerradas (log)
- Total señales registradas: 117
- Total señales cerradas: 117
- Señales cerradas con prob >=70%: 117
- Acierto real en señales >=70%: 45.30% (53/117) | IC95%=[36.57%,54.33%]
- Estado semáforo objetivo 70%: 🔴 Aún no

## Recomendaciones priorizadas para subir Prob IA real
1. Brecha principal: estás en 49.71% global vs objetivo 70%. En corto plazo, prioriza reducir exposición REAL y subir filtro de calidad antes de aumentar volumen.
2. Muestra IA >=70% insuficiente (n=117). No tomes decisiones estructurales hasta llegar al menos a n>=200 cierres IA >=70%.
3. Umbral operativo sugerido temporal: >= 80% (hit=50.00%, IC95%=[36.64%,63.36%], n=50).
4. Hay sobreconfianza en bins de probabilidad (gap pred-real >10 pts). Aplicar shrinkage recomendado: p_ajustada = 0.6*p_calibrada + 0.4*tasa_base_rolling.
5. Meta de modelo actual: reliable=True, auc=0.7703148347760467, brier=0.20568506652585608. Monitorear semanalmente ECE/Brier y recalibrar más frecuente que reentrenar base.

## Resumen por bot (cierres)
| Bot | Cerrados | Ganancias | Pérdidas | % Éxito |
|---|---:|---:|---:|---:|
| fulll45 | 5326 | 2620 | 2706 | 49.19% |
| fulll46 | 6118 | 3071 | 3047 | 50.20% |
| fulll47 | 6118 | 3056 | 3062 | 49.95% |
| fulll48 | 6102 | 3035 | 3067 | 49.74% |
| fulll49 | 6117 | 3080 | 3037 | 50.35% |
| fulll50 | 6133 | 2992 | 3141 | 48.79% |

## Sensibilidad por umbral (señales IA cerradas)
| Umbral | n | hit rate | IC95% | Muestra suficiente |
|---:|---:|---:|---:|:---:|
| 55% | 117 | 45.30% | [36.57%,54.33%] | ✅ |
| 60% | 117 | 45.30% | [36.57%,54.33%] | ✅ |
| 65% | 117 | 45.30% | [36.57%,54.33%] | ✅ |
| 70% | 117 | 45.30% | [36.57%,54.33%] | ✅ |
| 75% | 104 | 44.23% | [35.06%,53.81%] | ✅ |
| 80% | 50 | 50.00% | [36.64%,63.36%] | ⚠️ |
