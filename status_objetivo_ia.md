# Estado IA y avance al objetivo

- Actualizado (UTC): 2026-02-12T22:58:04.923116+00:00
- Objetivo principal (Prob IA real): 70%
- Efectividad real global de cierres (bots 45-50): 49.59% (17017/34316)
- Brecha vs objetivo: -20.41%

## Señales IA cerradas (log)
- Total señales registradas: 82
- Total señales cerradas: 82
- Señales cerradas con prob >=70%: 82
- Acierto real en señales >=70%: 45.12% (37/82) | IC95%=[34.81%,55.87%]
- Estado semáforo objetivo 70%: 🔴 Aún no

## Recomendaciones priorizadas para subir Prob IA real
1. Brecha principal: estás en 49.59% global vs objetivo 70%. En corto plazo, prioriza reducir exposición REAL y subir filtro de calidad antes de aumentar volumen.
2. Muestra IA >=70% insuficiente (n=82). No tomes decisiones estructurales hasta llegar al menos a n>=200 cierres IA >=70%.
3. Umbral operativo sugerido temporal: >= 80% (hit=50.00%, IC95%=[34.85%,65.15%], n=38).
4. Hay sobreconfianza en bins de probabilidad (gap pred-real >10 pts). Aplicar shrinkage recomendado: p_ajustada = 0.6*p_calibrada + 0.4*tasa_base_rolling.
5. Meta de modelo actual: reliable=True, auc=0.7165776123030029, brier=0.21859219409812153. Monitorear semanalmente ECE/Brier y recalibrar más frecuente que reentrenar base.

## Resumen por bot (cierres)
| Bot | Cerrados | Ganancias | Pérdidas | % Éxito |
|---|---:|---:|---:|---:|
| fulll45 | 5062 | 2480 | 2582 | 48.99% |
| fulll46 | 5848 | 2935 | 2913 | 50.19% |
| fulll47 | 5856 | 2925 | 2931 | 49.95% |
| fulll48 | 5836 | 2900 | 2936 | 49.69% |
| fulll49 | 5851 | 2930 | 2921 | 50.08% |
| fulll50 | 5863 | 2847 | 3016 | 48.56% |

## Sensibilidad por umbral (señales IA cerradas)
| Umbral | n | hit rate | IC95% | Muestra suficiente |
|---:|---:|---:|---:|:---:|
| 55% | 82 | 45.12% | [34.81%,55.87%] | ✅ |
| 60% | 82 | 45.12% | [34.81%,55.87%] | ✅ |
| 65% | 82 | 45.12% | [34.81%,55.87%] | ✅ |
| 70% | 82 | 45.12% | [34.81%,55.87%] | ✅ |
| 75% | 69 | 43.48% | [32.43%,55.21%] | ✅ |
| 80% | 38 | 50.00% | [34.85%,65.15%] | ⚠️ |
