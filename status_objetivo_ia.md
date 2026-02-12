# Estado IA y avance al objetivo

- Actualizado (UTC): 2026-02-12T09:53:26.995299+00:00
- Objetivo principal (Prob IA real): 70%
- Efectividad real global de cierres (bots 45-50): 49.59% (16426/33125)
- Brecha vs objetivo: -20.41%

## Señales IA cerradas (log)
- Total señales registradas: 78
- Total señales cerradas: 78
- Señales cerradas con prob >=70%: 78
- Acierto real en señales >=70%: 46.15% (36/78) | IC95%=[35.53%,57.14%]
- Estado semáforo objetivo 70%: 🔴 Aún no

## Recomendaciones priorizadas para subir Prob IA real
1. Brecha principal: estás en 49.59% global vs objetivo 70%. En corto plazo, prioriza reducir exposición REAL y subir filtro de calidad antes de aumentar volumen.
2. Muestra IA >=70% insuficiente (n=78). No tomes decisiones estructurales hasta llegar al menos a n>=200 cierres IA >=70%.
3. Umbral operativo sugerido temporal: >= 80% (hit=51.43%, IC95%=[35.57%,67.01%], n=35).
4. Hay sobreconfianza en bins de probabilidad (gap pred-real >10 pts). Aplicar shrinkage recomendado: p_ajustada = 0.6*p_calibrada + 0.4*tasa_base_rolling.
5. Meta de modelo actual: reliable=True, auc=0.6795974077455316, brier=0.2272919700942694. Monitorear semanalmente ECE/Brier y recalibrar más frecuente que reentrenar base.

## Resumen por bot (cierres)
| Bot | Cerrados | Ganancias | Pérdidas | % Éxito |
|---|---:|---:|---:|---:|
| fulll45 | 4864 | 2377 | 2487 | 48.87% |
| fulll46 | 5649 | 2840 | 2809 | 50.27% |
| fulll47 | 5654 | 2836 | 2818 | 50.16% |
| fulll48 | 5642 | 2802 | 2840 | 49.66% |
| fulll49 | 5652 | 2829 | 2823 | 50.05% |
| fulll50 | 5664 | 2742 | 2922 | 48.41% |

## Sensibilidad por umbral (señales IA cerradas)
| Umbral | n | hit rate | IC95% | Muestra suficiente |
|---:|---:|---:|---:|:---:|
| 55% | 78 | 46.15% | [35.53%,57.14%] | ✅ |
| 60% | 78 | 46.15% | [35.53%,57.14%] | ✅ |
| 65% | 78 | 46.15% | [35.53%,57.14%] | ✅ |
| 70% | 78 | 46.15% | [35.53%,57.14%] | ✅ |
| 75% | 65 | 44.62% | [33.17%,56.66%] | ✅ |
| 80% | 35 | 51.43% | [35.57%,67.01%] | ⚠️ |
