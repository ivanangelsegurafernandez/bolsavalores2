# Estado IA y avance al objetivo

- Actualizado (UTC): 2026-02-13T20:12:11.423361+00:00
- Objetivo principal (Prob IA real): 70%
- Efectividad real global de cierres (bots 45-50): 49.67% (17963/36167)
- Brecha vs objetivo: -20.33%

## Señales IA cerradas (log)
- Total señales registradas: 117
- Total señales cerradas: 117
- Señales cerradas con prob >=70%: 117
- Acierto real en señales >=70%: 45.30% (53/117) | IC95%=[36.57%,54.33%]
- Estado semáforo objetivo 70%: 🔴 Aún no

## Recomendaciones priorizadas para subir Prob IA real
1. Brecha principal: estás en 49.67% global vs objetivo 70%. En corto plazo, prioriza reducir exposición REAL y subir filtro de calidad antes de aumentar volumen.
2. Muestra IA >=70% insuficiente (n=117). No tomes decisiones estructurales hasta llegar al menos a n>=200 cierres IA >=70%.
3. Umbral operativo sugerido temporal: >= 80% (hit=50.00%, IC95%=[36.64%,63.36%], n=50).
4. Hay sobreconfianza en bins de probabilidad (gap pred-real >10 pts). Aplicar shrinkage recomendado: p_ajustada = 0.6*p_calibrada + 0.4*tasa_base_rolling.
5. Bot más inflado del log IA: fulll46 (inflación=+54.4%, n=18). Aplicar penalización por bot (beta_bot) y bajar stake hasta salir de ALERTA/CRITICO.
6. Meta de modelo actual: reliable=True, auc=0.7307109160208386, brier=0.2145385100315436. Monitorear semanalmente ECE/Brier y recalibrar más frecuente que reentrenar base.

## Riesgo de calibración por bot (log IA)
| Bot | n | %Real | %Pred media | Inflación | beta_bot sugerido | IC95% real | Semáforo |
|---|---:|---:|---:|---:|---:|---:|:---:|
| fulll47 | 13 | 69.23% | 81.90% | +12.67% | 7.67% | [42.37%,87.32%] | CRITICO |
| fulll48 | 13 | 53.85% | 77.43% | +23.59% | 18.59% | [29.14%,76.79%] | CRITICO |
| fulll50 | 27 | 51.85% | 78.70% | +26.84% | 21.84% | [33.99%,69.26%] | CRITICO |
| fulll45 | 22 | 50.00% | 79.78% | +29.78% | 24.78% | [30.72%,69.28%] | CRITICO |
| fulll49 | 24 | 29.17% | 80.35% | +51.18% | 46.18% | [14.91%,49.17%] | CRITICO |
| fulll46 | 18 | 27.78% | 82.19% | +54.41% | 49.41% | [12.50%,50.87%] | CRITICO |

## Resumen por bot (cierres)
| Bot | Cerrados | Ganancias | Pérdidas | % Éxito |
|---|---:|---:|---:|---:|
| fulll45 | 5369 | 2636 | 2733 | 49.10% |
| fulll46 | 6160 | 3087 | 3073 | 50.11% |
| fulll47 | 6160 | 3072 | 3088 | 49.87% |
| fulll48 | 6144 | 3051 | 3093 | 49.66% |
| fulll49 | 6159 | 3100 | 3059 | 50.33% |
| fulll50 | 6175 | 3017 | 3158 | 48.86% |

## Sensibilidad por umbral (señales IA cerradas)
| Umbral | n | hit rate | IC95% | Muestra suficiente |
|---:|---:|---:|---:|:---:|
| 55% | 117 | 45.30% | [36.57%,54.33%] | ✅ |
| 60% | 117 | 45.30% | [36.57%,54.33%] | ✅ |
| 65% | 117 | 45.30% | [36.57%,54.33%] | ✅ |
| 70% | 117 | 45.30% | [36.57%,54.33%] | ✅ |
| 75% | 104 | 44.23% | [35.06%,53.81%] | ✅ |
| 80% | 50 | 50.00% | [36.64%,63.36%] | ⚠️ |
