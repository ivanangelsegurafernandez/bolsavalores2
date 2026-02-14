# Estado IA y avance al objetivo

- Actualizado (UTC): 2026-02-14T01:17:52.364953+00:00
- Objetivo principal (Prob IA real): 70%
- Efectividad real global de cierres (bots 45-50): 49.55% (15210/30698)
- Brecha vs objetivo: -20.45%

## Señales IA cerradas (log)
- Total señales registradas: 82
- Total señales cerradas: 82
- Señales cerradas con prob >=70%: 82
- Acierto real en señales >=70%: 45.12% (37/82) | IC95%=[34.81%,55.87%]
- Estado semáforo objetivo 70%: 🔴 Aún no

## Recomendaciones priorizadas para subir Prob IA real
1. Brecha principal: estás en 49.55% global vs objetivo 70%. En corto plazo, prioriza reducir exposición REAL y subir filtro de calidad antes de aumentar volumen.
2. Muestra IA >=70% insuficiente (n=82). No tomes decisiones estructurales hasta llegar al menos a n>=200 cierres IA >=70%.
3. Umbral operativo sugerido temporal: >= 80% (hit=50.00%, IC95%=[34.85%,65.15%], n=38).
4. Hay sobreconfianza en bins de probabilidad (gap pred-real >10 pts). Aplicar shrinkage recomendado: p_ajustada = 0.6*p_calibrada + 0.4*tasa_base_rolling.
5. Bots a intervenir primero (impacto ponderado): fulll49(+50.9%, n=20, prioridad=0.34), fulll46(+54.7%, n=11, prioridad=0.20). Aplicar beta_bot y reducción de stake según semáforo.
6. Meta de modelo actual: reliable=True, auc=0.8039844638651692, brier=0.197933841371934. Monitorear semanalmente ECE/Brier y recalibrar más frecuente que reentrenar base.
7. Variables con señal débil + alta constancia detectadas: es_rebote, puntaje_estrategia, rsi_reversion, breakout, cruce_sma. Reingenierizar umbrales/estados para que no queden casi constantes.

## Riesgo de calibración por bot (log IA)
| Bot | n | Madurez | %Real | %Pred media | Inflación | beta_bot | Prioridad | Semáforo | Acción sugerida |
|---|---:|:---:|---:|---:|---:|---:|---:|:---:|---|
| fulll49 | 20 | MEDIA_MUESTRA | 30.00% | 80.90% | +50.90% | 45.90% | 0.34 | CRITICO | Reducir stake 50% y aplicar beta_bot completo |
| fulll46 | 11 | BAJA_MUESTRA | 27.27% | 82.01% | +54.74% | 49.74% | 0.20 | CRITICO | Reducir stake 50% y aplicar beta_bot completo |
| fulll45 | 17 | MEDIA_MUESTRA | 52.94% | 80.27% | +27.33% | 22.33% | 0.15 | CRITICO | Reducir stake 50% y aplicar beta_bot completo |
| fulll50 | 17 | MEDIA_MUESTRA | 52.94% | 79.62% | +26.68% | 21.68% | 0.15 | CRITICO | Reducir stake 50% y aplicar beta_bot completo |
| fulll48 | 8 | BAJA_MUESTRA | 50.00% | 76.65% | +26.65% | 21.65% | 0.07 | CRITICO | Reducir stake 50% y aplicar beta_bot completo |
| fulll47 | 9 | BAJA_MUESTRA | 66.67% | 83.04% | +16.37% | 11.37% | 0.05 | ALERTA | Reducir stake 25% y aplicar beta_bot parcial |

## Resumen por bot (cierres)
| Bot | Cerrados | Ganancias | Pérdidas | % Éxito |
|---|---:|---:|---:|---:|
| fulll45 | 5369 | 2636 | 2733 | 49.10% |
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

## Diagnóstico rápido de las 13 variables (cierres reales)
- Muestra usada: 30698 filas cerradas con `result_bin`.
| Feature | AUC univariado | Corr(y) | % valor dominante | Únicos | Señal |
|---|---:|---:|---:|---:|:---:|
| racha_actual | 0.6313 | +0.2148 | 25.00% | 27 | fuerte |
| es_rebote | 0.5045 | +0.0187 | 93.83% | 2 | débil |
| rsi_14 | 0.5044 | -0.0073 | 0.08% | 5811 | débil |
| sma_20 | 0.5029 | +0.0064 | 0.02% | 28628 | débil |
| sma_5 | 0.5029 | +0.0064 | 0.02% | 28217 | débil |
| payout | 0.5028 | +0.0052 | 54.99% | 11 | débil |
| puntaje_estrategia | 0.5024 | +0.0138 | 96.87% | 2 | débil |
| hora_bucket | 0.5015 | -0.0027 | 31.31% | 4 | débil |
| rsi_reversion | 0.5014 | +0.0047 | 90.12% | 2 | débil |
| volatilidad | 0.5012 | +0.0031 | 0.02% | 28699 | débil |
| breakout | 0.5009 | +0.0040 | 94.57% | 2 | débil |
| rsi_9 | 0.5004 | +0.0022 | 0.31% | 6746 | débil |
| cruce_sma | 0.5001 | +0.0007 | 98.68% | 2 | débil |

**Variables candidatas a reingeniería (señal débil + constancia alta):** es_rebote, puntaje_estrategia, rsi_reversion, breakout, cruce_sma
