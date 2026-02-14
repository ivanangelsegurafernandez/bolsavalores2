# Estado IA y avance al objetivo

- Actualizado (UTC): 2026-02-14T00:09:46.908033+00:00
- Objetivo principal (Prob IA real): 70%
- Efectividad real global de cierres (bots 45-50): 49.69% (18133/36492)
- Brecha vs objetivo: -20.31%

## Señales IA cerradas (log)
- Total señales registradas: 117
- Total señales cerradas: 117
- Señales cerradas con prob >=70%: 117
- Acierto real en señales >=70%: 45.30% (53/117) | IC95%=[36.57%,54.33%]
- Estado semáforo objetivo 70%: 🔴 Aún no

## Recomendaciones priorizadas para subir Prob IA real
1. Brecha principal: estás en 49.69% global vs objetivo 70%. En corto plazo, prioriza reducir exposición REAL y subir filtro de calidad antes de aumentar volumen.
2. Muestra IA >=70% insuficiente (n=117). No tomes decisiones estructurales hasta llegar al menos a n>=200 cierres IA >=70%.
3. Umbral operativo sugerido temporal: >= 80% (hit=50.00%, IC95%=[36.64%,63.36%], n=50).
4. Hay sobreconfianza en bins de probabilidad (gap pred-real >10 pts). Aplicar shrinkage recomendado: p_ajustada = 0.6*p_calibrada + 0.4*tasa_base_rolling.
5. Bots a intervenir primero (impacto ponderado): fulll49(+51.2%, n=24, prioridad=0.41), fulll46(+54.4%, n=18, prioridad=0.33). Aplicar beta_bot y reducción de stake según semáforo.
6. Meta de modelo actual: reliable=True, auc=0.766671274412287, brier=0.20655362589713477. Monitorear semanalmente ECE/Brier y recalibrar más frecuente que reentrenar base.

## Riesgo de calibración por bot (log IA)
| Bot | n | Madurez | %Real | %Pred media | Inflación | beta_bot | Prioridad | Semáforo | Acción sugerida |
|---|---:|:---:|---:|---:|---:|---:|---:|:---:|---|
| fulll49 | 24 | MEDIA_MUESTRA | 29.17% | 80.35% | +51.18% | 46.18% | 0.41 | CRITICO | Reducir stake 50% y aplicar beta_bot completo |
| fulll46 | 18 | MEDIA_MUESTRA | 27.78% | 82.19% | +54.41% | 49.41% | 0.33 | CRITICO | Reducir stake 50% y aplicar beta_bot completo |
| fulll50 | 27 | MEDIA_MUESTRA | 51.85% | 78.70% | +26.84% | 21.84% | 0.24 | CRITICO | Reducir stake 50% y aplicar beta_bot completo |
| fulll45 | 22 | MEDIA_MUESTRA | 50.00% | 79.78% | +29.78% | 24.78% | 0.22 | CRITICO | Reducir stake 50% y aplicar beta_bot completo |
| fulll48 | 13 | BAJA_MUESTRA | 53.85% | 77.43% | +23.59% | 18.59% | 0.10 | ALERTA | Reducir stake 25% y aplicar beta_bot parcial |
| fulll47 | 13 | BAJA_MUESTRA | 69.23% | 81.90% | +12.67% | 7.67% | 0.05 | OK | Mantener stake, monitoreo semanal |

## Resumen por bot (cierres)
| Bot | Cerrados | Ganancias | Pérdidas | % Éxito |
|---|---:|---:|---:|---:|
| fulll45 | 5421 | 2666 | 2755 | 49.18% |
| fulll46 | 6215 | 3118 | 3097 | 50.17% |
| fulll47 | 6213 | 3098 | 3115 | 49.86% |
| fulll48 | 6199 | 3080 | 3119 | 49.69% |
| fulll49 | 6215 | 3125 | 3090 | 50.28% |
| fulll50 | 6229 | 3046 | 3183 | 48.90% |

## Sensibilidad por umbral (señales IA cerradas)
| Umbral | n | hit rate | IC95% | Muestra suficiente |
|---:|---:|---:|---:|:---:|
| 55% | 117 | 45.30% | [36.57%,54.33%] | ✅ |
| 60% | 117 | 45.30% | [36.57%,54.33%] | ✅ |
| 65% | 117 | 45.30% | [36.57%,54.33%] | ✅ |
| 70% | 117 | 45.30% | [36.57%,54.33%] | ✅ |
| 75% | 104 | 44.23% | [35.06%,53.81%] | ✅ |
| 80% | 50 | 50.00% | [36.64%,63.36%] | ⚠️ |
