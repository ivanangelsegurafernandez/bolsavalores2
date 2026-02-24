# Reporte Integral de Salud IA

Generado UTC: `2026-02-24T10:32:08.280117+00:00`

## 1) Calibración real de probabilidades
- Señales cerradas: **1**
- Precisión @>=70%: **100.0%** (n=1)
- Precisión @>=85%: **100.0%** (n=1)
- ⚠️ Muestra cerrada muy baja: estas precisiones son orientativas, no concluyentes.

## 2) Desalineación Prob IA vs hitrate por bot (last_n=40)
| Bot | WR last40 (csv) | n señales IA | Hit last40 (señales) | Prob media last40 (señales) | Gap Prob-Hit señales | Gap Prob-WR csv | Muestra señales |
|---|---:|---:|---:|---:|---:|---:|---|
| fulll45 | 62.5% | 0 | N/A | N/A | N/A | N/A | BAJA(<5) |
| fulll46 | 62.5% | 0 | N/A | N/A | N/A | N/A | BAJA(<5) |
| fulll47 | 55.0% | 1 | 100.0% | 92.5% | N/A | N/A | BAJA(<5) |
| fulll48 | 60.0% | 0 | N/A | N/A | N/A | N/A | BAJA(<5) |
| fulll49 | 72.5% | 0 | N/A | N/A | N/A | N/A | BAJA(<5) |
| fulll50 | 72.5% | 0 | N/A | N/A | N/A | N/A | BAJA(<5) |

## 3) Salud de ejecución (auth/ws/timeout)
- No auditado en este run (falta `--runtime-log`).

## 4) Recomendación de cuándo correr este programa
- **Recomendado siempre**: al iniciar sesión y luego cada 30-60 min.
- **Corte de calidad fuerte**: después de cada bloque de +20 cierres nuevos.
- **Punto mínimo para decisiones estructurales**:
  - ✅ n_samples>=250
  - ❌ closed_signals>=80
  - ❌ reliable=true
  - ✅ auc>=0.53
- Ready for full diagnosis: **False**

## 5) Qué falta corregir si no está “bien”
- Nota: `Gap Prob-Hit señales` usa SOLO señales cerradas en `ia_signals_log.csv` y puede diferir de `WR last40 (csv)` del bot.
- Gaps por bot se publican solo si `n señales IA >= 5` para evitar conclusiones con muestra mínima.
- Si `precision@85` baja o n es pequeño: recalibrar/proteger compuerta.
- Si gap Prob-Hit por bot es alto: bajar exposición o bloquear bot temporalmente.
- Si auth/ws/timeouts suben: estabilizar conectividad antes de evaluar modelo.
- Si WHY-NO se concentra en `trigger_no`/`confirm_pending`: revisar timing de señales y trigger.
