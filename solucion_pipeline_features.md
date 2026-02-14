# Solución propuesta para reparar features y entrenamiento IA

## Objetivo 0 — Evitar entrenar con columnas rotas/muertas

Implementado en entrenamiento (`maybe_retrain`):
- Auditoría por feature con:
  - `nunique` (variación)
  - `dominance` (proporción del valor más común)
- Reglas activas:
  - `ROTA` si `nunique <= 1`
  - `CASI_CONSTANTE` si `dominance > 0.97`
- Variables `ROTA`/`CASI_CONSTANTE` se excluyen temporalmente del fit.
- Se guarda evidencia en `model_meta.json`:
  - `feature_health`
  - `dropped_features`

## Objetivo 1 — Reparar `hora_bucket`

Implementado:
- `hora_bucket` ahora usa bucket de **30 minutos** (48 estados) normalizado 0..1.
- Fuente temporal priorizada y robusta:
  1. `ts` ISO
  2. `epoch`/`timestamp` (segundos o ms)
  3. `fecha`
  4. `hora`
- Se corrigen casos ms/segundos y vacíos.

Validación esperada:
- `hora_bucket` ya no debe tener `nunique=1`.
- En muestras reales debe acercarse a ~24–48 buckets activos según cobertura horaria.

## Objetivo 2 — Descongelar features booleanas de evento

Implementado helper `enriquecer_features_evento`:
- `cruce_sma`: pasa a intensidad por separación relativa SMA5 vs SMA20.
- `breakout`: mezcla flag original + intensidad por spread/volatilidad.
- `rsi_reversion`: intensidad por distancia a extremos RSI (<30 o >70).
- `es_rebote`: pasa a intensidad 0..1 (racha + señal de giro), no solo 0/1.
- `puntaje_estrategia`: se recalcula sobre señales enriquecidas.

## Objetivo 3 — Alineación temporal

Se refuerza el uso de timestamp real para `hora_bucket`.

Siguiente paso recomendado (operativo):
- Confirmar que RSI/SMA se calculan con el **mismo timeframe** de la decisión del trade.
- Si el trade decide en 1m, los indicadores deben venir del mismo 1m (no 5m mezclado).

## Objetivo 4 — Simplificación temporal del modelo

Implementado:
- Si tras filtros saludables quedan muy pocas columnas (`<=2`), activa modo simplificado y entrena con `racha_actual` únicamente hasta reparar el resto.

## Objetivo 5 — Criterios de éxito

Checklist post-fix:
- `hora_bucket` con variación real (no constante).
- Menos columnas en estado `CASI_CONSTANTE`.
- 2–5 features con señal útil real (además de `racha_actual`).
- Menor sobreconfianza de probabilidades (mejor calibración en muestra fuera de train).

