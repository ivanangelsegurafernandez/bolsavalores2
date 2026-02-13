# Análisis rápido (antes de cambios) — Prob IA objetivo 70%

Fecha: 2026-02-13.
Base usada: `registro_enriquecido_fulll45.csv` ... `registro_enriquecido_fulll50.csv` (solo `result_bin` cerrado).

## Hallazgos clave

- Muestra analizada: **30,698 cierres**, winrate base **49.55%**.
- En el tablero IA actual, las señales >=70% van en **45.3% real (53/117)**, con brecha fuerte vs objetivo 70%.
- La mayoría de features binarias están muy desbalanceadas (casi siempre 1 o casi siempre 0), por lo que aportan poca separación real.
- **`racha_actual` es la variable con más señal** en esta foto de datos.

## Señal estadística rápida por feature (aprox)

> Métricas orientativas: correlación con `result_bin` y AUC univariado (sin modelo multivariable).

- `rsi_9`: corr +0.0022 | AUC 0.5004
- `rsi_14`: corr -0.0073 | AUC 0.5044
- `sma_5`: corr +0.0064 | AUC 0.5029
- `sma_20`: corr +0.0064 | AUC 0.5029
- `cruce_sma`: corr +0.0007 | AUC 0.5001 (98.7% de filas = 1)
- `breakout`: corr +0.0040 | AUC 0.5009 (94.6% = 1)
- `rsi_reversion`: corr +0.0047 | AUC 0.5014 (90.1% = 0)
- **`racha_actual`**: **corr +0.2148 | AUC 0.6313**
- `payout` (ROI derivado): corr +0.0052 | AUC 0.5028
- `puntaje_estrategia`: corr +0.0138 | AUC 0.5024 (96.9% mismo valor)
- `volatilidad` (proxy SMA): corr +0.0031 | AUC 0.5012
- `es_rebote`: corr +0.0187 | AUC 0.5045 (6.2% = 1)
- `hora_bucket`: corr +0.0009 | AUC 0.5006

## Qué sí parece funcional vs qué hoy aporta poco

### Funcional (ahora)
- **`racha_actual`**: separa bien. Ejemplo por tramos:
  - racha [-4,-2] => ~36.5% acierto
  - racha [2,4] => ~62.1% acierto
  - racha >=5 => ~65.2% acierto

### Funcionales pero débiles / con señal marginal
- `es_rebote`: mejora moderada cuando vale 1 (~53.2% vs ~49.3%), pero poca cobertura.

### Probablemente redundantes o con poco valor en su estado actual
- `sma_5` y `sma_20`: casi colineales entre sí (muy alta correlación).
- `cruce_sma`, `breakout`, `rsi_reversion`, `puntaje_estrategia`: alta concentración en 1 o 0, poca variabilidad útil.
- `hora_bucket`: distribución razonable, pero sin separación fuerte.
- `payout`: demasiados niveles repetidos; por sí solo separa casi nada.

## Riesgos de calidad de datos detectados

- Hay patrones de features repetidos con etiqueta distinta (`result_bin` cambia) en una fracción pequeña: puede meter ruido de aprendizaje.
- El tablero reporta sobreconfianza (prob alta, hit real bajo), por lo que el problema principal hoy parece ser **calibración + filtro de entrada**, no falta de complejidad del modelo.

## Opinión antes de tocar código

Para empujar la Prob IA real hacia ~70% con data real, primero atacaría esto en orden:

1. **Gating operativo estricto por calidad** (no por volumen):
   - bloquear rachas negativas fuertes (ej. <= -2),
   - priorizar ventanas con racha positiva y rebote confirmado.
2. **Recalibrar probabilidades** (shrinkage o isotónica) para bajar sobreconfianza.
3. **Reducir features casi constantes** o reingenierizarlas (que produzcan más estados reales).
4. **Agregar interacciones dirigidas** (ej. `racha_actual x es_rebote`, `rsi_reversion x breakout`) en vez de añadir muchas columnas nuevas.
5. **Validar por régimen/activo** (HZ10/HZ25/HZ50/HZ75) para no mezclar contextos distintos.

Conclusión: **No, hoy no todas las 13 variables están aportando de forma similar**. La mayoría funcionan técnicamente, pero varias aportan muy poco poder predictivo en su estado actual; `racha_actual` domina la señal en esta muestra.
