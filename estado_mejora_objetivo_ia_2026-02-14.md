# Estado de mejora con data actualizada (2026-02-14)

## ¿Logramos mejorar el problema?

Parcialmente. Ya están implementados los 5 frentes técnicos pedidos, pero **todavía no hay evidencia suficiente de 70% real sostenido**.

- Se reforzó el gate de calidad por racha con versión **contextual por activo**.
- Se mantiene la recalibración (shrinkage + calibración del modelo en reentreno).
- Se filtran features casi constantes/redundantes antes de fit.
- Se agregaron interacciones dirigidas (`racha_x_rebote`, `rev_x_breakout`).
- Se valida por régimen/activo para no mezclar HZ10/HZ25/HZ50/HZ75.

## Qué muestran tus filas nuevas

En tu bloque reciente hay señales ganadoras incluso con racha negativa (ej. -3 y -2), lo cual indica que un gate rígido por racha podía sobrebloquear entradas válidas. Por eso se cambió a gate contextual:

1. Permite excepción por rebote confirmado.
2. Permite excepción por reversión RSI fuerte + breakout.
3. Si no hay excepción, solo permite racha negativa cuando el histórico reciente de ese **bot+activo** en racha<=-2 demuestra WR suficiente.

## Qué falta para acercarse de verdad a 70%

1. **Aumentar muestra cerrada IA>=70** (ideal >200) para validar estabilidad.
2. **Auditar por activo** semanalmente:
   - WR real por activo y por tramo de racha,
   - inflación Pred-Real,
   - tasa de bloqueo del gate (para no sobrefiltrar).
3. **Ajuste fino de thresholds del gate contextual** (`GATE_RACHA_NEG_MIN_WR`, `GATE_RSI_REV_MIN_NEG`) con datos cerrados de 7–14 días.
4. **Subir calidad de señal de features binarias** (hoy muchas siguen casi constantes).

## Conclusión ejecutiva

- Sí hubo mejora de arquitectura y control de riesgo de sobreconfianza.
- Pero para afirmar “ya llegamos” al objetivo 70% falta validación con más cierres reales y monitoreo por activo.
