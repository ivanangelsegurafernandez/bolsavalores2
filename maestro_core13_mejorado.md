# Maestro CORE-13 Mejorado (v2)

Generado: 2026-02-14
Base de decisión: veredicto actualizado sobre **37,036 cierres**.

## 1) Diagnóstico ejecutivo

1. **`hora_bucket` quedó sano**: tiene 24 buckets y baja dominancia; ya no está roto ni en fallback.
2. **Las supuestas features “de evento” están pegadas**:
   - `cruce_sma` ≈ 98.77% no-cero (dominancia 0.988)
   - `breakout` ≈ 94.97% no-cero (dominancia 0.950)
   - `puntaje_estrategia` ≈ 100% no-cero (dominancia 0.967)
3. **`racha_actual` sigue siendo el núcleo de señal**: AUC global univariado ~0.61 y mejora por segmentos (hasta ~0.67–0.70).
4. **Hay inestabilidad temporal severa**: fold inicial muy alto (~0.80+) y caída posterior hasta ~0.51.
5. **La mejora por nuevas rediseñadas es marginal**: delta ablation ~+0.0022 (potencial ruido).

---

## 2) Definición del nuevo “maestro” operativo

### Pilar A — Semántica correcta de features (estado vs evento)

- **Regla maestra**: ninguna feature se usa si su definición no coincide con su nombre.
- `cruce_sma`:
  - Si representa `fast > slow`, renombrarla como **estado_tendencia_sma**.
  - Si se desea evento real de cruce, definirla como **transición** (0→1 o 1→0), no como estado continuo.
- `breakout`:
  - Debe activarse al romper nivel/umbral en vela actual respecto a referencia previa, no permanecer “encendido” casi siempre.

**Meta operativa**: evitar binarias críticas con dominancia >95–97% salvo justificación explícita.

### Pilar B — Pasar de switch binario a señal continua

Para `rsi_reversion`, `es_rebote` y futuras binarias:
- usar **intensidad/proximidad** al umbral,
- usar **distancia normalizada** (z-score o porcentaje),
- usar **pendiente/velocidad** del indicador,
- mantener binaria solo como flag auxiliar.

**Meta operativa**: que la feature aporte gradiente útil y no solo encendido/apagado.

### Pilar C — `racha_actual` como core + segmentación como filtro de ejecución

No usar segmentación solo para reportes: usarla para decidir exposición.

- Priorizar segmentos donde `AUC(racha_actual)` es mayor (ej. payout medio/alto cuando sostengan ventaja).
- Reducir stake/frecuencia donde señal cae a zona plana.
- Aplicar ranking dinámico por segmento: payout, activo, hora, volatilidad.

**Meta operativa**: más operaciones donde hay señal y menos donde no la hay.

### Pilar D — Arquitectura anti-drift

- Entrenamiento con **rolling window**.
- Reentreno frecuente (cadencia fija + disparador por degradación).
- Validación siempre **walk-forward**.
- Monitoreo online de drift (caída de AUC, cambio de distribución, PSI/KS según disponibilidad).

**Meta operativa**: evitar que el sistema “funcione solo en un período”.

### Pilar E — Auditoría anti-leakage y anti-ruido

- Verificar que `racha_actual` se calcule solo con pasado estricto.
- Validar cortes temporales sin contaminación entre train/test.
- Rechazar mejoras de features por fold aislado.
- Exigir consistencia multfold para aceptar cambios.

**Meta operativa**: credibilidad estadística y control de sobreajuste.

---

## 3) Protocolo de decisión (Go / No-Go)

### Go
Se permite promover cambios al maestro solo si:
1. Mejoran baseline (`solo racha_actual`) en walk-forward agregado.
2. Mejoran en mayoría de folds (no solo en uno).
3. No empeoran estabilidad temporal.
4. Pasan chequeo de fuga temporal.

### No-Go
Se rechaza promoción si:
1. Delta total ≤ ruido operativo (ej. +0.002 sin estabilidad).
2. Hay fold inflado sospechoso sin explicación robusta.
3. La mejora depende de features dominantes/casi constantes.

---

## 4) Política de ejecución real (trading)

1. **Core de señal**: `racha_actual`.
2. **Gating por segmento**:
   - subir tamaño/frecuencia en segmentos top,
   - reducir exposición fuera de segmentos con ventaja.
3. **Control de riesgo adaptativo**:
   - freno automático cuando cae desempeño del segmento,
   - recuperación gradual tras validación de estabilidad.
4. **Disciplina de despliegue**:
   - cada cambio pasa por ablation walk-forward,
   - nada entra a producción por intuición o por un único fold brillante.

---

## 5) Backlog priorizado del maestro v2

### Prioridad P0 (obligatorio)
- Auditoría de fuga temporal en cálculo de `racha_actual` y en armado de folds.
- Redefinición formal de `cruce_sma` y `breakout` (evento vs estado).

### Prioridad P1
- Rediseño de binarias a continuas de intensidad/proximidad.
- Integrar segmentación en la lógica de sizing/gating operativo.

### Prioridad P2
- Monitoreo de drift y recalibración/reentreno adaptativo.
- Framework de aceptación automática de features por consistencia multfold.

---

## 6) Veredicto final del Maestro CORE-13 v2

Con el estado actual, el enfoque rentable y robusto es:
- **`racha_actual` como columna vertebral**,
- **filtros por segmento como ventaja táctica**,
- **cero tolerancia a fuga temporal**,
- **cero promoción de features sin ganancia estable real**.

Este maestro prioriza **resultado real + estabilidad temporal + credibilidad estadística** por encima de mejoras cosméticas.
