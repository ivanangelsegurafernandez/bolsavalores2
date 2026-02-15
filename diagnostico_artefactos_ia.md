# Diagnóstico de artefactos IA (alineación + causa raíz)

Generado: 2026-02-15 (UTC)

## 1) Alineación de artefactos

### Resultado
- `model_meta.json` y `feature_names.pkl` están alineados en features: ambos reportan `['racha_actual']`.
- Timestamps de artefactos (`modelo_xgb.pkl`, `scaler.pkl`, `feature_names.pkl`, `model_meta.json`) son consistentes en el mismo bloque temporal de actualización.
- Validación profunda de carga de `modelo_xgb.pkl`/`scaler.pkl` quedó limitada por dependencias ausentes en este entorno (`sklearn`, `joblib`, `xgboost`).

### Evidencia
- `model_meta.feature_names = ['racha_actual']` y `feature_names.pkl = ['racha_actual']`.
- `model_meta`: `auc=0.5667`, `reliable=false`, `brier=0.2586`.

## 2) Por qué quedó casi solo `racha_actual`

Se observaron dos causas simultáneas:

1. **Calidad predictiva real baja del resto de variables** en la muestra cerrada grande.
   - En `veredicto_variables_core13.md`, `racha_actual` aparece como única variable **UTIL** (AUC univariado ~0.629).
   - El resto está en **BAJA** o **CASI_CONSTANTE** (por dominancia alta), por lo que el quality-gate las filtra.

2. **Modo simplificado del pipeline de entrenamiento**.
   - El maestro activa explícitamente "modo simplificado" cuando tras filtros quedan <=2 features y existe `racha_actual`.
   - Esto consolida `feature_names=['racha_actual']` en los artefactos finales.

## 3) Recalibración y medición previa a exigir 75%

Se recalculó el estado usando objetivo `75%` y muestra mínima fuerte `n>=200`:

- Señales cerradas con prob `>=75%`: **69**
- Acierto real en `>=75%`: **43.48%** (30/69), IC95% **[32.43%, 55.21%]**
- Estado: **NO cumple** para exigir 75% como gatillo duro
- Sobreconfianza clara por bins:
  - [0.70,0.80): gap +34.38 pp
  - [0.80,0.90): gap +33.99 pp
  - [0.90,1.00): gap +44.44 pp

Conclusión operacional: mantener 75% como objetivo de calidad, pero no como gatillo duro inmediato mientras no suba la calibración real y la muestra fuerte.

## 4) Acciones recomendadas inmediatas

1. Mantener auditoría con objetivo 75% (ya actualizada en `status_objetivo_ia.md`).
2. Corregir variables casi-constantes por diseño (cruce/breakout/rebote/puntaje) para devolver variabilidad útil.
3. Reentrenar y validar con walk-forward comparando:
   - baseline: `racha_actual`
   - combo: `racha_actual + variables rediseñadas`
4. Solo endurecer gatillo REAL a 75% cuando se cumpla:
   - n fuerte >= 200 (>=75%), y
   - hit real >= 75% con IC inferior aceptable.
