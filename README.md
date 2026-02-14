# bolsavalores2

## Afinar `Prob IA` para que represente probabilidad real

Si la interfaz muestra `Prob IA = 65%`, lo ideal es que, en promedio, **~65 de cada 100 señales con ese score terminen bien**. Para acercarse a eso, aplica este flujo:

### 1) Separar predicción vs calibración
- Mantén tu modelo principal (XGBoost) para **rankear** señales.
- Encima del score bruto, entrena un **calibrador** (`isotonic` o `Platt/sigmoid`) con datos fuera de muestra.
- Nunca calibres y evalúes con el mismo tramo temporal.

### 2) Validación temporal estricta (walk-forward)
- Evita mezclar pasado/futuro al partir datos.
- Usa bloques por tiempo: entrenas en ventana A, calibras en B, validas en C.
- Repite por múltiples ventanas y promedia métricas.

### 3) Medir calibración, no solo AUC/F1
Añade estas métricas en cada ciclo:
- **Brier Score** (más bajo = mejor calibración).
- **ECE/MCE** (error de calibración por bins).
- **Reliability curve** (probabilidad predicha vs frecuencia real).

### 4) Corregir sobreconfianza con “shrinkage”
Si detectas inflación sistemática (por ejemplo predicho 75%, real 45%):
- Aplica un ajuste conservador: `p_ajustada = alpha * p_calibrada + (1-alpha) * p_base`.
- Donde `p_base` puede ser la tasa histórica de acierto (ej. 0.45).
- Empieza con `alpha` entre `0.5` y `0.8` y optimiza por Brier/ECE.

### 5) Reglas por tamaño de muestra
- No mostrar probabilidades “fuertes” con `n` muy bajo.
- Si `n` del bot/ciclo en ventana reciente es pequeño, muestra etiqueta: `baja confianza estadística`.
- Usa intervalos de confianza (Wilson/Beta) junto al porcentaje.

### 6) Drift y recalibración continua
- Monitorea drift de features y drift de calibración (ECE por semana).
- Recalibra con mayor frecuencia que el reentrenamiento completo.
- Si hay cambio de régimen, baja temporalmente exposición y umbrales.

### 7) Ajustar umbral de entrada por valor esperado, no por score fijo
- No usar siempre `>=70%` por costumbre.
- Usa umbral que maximice `EV = p*ganancia_media - (1-p)*perdida_media - costos`.
- Reestimar ese umbral por bot/ciclo y por régimen.

### 8) Exponer en UI “Prob IA real” y “desviación”
Además de `Prob IA`, muestra:
- `Prob IA calibrada` (post-calibración).
- `Hit-rate real rolling` (últimos N trades cerrados).
- `Inflación = Prob media predicha - éxito real`.

---

## Checklist mínimo recomendado
1. Entrenar modelo base en `train`.
2. Calibrar en `calibration` con `isotonic`.
3. Validar en `test` con Brier + ECE + curva de confiabilidad.
4. Aplicar shrinkage si persiste sobreconfianza.
5. Publicar en UI: prob calibrada + intervalo + inflación rolling.

Con esto, la probabilidad que se ve en pantalla deja de ser “optimista” y se vuelve una estimación estadística más cercana al resultado real observado.


## Orden por etapas en el programa (para depurar más rápido)

En `5R6M-1-2-4-8-16.py` se añadió un flujo explícito por etapas para operación y diagnóstico:

- `BOOT_01` → Arranque y validación de entorno.
- `BOOT_02` → Tokens/audio/reset opcional.
- `BOOT_03` → Backfill + primer entrenamiento IA.
- `BOOT_04` → Sincronización inicial HUD/CSV.
- `TICK_01` → Lectura de token + carga incremental por bot.
- `TICK_02` → Watchdog REAL + detección de cierres.
- `TICK_03` → Selección IA / ventana manual / asignación REAL.
- `TICK_04` → Refresh de saldo + render HUD.
- `STOP` → Salida controlada.

Además, el HUD muestra la etapa activa (`ETAPA ...`) con segundos transcurridos para localizar más rápido dónde se atora el ciclo.

## Qué hacer cuando la pantalla sigue en rojo (plan práctico)

Si ves varios bots con `Prob IA` alta pero `Real` bajo (inflación grande), no significa que el sistema esté “roto”; normalmente significa **descalibración + poca muestra útil**. Usa este orden:

1. **Congelar decisiones por score alto con poca muestra**
   - Si `n < 200` por bot/ciclo, evita usar umbrales agresivos (`>=70%`) para operar fuerte.
   - Mantén modo conservador hasta reunir muestra más estable.

2. **Entrenar más, pero con calidad y ventana temporal correcta**
   - Sí, conviene dejar entrenar más, pero en modo walk-forward.
   - Objetivo mínimo por bot: `n>=200` cierres útiles recientes (mejor `300-500`) antes de confiar en probas altas.

3. **Subir umbral de “CONFIABLE” mientras haya sobreestimación**
   - Si `Pred - Real > 15pp` de forma persistente, sube el filtro de entrada y reduce tamaño de apuesta.
   - Ejemplo: pasar temporalmente de `>=60%` a `>=72%` hasta que baje la inflación.

4. **Aplicar penalización por bot (bot-specific shrinkage)**
   - Ajuste recomendado: `p_final = p_calibrada - beta_bot`, con `beta_bot` ligado a la inflación rolling de ese bot.
   - Si un bot infla +25pp, no debe mostrar señal fuerte aunque el score bruto sea alto.

5. **Separar ranking de ejecución**
   - Usa el modelo para rankear candidatos.
   - Pero ejecuta solo los top con `EV positivo` y con intervalo de confianza aceptable.

6. **Control de régimen de mercado**
   - Etiqueta sesión/tendencia/volatilidad y evalúa métricas por régimen.
   - Si cambia el régimen, recalibra primero; después reentrena.

7. **Meta realista de mejora**
   - No busques que todo sea verde en horas.
   - Señal saludable: inflación bajando semana a semana y `Real` acercándose a `Pred` en cada bin de probabilidad.

### Regla rápida para decidir hoy
- Si la calibración está en crítico y hay pocos cierres por bot, **síguelo entrenando**, pero en modo prudente.
- Reduce exposición, sube umbral temporalmente y exige más muestra antes de declarar “IA confiable”.
- Prioriza estabilidad de calibración (Brier/ECE) sobre ganar “ticks” aislados.

## Script rápido para revisar las 13 variables + result_bin

Puedes validar calidad del CSV (NaN, duplicados, balance de clases y conflictos de etiqueta) con:

```bash
python analizar_13_variables.py registro_enriquecido_fulll49.csv
```

## Plan de cambio: priorizar calidad de información (ya no cantidad)

Sí, **se puede** pasar del enfoque de 13 variables al enfoque de calidad. Con la evidencia actual, el plan base es operar con un núcleo mínimo y exigir que toda variable adicional demuestre aporte real fuera de muestra.

### Objetivo operativo
- Reducir ruido y sobreajuste.
- Subir estabilidad temporal de la `Prob IA`.
- Permitir que solo entren al modelo variables con impacto medible en walk-forward.

### Cambios concretos a implementar

1. **Congelar baseline en variable núcleo**
   - Baseline inicial: `racha_actual` como señal principal.
   - Las otras 12 pasan a estado `shadow` (se calculan solo para análisis, no para decidir).

2. **Crear “Feature Gate” (puerta de calidad)**
   - Cada variable candidata deberá cumplir criterios mínimos para entrar al modelo:
     - Delta AUC walk-forward positivo y estable.
     - Mejora en Brier/ECE (no solo AUC).
     - Consistencia por segmentos (activo/hora/payout/volatilidad).
   - Si no cumple, queda fuera automáticamente.

3. **Separar variables de producción vs investigación**
   - `features_prod`: solo variables aprobadas por gate.
   - `features_shadow`: variables en evaluación.
   - El modelo de ejecución usa únicamente `features_prod`.

4. **Agregar control de drift por variable**
   - Monitorear PSI/KS por feature y alertar cuando cambie régimen.
   - Si una feature aprobada pierde estabilidad, vuelve a `shadow`.

5. **Ajustar reportes para decisiones de calidad**
   - Reporte semanal por feature con:
     - AUC/Brier/ECE walk-forward.
     - Delta contra baseline (`solo racha_actual`).
     - Recomendación automática: `aprobar`, `mantener en shadow`, `retirar`.

### Criterio de éxito
- Menor inflación (`Pred - Real`) en bins altos.
- Brier/ECE mejores o iguales con menos variables.
- Menor variabilidad de métricas entre folds temporales.

### Implementación por fases
- **Fase 1 (rápida):** ejecución solo con `racha_actual` + calibración estricta.
- **Fase 2:** incorporar feature gate y listas `prod/shadow`.
- **Fase 3:** reingreso progresivo de variables rediseñadas que sí pasen el gate.

### Regla de gobierno (simple)
Una variable nueva **no entra** por intuición; entra únicamente si gana al baseline en walk-forward y mejora calibración real.
