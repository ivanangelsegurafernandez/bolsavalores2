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
