# Análisis incremental HUD IA (capturas de 18:17 a 18:32)

Fecha de lectura: 2026-02-23.
Fuente: 10 capturas de consola del modo DEMO (`Últimos 40`).

## Resumen ejecutivo

- El sistema está **activo y alimentándose**, pero sigue en **warmup real** (`n=100<250` y luego `n=122<250`), por lo que el maestro opera en modo conservador.
- El bloqueo dominante no es técnico: es de **compuerta lógica (`WHY-NO`)** por diseño (`warmup=sí`, `confirm_pending(0/2)`, `trigger_no`, `gate_consumed=no`).
- El umbral dinámico se ve más estricto en este tramo (`CAP≈85%`, `ROOF mode=B`, alternando `AUTO=BLOCK/AUTO=ADAPT`).
- Las probabilidades IA se mantienen en banda **compacta** (~45% a ~68.3%), sin señales >70% sostenidas y lejos de un nivel de entrada real alto.
- Se ve **desalineación** entre `% ÉXITO` y `Prob IA` en varios bots, consistente con calibración floja en warmup o cambio de régimen reciente.
- Hubo reentrenamiento con métricas débiles (`AUC=0.452`, `F1=0.353`), coherente con el estado `Confianza IA: BAJA`.

## Hallazgos detallados del tramo

### 1) Warmup real sigue mandando

- El HUD muestra warmup activo en toda la secuencia (`n=100→122`, objetivo de cierre `n=250`).
- Aunque `n_min_real` supera el umbral (de `17/15` a `20/15`), eso **no habilita entrada por sí solo**: siguen pesando warmup y confirmación.

**Lectura práctica:** el sistema está aprendiendo todavía; no está en fase de promoción/agresividad.

### 2) La compuerta (`WHY-NO`) es el freno principal

En múltiples capturas se repite:
- `warmup=sí`
- `confirm_pending(0/2)`
- `trigger_no`
- `gate_consumed=no`

**Traducción operativa:** no hay señal fuerte + confirmada para disparo. El motor no está trabado: está bloqueando deliberadamente.

### 3) CAP más alto = entrada más estricta

- En este bloque aparece `CAP≈85.0%` (frente a ventanas previas más laxas).
- Se mantiene `ROOF mode=B` con alternancia `AUTO=BLOCK/AUTO=ADAPT`.

**Lectura:** la lógica adaptativa está viva, pero hoy ajusta hacia prudencia, no hacia apertura.

### 4) Prob IA compacta, edge aún limitado

Rango observado:
- mínimo aproximado en la zona media de 40s,
- máximo puntual ~`68.3%` (`fulll46`),
- otro pico visible ~`63.4%` (`fulll50`).

**Lectura:** hay scoring activo, pero sin convicción estadística robusta para disparos de alta confianza.

### 5) Desalineación `% ÉXITO` vs `Prob IA`

Ejemplos visibles en la secuencia:
- `fulll45`: `% ÉXITO` en torno a ~63–66%, con `Prob IA` que en una toma cae cerca de ~39.9%.
- `fulll50`: `% ÉXITO` en torno a ~39–43%, con `Prob IA` que sube hasta ~63.4%.

**Interpretación probable (sin concluir bug):**
- mayor peso del estado/patrón reciente sobre histórico bruto,
- calibración inestable propia de warmup,
- posible deriva/régimen nuevo no estabilizado.

### 6) Reentrenamiento visible, pero sin fuerza estadística

Evento registrado:
- `IA reentrenada | AUC=0.452 F1=0.353 thr=0.50 calib=sigmoid`

Lectura técnica:
- `AUC<0.5` implica desempeño por debajo de azar en ese corte de test,
- en warmup puede ocurrir por muestra limitada o distribución de clases,
- valida el estado del HUD: `Confianza IA: BAJA`.

### 7) SENSOR_PLANO mejora leve

- En parte del tramo pasa de `6/6` a `5/6`.

**Lectura:** mini-señal positiva de que al menos un bot deja condición plana, aunque todavía insuficiente para habilitar entradas.

### 8) Ingesta incremental funcionando correctamente

En `Eventos recientes` aparece flujo constante:
- `Incremental: +1 fila desde fulll50/49/45/46/47...`

**Lectura:**
- bots alimentan,
- maestro lee,
- no hay cuello de botella de ingestión,
- el bloqueo real es de calidad/confirmación de señal.

## Diagnóstico franco del estado actual

El sistema está en modo:

> **“Estoy vivo, estoy aprendiendo, todavía no tengo permiso estadístico para disparar.”**

No se observa atasco técnico en estas capturas. Lo que se observa es:
- warmup vigente,
- compuerta estricta,
- calibración todavía inestable.

## Recomendaciones inmediatas (solo operativas, sin tocar código)

- Mantener monitoreo hasta cerrar warmup (`n>=250`) antes de evaluar promoción real.
- Seguir comparando `Prob IA` vs `% ÉXITO` por bot para confirmar si la desalineación converge o empeora tras más muestra.
- Tratar `AUC<0.5` durante warmup como señal de cautela (no de fallo fatal), y exigir estabilización antes de abrir compuerta.
- Vigilar `SENSOR_PLANO` como indicador temprano: sostener `5/6` o mejor sería señal de régimen más útil.

## Conclusión

En este tramo, el pipeline se ve sano en operación (ingesta y ciclo vivos), pero conservador por diseño. El freno no es técnico: es estadístico/operativo (**warmup + confirmación + compuerta estricta**), con calibración aún débil para confiar en disparos de alta probabilidad.
