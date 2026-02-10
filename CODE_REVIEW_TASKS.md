# Revisión rápida y propuesta de tareas

## Hallazgos

1. **Error tipográfico**: hay textos con repetición y acentos mal escritos, por ejemplo `si se se habilita` y `ia_seniales`.
2. **Fallo funcional potencial**: la martingala del maestro (5 pasos) no coincide con la de los bots (6 pasos), lo que puede desalinear órdenes por ciclo.
3. **Discrepancia de documentación**: el README no documenta ejecución, dependencias ni arquitectura pese a que el proyecto contiene varios scripts de trading coordinados.
4. **Calidad de pruebas insuficiente**: no hay carpeta `tests/` ni pruebas automáticas para funciones críticas (normalización CSV, handshake de orden REAL, martingala).

## Tareas propuestas

### Tarea 1 — Corregir error tipográfico
- Corregir el comentario `si se se habilita` por `si se habilita`.
- Estandarizar `seniales` a `senales` (o `señales` en comentarios) para mantener consistencia semántica.
- Aplicar una pasada de lint ortográfico en comentarios/labels del HUD para evitar regresiones de texto.

### Tarea 2 — Solucionar fallo de desalineación de martingala
- Definir una única fuente de verdad para escalado de martingala (`[1,2,4,8,16]` o `[1,2,4,8,16,32]`).
- Unificar maestro y bots para que `MAX_CICLOS` y los índices de ciclo usen la misma longitud.
- Añadir validación al iniciar: abortar con mensaje claro si se detecta configuración inconsistente entre procesos.

### Tarea 3 — Corregir discrepancia de documentación
- Ampliar `README.md` con:
  - propósito de cada script,
  - flujo maestro↔bots,
  - requisitos e instalación,
  - variables/archivos compartidos (`token_actual.txt`, `orden_real/`, CSV).
- Incluir una sección de “riesgos operativos” y modo simulación para pruebas seguras.

### Tarea 4 — Mejorar una prueba
- Crear tests unitarios para:
  - normalización de filas CSV (padding/truncado),
  - deduplicación de última fila,
  - parseo de orden REAL con TTL/ciclo válidos.
- Añadir un test de contrato que falle si la longitud de martingala difiere entre maestro y bots.
- Ejecutar tests en CI (por ejemplo `pytest`) para bloquear merges con regresiones.
