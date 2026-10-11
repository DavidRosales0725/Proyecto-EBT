# CHANGELOG

Todos los cambios importantes del proyecto EBT se documentan en este archivo.

---

# [Fase 2] - 2026-07-08

## ✅ Data Loader (COMPLETADO)

### Implementado

- Creación del módulo `data_loader.py`.
- Detección automática de columnas mediante sinónimos.
- Soporte para mapeo personalizado de columnas.
- Limpieza automática de datos.
- Eliminación de registros inválidos.
- Registro detallado de los motivos de descarte.
- Ordenamiento cronológico garantizado.
- Compatibilidad con archivos CSV de football-data.co.uk.

### Validación

- Probado con CSV sintéticos.
- Probado con CSVs corruptos.
- Probado con la temporada real Premier League 2024/25.
- Comparado con la temporada 2023/24 para verificar consistencia.
- Auditoría independiente realizada con pandas.
- Resultado: **380 partidos cargados correctamente, 0 descartados.**

---

## ✅ Históricos (COMPLETADO)

### Arquitectura

- Creación del módulo `historicos.py`.
- Diseño revisado y aprobado antes de implementar.
- Protección contra Data Leakage mediante filtro temporal estricto.
- Índices por equipo y liga.
- Históricos separados por rol (Local / Visitante).
- Incorporación del indicador `confianza_historico`.
- Nuevo módulo compartido `tipos_comunes.py`.

### Compatibilidad

- Refactor de `ResultadoPartido` sin romper la Fase 1.
- Compatibilidad total verificada mediante pruebas de regresión.

### Pruebas

- 14 pruebas unitarias implementadas.
- Validación de propiedad anti-Data Leakage.
- Comparación contra algoritmo de fuerza bruta.
- Validación utilizando datos reales de Premier League.

### Hallazgos

- Se confirmó que utilizando una única temporada y
  `cruzar_temporadas=False`, únicamente el 46.3% de los partidos
  dispone de histórico suficiente en ambos equipos.

- Se concluyó que el backtesting definitivo deberá utilizar múltiples
  temporadas para obtener una muestra estadísticamente robusta.

---

## Estado del proyecto

### Fase 1

✅ Finalizada

### Fase 2

- ✅ Arquitectura
- ✅ Data Loader
- ✅ Históricos
- ⏳ Backtesting
- ⏳ Validación completa

---

Próximo objetivo:

➡ Implementar el módulo de Backtesting.