# Reporte Final de Calibración del Motor

Responde directamente a las 4 preguntas planteadas: mejor α, mejor tamaño de
histórico, mejor versión del modelo, y límites conocidos.

---

## 1. ¿Cuál es el mejor valor de α?

**Respuesta corta: depende de qué tan volátil sea la forma real de los
equipos que analices — y eso no lo podemos saber sin datos reales.**

Este fue el hallazgo más importante del proceso. El primer barrido (con
fuerzas de equipo constantes en el tiempo) dio como "mejor" α=0.95-1.0,
es decir, casi sin ponderación por recencia. Esto era sospechoso: contradice
la intuición de que la forma reciente importa.

Diagnóstico: si en la simulación la fuerza real de un equipo **no cambia**
de un partido a otro, no hay ninguna "forma reciente" que capturar —
ponderar más los partidos recientes solo añade ruido. Un promedio simple
(α≈1.0) es la estimación óptima en ese escenario.

Para probar correctamente la utilidad de α<1, construí un segundo simulador
donde la fuerza de los equipos **sí cambia con el tiempo** (random walk:
lesiones, fichajes, rachas). Resultado:

| Volatilidad de forma | Mejor α encontrado |
|---|---|
| Baja (equipos estables) | 0.95 |
| Media | 0.85 |
| Alta (forma muy cambiante) | 0.75 |

**Conclusión práctica**: no existe un α universalmente óptimo — depende de
cuánto cambia realmente la forma de los equipos en la liga que se analice.
Recomendación para la Fase 1: usar **α = 0.90 como valor por defecto**
(punto medio razonable), pero exponerlo como parámetro configurable en la
interfaz, y — más importante — cuando tengamos datos reales, medir la
volatilidad real de forma y ajustar α en consecuencia. Esto ya no es una
suposición de diseño, es algo que se puede calcular empíricamente después.

## 2. ¿Cuántos partidos históricos conviene usar?

De la validación anterior y este barrido: **la mejora es monótona con más
partidos** (5 → 30 partidos siempre mejoró el Brier Score en el escenario
de fuerzas estables). Pero:

- Con **menos de 10 partidos**, el motor casi no le gana a una frecuencia
  histórica simple (confirmado en el reporte anterior).
- Con **15-20 partidos**, ya hay una mejora clara y estable.
- Con **25-30 partidos**, sigue mejorando, pero con retornos decrecientes.

**Recomendación**: 15-20 partidos como default, permitir hasta 30 si el
usuario los tiene disponibles, advertir explícitamente cuando el usuario
ingrese menos de 10 (el mercado que se calcule ahí tendrá "confianza baja"
por diseño, no por falla del sistema).

## 3. ¿Qué versión del modelo predice mejor: Poisson puro o Dixon-Coles?

**Resultado honesto: en esta validación, Dixon-Coles no mostró una mejora
significativa** (entre -0.3% y +0.7% en Brier Score), incluso probando
específicamente en el subconjunto de partidos de bajo scoring donde la
corrección debería importar más.

Esto no significa que Dixon-Coles esté mal fundamentado teóricamente (la
literatura académica sí documenta el efecto en datos reales). Significa que:

- El tamaño del efecto en datos reales de fútbol es **pequeño en términos
  absolutos** — típicamente corrige unos pocos puntos porcentuales en un
  puñado de celdas de marcador específicas (0-0, 1-0, 0-1, 1-1), y solo
  se refleja en el Brier Score agregado cuando el λ del partido es bajo.
- El parámetro ρ (rho) normalmente se estima con **cientos o miles de
  partidos de liga** vía máxima verosimilitud, no con la muestra de 15-20
  partidos por equipo que tenemos en Fase 1. Usar un ρ "de la literatura"
  sin ajustar a los datos reales del proyecto (como hicimos aquí) diluye
  el beneficio.

**Recomendación para Fase 1: no implementar Dixon-Coles todavía.** El
Poisson independiente es suficientemente bueno dado el tamaño de muestra
disponible, y añadir Dixon-Coles con un ρ no calibrado agrega complejidad
sin beneficio medible. Dejarlo documentado como mejora de Fase 2, cuando
haya volumen de datos reales suficiente para estimar ρ correctamente.

## 4. Límites del modelo (resumen consolidado)

| Límite | Evidencia |
|---|---|
| Con equipos de calidad muy pareja, el modelo apenas mejora sobre la frecuencia histórica | Confirmado con modelo "oráculo": techo de mejora ~10% incluso con fuerzas verdaderas conocidas |
| Con menos de 10 partidos históricos, casi no hay ventaja sobre un modelo ingenuo | Confirmado en barrido N_HISTORICO=5-10 |
| El valor óptimo de α depende de la volatilidad real de forma, que no conocemos de antemano | Confirmado con simulador de deriva temporal |
| Dixon-Coles no aporta mejora medible sin calibrar ρ con datos reales de volumen | Confirmado probando específicamente en partidos de bajo scoring |
| El modelo asume independencia entre ataque/defensa de los dos equipos y estacionariedad de fuerza en la ventana analizada | Supuesto de diseño, no probado aquí |
| Calibración se degrada levemente en probabilidades altas (>70%) | Confirmado en el reporte de validación anterior (bins de calibración) |

## Recomendación final para pasar a la siguiente fase

El motor está en un punto sólido para justificar avanzar:

- **α = 0.90 por defecto**, configurable.
- **15-20 partidos históricos** como rango recomendado, advertencia si <10.
- **Poisson puro** (no Dixon-Coles todavía) como versión de producción de Fase 1.
- Documentar los límites de la tabla anterior directamente en la interfaz
  (ej. mostrar "confianza baja" cuando la muestra es pequeña o los equipos
  son de nivel muy similar).

El siguiente paso ideal sigue siendo conseguir un CSV con partidos reales
(100+) para repetir esta misma batería de pruebas con datos de verdad, en
vez de solo simulación. Si no es posible ahora, la interfaz en Streamlit ya
puede construirse con confianza sobre esta base, dejando estos parámetros
como configurables para recalibrarlos después con datos reales.

## Archivos añadidos en esta fase

```
motor/
├── core/
│   └── dixon_coles.py           # corrección Dixon-Coles (implementada, no activada por defecto)
├── tests/
│   └── simulador_dinamico.py    # simulador con deriva temporal de forma
├── barrido_validacion.py         # barrido alpha x histórico + comparación Dixon-Coles
└── barrido_alpha_dinamico.py     # barrido alpha con volatilidad de forma real
```
