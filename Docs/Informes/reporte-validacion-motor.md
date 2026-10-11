# Reporte de Validación del Motor Estadístico

## Metodología

Como no tengo acceso a una API de datos históricos de fútbol reales (mi entorno
solo permite conexión a repositorios de paquetes, y la herramienta de datos
deportivos disponible solo devuelve partidos **programados**, no resultados
históricos en volumen), la validación se hizo por **simulación controlada**:

1. Se genera un universo de 30 equipos con fuerzas de ataque/defensa
   **verdaderas** (conocidas por el simulador, ocultas para el motor).
2. Para cada uno de 500 partidos de prueba, se genera un histórico de 15
   partidos previos para el local (solo como local) y el visitante (solo
   como visitante) — igual que un usuario real ingresándolos a mano.
3. El motor **solo ve esos históricos**, nunca las fuerzas verdaderas.
4. Se simula el resultado real del partido de prueba y se compara contra
   la predicción del motor.

Esto es lo más riguroso que se puede hacer sin datos reales: como conocemos
la "verdad" del proceso generador, podemos medir objetivamente si el motor
la recupera correctamente a partir de la información limitada que tendría
un usuario real.

## Resultado 1 — el motor funciona, pero su ventaja depende de la dispersión de calidad de la liga

Primer hallazgo importante (no es un error, es un descubrimiento honesto):
con una liga muy pareja (poca diferencia de calidad entre equipos), el motor
apenas mejora sobre un modelo ingenuo que simplemente predice la frecuencia
histórica general. Esto **no es un fallo del código** — lo confirmamos
probando un modelo "oráculo" que usa las fuerzas verdaderas exactas (sin
estimación): incluso con información perfecta, la mejora máxima posible en
una liga pareja es de solo ~10%. Es decir, hay un techo teórico impuesto por
cuánto varía la calidad real entre equipos, no por el modelo.

Con una dispersión de calidad más realista (equivalente a una liga real con
diferencia clara entre grandes y chicos), el motor sí mejora consistentemente:

| Mercado | Brier Motor | Brier Ingenuo | Mejora |
|---|---|---|---|
| Over 0.5 | 0.0832 | 0.0835 | +0.5% |
| Over 1.5 | 0.1933 | 0.2092 | +7.6% |
| Over 2.5 | 0.2276 | 0.2499 | +8.9% |
| Over 3.5 | 0.2010 | 0.2224 | +9.6% |
| BTTS Sí | 0.2381 | 0.2500 | +4.7% |

(500 partidos simulados, histórico de 15 partidos por rol, α=0.90)

**Conclusión práctica**: el motor aporta más valor cuando hay diferencias
reales de calidad entre los equipos que estás analizando (lo cual es el caso
típico en fútbol real — hay grandes y chicos). En enfrentamientos entre
equipos de nivel muy similar, no esperes que el modelo sea mucho mejor que
la frecuencia histórica simple, y eso es matemáticamente esperable, no un
defecto a corregir.

## Resultado 2 — calibración por bins

Cuando el motor dice "60% de probabilidad", ¿realmente ocurre ~60% de las
veces? Tabla de calibración para Over 2.5:

| Rango predicho | n | Predicho (prom.) | Real |
|---|---|---|---|
| 0-30% | 52 | 21.3% | 21.2% |
| 30-50% | 135 | 40.9% | 40.7% |
| 50-70% | 178 | 60.5% | 55.1% |
| 70-100% | 135 | 80.6% | 67.4% |

La calibración es buena en probabilidades bajas y medias. En el rango alto
(70-100%) el motor tiende a **sobreestimar ligeramente** la confianza —
algo esperable con muestras de 15 partidos, y coherente con la limitación
de Poisson puro descrita en el documento técnico anterior (no corrige la
correlación en marcadores extremos). Esto es una señal real a tener en
cuenta para la interfaz: mostrar un "nivel de confianza" más conservador
cuando la probabilidad estimada supera 70%.

## Resultado 3 — sensibilidad al tamaño de muestra histórica

| Partidos históricos | Mejora sobre modelo ingenuo (Over 2.5) |
|---|---|
| 10 | -4.6% (peor que el ingenuo) |
| 20 | -0.2% (empate) |
| Con dispersión realista de liga y 20 partidos | +6.8% a +14.5% |

Esto confirma cuantitativamente lo que el documento técnico anterior advertía
cualitativamente: con menos de ~15-20 partidos históricos, el motor tiene
poco margen para superar una simple frecuencia histórica. Recomendación
práctica para la interfaz: advertir al usuario cuando ingrese menos de 10
partidos, y considerar 15-20 como el rango ideal por defecto.

## Qué NO pudimos validar todavía

No se validó contra partidos reales jugados, porque no tengo una fuente de
datos históricos reales conectada (solo partidos programados a futuro). Si
tienes un CSV con resultados reales (puede ser de cualquier liga, cuantos
más partidos mejor), el mismo script de backtesting (`validar_motor.py`)
se puede adaptar para leerlo directamente y correr la misma validación con
datos 100% reales — sería el siguiente paso ideal antes de construir la
interfaz.

## Archivos del motor (listos para revisión)

```
motor/
├── core/
│   ├── pesos.py              # ponderación exponencial
│   ├── fuerzas.py             # fuerza ofensiva/defensiva relativa
│   ├── expected_goals.py      # cálculo de lambda (goles esperados)
│   ├── poisson_model.py       # probabilidades Over/Under y BTTS
│   ├── value_bet.py           # probabilidad implícita, value, overround
│   └── metrics.py             # Brier Score, Log Loss, hit rate
├── tests/
│   └── simulador.py           # generador de datos sintéticos con verdad conocida
└── validar_motor.py           # script de validación (500 partidos)
```
