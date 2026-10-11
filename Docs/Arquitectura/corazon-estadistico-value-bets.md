# Corazón Estadístico del Sistema de Value Bets
## Documento técnico de diseño matemático — Fase 1

---

## 0. Corrección al pipeline propuesto

Tu diagrama es correcto en el orden general, pero le falta un paso. Lo dejo así:

```
Datos históricos
    ↓
Limpieza de datos
    ↓
Ponderación de forma reciente
    ↓
Normalización contra el contexto de la muestra   ← PASO AÑADIDO
    ↓
Fuerza ofensiva y defensiva
    ↓
Estimación de goles esperados (λ)
    ↓
Modelo de Poisson
    ↓
Probabilidades reales
    ↓
Comparación con cuotas
    ↓
Value Bet
    ↓
Registro del resultado
    ↓
Evaluación del rendimiento del modelo
    ↓
Recalibración de parámetros (α, umbrales de confianza)   ← PASO AÑADIDO
```

**Por qué falta ese paso**: sin normalizar contra un promedio de referencia (el promedio de goles de la propia muestra, ya que no tenemos liga completa en Fase 1), la "fuerza ofensiva" de un equipo es solo su promedio de goles anotados — un número absoluto sin contexto. Esto funciona, pero es más débil que expresar la fuerza como un **factor relativo** (ver sección 1). El segundo paso añadido, la recalibración, es lo que convierte esto de "un cálculo que se hace una vez" a "un sistema que aprende de sus propios aciertos y errores" — coherente con que ya vas a registrar histórico en CSV.

---

## 1. Fuerza ofensiva y defensiva

### 1.1 El error de usar promedios absolutos

Si defines "fuerza ofensiva del Barcelona como local" = promedio de goles anotados como local, tienes un número en el vacío. 2.1 goles/partido, ¿es mucho o poco? Depende del promedio general de goles en esa muestra combinada.

### 1.2 Método recomendado: fuerza relativa (estilo Dixon-Coles simplificado)

Se calcula un promedio de referencia con los datos que sí tenemos disponibles — el conjunto de partidos del local + del visitante que el usuario ingresó (en Fase 1, sin datos de liga, esta es la mejor referencia disponible):

```
PromedioGoles_Referencia = (Σ goles anotados en todos los partidos ingresados) / (número total de partidos)
```

Con esa referencia:

```
FuerzaAtaque_Local = PromedioGolesAnotados_Local / PromedioGoles_Referencia
FuerzaDefensa_Local = PromedioGolesRecibidos_Local / PromedioGoles_Referencia

FuerzaAtaque_Visitante = PromedioGolesAnotados_Visitante / PromedioGoles_Referencia
FuerzaDefensa_Visitante = PromedioGolesRecibidos_Visitante / PromedioGoles_Referencia
```

**Interpretación**:
- `FuerzaAtaque = 1.3` → el equipo anota 30% más de lo esperado en ese contexto.
- `FuerzaDefensa = 0.7` → el equipo recibe 30% menos de lo esperado (buena defensa).
- `FuerzaDefensa = 1.4` → recibe 40% más de lo esperado (defensa débil).

Este es exactamente el principio del modelo de Maher (1982), que es la base estadística de la mayoría de los modelos de fútbol comerciales, incluidos los que usan las casas de apuestas para fijar sus líneas iniciales. Adaptado a que no tenemos liga completa, usamos la propia muestra ingresada como referencia — es una simplificación honesta, no un error, siempre que quede documentada en el código.

### 1.3 Por qué esto es mejor que un promedio simple

Con promedios simples, "Ataque Local + Defensa Visitante" mezcla unidades sin normalizar — puede sobreestimar o subestimar sistemáticamente según qué tan ofensiva sea la muestra completa. Con fuerzas relativas, multiplicar factores (en vez de sumar promedios) respeta la naturaleza multiplicativa del problema: un equipo con ataque 1.3 contra una defensa 1.4 debería anotar más que ambos factores por separado sugieren de forma aditiva.

---

## 2. Ponderación de forma reciente

### 2.1 Comparación de métodos

**Método A — Pesos lineales** (ej. partido 1 vale 1, partido 2 vale 2... partido n vale n)

```
peso_i = i / Σ(1..n)
```

Ventaja: simple, fácil de explicar.
Debilidad: el diferencial de peso entre el partido más antiguo y el más reciente es limitado y fijo — no se puede ajustar la "agresividad" de la ponderación sin cambiar la fórmula completa.

**Método B — Pesos en bloques** (ej. "últimos 3 valen doble")

Debilidad: arbitrario, genera saltos discontinuos sin justificación estadística. No lo recomiendo.

**Método C — Ponderación exponencial (recomendado)**

```
peso_i = α^(n - i)       donde i = 1 es el partido más antiguo, i = n el más reciente
```

Luego se normaliza:

```
peso_normalizado_i = peso_i / Σ(peso_1...peso_n)
```

Y el promedio ponderado de goles:

```
PromedioPonderado = Σ(peso_normalizado_i × goles_i)
```

**Por qué es superior**:
- El parámetro `α` (típicamente entre 0.85 y 0.95) controla de forma continua qué tanto "importa" la racha reciente vs. el histórico completo, sin cambiar la estructura de la fórmula.
- Es el mismo principio matemático detrás de un EWMA (Exponentially Weighted Moving Average), ampliamente usado en series de tiempo financieras y deportivas por su estabilidad.
- No genera discontinuidades: la diferencia de peso entre partido 5 y partido 6 es proporcional, no un salto arbitrario.

**Recomendación concreta**: usar `α = 0.90` como valor por defecto, configurable por el usuario. Con `α = 0.90` y `n = 10`, el partido más reciente pesa aproximadamente 3.5 veces más que el más antiguo — una ponderación notable pero no extrema.

### 2.2 Advertencia importante

Con muestras pequeñas (5 partidos), la ponderación exponencial tiene menos margen para actuar — con pocos datos, cualquier método de ponderación se acerca al promedio simple. Esto no es un defecto del método, es una limitación estructural de tener pocos datos, y debe comunicarse en la interfaz (ej. "confianza baja: muestra pequeña").

---

## 3. Estimación de goles esperados sin datos oficiales de xG

### 3.1 Qué es y qué no es esto

Lo que vamos a calcular **no es xG real** (que se basa en calidad de cada disparo: ángulo, distancia, tipo de jugada). Es una **aproximación basada en resultados históricos**, a veces llamada "goles esperados estadísticos" o "pseudo-xG". Es importante ser honesto sobre esta distinción en la interfaz para no generar falsas expectativas de precisión.

### 3.2 Fórmula

Con las fuerzas relativas de la sección 1:

```
λ_local = FuerzaAtaque_Local × FuerzaDefensa_Visitante × PromedioGoles_Referencia
λ_visitante = FuerzaAtaque_Visitante × FuerzaDefensa_Local × PromedioGoles_Referencia
```

Esto reemplaza tu propuesta original de "promediar Ataque Local + Defensa Visitante" por una versión **multiplicativa normalizada**, que es la estándar en la literatura (Maher 1982; Dixon-Coles 1997). Es más precisa porque:

- Evita que un ataque fuerte y una defensa débil "se cancelen" de forma aditiva ingenua.
- Mantiene todas las magnitudes ancladas al mismo promedio de referencia, evitando que unidades distintas se mezclen sin sentido.

### 3.3 Ejemplo numérico

Si `PromedioGoles_Referencia = 1.4`, `FuerzaAtaque_Local = 1.3`, `FuerzaDefensa_Visitante = 1.2`:

```
λ_local = 1.3 × 1.2 × 1.4 = 2.184
```

Es decir, se espera que el local anote ≈2.18 goles en este partido específico, dado su ataque y la debilidad defensiva del rival, en el contexto de esta muestra.

---

## 4. Distribución de Poisson aplicada a los mercados

### 4.1 Por qué Poisson

La distribución de Poisson modela el número de eventos discretos (goles) en un intervalo fijo (90 minutos) cuando estos ocurren de forma independiente y a una tasa promedio conocida (λ). Es el modelo estándar en fútbol desde Maher (1982) precisamente porque los goles cumplen razonablemente estas condiciones.

### 4.2 Fórmula general

```
P(X = k) = (e^−λ × λ^k) / k!
```

Donde `X` es el número de goles y `λ` la tasa esperada.

### 4.3 Goles totales del partido

Asumiendo independencia entre los goles del local y del visitante (simplificación estándar en Fase 1):

```
λ_total = λ_local + λ_visitante
```

Los goles totales del partido también siguen una distribución Poisson con parámetro `λ_total` (propiedad de aditividad de Poisson).

### 4.4 Over/Under

```
P(Under k.5) = P(X ≤ k) = Σ[j=0 hasta k] (e^−λ_total × λ_total^j) / j!
P(Over k.5) = 1 − P(Under k.5)
```

Aplicado a cada línea:

| Mercado | Fórmula |
|---|---|
| Over/Under 0.5 | usa `k=0` |
| Over/Under 1.5 | usa `k=1` |
| Over/Under 2.5 | usa `k=2` |
| Over/Under 3.5 | usa `k=3` |

### 4.5 BTTS (Ambos Marcan)

Asumiendo independencia entre goles del local y del visitante:

```
P(Local anota ≥1) = 1 − e^−λ_local
P(Visitante anota ≥1) = 1 − e^−λ_visitante

BTTS_Sí = P(Local anota ≥1) × P(Visitante anota ≥1)
BTTS_No = 1 − BTTS_Sí
```

---

## 5. Limitaciones del modelo de Poisson y mejoras futuras

### 5.1 Limitaciones conocidas

1. **Subestimación de marcadores bajos correlacionados**: Poisson puro asume independencia total entre goles del local y visitante, pero empíricamente los resultados 0-0, 1-0, 0-1 y 1-1 ocurren con más frecuencia de la que Poisson puro predice. Esto es un hallazgo documentado desde Dixon & Coles (1997).

2. **No captura "clima de partido"**: tarjetas rojas, lesiones, motivación (ej. un equipo ya clasificado que rota jugadores) no están en el modelo.

3. **Estacionariedad asumida**: el modelo asume que la fuerza de un equipo es constante en la ventana de partidos analizada, cuando en realidad las plantillas cambian (fichajes, lesiones de jugadores clave).

4. **Sensibilidad a muestras pequeñas**: con 5-10 partidos, los intervalos de confianza de λ son amplios, aunque el punto estimado sea el mismo.

### 5.2 Mejoras futuras (en orden de complejidad creciente)

- **Dixon-Coles (1997)**: introduce un parámetro de corrección `τ(x,y)` que ajusta específicamente las probabilidades de los marcadores 0-0, 1-0, 0-1 y 1-1, corrigiendo la limitación #1. Es la mejora más directa y de mayor retorno para este proyecto.
- **Bivariate Poisson**: modela explícitamente una covarianza entre goles del local y visitante (en vez de asumir independencia total), captando efectos como el ritmo de partido compartido.
- **Rating tipo Elo**: en vez de solo promedios de la muestra ingresada, mantener una fuerza acumulada de cada equipo a lo largo del tiempo, actualizada partido a partido.
- **xG real vía API**: si en el futuro se integran datos oficiales de xG (Understat, Opta, etc.), reemplazar el pseudo-xG de la sección 3 por datos reales mejora sustancialmente la precisión, porque xG real captura calidad de ocasión y no solo resultado final.

---

## 6. De probabilidades a Value Bet

### 6.1 Probabilidad implícita de la cuota

```
Prob_implícita (%) = (1 / cuota) × 100
```

### 6.2 Comparación

```
Value (%) = Prob_sistema (%) − Prob_implícita (%)
```

- `Value > 0` → el sistema considera el evento más probable de lo que la cuota sugiere → **hay Value Bet**.
- `Value ≤ 0` → no hay value, la cuota ya está "cara" o justa respecto al modelo.

### 6.3 Advertencia necesaria (overround)

Las cuotas reales de una casa de apuestas casi nunca sirven una probabilidad "justa" — incluyen un margen (overround) que hace que la suma de probabilidades implícitas de todos los resultados posibles supere el 100%. Esto significa que un value bet detectado por el sistema puede en realidad estar compensando ese margen, no reflejando un verdadero error de la casa. Vale la pena, si el usuario tiene las cuotas de ambos lados de un mercado (ej. Over 2.5 y Under 2.5), calcular el overround explícito:

```
Overround = Prob_implícita(Over) + Prob_implícita(Under) − 100
```

Esto da una medida directa de cuánto margen tiene la casa en ese mercado específico, y ayuda a calibrar qué tan exigente debe ser el umbral de "value" para considerarlo confiable (ej. exigir Value > overround/2 antes de marcarlo como oportunidad real).

### 6.4 Nivel de confianza (versión objetiva, no subjetiva)

Comparando la probabilidad Poisson contra la empírica (histórica) del mismo mercado:

| Condición | Confianza |
|---|---|
| Diferencia ≤ 10 puntos y ≥10 partidos en la muestra | Alta |
| Diferencia 10-20 puntos, o 5-9 partidos | Media |
| Diferencia > 20 puntos, o <5 partidos | Baja |

---

## 7. Métricas de rendimiento del modelo

### 7.1 Métricas financieras (las que ya tenías)

```
ROI (%) = (Ganancia_total / Total_apostado) × 100
Yield (%) = mismo cálculo que ROI (son sinónimos en apuestas deportivas)
```

### 7.2 Métricas de precisión probabilística (recomendación de añadir)

Estas son más importantes que el ROI para saber si el **modelo en sí** es bueno, independientemente de si tuviste suerte en una racha:

**Brier Score** — mide qué tan calibradas están tus probabilidades:

```
Brier = (1/N) × Σ(prob_estimada_i − resultado_real_i)²
```

Donde `resultado_real_i` es 1 si el mercado se cumplió, 0 si no. Un Brier Score cercano a 0 indica buena calibración; cercano a 1, mala.

**Log Loss** — penaliza con más fuerza los errores confiados:

```
LogLoss = −(1/N) × Σ[y_i × log(p_i) + (1−y_i) × log(1−p_i)]
```

**Por qué recomiendo priorizar Brier Score sobre Log Loss para este proyecto**: Log Loss castiga desproporcionadamente los casos donde el modelo dijo "99% seguro" y falló — útil en producción de alto volumen, pero con pocas apuestas (decenas, no miles) puede distorsionar la evaluación por un solo caso extremo. Brier Score es más estable con muestras pequeñas y es igual de interpretable para un proyecto académico.

### 7.3 Tabla de métricas recomendadas para el sistema

| Métrica | Qué mide | Prioridad |
|---|---|---|
| % Acierto | Aciertos / Total apuestas | Básica, ya la tenías |
| ROI / Yield | Rentabilidad real | Básica, ya la tenías |
| Racha ganadora/perdedora | Rachas consecutivas | Básica, ya la tenías |
| **Brier Score** | Calibración del modelo | **Añadir — clave para validar si el modelo "sabe lo que dice"** |
| Mejor/peor mercado | Qué mercado rinde más | Ya la tenías |
| Log Loss | Penalización de errores confiados | Opcional, útil cuando haya más volumen de datos |

---

## Resumen de las correcciones respecto a tu planteamiento original

1. Fuerza ofensiva/defensiva: de promedios absolutos → **factores relativos normalizados** (sección 1).
2. Goles esperados: de suma/promedio aditivo → **producto de factores relativos** (sección 3), consistente con Maher (1982).
3. Ponderación: de "últimos partidos valen más" (vago) → **ponderación exponencial con α configurable** (sección 2).
4. Probabilidades: de conteo de frecuencia histórica → **Poisson como cálculo principal**, frecuencia histórica como contraste de confianza.
5. Value Bet: se añade el concepto de **overround** para no sobreestimar oportunidades reales.
6. Métricas: se añade **Brier Score** como métrica de calibración, complementaria al ROI.
7. Pipeline: se añaden los pasos de **normalización contra referencia** y **recalibración**.

---

¿Seguimos con el diseño de la interfaz en Streamlit, o prefieres que primero valide contigo algún ejemplo numérico completo (con datos reales de dos equipos) para confirmar que la lógica da resultados que tienen sentido antes de pasar a la interfaz?
