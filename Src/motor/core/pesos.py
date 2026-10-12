"""
Ponderación de forma reciente.

Implementa ponderación exponencial: los partidos más recientes pesan más,
de forma continua (sin saltos arbitrarios tipo "los últimos 3 valen doble").
"""
from typing import List
import math

# Decaimiento por partido: alpha=0.90 conserva el 90% del peso relativo
# por cada jornada de antigüedad (vida media aproximada: 6.6 partidos).
# Es un default estable; debe recalibrarse con validación temporal por liga.
DEFAULT_ALPHA = 0.90


def pesos_exponenciales(n: int, alpha: float = DEFAULT_ALPHA) -> List[float]:
    """
    Genera pesos normalizados para n partidos ordenados del más antiguo (i=1)
    al más reciente (i=n).

    peso_i = alpha^(n - i), luego normalizado para sumar 1.
    """
    if not math.isfinite(alpha) or not (0.0 < alpha <= 1.0):
        raise ValueError("alpha debe ser finito y estar en el rango (0, 1]")
    if n <= 0:
        return []
    crudos = [alpha ** (n - i) for i in range(1, n + 1)]
    total = sum(crudos)
    return [p / total for p in crudos]


def promedio_ponderado(valores: List[float], alpha: float = DEFAULT_ALPHA) -> float:
    """
    Calcula el promedio ponderado exponencialmente de una lista de valores,
    donde valores[0] es el partido más antiguo y valores[-1] el más reciente.
    """
    n = len(valores)
    if n == 0:
        return 0.0
    if any(not math.isfinite(float(v)) for v in valores):
        raise ValueError("Los valores ponderados deben ser numéricos y finitos")
    pesos = pesos_exponenciales(n, alpha)
    return sum(p * v for p, v in zip(pesos, valores))
