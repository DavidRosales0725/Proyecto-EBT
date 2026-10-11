"""
Ponderación de forma reciente.

Implementa ponderación exponencial: los partidos más recientes pesan más,
de forma continua (sin saltos arbitrarios tipo "los últimos 3 valen doble").
"""
from typing import List


def pesos_exponenciales(n: int, alpha: float = 0.90) -> List[float]:
    """
    Genera pesos normalizados para n partidos ordenados del más antiguo (i=1)
    al más reciente (i=n).

    peso_i = alpha^(n - i), luego normalizado para sumar 1.
    """
    if n <= 0:
        return []
    crudos = [alpha ** (n - i) for i in range(1, n + 1)]
    total = sum(crudos)
    return [p / total for p in crudos]


def promedio_ponderado(valores: List[float], alpha: float = 0.90) -> float:
    """
    Calcula el promedio ponderado exponencialmente de una lista de valores,
    donde valores[0] es el partido más antiguo y valores[-1] el más reciente.
    """
    n = len(valores)
    if n == 0:
        return 0.0
    pesos = pesos_exponenciales(n, alpha)
    return sum(p * v for p, v in zip(pesos, valores))
