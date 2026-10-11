"""
Métricas de evaluación del modelo: Brier Score, Log Loss, hit rate.
"""
import math
from typing import List


def brier_score(probs: List[float], resultados: List[int]) -> float:
    """
    probs: probabilidades estimadas (0-1) de que el evento ocurra.
    resultados: 1 si ocurrió, 0 si no.
    Menor es mejor. 0 = perfecto, 1 = pésimo.
    """
    n = len(probs)
    if n == 0:
        return float("nan")
    return sum((p - r) ** 2 for p, r in zip(probs, resultados)) / n


def log_loss(probs: List[float], resultados: List[int], eps: float = 1e-15) -> float:
    n = len(probs)
    if n == 0:
        return float("nan")
    total = 0.0
    for p, r in zip(probs, resultados):
        p = min(max(p, eps), 1 - eps)
        total += r * math.log(p) + (1 - r) * math.log(1 - p)
    return -total / n


def hit_rate(probs: List[float], resultados: List[int], umbral: float = 0.5) -> float:
    """
    Porcentaje de aciertos si se "apostara" siempre que la probabilidad
    estimada supere el umbral (por defecto 50%).
    """
    aciertos = 0
    apuestas = 0
    for p, r in zip(probs, resultados):
        if p >= umbral:
            apuestas += 1
            if r == 1:
                aciertos += 1
    if apuestas == 0:
        return float("nan")
    return aciertos / apuestas
