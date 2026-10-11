"""
Conversión de cuotas a probabilidad implícita, cálculo de value y overround.
"""
from dataclasses import dataclass


@dataclass
class ResultadoValueBet:
    prob_sistema_pct: float
    prob_implicita_pct: float
    value_pct: float
    hay_value: bool


def probabilidad_implicita(cuota: float) -> float:
    """Convierte una cuota decimal a probabilidad implícita (%)."""
    if cuota <= 1.0:
        raise ValueError("La cuota debe ser mayor a 1.0")
    return (1 / cuota) * 100


def evaluar_value_bet(prob_sistema: float, cuota: float) -> ResultadoValueBet:
    """prob_sistema debe estar en escala 0-1 (ej. 0.62 para 62%)."""
    prob_sistema_pct = prob_sistema * 100
    prob_implicita_pct = probabilidad_implicita(cuota)
    value = prob_sistema_pct - prob_implicita_pct
    return ResultadoValueBet(
        prob_sistema_pct=prob_sistema_pct,
        prob_implicita_pct=prob_implicita_pct,
        value_pct=value,
        hay_value=value > 0,
    )


def overround(cuota_a: float, cuota_b: float) -> float:
    """Overround de un mercado de dos resultados (ej. Over/Under de la misma línea)."""
    return probabilidad_implicita(cuota_a) + probabilidad_implicita(cuota_b) - 100
