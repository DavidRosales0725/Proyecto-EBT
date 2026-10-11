"""
Corrección Dixon-Coles (1997) para la subestimación de marcadores bajos
correlacionados (0-0, 1-0, 0-1, 1-1) que produce el Poisson independiente puro.

tau(x, y) ajusta solo esas 4 celdas de la distribución conjunta; el resto
de la matriz queda igual (proporcionalmente reescalada al normalizar).
"""
import math
from dataclasses import dataclass
from scipy.stats import poisson as _poisson


def tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1 - (lam * mu * rho)
    elif x == 0 and y == 1:
        return 1 + (lam * rho)
    elif x == 1 and y == 0:
        return 1 + (mu * rho)
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def matriz_conjunta(lam: float, mu: float, rho: float = 0.0, max_goles: int = 10):
    """
    Matriz [i][j] = P(local anota i, visitante anota j), con corrección
    Dixon-Coles aplicada a las 4 celdas de marcador bajo, y renormalizada
    para que la matriz completa vuelva a sumar 1.
    """
    matriz = [[0.0] * (max_goles + 1) for _ in range(max_goles + 1)]
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            base = _poisson.pmf(i, lam) * _poisson.pmf(j, mu)
            matriz[i][j] = base * tau(i, j, lam, mu, rho)

    total = sum(sum(fila) for fila in matriz)
    if total > 0:
        for i in range(max_goles + 1):
            for j in range(max_goles + 1):
                matriz[i][j] /= total
    return matriz


@dataclass
class ProbabilidadesMercadoDC:
    over_05: float
    over_15: float
    over_25: float
    over_35: float
    btts_si: float


def calcular_probabilidades_dc(lam: float, mu: float, rho: float = 0.0,
                                max_goles: int = 10) -> ProbabilidadesMercadoDC:
    m = matriz_conjunta(lam, mu, rho, max_goles)

    def prob_over(k):
        return sum(m[i][j] for i in range(max_goles + 1) for j in range(max_goles + 1)
                   if i + j > k)

    btts_si = sum(m[i][j] for i in range(1, max_goles + 1) for j in range(1, max_goles + 1))

    return ProbabilidadesMercadoDC(
        over_05=prob_over(0.5), over_15=prob_over(1.5),
        over_25=prob_over(2.5), over_35=prob_over(3.5),
        btts_si=btts_si,
    )
