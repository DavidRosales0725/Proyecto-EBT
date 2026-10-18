"""
Cálculo de probabilidades de mercado vía distribución de Poisson.
"""
import math
from dataclasses import dataclass
from scipy.stats import poisson
from .expected_goals import GolesEsperados


@dataclass
class ProbabilidadesMercado:
    over_05: float
    under_05: float
    over_15: float
    under_15: float
    over_25: float
    under_25: float
    over_35: float
    under_35: float
    btts_si: float
    btts_no: float


def _prob_under(k: int, lam_total: float) -> float:
    """P(goles totales <= k) usando la CDF de Poisson."""
    return poisson.cdf(k, lam_total)


def calcular_probabilidades(goles: GolesEsperados) -> ProbabilidadesMercado:
    lam_total = goles.lambda_local + goles.lambda_visitante

    under_05 = _prob_under(0, lam_total)
    under_15 = _prob_under(1, lam_total)
    under_25 = _prob_under(2, lam_total)
    under_35 = _prob_under(3, lam_total)

    btts_si = (1 - math.exp(-goles.lambda_local)) * (1 - math.exp(-goles.lambda_visitante))

    return ProbabilidadesMercado(
        over_05=1 - under_05, under_05=under_05,
        over_15=1 - under_15, under_15=under_15,
        over_25=1 - under_25, under_25=under_25,
        over_35=1 - under_35, under_35=under_35,
        btts_si=btts_si, btts_no=1 - btts_si,
    )
def calcular_matriz_marcadores(
    goles: GolesEsperados,
    max_goles: int = 6,
) -> dict[tuple[int, int], float]:
    """
    Calcula la probabilidad de cada marcador 0..max_goles
    usando independencia entre los goles locales y visitantes.
    """
    matriz = {}

    for goles_local in range(max_goles + 1):
        prob_local = poisson.pmf(
            goles_local,
            goles.lambda_local,
        )

        for goles_visitante in range(max_goles + 1):
            prob_visitante = poisson.pmf(
                goles_visitante,
                goles.lambda_visitante,
            )

            matriz[(goles_local, goles_visitante)] = (
                prob_local * prob_visitante
            )

    return matriz


def calcular_1x2(
    matriz: dict[tuple[int, int], float],
) -> dict[str, float]:
    """
    Calcula las probabilidades de victoria local,
    empate y victoria visitante a partir de la matriz.
    """
    victoria_local = 0.0
    empate = 0.0
    victoria_visitante = 0.0

    for (goles_local, goles_visitante), probabilidad in matriz.items():

        if goles_local > goles_visitante:
            victoria_local += probabilidad

        elif goles_local == goles_visitante:
            empate += probabilidad

        else:
            victoria_visitante += probabilidad

    return {
        "local": victoria_local,
        "empate": empate,
        "visitante": victoria_visitante,
    }
