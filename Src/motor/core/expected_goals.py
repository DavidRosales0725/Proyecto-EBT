"""
Estimación de goles esperados (pseudo-xG estadístico), sin datos oficiales de xG.

lambda_local = FuerzaAtaque_Local * FuerzaDefensa_Visitante * PromedioReferencia
lambda_visitante = FuerzaAtaque_Visitante * FuerzaDefensa_Local * PromedioReferencia

Modelo multiplicativo normalizado (Maher, 1982), no aditivo.
"""
from dataclasses import dataclass
from .fuerzas import FuerzaEquipo

@dataclass
class GolesEsperados:
    lambda_local: float
    lambda_visitante: float


def calcular_goles_esperados(fuerza_local: FuerzaEquipo, fuerza_visitante: FuerzaEquipo,
                              promedio_ref: float) -> GolesEsperados:
    lambda_local = fuerza_local.ataque_relativo * fuerza_visitante.defensa_relativa * promedio_ref
    lambda_visitante = fuerza_visitante.ataque_relativo * fuerza_local.defensa_relativa * promedio_ref

    # Piso de seguridad: lambda no puede ser 0 o negativo (rompería Poisson)
    lambda_local = max(lambda_local, 0.05)
    lambda_visitante = max(lambda_visitante, 0.05)

    return GolesEsperados(lambda_local=lambda_local, lambda_visitante=lambda_visitante)
