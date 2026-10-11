"""
Cálculo de fuerza ofensiva y defensiva relativa (estilo Maher 1982).

Regla fundamental respetada: nunca se mezclan partidos de local con partidos
de visitante. Cada equipo se analiza únicamente en su rol correspondiente.
"""
from dataclasses import dataclass
from typing import List
from .pesos import promedio_ponderado


@dataclass
class ResultadoPartido:
    goles_a_favor: int
    goles_en_contra: int


@dataclass
class FuerzaEquipo:
    ataque: float          # promedio ponderado de goles anotados
    defensa: float          # promedio ponderado de goles recibidos
    ataque_relativo: float  # ataque / promedio_referencia
    defensa_relativa: float  # defensa / promedio_referencia


def promedio_referencia(historico_local: List[ResultadoPartido],
                         historico_visitante: List[ResultadoPartido]) -> float:
    """
    Promedio de goles anotados combinando ambas muestras (local + visitante).
    Sirve como contexto para normalizar las fuerzas relativas.
    En Fase 1, sin datos de liga completa, esta es la mejor referencia disponible.
    """
    todos_goles = (
        [r.goles_a_favor for r in historico_local] +
        [r.goles_a_favor for r in historico_visitante]
    )
    if not todos_goles:
        return 1.0
    return sum(todos_goles) / len(todos_goles)


def calcular_fuerza(historico: List[ResultadoPartido], promedio_ref: float,
                     alpha: float = 0.90) -> FuerzaEquipo:
    """
    Calcula la fuerza ofensiva y defensiva de un equipo en un rol específico
    (ya sea SOLO como local o SOLO como visitante).
    """
    goles_favor = [r.goles_a_favor for r in historico]
    goles_contra = [r.goles_en_contra for r in historico]

    ataque = promedio_ponderado(goles_favor, alpha)
    defensa = promedio_ponderado(goles_contra, alpha)

    ataque_rel = ataque / promedio_ref if promedio_ref > 0 else 1.0
    defensa_rel = defensa / promedio_ref if promedio_ref > 0 else 1.0

    return FuerzaEquipo(
        ataque=ataque,
        defensa=defensa,
        ataque_relativo=ataque_rel,
        defensa_relativa=defensa_rel,
    )
