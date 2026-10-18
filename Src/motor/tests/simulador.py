"""
Genera un universo de equipos con fuerzas VERDADERAS conocidas, y simula
partidos muestreando de una Poisson real con esas fuerzas.

Esto permite validar el motor: como conocemos la "verdad" del proceso
generador, podemos medir si el motor la recupera correctamente a partir
de únicamente los resultados observados (igual que haría un usuario real).
"""
import random
import math
from dataclasses import dataclass
from ..core.fuerzas import ResultadoPartido


@dataclass
class EquipoVerdadero:
    nombre: str
    ataque_verdadero: float   # multiplicador real de fuerza ofensiva
    defensa_verdadera: float  # multiplicador real de fuerza defensiva


def generar_universo_equipos(n_equipos: int, seed: int = 42, sigma: float = 0.45) -> list:
    """
    sigma controla la dispersión de calidad entre equipos.
    ~0.28 = liga muy pareja; ~0.45 = dispersión típica de una liga real;
    ~0.6+ = liga con brecha grande entre grandes y chicos.
    """
    rnd = random.Random(seed)
    equipos = []
    for i in range(n_equipos):
        ataque = math.exp(rnd.gauss(0, sigma))
        defensa = math.exp(rnd.gauss(0, sigma))
        equipos.append(EquipoVerdadero(f"Equipo_{i+1}", ataque, defensa))
    return equipos


def simular_partido(local: EquipoVerdadero, visitante: EquipoVerdadero,
                     promedio_liga: float, rnd: random.Random):
    """Muestrea un resultado real usando el proceso Poisson VERDADERO."""
    lam_local = local.ataque_verdadero * visitante.defensa_verdadera * promedio_liga
    lam_visitante = visitante.ataque_verdadero * local.defensa_verdadera * promedio_liga

    goles_local = rnd_poisson(lam_local, rnd)
    goles_visitante = rnd_poisson(lam_visitante, rnd)
    return goles_local, goles_visitante, lam_local, lam_visitante


def rnd_poisson(lam: float, rnd: random.Random) -> int:
    """Muestreo de Poisson vía algoritmo de Knuth (sin depender de numpy)."""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rnd.random()
        if p <= L:
            return k - 1


def _tau_verdadero(x, y, lam, mu, rho):
    if x == 0 and y == 0:
        return 1 - (lam * mu * rho)
    elif x == 0 and y == 1:
        return 1 + (lam * rho)
    elif x == 1 and y == 0:
        return 1 + (mu * rho)
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def simular_partido_correlacionado(local, visitante, promedio_liga: float,
                                    rnd: random.Random, rho_verdadero: float = -0.13,
                                    max_goles: int = 10):
    """
    Simula un partido cuyo proceso generador REAL tiene la correlación de
    marcadores bajos descrita por Dixon-Coles (rho_verdadero < 0 = más
    0-0 y 1-1 de lo que el Poisson independiente predice, menos 1-0 y 0-1).

    Usa muestreo directo por matriz de probabilidad acumulada (exacto,
    no requiere rechazo).
    """
    lam_local = local.ataque_verdadero * visitante.defensa_verdadera * promedio_liga
    lam_visitante = visitante.ataque_verdadero * local.defensa_verdadera * promedio_liga

    # Construir la distribución conjunta verdadera con la correlación real
    from scipy.stats import poisson as _poisson
    matriz = [[0.0] * (max_goles + 1) for _ in range(max_goles + 1)]
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            base = _poisson.pmf(i, lam_local) * _poisson.pmf(j, lam_visitante)
            matriz[i][j] = base * _tau_verdadero(i, j, lam_local, lam_visitante, rho_verdadero)
    total = sum(sum(fila) for fila in matriz)
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            matriz[i][j] /= total

    # Muestreo por CDF acumulada
    r = rnd.random()
    acumulado = 0.0
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            acumulado += matriz[i][j]
            if r <= acumulado:
                return i, j, lam_local, lam_visitante
    return max_goles, max_goles, lam_local, lam_visitante  # fallback numérico


def generar_historico(equipo: EquipoVerdadero, rivales: list, promedio_liga: float,
                       n_partidos: int, rnd: random.Random, rol_local: bool):
    """
    Genera n_partidos históricos para un equipo en UN SOLO rol (local o visitante),
    respetando la regla de no mezclar roles.
    """
    historico = []
    for _ in range(n_partidos):
        rival = rnd.choice(rivales)
        if rol_local:
            gl, gv, _, _ = simular_partido(equipo, rival, promedio_liga, rnd)
            historico.append(ResultadoPartido(goles_a_favor=gl, goles_en_contra=gv))
        else:
            gl, gv, _, _ = simular_partido(rival, equipo, promedio_liga, rnd)
            historico.append(ResultadoPartido(goles_a_favor=gv, goles_en_contra=gl))
    return historico
