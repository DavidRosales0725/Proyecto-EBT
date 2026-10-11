"""
Simulador con DERIVA temporal de fuerza (random walk sobre ataque/defensa).

A diferencia de tests/simulador.py (fuerzas constantes), aquí la fuerza real
de cada equipo cambia partido a partido -- esto es lo único que justifica
estadísticamente ponderar más los partidos recientes. Si alpha < 1 no ayuda
ni siquiera aquí, la ponderación por recencia no tendría base real.
"""
import random
import math
from dataclasses import dataclass, field
from core.fuerzas import ResultadoPartido


@dataclass
class EquipoDinamico:
    nombre: str
    log_ataque: float
    log_defensa: float

    def paso(self, rnd: random.Random, volatilidad: float):
        """Un paso de random walk en log-fuerza (mantiene la fuerza en escala multiplicativa)."""
        self.log_ataque += rnd.gauss(0, volatilidad)
        self.log_defensa += rnd.gauss(0, volatilidad)

    @property
    def ataque_verdadero(self):
        return math.exp(self.log_ataque)

    @property
    def defensa_verdadera(self):
        return math.exp(self.log_defensa)


def generar_universo_dinamico(n_equipos: int, seed: int, sigma_inicial: float = 0.45):
    rnd = random.Random(seed)
    return [EquipoDinamico(f"E{i}", rnd.gauss(0, sigma_inicial), rnd.gauss(0, sigma_inicial))
            for i in range(n_equipos)]


def _rnd_poisson(lam: float, rnd: random.Random) -> int:
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rnd.random()
        if p <= L:
            return k - 1


def generar_historico_dinamico(equipo: EquipoDinamico, rivales: list, promedio_liga: float,
                                n_partidos: int, rnd: random.Random, rol_local: bool,
                                volatilidad: float):
    """
    Genera n_partidos históricos EN ORDEN CRONOLÓGICO (el primero es el más
    antiguo), aplicando deriva de fuerza en cada paso. El equipo termina el
    histórico con una fuerza distinta a la que tenía al principio -- así, un
    promedio simple (que no distingue antiguo de reciente) queda en desventaja
    frente a un promedio ponderado que sí le da más peso a los partidos finales.
    """
    historico = []
    for _ in range(n_partidos):
        equipo.paso(rnd, volatilidad)
        rival = rnd.choice(rivales)
        rival.paso(rnd, volatilidad)
        if rol_local:
            lam_l = equipo.ataque_verdadero * rival.defensa_verdadera * promedio_liga
            lam_v = rival.ataque_verdadero * equipo.defensa_verdadera * promedio_liga
            gl, gv = _rnd_poisson(lam_l, rnd), _rnd_poisson(lam_v, rnd)
            historico.append(ResultadoPartido(goles_a_favor=gl, goles_en_contra=gv))
        else:
            lam_l = rival.ataque_verdadero * equipo.defensa_verdadera * promedio_liga
            lam_v = equipo.ataque_verdadero * rival.defensa_verdadera * promedio_liga
            gl, gv = _rnd_poisson(lam_l, rnd), _rnd_poisson(lam_v, rnd)
            historico.append(ResultadoPartido(goles_a_favor=gv, goles_en_contra=gl))
    return historico


def jugar_partido_actual(local: EquipoDinamico, visitante: EquipoDinamico,
                          promedio_liga: float, rnd: random.Random):
    """El partido de prueba: usa la fuerza ACTUAL (post-deriva) de cada equipo."""
    lam_local = local.ataque_verdadero * visitante.defensa_verdadera * promedio_liga
    lam_visitante = visitante.ataque_verdadero * local.defensa_verdadera * promedio_liga
    gl = _rnd_poisson(lam_local, rnd)
    gv = _rnd_poisson(lam_visitante, rnd)
    return gl, gv
