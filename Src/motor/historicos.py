"""
historicos.py

Recuperación de contextos históricos "punto en el tiempo" para partidos de
fútbol, con garantías estrictas de ausencia de data leakage.

Capa arquitectónica: Preparación. No importa nada de core/ (Motor) -- usa
tipos_comunes.py como contrato de datos compartido, respetando la dirección
de dependencia Datos -> Preparación -> Motor -> Decisión -> Evaluación.

Diseño aprobado en `diseno-historicos.md`. Decisiones confirmadas en la
aprobación:
  - ResultadoPartido vive en tipos_comunes.py (no en core/fuerzas.py).
  - El filtro por `liga` es OBLIGATORIO en esta V1 (no opcional).
  - Defaults: n_partidos=15, n_minimo=10, cruzar_temporadas=False.
  - Se añade `confianza_historico` ("alta"/"media"/"baja") a HistoricoEquipo.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd

from motor.tipos_comunes import ResultadoPartido

# ---------------------------------------------------------------------------
# Defaults del módulo. NOTA: cuando exista config.py (aprobado en la revisión
# arquitectónica, pendiente de ejecutar), estos valores deberían migrar allí
# y este módulo debería importarlos en vez de definirlos localmente.
# ---------------------------------------------------------------------------
N_HISTORICO_RECOMENDADO = 15
N_HISTORICO_MINIMO = 10
CRUZAR_TEMPORADAS_DEFAULT = False


class Rol(Enum):
    LOCAL = "local"
    VISITANTE = "visitante"


class ErrorDataFrameNoOrdenado(Exception):
    """Se lanza cuando el DataFrame de entrada no viene ordenado cronológicamente."""
    pass


# ---------------------------------------------------------------------------
# Estructuras de salida
# ---------------------------------------------------------------------------

@dataclass
class HistoricoEquipo:
    equipo: str
    rol: Rol
    partidos: List[ResultadoPartido]
    n_solicitados: int
    n_encontrados: int
    suficiente: bool
    confianza_historico: str  # "alta" | "media" | "baja"
    fecha_mas_antigua: Optional[pd.Timestamp]
    fecha_mas_reciente: Optional[pd.Timestamp]
    incluye_temporada_anterior: bool = False


@dataclass
class ContextoPartido:
    fecha_partido: pd.Timestamp
    equipo_local: str
    equipo_visitante: str
    liga: str
    historico_local: HistoricoEquipo
    historico_visitante: HistoricoEquipo
    advertencias: List[str] = field(default_factory=list)
    # Uso EXCLUSIVO de evaluación/backtesting posterior. El código que
    # construye historico_local/historico_visitante NUNCA debe leer este
    # campo (ver test_resultado_real_no_contamina_historico).
    resultado_real: Optional[Tuple[int, int]] = None


# ---------------------------------------------------------------------------
# Estructura interna del índice (no se expone fuera del módulo)
# ---------------------------------------------------------------------------

@dataclass
class _EntradaHistorica:
    fecha: pd.Timestamp
    rival: str
    goles_a_favor: int
    goles_en_contra: int
    liga: str
    temporada: str


def _calcular_confianza(n_encontrados: int, n_solicitados: int, n_minimo: int) -> str:
    if n_encontrados >= n_solicitados:
        return "alta"
    elif n_encontrados >= n_minimo:
        return "media"
    return "baja"


class IndiceHistorico:
    """
    Estructura de datos pre-agrupada para consultas O(log n) de históricos
    "punto en el tiempo". Se construye UNA sola vez por dataset y se
    reutiliza para todas las consultas (ej. todos los partidos de un
    backtesting).

    Clave de agrupación: (equipo, liga) -- el filtro por liga es obligatorio
    en esta V1, así que se aplica desde la construcción del índice, no como
    un filtro posterior.
    """

    def __init__(self, df_partidos: pd.DataFrame):
        self._validar_orden(df_partidos)
        self._local: Dict[Tuple[str, str], List[_EntradaHistorica]] = {}
        self._visitante: Dict[Tuple[str, str], List[_EntradaHistorica]] = {}
        self._local_fechas: Dict[Tuple[str, str], List[pd.Timestamp]] = {}
        self._visitante_fechas: Dict[Tuple[str, str], List[pd.Timestamp]] = {}
        self._construir(df_partidos)

    @staticmethod
    def _validar_orden(df: pd.DataFrame) -> None:
        requeridas = {"fecha", "equipo_local", "equipo_visitante", "goles_local", "goles_visitante", "liga", "temporada"}
        faltantes = requeridas.difference(df.columns)
        if faltantes:
            raise ErrorDataFrameNoOrdenado(
                "Faltan columnas requeridas para el histórico: "
                f"{sorted(faltantes)}. Se espera el formato interno producido por data_loader.cargar_datos()."
            )
        if df["fecha"].isna().any():
            raise ErrorDataFrameNoOrdenado("Hay fechas vacías; limpia el CSV con data_loader.cargar_datos() antes de crear el índice.")
        if not df["fecha"].is_monotonic_increasing:
            raise ErrorDataFrameNoOrdenado(
                "El DataFrame de partidos no está ordenado cronológicamente. "
                "historicos.py exige recibir datos ya ordenados (contrato de "
                "data_loader.py) y no los reordena silenciosamente, para no "
                "enmascarar un posible error aguas arriba en el pipeline."
            )

    def _construir(self, df: pd.DataFrame) -> None:
        # Como el DataFrame ya viene ordenado por fecha ascendente, simplemente
        # agrupar en orden de aparición preserva el orden cronológico dentro
        # de cada grupo -- no hace falta volver a ordenar (O(n) en total).
        for row in df.itertuples(index=False):
            clave_local = (row.equipo_local, row.liga)
            clave_visitante = (row.equipo_visitante, row.liga)

            self._local.setdefault(clave_local, []).append(_EntradaHistorica(
                fecha=row.fecha, rival=row.equipo_visitante,
                goles_a_favor=row.goles_local, goles_en_contra=row.goles_visitante,
                liga=row.liga, temporada=row.temporada,
            ))
            self._visitante.setdefault(clave_visitante, []).append(_EntradaHistorica(
                fecha=row.fecha, rival=row.equipo_local,
                goles_a_favor=row.goles_visitante, goles_en_contra=row.goles_local,
                liga=row.liga, temporada=row.temporada,
            ))

        self._local_fechas = {k: [e.fecha for e in v] for k, v in self._local.items()}
        self._visitante_fechas = {k: [e.fecha for e in v] for k, v in self._visitante.items()}

    def _candidatos(self, equipo: str, liga: str, rol: Rol,
                     fecha_corte: pd.Timestamp) -> Tuple[List[_EntradaHistorica], int]:
        """
        Retorna (lista_completa_de_entradas, indice_de_corte) donde
        lista_completa_de_entradas[:indice_de_corte] son estrictamente
        anteriores a fecha_corte. bisect_left garantiza fecha < fecha_corte
        (nunca <=), cumpliendo la garantía anti-leakage de Nivel 2.
        """
        clave = (equipo, liga)
        if rol == Rol.LOCAL:
            entradas = self._local.get(clave, [])
            fechas = self._local_fechas.get(clave, [])
        else:
            entradas = self._visitante.get(clave, [])
            fechas = self._visitante_fechas.get(clave, [])

        idx_corte = bisect.bisect_left(fechas, fecha_corte)
        return entradas, idx_corte

    def historico_equipo(self, equipo: str, liga: str, rol: Rol,
                          fecha_corte: pd.Timestamp, temporada_actual: str,
                          n_partidos: int = N_HISTORICO_RECOMENDADO,
                          n_minimo: int = N_HISTORICO_MINIMO,
                          cruzar_temporadas: bool = CRUZAR_TEMPORADAS_DEFAULT
                          ) -> HistoricoEquipo:
        entradas, idx_corte = self._candidatos(equipo, liga, rol, fecha_corte)

        seleccionados: List[_EntradaHistorica] = []
        incluye_temporada_anterior = False
        i = idx_corte - 1
        while i >= 0 and len(seleccionados) < n_partidos:
            entrada = entradas[i]
            es_temporada_actual = entrada.temporada == temporada_actual
            if es_temporada_actual or cruzar_temporadas:
                seleccionados.append(entrada)
                if not es_temporada_actual:
                    incluye_temporada_anterior = True
            elif not cruzar_temporadas:
                # Lista ordenada ascendente por fecha: si sin cruzar_temporadas
                # encontramos la primera entrada de una temporada distinta
                # caminando hacia atrás, todo lo anterior también lo será.
                break
            i -= 1

        seleccionados.reverse()  # orden ascendente (más antiguo primero),
                                 # requerido por pesos.py para la ponderación exponencial

        n_encontrados = len(seleccionados)
        confianza = _calcular_confianza(n_encontrados, n_partidos, n_minimo)

        return HistoricoEquipo(
            equipo=equipo,
            rol=rol,
            partidos=[ResultadoPartido(goles_a_favor=e.goles_a_favor,
                                        goles_en_contra=e.goles_en_contra)
                      for e in seleccionados],
            n_solicitados=n_partidos,
            n_encontrados=n_encontrados,
            suficiente=(confianza != "baja"),
            confianza_historico=confianza,
            fecha_mas_antigua=seleccionados[0].fecha if seleccionados else None,
            fecha_mas_reciente=seleccionados[-1].fecha if seleccionados else None,
            incluye_temporada_anterior=incluye_temporada_anterior,
        )


# ---------------------------------------------------------------------------
# API de alto nivel
# ---------------------------------------------------------------------------

def construir_contexto_partido(indice: IndiceHistorico,
                                fecha_partido: pd.Timestamp,
                                equipo_local: str,
                                equipo_visitante: str,
                                liga: str,
                                temporada: str,
                                n_partidos: int = N_HISTORICO_RECOMENDADO,
                                n_minimo: int = N_HISTORICO_MINIMO,
                                cruzar_temporadas: bool = CRUZAR_TEMPORADAS_DEFAULT,
                                resultado_real: Optional[Tuple[int, int]] = None
                                ) -> ContextoPartido:
    """
    Punto de entrada principal para un único partido. `fecha_partido` actúa
    como fecha_corte para ambos históricos -- ni el local ni el visitante
    pueden usar información de fecha >= fecha_partido.
    """
    historico_local = indice.historico_equipo(
        equipo_local, liga, Rol.LOCAL, fecha_partido, temporada,
        n_partidos, n_minimo, cruzar_temporadas,
    )
    historico_visitante = indice.historico_equipo(
        equipo_visitante, liga, Rol.VISITANTE, fecha_partido, temporada,
        n_partidos, n_minimo, cruzar_temporadas,
    )

    advertencias: List[str] = []
    if not historico_local.suficiente:
        advertencias.append(
            f"Histórico insuficiente para {equipo_local} como local "
            f"({historico_local.n_encontrados}/{n_partidos}, confianza={historico_local.confianza_historico})"
        )
    if not historico_visitante.suficiente:
        advertencias.append(
            f"Histórico insuficiente para {equipo_visitante} como visitante "
            f"({historico_visitante.n_encontrados}/{n_partidos}, confianza={historico_visitante.confianza_historico})"
        )

    return ContextoPartido(
        fecha_partido=fecha_partido,
        equipo_local=equipo_local,
        equipo_visitante=equipo_visitante,
        liga=liga,
        historico_local=historico_local,
        historico_visitante=historico_visitante,
        advertencias=advertencias,
        resultado_real=resultado_real,
    )


def generar_contextos_para_dataframe(df_partidos: pd.DataFrame,
                                      n_partidos: int = N_HISTORICO_RECOMENDADO,
                                      n_minimo: int = N_HISTORICO_MINIMO,
                                      cruzar_temporadas: bool = CRUZAR_TEMPORADAS_DEFAULT
                                      ) -> List[ContextoPartido]:
    """
    Helper de conveniencia para backtesting: construye el índice una sola
    vez y genera un ContextoPartido para CADA partido del DataFrame, usando
    el propio dataset como fuente de históricos (cada partido solo puede
    ver los partidos anteriores a su propia fecha, incluidos los de otros
    partidos del mismo DataFrame).
    """
    indice = IndiceHistorico(df_partidos)
    contextos = []
    for row in df_partidos.itertuples(index=False):
        contexto = construir_contexto_partido(
            indice, row.fecha, row.equipo_local, row.equipo_visitante,
            row.liga, row.temporada, n_partidos, n_minimo, cruzar_temporadas,
            resultado_real=(row.goles_local, row.goles_visitante),
        )
        contextos.append(contexto)
    return contextos
