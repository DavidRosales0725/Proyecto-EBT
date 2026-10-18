"""
data_loader.py

Módulo independiente de importación y preparación de datos históricos de
fútbol para el motor estadístico. NO modifica ni depende del motor
matemático (core/) -- su única responsabilidad es dejar los datos limpios,
validados y en un formato interno único, listos para que el motor los
consuma en la Fase 2 (backtesting con datos reales).

Diseño pensado para soportar distintos formatos de CSV en el futuro:
  - La detección de columnas es por SINÓNIMOS, no por posición ni por un
    único nombre fijo. Añadir soporte a un nuevo proveedor de datos es
    agregar sus nombres de columna a SINONIMOS_CAMPOS, sin tocar el resto
    del módulo.
  - También se puede pasar un mapeo manual (`mapeo_personalizado`) para
    csvs que no sigan ningún patrón conocido.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
try:
    import streamlit as st
except ModuleNotFoundError:  # permite reutilizar el cargador fuera de la app
    st = None
try:  # paquete normal de la aplicación
    from motor.team_names import normalizar_nombre_equipo
except ModuleNotFoundError:  # compatibilidad con importación directa en scripts/tests
    from team_names import normalizar_nombre_equipo
from dataclasses import dataclass, field
from typing import Optional, Dict, List


# ---------------------------------------------------------------------------
# 1. CONFIGURACIÓN: campos requeridos y sinónimos conocidos
# ---------------------------------------------------------------------------

CAMPOS_REQUERIDOS = [
    "fecha",
    "equipo_local",
    "equipo_visitante",
    "goles_local",
    "goles_visitante",
]

CAMPOS_OPCIONALES = ["liga", "temporada"]

# Sinónimos conocidos por campo interno. Añadir aquí nuevos formatos de CSV
# (por ejemplo Kaggle, u otra fuente) sin modificar el resto del módulo.
SINONIMOS_CAMPOS: Dict[str, List[str]] = {
    "fecha": ["Date", "date", "Fecha", "MatchDate", "match_date"],
    "equipo_local": ["HomeTeam", "home_team", "Local", "EquipoLocal", "home", "Home"],
    "equipo_visitante": ["AwayTeam", "away_team", "Visitante", "EquipoVisitante", "away", "Away"],
    "goles_local": ["FTHG", "HG", "home_goals", "GolesLocal", "goles_local", "HomeGoals"],
    "goles_visitante": ["FTAG", "AG", "away_goals", "GolesVisitante", "goles_visitante", "AwayGoals"],
    "liga": ["Div", "League", "Liga", "competition", "Competition"],
    "temporada": ["Season", "Temporada", "season"],
}


# ---------------------------------------------------------------------------
# 2. ESTRUCTURAS DE RESULTADO
# ---------------------------------------------------------------------------

@dataclass
class ResumenCarga:
    n_partidos_validos: int
    n_partidos_descartados: int
    n_equipos: int
    fecha_min: Optional[pd.Timestamp]
    fecha_max: Optional[pd.Timestamp]
    promedio_goles_totales: float
    columnas_detectadas: Dict[str, str]
    motivos_descarte: Dict[str, int] = field(default_factory=dict)
    columnas_con_faltantes_originales: Dict[str, int] = field(default_factory=dict)

    def imprimir(self) -> None:
        print("=" * 60)
        print("RESUMEN DE CARGA DE DATOS")
        print("=" * 60)
        print(f"Partidos válidos:      {self.n_partidos_validos}")
        print(f"Partidos descartados:  {self.n_partidos_descartados}")
        if self.motivos_descarte:
            for motivo, n in self.motivos_descarte.items():
                print(f"    - {motivo}: {n}")
        print(f"Equipos distintos:     {self.n_equipos}")
        if self.fecha_min is not None and self.fecha_max is not None:
            print(f"Rango de fechas:       {self.fecha_min.date()} a {self.fecha_max.date()}")
        else:
            print("Rango de fechas:       N/D (sin partidos válidos)")
        print(f"Promedio goles/partido: {self.promedio_goles_totales:.3f}")
        print("\nColumnas detectadas:")
        for campo, columna in self.columnas_detectadas.items():
            print(f"    {campo:<20} <- '{columna}'")
        if self.columnas_con_faltantes_originales:
            print("\nDatos faltantes detectados en el CSV original:")
            for col, n in self.columnas_con_faltantes_originales.items():
                print(f"    {col}: {n} valores vacíos")
        print("=" * 60)


@dataclass
class DatosPreparados:
    df: pd.DataFrame               # datos válidos, en formato interno, ordenados cronológicamente
    df_descartados: pd.DataFrame   # filas descartadas, con columna 'motivo_descarte'
    resumen: ResumenCarga


class ErrorColumnasFaltantes(Exception):
    """Se lanza cuando no se pueden detectar todas las columnas obligatorias."""
    pass


# ---------------------------------------------------------------------------
# 3. DETECCIÓN Y VALIDACIÓN DE COLUMNAS
# ---------------------------------------------------------------------------

def detectar_columnas(df: pd.DataFrame,
                       mapeo_personalizado: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Detecta automáticamente qué columna del CSV corresponde a cada campo
    interno, usando la tabla de sinónimos. Si se pasa `mapeo_personalizado`
    (ej. {"fecha": "MiColumnaDeFecha"}), ese mapeo tiene prioridad sobre
    la detección automática para esos campos específicos.

    Retorna un diccionario {campo_interno: nombre_columna_real}. Los campos
    que no se logren detectar simplemente no aparecen en el resultado.
    """
    columnas_normalizadas = {c.strip().lower(): c for c in df.columns}
    mapeo: Dict[str, str] = {}

    todos_los_campos = CAMPOS_REQUERIDOS + CAMPOS_OPCIONALES
    for campo in todos_los_campos:
        if mapeo_personalizado and campo in mapeo_personalizado:
            columna_forzada = mapeo_personalizado[campo]
            if columna_forzada in df.columns:
                mapeo[campo] = columna_forzada
                continue

        for candidato in SINONIMOS_CAMPOS.get(campo, []):
            if candidato in df.columns:
                mapeo[campo] = candidato
                break
            # también probar comparación case-insensitive
            candidato_norm = candidato.strip().lower()
            if candidato_norm in columnas_normalizadas:
                mapeo[campo] = columnas_normalizadas[candidato_norm]
                break

    return mapeo


def validar_columnas(mapeo: Dict[str, str], columnas_disponibles: Optional[List[str]] = None) -> None:
    """
    Lanza ErrorColumnasFaltantes con un mensaje claro si falta algún campo
    obligatorio. No valida los campos opcionales (liga, temporada).
    """
    faltantes = [c for c in CAMPOS_REQUERIDOS if c not in mapeo]
    if faltantes:
        columnas_info = f" Columnas disponibles en el CSV: {list(columnas_disponibles)}." if columnas_disponibles is not None else ""
        raise ErrorColumnasFaltantes(
            f"No se pudieron detectar las siguientes columnas obligatorias: "
            f"{faltantes}.{columnas_info} "
            f"Si el CSV usa nombres no estándar, pasa 'mapeo_personalizado' "
            f"indicando explícitamente qué columna corresponde a cada campo, "
            f"ej. mapeo_personalizado={{'fecha': 'MiColumnaFecha'}}."
        )


# ---------------------------------------------------------------------------
# 4. TRANSFORMACIÓN A FORMATO INTERNO
# ---------------------------------------------------------------------------

def _parsear_fechas(serie: pd.Series) -> pd.Series:
    """
    Parsea fechas probando primero los formatos explícitos más comunes en
    fuentes de datos de fútbol (dd/mm/yy y dd/mm/yyyy, como football-data.co.uk),
    y solo cae a inferencia genérica ('mixed') para lo que no calce, evitando
    además la ambigüedad día/mes que puede producir un parseo silenciosamente
    incorrecto.
    """
    serie_str = serie.astype(str)
    resultado = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")

    formatos_conocidos = ["%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d"]
    pendientes = resultado.isna()
    for fmt in formatos_conocidos:
        if not pendientes.any():
            break
        parseado = pd.to_datetime(serie_str[pendientes], format=fmt, errors="coerce")
        resultado.loc[pendientes] = parseado
        pendientes = resultado.isna()

    if pendientes.any():
        resultado.loc[pendientes] = pd.to_datetime(
            serie_str[pendientes], dayfirst=True, format="mixed", errors="coerce"
        )

    return resultado


# Nombres de campo internos ya reservados por el formato estándar -- se usan
# para evitar que una columna passthrough del CSV original colisione por
# accidente con un nombre interno ya definido.
_NOMBRES_INTERNOS_RESERVADOS = {
    "fecha", "equipo_local", "equipo_visitante", "goles_local",
    "goles_visitante", "liga", "temporada", "_indice_original",
}


def estandarizar(df: pd.DataFrame, mapeo: Dict[str, str]) -> pd.DataFrame:
    """
    Construye un DataFrame nuevo con nombres de columna internos fijos,
    parseando tipos (fecha a datetime, goles a numérico). No descarta
    filas inválidas todavía -- eso ocurre en limpiar().

    Además de los campos estandarizados, conserva como "passthrough" el
    resto de columnas del CSV original que no formen parte del mapeo
    (ej. columnas de cuotas, estadísticas de partido) -- cambio aditivo:
    no afecta ninguna de las columnas estandarizadas ya validadas, solo
    evita que datos adicionales del CSV fuente se pierdan silenciosamente
    para módulos aguas abajo (ej. backtesting.py, que los necesita para
    resolver cuotas).
    """
    interno = pd.DataFrame()

    interno["fecha"] = _parsear_fechas(df[mapeo["fecha"]])
    interno["equipo_local"] = df[mapeo["equipo_local"]].map(normalizar_nombre_equipo)
    interno["equipo_visitante"] = df[mapeo["equipo_visitante"]].map(normalizar_nombre_equipo)
    interno["goles_local"] = pd.to_numeric(df[mapeo["goles_local"]], errors="coerce")
    interno["goles_visitante"] = pd.to_numeric(df[mapeo["goles_visitante"]], errors="coerce")

    interno["liga"] = df[mapeo["liga"]].astype(str).str.strip() if "liga" in mapeo else "SIN_LIGA"
    interno["temporada"] = df[mapeo["temporada"]].astype(str).str.strip() if "temporada" in mapeo else "SIN_TEMPORADA"

    # conservar el índice original para trazabilidad en caso de descarte
    interno["_indice_original"] = df.index

    # Consolidar el bloque de columnas estandarizadas ANTES de concatenar el
    # passthrough evita el patrón que dispara PerformanceWarning de pandas
    # (insertar una columna suelta después de un concat grande sobre un
    # frame ya fragmentado en varios bloques de una sola columna).
    interno = interno.copy()

    columnas_usadas_en_mapeo = set(mapeo.values())
    columnas_passthrough = [
        c for c in df.columns
        if c not in columnas_usadas_en_mapeo and c not in _NOMBRES_INTERNOS_RESERVADOS
    ]
    if columnas_passthrough:
        interno = pd.concat([interno, df[columnas_passthrough]], axis=1)

    return interno.copy()


# ---------------------------------------------------------------------------
# 5. LIMPIEZA / VALIDACIÓN DE REGISTROS
# ---------------------------------------------------------------------------

def limpiar(df_estandar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa el DataFrame estandarizado en (válidos, descartados).

    Un registro se descarta si:
      - la fecha no se pudo parsear (NaT)
      - falta el nombre de algún equipo (vacío o 'nan')
      - los goles no son numéricos, son negativos, o no son enteros
      - el equipo local y visitante son el mismo (dato corrupto)

    Cada fila descartada queda con una columna 'motivo_descarte' explicando
    la razón (una fila puede tener varios motivos; se listan todos).
    """
    df = df_estandar.copy()
    motivos = pd.Series([""] * len(df), index=df.index, dtype=object)

    def agregar_motivo(mask: pd.Series, texto: str):
        nonlocal motivos
        motivos.loc[mask] = motivos.loc[mask].apply(
            lambda actual: f"{actual};{texto}" if actual else texto
        )

    mask_fecha_invalida = df["fecha"].isna()
    agregar_motivo(mask_fecha_invalida, "fecha_invalida")

    mask_local_vacio = df["equipo_local"].isna() | (df["equipo_local"].str.lower().isin(["", "nan", "none"]))
    agregar_motivo(mask_local_vacio, "equipo_local_vacio")

    mask_visitante_vacio = df["equipo_visitante"].isna() | (df["equipo_visitante"].str.lower().isin(["", "nan", "none"]))
    agregar_motivo(mask_visitante_vacio, "equipo_visitante_vacio")

    mask_goles_local_invalido = df["goles_local"].isna() | (df["goles_local"] < 0) | (df["goles_local"] % 1 != 0)
    agregar_motivo(mask_goles_local_invalido, "goles_local_invalidos")

    mask_goles_visitante_invalido = df["goles_visitante"].isna() | (df["goles_visitante"] < 0) | (df["goles_visitante"] % 1 != 0)
    agregar_motivo(mask_goles_visitante_invalido, "goles_visitante_invalidos")

    mask_mismo_equipo = (
        df["equipo_local"].str.lower() == df["equipo_visitante"].str.lower()
    ) & (~mask_local_vacio) & (~mask_visitante_vacio)
    agregar_motivo(mask_mismo_equipo, "equipo_local_igual_a_visitante")

    mask_duplicado = df.duplicated(subset=["fecha", "equipo_local", "equipo_visitante"], keep="first")
    agregar_motivo(mask_duplicado, "duplicado")

    mask_invalido = motivos != ""

    df_validos = df.loc[~mask_invalido].copy()
    df_descartados = df.loc[mask_invalido].copy()
    df_descartados["motivo_descarte"] = motivos.loc[mask_invalido]

    # tipar goles como enteros ya garantizado válidos
    df_validos["goles_local"] = df_validos["goles_local"].astype(int)
    df_validos["goles_visitante"] = df_validos["goles_visitante"].astype(int)

    return df_validos, df_descartados


# ---------------------------------------------------------------------------
# 6. ORDENAMIENTO CRONOLÓGICO
# ---------------------------------------------------------------------------

def ordenar_cronologicamente(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena por fecha ascendente. Requisito base para cualquier backtesting futuro."""
    return df.sort_values("fecha").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. DETECCIÓN DE FALTANTES EN EL CSV ORIGINAL (antes de limpiar)
# ---------------------------------------------------------------------------

def detectar_faltantes_originales(df: pd.DataFrame, mapeo: Dict[str, str]) -> Dict[str, int]:
    """
    Cuenta valores faltantes en las columnas ORIGINALES del CSV (antes de
    cualquier transformación), para las columnas que sí se detectaron.
    Ayuda a diagnosticar si el CSV fuente tiene huecos sistemáticos.
    """
    faltantes = {}
    for campo, columna in mapeo.items():
        n_faltantes = df[columna].isna().sum()
        if n_faltantes > 0:
            faltantes[columna] = int(n_faltantes)
    return faltantes


# ---------------------------------------------------------------------------
# 8. RESUMEN
# ---------------------------------------------------------------------------

def generar_resumen(df_validos: pd.DataFrame, df_descartados: pd.DataFrame,
                     mapeo: Dict[str, str],
                     faltantes_originales: Dict[str, int]) -> ResumenCarga:
    equipos = set(df_validos["equipo_local"]).union(set(df_validos["equipo_visitante"]))

    if len(df_validos) > 0:
        fecha_min = df_validos["fecha"].min()
        fecha_max = df_validos["fecha"].max()
        promedio_goles = (df_validos["goles_local"] + df_validos["goles_visitante"]).mean()
    else:
        fecha_min = fecha_max = None
        promedio_goles = float("nan")

    motivos_conteo: Dict[str, int] = {}
    if len(df_descartados) > 0:
        for lista_motivos in df_descartados["motivo_descarte"]:
            for motivo in lista_motivos.split(";"):
                motivos_conteo[motivo] = motivos_conteo.get(motivo, 0) + 1

    return ResumenCarga(
        n_partidos_validos=len(df_validos),
        n_partidos_descartados=len(df_descartados),
        n_equipos=len(equipos),
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        promedio_goles_totales=promedio_goles,
        columnas_detectadas=mapeo,
        motivos_descarte=motivos_conteo,
        columnas_con_faltantes_originales=faltantes_originales,
    )


# ---------------------------------------------------------------------------
# 9. FUNCIÓN PRINCIPAL (orquesta todo el pipeline)
# ---------------------------------------------------------------------------

def cargar_datos(ruta_csv: str,
                  mapeo_personalizado: Optional[Dict[str, str]] = None,
                  encoding: str = "utf-8",
                  imprimir_resumen: bool = True) -> DatosPreparados:
    """
    Punto de entrada único del módulo. Lee un CSV, detecta columnas, valida,
    estandariza, limpia, ordena cronológicamente y genera el resumen.

    Parámetros
    ----------
    ruta_csv : ruta al archivo CSV a cargar.
    mapeo_personalizado : opcional, para csvs con nombres de columna no
        reconocidos automáticamente. Ej: {"fecha": "FechaPartido"}.
    encoding : football-data.co.uk usa a veces 'latin-1'/'cp1252' en vez
        de utf-8 -- si falla la lectura, probar encoding="latin-1".
    imprimir_resumen : si True, imprime el resumen al finalizar.

    Retorna
    -------
    DatosPreparados con:
        .df               -> datos válidos, formato interno, ordenados
        .df_descartados   -> filas descartadas con motivo
        .resumen          -> objeto ResumenCarga
    """
    ruta = Path(ruta_csv).expanduser().resolve()
    stat = ruta.stat()
    # La metadata forma parte de la clave, así los cambios de contenido
    # invalidan automáticamente el resultado sin releer el archivo entero.
    firma_archivo = (stat.st_mtime_ns, stat.st_size)
    resultado = _cargar_datos_cacheado(
        str(ruta), encoding, tuple(sorted((mapeo_personalizado or {}).items())),
        firma_archivo,
    )
    if imprimir_resumen:
        resultado.resumen.imprimir()
    return resultado


def _cargar_datos_sin_cache(ruta_csv: str, encoding: str,
                            mapeo_items: tuple[tuple[str, str], ...],
                            firma_archivo: tuple[int, int]) -> DatosPreparados:
    """Carga/limpia un CSV; si falla, la excepción no se almacena en caché."""
    del firma_archivo  # solo se usa como clave de invalidación del caché
    df_crudo = pd.read_csv(ruta_csv, encoding=encoding)

    mapeo = detectar_columnas(df_crudo, dict(mapeo_items) or None)
    validar_columnas(mapeo, columnas_disponibles=list(df_crudo.columns))

    faltantes_originales = detectar_faltantes_originales(df_crudo, mapeo)

    df_estandar = estandarizar(df_crudo, mapeo)
    df_validos, df_descartados = limpiar(df_estandar)
    df_validos = ordenar_cronologicamente(df_validos)

    resumen = generar_resumen(df_validos, df_descartados, mapeo, faltantes_originales)

    return DatosPreparados(df=df_validos, df_descartados=df_descartados, resumen=resumen)


if st is not None:
    _cargar_datos_cacheado = st.cache_data(ttl=3600, show_spinner=False)(_cargar_datos_sin_cache)
else:
    _cargar_datos_cacheado = _cargar_datos_sin_cache
