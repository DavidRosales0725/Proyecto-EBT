"""
Suite de pruebas para historicos.py, siguiendo el plan de 14 pruebas
comprometido en diseno-historicos.md (Sección 13).
"""
import random
import pandas as pd
import pytest

from motor.historicos import (
    IndiceHistorico, Rol, construir_contexto_partido,
    ErrorDataFrameNoOrdenado, N_HISTORICO_RECOMENDADO, N_HISTORICO_MINIMO,
)


# ---------------------------------------------------------------------------
# Fixtures: construcción de un dataset sintético controlado
# ---------------------------------------------------------------------------

def _construir_df(filas):
    """filas: lista de tuplas (fecha_str, local, visitante, gl, gv, liga, temporada)"""
    df = pd.DataFrame(filas, columns=[
        "fecha", "equipo_local", "equipo_visitante",
        "goles_local", "goles_visitante", "liga", "temporada"
    ])
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.sort_values("fecha").reset_index(drop=True)


@pytest.fixture
def df_basico():
    """
    Liga E0, temporada 2023, 4 equipos, suficientes partidos para tener
    histórico completo hacia el final de la temporada.
    """
    equipos = ["A", "B", "C", "D"]
    filas = []
    fecha = pd.Timestamp("2023-08-01")
    rnd = random.Random(42)
    # round-robin simplificado: cada equipo juega contra cada otro, ida y vuelta,
    # repetido varias veces para acumular suficiente histórico
    for ronda in range(6):
        for i, local in enumerate(equipos):
            visitante = equipos[(i + 1 + ronda) % len(equipos)]
            if local == visitante:
                continue
            gl, gv = rnd.randint(0, 4), rnd.randint(0, 4)
            filas.append((fecha.strftime("%Y-%m-%d"), local, visitante, gl, gv, "E0", "2023"))
            fecha += pd.Timedelta(days=3)
    return _construir_df(filas)


@pytest.fixture
def df_dos_temporadas():
    """Mismo esquema, pero con partidos repartidos en temporada 2022 y 2023."""
    equipos = ["A", "B", "C", "D"]
    filas = []
    fecha = pd.Timestamp("2022-08-01")
    rnd = random.Random(7)
    temporada = "2022"
    for ronda in range(8):
        if ronda == 4:
            temporada = "2023"
            fecha = pd.Timestamp("2023-08-01")
        for i, local in enumerate(equipos):
            visitante = equipos[(i + 1 + ronda) % len(equipos)]
            if local == visitante:
                continue
            gl, gv = rnd.randint(0, 4), rnd.randint(0, 4)
            filas.append((fecha.strftime("%Y-%m-%d"), local, visitante, gl, gv, "E0", temporada))
            fecha += pd.Timedelta(days=3)
    return _construir_df(filas)


# ---------------------------------------------------------------------------
# Test 1: sin partidos futuros (la prueba de propiedad central)
# ---------------------------------------------------------------------------

def test_sin_partidos_futuros(df_basico):
    indice = IndiceHistorico(df_basico)
    rnd = random.Random(123)
    filas = df_basico.to_dict("records")

    for _ in range(200):
        fila = rnd.choice(filas)
        equipo = rnd.choice([fila["equipo_local"], fila["equipo_visitante"]])
        rol = Rol.LOCAL if equipo == fila["equipo_local"] else Rol.VISITANTE
        fecha_corte = fila["fecha"]

        hist = indice.historico_equipo(equipo, "E0", rol, fecha_corte, fila["temporada"])
        # No podemos comparar fechas directamente porque HistoricoEquipo no
        # expone las fechas individuales de cada partido, así que verificamos
        # vía fecha_mas_reciente, que es la más "peligrosa" (la más cercana al corte)
        if hist.fecha_mas_reciente is not None:
            assert hist.fecha_mas_reciente < fecha_corte, (
                f"LEAKAGE: histórico de {equipo} incluye fecha "
                f"{hist.fecha_mas_reciente} >= corte {fecha_corte}"
            )


# ---------------------------------------------------------------------------
# Test 2: excluye el mismo día
# ---------------------------------------------------------------------------

def test_excluye_mismo_dia():
    filas = [
        ("2023-08-01", "A", "B", 1, 0, "E0", "2023"),
        ("2023-08-01", "A", "C", 2, 2, "E0", "2023"),  # mismo día, A como local otra vez
        ("2023-08-08", "A", "D", 0, 0, "E0", "2023"),
    ]
    df = _construir_df(filas)
    indice = IndiceHistorico(df)

    fecha_corte = pd.Timestamp("2023-08-01")
    hist = indice.historico_equipo("A", "E0", Rol.LOCAL, fecha_corte, "2023")
    assert hist.n_encontrados == 0, "No debería incluir ningún partido del mismo día del corte"


# ---------------------------------------------------------------------------
# Test 3: separación estricta de rol
# ---------------------------------------------------------------------------

def test_separacion_estricta_de_rol(df_basico):
    indice = IndiceHistorico(df_basico)
    fecha_corte = df_basico["fecha"].max() + pd.Timedelta(days=1)

    hist_local = indice.historico_equipo("A", "E0", Rol.LOCAL, fecha_corte, "2023", n_partidos=100)
    hist_visitante = indice.historico_equipo("A", "E0", Rol.VISITANTE, fecha_corte, "2023", n_partidos=100)

    # Verificar contra el DataFrame crudo: todos los partidos que el histórico
    # de "local" atribuye a A deben corresponder a filas donde A es equipo_local
    partidos_reales_como_local = df_basico[df_basico["equipo_local"] == "A"]
    partidos_reales_como_visitante = df_basico[df_basico["equipo_visitante"] == "A"]

    assert hist_local.n_encontrados == len(partidos_reales_como_local)
    assert hist_visitante.n_encontrados == len(partidos_reales_como_visitante)


# ---------------------------------------------------------------------------
# Test 4: orden cronológico ascendente
# ---------------------------------------------------------------------------

def test_orden_cronologico_ascendente(df_basico):
    indice = IndiceHistorico(df_basico)
    fecha_corte = df_basico["fecha"].max() + pd.Timedelta(days=1)

    # Reconstruir fechas desde las entradas internas vía un acceso controlado:
    # usamos el propio método interno para verificar orden (aceptable en un
    # test de caja blanca de este módulo)
    entradas, idx_corte = indice._candidatos("A", "E0", Rol.LOCAL, fecha_corte)
    fechas = [e.fecha for e in entradas[:idx_corte]]
    assert fechas == sorted(fechas), "Las entradas internas deben estar en orden ascendente"


# ---------------------------------------------------------------------------
# Test 5: N partidos exactos cuando hay suficientes
# ---------------------------------------------------------------------------

def test_n_partidos_exactos_cuando_hay_suficientes(df_basico):
    indice = IndiceHistorico(df_basico)
    fecha_corte = df_basico["fecha"].max() + pd.Timedelta(days=1)

    hist = indice.historico_equipo("A", "E0", Rol.LOCAL, fecha_corte, "2023", n_partidos=3)
    total_disponible = len(df_basico[df_basico["equipo_local"] == "A"])
    assert total_disponible >= 3, "El fixture debe tener suficientes partidos para esta prueba"
    assert hist.n_encontrados == 3
    assert hist.confianza_historico == "alta"


# ---------------------------------------------------------------------------
# Test 6: histórico parcial marca insuficiente
# ---------------------------------------------------------------------------

def test_historico_parcial_marca_insuficiente():
    filas = [(f"2023-08-{i:02d}", "A", "B", 1, 1, "E0", "2023") for i in range(1, 4)]
    df = _construir_df(filas)
    indice = IndiceHistorico(df)
    fecha_corte = pd.Timestamp("2023-09-01")

    hist = indice.historico_equipo("A", "E0", Rol.LOCAL, fecha_corte, "2023",
                                    n_partidos=15, n_minimo=10)
    assert hist.n_encontrados == 3
    assert hist.suficiente is False
    assert hist.confianza_historico == "baja"


# ---------------------------------------------------------------------------
# Test 7: equipo debutante, histórico vacío
# ---------------------------------------------------------------------------

def test_equipo_debutante_historico_vacio(df_basico):
    indice = IndiceHistorico(df_basico)
    fecha_corte = df_basico["fecha"].max() + pd.Timedelta(days=1)

    hist = indice.historico_equipo("EQUIPO_NUEVO", "E0", Rol.LOCAL, fecha_corte, "2023")
    assert hist.n_encontrados == 0
    assert hist.partidos == []
    assert hist.suficiente is False
    assert hist.fecha_mas_antigua is None
    assert hist.fecha_mas_reciente is None


# ---------------------------------------------------------------------------
# Test 8: filtrado por liga
# ---------------------------------------------------------------------------

def test_filtrado_por_liga():
    filas = [
        ("2023-08-01", "A", "B", 2, 0, "E0", "2023"),
        ("2023-08-08", "A", "B", 1, 1, "E1", "2023"),  # mismo par, otra liga
        ("2023-08-15", "A", "B", 3, 0, "E0", "2023"),
    ]
    df = _construir_df(filas)
    indice = IndiceHistorico(df)
    fecha_corte = pd.Timestamp("2023-09-01")

    hist_e0 = indice.historico_equipo("A", "E0", Rol.LOCAL, fecha_corte, "2023", n_partidos=10)
    hist_e1 = indice.historico_equipo("A", "E1", Rol.LOCAL, fecha_corte, "2023", n_partidos=10)

    assert hist_e0.n_encontrados == 2
    assert hist_e1.n_encontrados == 1


# ---------------------------------------------------------------------------
# Test 9: cruzar_temporadas activado vs desactivado
# ---------------------------------------------------------------------------

def test_cruzar_temporadas_activado_vs_desactivado(df_dos_temporadas):
    indice = IndiceHistorico(df_dos_temporadas)
    # Corte cerca del inicio de la temporada 2023, cuando aún no hay 15
    # partidos disponibles solo en esa temporada
    partidos_2023 = df_dos_temporadas[df_dos_temporadas["temporada"] == "2023"]
    fecha_corte = partidos_2023["fecha"].iloc[2]  # tras solo 2 fechas de la nueva temporada

    hist_sin_cruzar = indice.historico_equipo("A", "E0", Rol.LOCAL, fecha_corte, "2023",
                                               n_partidos=15, n_minimo=10, cruzar_temporadas=False)
    hist_cruzando = indice.historico_equipo("A", "E0", Rol.LOCAL, fecha_corte, "2023",
                                             n_partidos=15, n_minimo=10, cruzar_temporadas=True)

    assert hist_cruzando.n_encontrados >= hist_sin_cruzar.n_encontrados
    assert hist_sin_cruzar.incluye_temporada_anterior is False
    if hist_cruzando.n_encontrados > hist_sin_cruzar.n_encontrados:
        assert hist_cruzando.incluye_temporada_anterior is True


# ---------------------------------------------------------------------------
# Test 10: equivalencia contra filtrado ingenuo (fuerza bruta)
# ---------------------------------------------------------------------------

def _historico_fuerza_bruta(df, equipo, liga, rol, fecha_corte, temporada,
                             n_partidos, cruzar_temporadas):
    columna_equipo = "equipo_local" if rol == Rol.LOCAL else "equipo_visitante"
    candidatos = df[
        (df[columna_equipo] == equipo) &
        (df["liga"] == liga) &
        (df["fecha"] < fecha_corte)
    ]
    if not cruzar_temporadas:
        candidatos = candidatos[candidatos["temporada"] == temporada]
    candidatos = candidatos.sort_values("fecha")
    return candidatos.tail(n_partidos)


def test_equivalencia_contra_filtrado_ingenuo(df_dos_temporadas):
    indice = IndiceHistorico(df_dos_temporadas)
    rnd = random.Random(99)
    filas = df_dos_temporadas.to_dict("records")

    for _ in range(100):
        fila = rnd.choice(filas)
        equipo = rnd.choice([fila["equipo_local"], fila["equipo_visitante"]])
        rol = Rol.LOCAL if equipo == fila["equipo_local"] else Rol.VISITANTE
        cruzar = rnd.choice([True, False])

        hist_optimizado = indice.historico_equipo(
            equipo, "E0", rol, fila["fecha"], fila["temporada"],
            n_partidos=15, n_minimo=10, cruzar_temporadas=cruzar
        )
        esperado = _historico_fuerza_bruta(
            df_dos_temporadas, equipo, "E0", rol, fila["fecha"], fila["temporada"],
            15, cruzar
        )
        assert hist_optimizado.n_encontrados == len(esperado), (
            f"Discrepancia para {equipo} rol={rol} fecha={fila['fecha']} cruzar={cruzar}: "
            f"optimizado={hist_optimizado.n_encontrados}, fuerza_bruta={len(esperado)}"
        )


# ---------------------------------------------------------------------------
# Test 11: error si el DataFrame no está ordenado
# ---------------------------------------------------------------------------

def test_error_si_dataframe_no_ordenado():
    filas = [
        ("2023-08-15", "A", "B", 1, 0, "E0", "2023"),
        ("2023-08-01", "C", "D", 2, 2, "E0", "2023"),  # fuera de orden
    ]
    df = pd.DataFrame(filas, columns=[
        "fecha", "equipo_local", "equipo_visitante",
        "goles_local", "goles_visitante", "liga", "temporada"
    ])
    df["fecha"] = pd.to_datetime(df["fecha"])
    # NO ordenamos a propósito

    with pytest.raises(ErrorDataFrameNoOrdenado):
        IndiceHistorico(df)


# ---------------------------------------------------------------------------
# Test 12: metadata consistente
# ---------------------------------------------------------------------------

def test_metadata_consistente(df_basico):
    indice = IndiceHistorico(df_basico)
    fecha_corte = df_basico["fecha"].max() + pd.Timedelta(days=1)

    hist = indice.historico_equipo("A", "E0", Rol.LOCAL, fecha_corte, "2023", n_partidos=5)
    assert hist.n_encontrados == len(hist.partidos)
    if hist.partidos:
        assert hist.fecha_mas_antigua <= hist.fecha_mas_reciente


# ---------------------------------------------------------------------------
# Test 13: pureza / determinismo
# ---------------------------------------------------------------------------

def test_pureza_determinismo(df_basico):
    indice = IndiceHistorico(df_basico)
    fecha_corte = df_basico["fecha"].max() + pd.Timedelta(days=1)

    hist1 = indice.historico_equipo("A", "E0", Rol.LOCAL, fecha_corte, "2023", n_partidos=5)
    hist2 = indice.historico_equipo("A", "E0", Rol.LOCAL, fecha_corte, "2023", n_partidos=5)

    assert hist1.n_encontrados == hist2.n_encontrados
    assert hist1.partidos == hist2.partidos
    assert hist1.fecha_mas_antigua == hist2.fecha_mas_antigua


# ---------------------------------------------------------------------------
# Test 14: resultado_real no contamina el histórico
# ---------------------------------------------------------------------------

def test_resultado_real_no_contamina_historico(df_basico):
    indice = IndiceHistorico(df_basico)
    fecha_partido = df_basico["fecha"].iloc[10]
    fila = df_basico.iloc[10]

    contexto_sin_resultado = construir_contexto_partido(
        indice, fecha_partido, fila["equipo_local"], fila["equipo_visitante"],
        fila["liga"], fila["temporada"], resultado_real=None,
    )
    contexto_con_resultado = construir_contexto_partido(
        indice, fecha_partido, fila["equipo_local"], fila["equipo_visitante"],
        fila["liga"], fila["temporada"],
        resultado_real=(fila["goles_local"], fila["goles_visitante"]),
    )

    assert contexto_sin_resultado.historico_local.partidos == contexto_con_resultado.historico_local.partidos
    assert contexto_sin_resultado.historico_visitante.partidos == contexto_con_resultado.historico_visitante.partidos
    assert contexto_sin_resultado.resultado_real is None
    assert contexto_con_resultado.resultado_real == (fila["goles_local"], fila["goles_visitante"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
