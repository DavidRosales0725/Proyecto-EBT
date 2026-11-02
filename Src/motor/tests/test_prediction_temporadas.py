import pandas as pd
import pytest

from motor.prediction import (
    evaluar_confianza,
    normalizar_xg,
    peso_temporada_actual,
    predecir_partido,
)


def test_peso_temporada_actual_crece_con_muestras_y_llega_a_uno():
    assert peso_temporada_actual(0) == 0
    assert peso_temporada_actual(1) == pytest.approx(0.7375)
    assert peso_temporada_actual(4) == pytest.approx(0.85)
    assert peso_temporada_actual(7) == pytest.approx(0.9625)
    assert peso_temporada_actual(8) == 1.0


def test_normalizacion_xg_acota_extremos_y_recupera_valores_no_finitos():
    assert normalizar_xg(4.5) == 3.5
    assert normalizar_xg(2.1) == 2.1
    assert normalizar_xg(float("nan"), 1.4) == 1.4


def test_indicador_confianza_usa_el_rol_menos_cubierto_y_menciona_mezcla():
    assert evaluar_confianza(8, 10)["nivel"] == "Alta"
    moderada = evaluar_confianza(7, 4)
    assert moderada["nivel"] == "Moderada"
    baja = evaluar_confianza(3, 4, partidos_local_anterior=10, partidos_visitante_anterior=9)
    assert baja["nivel"] == "Baja"
    assert "histórico anterior" in baja["detalle"]
    assert baja["partidos_local_anterior"] == "10"


def test_pronostico_acepta_temporadas_combinadas_y_reporta_muestras():
    filas = []
    for i, (fecha, temporada, gf, ga) in enumerate([
        ("2024-08-01", 2024, 1, 0),
        ("2024-09-01", 2024, 2, 0),
        ("2024-10-01", 2024, 1, 1),
        ("2025-08-01", 2025, 3, 0),
    ]):
        filas.append({"fecha": fecha, "_temporada": temporada,
                      "equipo_local": "Arsenal", "equipo_visitante": f"Rival{i}",
                      "goles_local": gf, "goles_visitante": ga})
        filas.append({"fecha": fecha, "_temporada": temporada,
                      "equipo_local": f"RivalAway{i}", "equipo_visitante": "Chelsea",
                      "goles_local": 0, "goles_visitante": gf})

    pronostico = predecir_partido(
        pd.DataFrame(filas), "Arsenal", "Chelsea", "2025-10-01",
        ventana=10, iteraciones=100,
    )

    assert pronostico.partidos_local == 4
    assert pronostico.partidos_visitante == 4
    assert pronostico.partidos_local_temporada_actual == 1
    assert pronostico.partidos_visitante_temporada_actual == 1
    assert pronostico.partidos_local_temporada_anterior == 3
    assert pronostico.partidos_visitante_temporada_anterior == 3
    assert pronostico.temporadas_combinadas
    assert pronostico.lambda_local > 0
    assert pronostico.lambda_visitante > 0
