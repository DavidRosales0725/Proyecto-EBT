"""
tipos_comunes.py

Tipos de datos compartidos entre las capas de Preparación (data_loader.py,
historicos.py) y Motor (core/). Deliberadamente sin lógica de negocio --
solo definiciones de datos, para que ninguna capa tenga que importar de
la otra y así se respete la dirección de dependencia de la arquitectura
(Datos -> Preparación -> Motor -> Decisión -> Evaluación).

ResultadoPartido vivía originalmente en core/fuerzas.py; se movió aquí
por decisión de diseño explícita al aprobar historicos.py, para que
historicos.py (Preparación) pueda producirlo sin depender de core/ (Motor).
core/fuerzas.py lo re-exporta para no romper compatibilidad con código
existente.
"""
from dataclasses import dataclass


@dataclass
class ResultadoPartido:
    goles_a_favor: int
    goles_en_contra: int
