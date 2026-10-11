"""
conftest.py (tests/)

Refuerzo del conftest.py de la raíz del proyecto: si pytest se invoca
apuntando directamente a esta carpeta (ej. `pytest tests/` o
`pytest /ruta/al/proyecto/tests/` desde otro directorio de trabajo),
pytest puede fijar esta misma carpeta como rootdir y no llegar a cargar
el conftest.py de la raíz del proyecto. Este archivo garantiza que el
import de los módulos del proyecto funcione en ambos escenarios.
"""
import sys
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))
