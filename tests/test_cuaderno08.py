"""Cuaderno 08: votación ponderada (clásicos) y la tarea caótica 20D."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from laboratorio.banco.clasicos import _votacion_ponderada  # noqa: E402
from laboratorio.banco.tareas import TAREAS, cargar_tarea  # noqa: E402


def test_tarea_caotica():
    assert "caotica_20d" in TAREAS
    t = cargar_tarea("caotica_20d")
    assert t.entradas == 20 and t.salidas == 4 and len(t.x_entreno) + len(t.x_prueba) == 4000
    assert set(np.unique(t.y_entreno.numpy())) == {0, 1, 2, 3}
    # Determinista: misma semilla de datos, mismos datos.
    from laboratorio.banco.tareas import caotica_20d
    assert np.array_equal(caotica_20d().x_entreno.numpy(), t.x_entreno.numpy())


def test_votacion_ponderada_pesos_y_prediccion():
    from sklearn.datasets import make_moons

    x, y = make_moons(300, noise=0.2, random_state=0)
    m = _votacion_ponderada(0).fit(x, y)
    assert len(m.pesos_) == 5 and abs(sum(m.pesos_) - 1) < 1e-9
    assert (m.predict(x) == y).mean() > 0.8


def test_stacking_del_cuaderno_completo():
    from sklearn.datasets import make_moons

    from laboratorio.banco.clasicos import _stacking

    x, y = make_moons(300, noise=0.2, random_state=0)
    assert (_stacking(0).fit(x, y).predict(x) == y).mean() > 0.8
