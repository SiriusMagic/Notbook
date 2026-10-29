"""Cuaderno 04: interpolación y predicción de fotogramas."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from laboratorio.banco import video as v  # noqa: E402


def test_clips_se_mueven_y_son_deterministas():
    a, b = v.clips(4, 0), v.clips(4, 0)
    assert a.shape == (4, 5, 3, 32, 32) and np.array_equal(a, b)
    assert 0 <= a.min() and a.max() <= 1
    assert np.abs(a[:, 0] - a[:, 4]).mean() > 0.01  # los dígitos cambian de lugar


def test_metricas():
    x = v.clips(2, 0)[:, 0]
    assert v.psnr(x, x) == 100.0 and v.ssim(x, x) == pytest.approx(1.0, abs=1e-4)
    assert v.ssim(x, 1 - x) < 0.5


@pytest.mark.parametrize("tarea", list(v.MODELOS))
def test_modelos_corren(tarea):
    d = v.clips(8, 0)
    x, y = v._xy(d, tarea)
    for nombre in v.MODELOS[tarea]:
        if nombre in ("copiar", "promedio"):
            assert v.sin_red(nombre, d, tarea).shape == y.shape
            continue
        m = v.entrenar(nombre, x, y, epocas=1, lote=4, ocultas_lstm=8, dispositivo="cpu")
        assert v.predecir(m, x).shape == y.shape
