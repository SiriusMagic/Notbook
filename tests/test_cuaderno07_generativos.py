"""Cuaderno 07: D-ReG y HCMG."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from laboratorio.banco import generativos_07 as g  # noqa: E402


@pytest.mark.parametrize("nombre", ["dreg_cuaderno", "dreg_arreglado", "gru", "lstm", "transformer"])
def test_modelos_de_texto(nombre):
    xe = np.random.default_rng(0).integers(4, 30, (16, 10))
    m = g.entrenar_texto(nombre, xe, 30, pasos=2, lote=4)
    assert m(torch.as_tensor(xe[:2])).shape == (2, 10, 30)
    assert g.perplejidad_texto(m, xe[:4], 30) > 1


@pytest.mark.parametrize("nombre", ["hcmg", "hcmg_sin_condicion", "unet_condiciones"])
def test_hcmg(nombre):
    x, c, k, y = g.datos_hcmg(4, 0)
    assert x.shape == y.shape == (4, 3, 16, 16) and c.shape == (4, 10) and k.shape == (4, 3)
    out = g._modelo_imagen(nombre)(x, c, k)
    assert out.shape == y.shape and out.abs().max() <= 1
