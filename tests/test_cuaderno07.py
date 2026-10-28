"""Cuaderno 07: capa anti-gradiente (inversión de gradiente)."""

import pytest

torch = pytest.importorskip("torch")

from laboratorio.banco.piezas import PIEZAS, Red  # noqa: E402
from laboratorio.banco.piezas_07 import _InvertirGradiente  # noqa: E402


def test_invierte_el_gradiente_sin_cambiar_los_datos():
    x = torch.randn(3, 4, requires_grad=True)
    y = _InvertirGradiente.apply(x, 1.0)
    assert torch.equal(y, x)
    y.sum().backward()
    assert torch.equal(x.grad, -torch.ones(3, 4))


def test_el_extractor_aprende_al_reves():
    """Con la GRL, un paso de gradiente sube la pérdida a través del extractor (lo que muestra el banco)."""
    torch.manual_seed(0)
    red = Red(PIEZAS["anti_gradiente"], 2, 1, ocultas=8)
    x, y = torch.randn(64, 2), torch.randn(64, 1)
    perdida = ((red(x) - y) ** 2).mean()
    perdida.backward()
    normal = Red(PIEZAS["estandar"], 2, 1, ocultas=8)
    normal.load_state_dict({k.replace("topologia.", "").replace("capa1.", "capa1.lineal.").replace("capa2.", "capa2.lineal."): v
                            for k, v in red.state_dict().items()})
    ((normal(x) - y) ** 2).mean().backward()
    assert torch.allclose(red.topologia.capa1.weight.grad, -normal.capa1.lineal.weight.grad, atol=1e-6)
    assert torch.allclose(red.topologia.salida.weight.grad, normal.salida.weight.grad, atol=1e-6)
