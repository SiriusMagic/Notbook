import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from laboratorio.banco.piezas import PIEZAS, Red  # noqa: E402
from laboratorio.banco.piezas_reparadas import (  # noqa: E402
    HiperReLUSuave,
    ResonanteSuave,
    datos_dominios,
    entrenar_dann,
    piezas_reparadas,
)


@pytest.mark.parametrize("nombre", [p.nombre for p in piezas_reparadas()])
def test_reparada_forma_y_gradiente(nombre):
    torch.manual_seed(0)
    red = Red(PIEZAS[nombre], 5, 3, ocultas=8)
    red.train()
    x = torch.randn(16, 5)
    y = red(x)
    assert y.shape == (16, 3)
    perdida = y.pow(2).mean() + red.perdida_extra(y, None)
    perdida.backward()
    con_grad = [p for p in red.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(con_grad) >= len(list(red.parameters())) // 2
    assert torch.isfinite(y).all()


def test_resonante_suave_deja_pasar_gradiente_bajo_el_umbral():
    z = torch.tensor([-1.0, 0.0, 0.2], requires_grad=True)
    ResonanteSuave()(z).sum().backward()
    assert (z.grad != 0).all()  # la original daba 0 bajo el umbral


def test_hiper_relu_suave_empieza_sin_amplificar():
    z = torch.tensor([1.0])
    assert abs(HiperReLUSuave()(z).item() - 1.0) < 0.05


def test_dann_datos_y_entrenamiento_corto():
    datos = datos_dominios(0)
    assert datos["origen_x"].shape[1] == 64 and datos["destino_x"].shape[1] == 64
    assert len(datos["prueba_x"]) == len(datos["prueba_y"])
    destino, origen = entrenar_dann(datos, True, 0, epocas=2)
    assert 0.0 <= destino <= 1.0 and 0.0 <= origen <= 1.0


@pytest.mark.parametrize("nombre", ["dendritica_dropout", "rbf_dropout", "dendritica_multi", "rbf_selectiva"])
def test_mejorada_forma_y_gradiente(nombre):
    torch.manual_seed(0)
    red = Red(PIEZAS[nombre], 6, 2, ocultas=8)
    red.train()
    y = red(torch.randn(16, 6))
    assert y.shape == (16, 2) and torch.isfinite(y).all()
    y.pow(2).mean().backward()
    assert all(p.grad is not None for p in red.parameters())
