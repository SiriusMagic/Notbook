"""Mediciones de hardware y de lo que pasa dentro de la red."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from torch import nn  # noqa: E402

from laboratorio.banco.evaluar import Config, entrenar_una  # noqa: E402
from laboratorio.banco.medicion import cka_entre, cka_lineal, medir_hardware, medir_internas  # noqa: E402
from laboratorio.banco.piezas import PIEZAS, Red  # noqa: E402


def test_cka_propiedades():
    torch.manual_seed(0)
    a = torch.randn(200, 10)
    q, _ = torch.linalg.qr(torch.randn(10, 10))
    assert cka_lineal(a, a) == pytest.approx(1.0)
    assert cka_lineal(a, a @ q * 3 + 5) == pytest.approx(1.0, abs=1e-6)  # no cambia con rotación, escala ni desplazamiento
    assert cka_lineal(a, torch.randn(200, 10)) < 0.2  # ruido sin relación: parecido bajo


def test_neuronas_muertas_se_detectan():
    torch.manual_seed(0)
    red = Red(PIEZAS["estandar"], 4, 2, ocultas=10)
    with torch.no_grad():
        red.capa1.lineal.bias[:5] = -1000  # 5 de 10 neuronas siempre en 0 tras ReLU
    m = medir_internas(red, torch.randn(300, 4), torch.randint(0, 2, (300,)), nn.functional.cross_entropy)
    assert m["capa1_muertas"] == 0.5 and m["capa1_inmoviles"] >= 0.5
    assert m["capa1_grad"] > 0 and "capa2_norma" in m


def test_hardware():
    red = Red(PIEZAS["estandar"], 4, 2, ocultas=10)
    m = medir_hardware(red, torch.randn(300, 4), tokens_por_ejemplo=12, repeticiones=5)
    assert m["latencia_ms"] > 0 and m["latencia_p95_ms"] >= m["latencia_ms"]
    assert m["ejemplos_por_s"] > 0 and m["tokens_por_s"] == pytest.approx(m["ejemplos_por_s"] * 12, rel=1e-3)
    params = sum(p.numel() for p in red.parameters())
    assert m["huella_mb"] == pytest.approx(params * 4 / 2**20, abs=1e-4)


def test_entrenar_mide_y_devuelve_modelo():
    c, (modelo, x) = entrenar_una("ida", "digitos", 0, Config(epocas=5, arquitectura="cnn", lote=128), devolver_modelo=True)
    for clave in ("latencia_ms", "ejemplos_por_s", "huella_mb", "capa1_muertas", "extractor_grad", "eficiencia"):
        assert clave in c.medidas, clave
    assert set(cka_entre(modelo, modelo, x)) == {"extractor", "capa1", "capa2"}
    sin = entrenar_una("ida", "digitos", 0, Config(epocas=5, medir=False))
    assert sin.medidas == {}
