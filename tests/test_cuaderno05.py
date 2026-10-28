"""Cuaderno 05: atención suave con RL (máscara en un Transformer), AttentiveLinear y atención local."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from laboratorio.banco.arquitecturas import INVESTIGACION, _CapaMascaraRL, crear_modelo  # noqa: E402
from laboratorio.banco.evaluar import Config, correr  # noqa: E402
from laboratorio.banco.piezas import PIEZAS  # noqa: E402
from laboratorio.banco.piezas_05 import AtencionLocal, AtencionSalida  # noqa: E402


def _tarea_texto():
    return SimpleNamespace(vocabulario=50, forma=(10,), dato="texto", salidas=3, extra={})


@pytest.mark.parametrize("arq", ["transformer_mascara_rl", "transformer_mascara_rl_log"])
def test_transformer_con_mascara(arq):
    assert arq in INVESTIGACION["texto"]
    torch.manual_seed(0)
    m = crear_modelo(PIEZAS["estandar"], _tarea_texto(), arq, 8)
    ids = torch.randint(1, 50, (4, 10))
    ids[0, 6:] = 0  # relleno
    m.train()
    m(ids).sum().backward()
    assert m.extractor.capas[0].mascara.grad is not None  # la máscara aprende con el gradiente
    m.eval()
    assert torch.isfinite(m(ids)).all()


def test_mascara_multiplicada_no_bloquea_y_log_si():
    """El fallo de diseño: con la máscara ≈ 0 en una posición, multiplicar no la apaga; sumar log(máscara) sí."""
    torch.manual_seed(0)
    resultados = {}
    for modo in ("original", "log"):
        capa = _CapaMascaraRL(8, 1, 3, modo).eval()
        with torch.no_grad():
            capa.mascara.fill_(0.0)
            capa.mascara[0, :, 2] = -30.0  # nadie debería mirar la posición 2
        capturado = {}
        original_softmax = torch.nn.functional.softmax

        def espia(t, dim=-1, **k):
            r = original_softmax(t, dim=dim, **k)
            if t.dim() == 4:
                capturado["atn"] = r
            return r

        torch.nn.functional.softmax = espia
        try:
            capa(torch.randn(2, 3, 8) * 3, torch.zeros(2, 3, dtype=torch.bool))
        finally:
            torch.nn.functional.softmax = original_softmax
        resultados[modo] = capturado["atn"][..., 2].max().item()
    assert resultados["log"] < 1e-6 < resultados["original"]


def test_atencion_salida_es_una_matriz_fija():
    capa = AtencionSalida(3, 4)
    x = torch.randn(5, 3)
    esperado = torch.relu(capa.lineal(x)) @ torch.sigmoid(capa.atencion)
    assert torch.allclose(capa(x), esperado)


def test_atencion_local_usa_la_perdida_real():
    capa = AtencionLocal(2, 3, memoria=2)
    capa.train()
    capa(torch.ones(4, 2))
    capa.despues_del_paso(perdida=0.5)
    capa(torch.zeros(4, 2))
    capa.despues_del_paso(perdida=2.0)
    assert torch.allclose(capa.errores, torch.tensor([0.5, 2.0]))
    assert torch.allclose(capa.memoria[0], torch.ones(2)) and torch.allclose(capa.memoria[1], torch.zeros(2))


def test_corre_en_el_banco():
    corr = correr("atencion_local", "lunas", config=Config(epocas=150, semillas=(0,), medir=False))
    assert not corr[0].fallo and corr[0].valor > 0.6
