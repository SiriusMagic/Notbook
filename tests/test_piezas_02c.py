"""Cuaderno 02, última parte: dendrítica, synaptic tagging, estado interno, ruteo adaptativo y CNN con entrada densa."""

import pytest

torch = pytest.importorskip("torch")

from laboratorio.banco.arquitecturas import INVESTIGACION, crear_modelo  # noqa: E402
from laboratorio.banco.evaluar import Config, correr  # noqa: E402
from laboratorio.banco.piezas import PIEZAS, Red  # noqa: E402
from laboratorio.banco.piezas_02c import CapaDendritica, CapaSynapticTagging  # noqa: E402
from laboratorio.banco.tareas import cargar_tarea  # noqa: E402

NUEVAS = ["dendritica", "synaptic_tagging", "estado_interno", "ruteo_adaptativo"]


def test_dendritica_dos_ramas():
    capa = CapaDendritica(4, 3)
    with torch.no_grad():
        capa.dendrita1.weight.fill_(1.0)
        capa.dendrita2.weight.zero_()
    # rama 1: relu(suma de las 2 primeras); rama 2: exp(0) = 1; sesgo 0 → sigmoide(rama1 + 1)
    x = torch.tensor([[1.0, 2.0, 5.0, -5.0], [-1.0, -2.0, 0.0, 0.0]])
    assert torch.allclose(capa(x), torch.sigmoid(torch.tensor([[4.0] * 3, [1.0] * 3])))
    assert CapaDendritica(1, 2)(torch.randn(5, 1)).shape == (5, 2)  # con 1 entrada no falla


def test_synaptic_tagging_consolida_cada_periodo():
    torch.manual_seed(0)
    capa = CapaSynapticTagging(3, 2, umbral=0.0, periodo=2)
    for paso in range(2):
        capa(torch.randn(8, 3)).sum().backward()
        antes = capa.pesos.detach().clone()
        capa.despues_del_paso()
        cambio = not torch.equal(antes, capa.pesos)
        assert cambio == (paso == 1)  # solo en el paso 2 (periodo) se consolida
        capa.zero_grad()
    assert capa.consolidados > 0


def test_ruteo_y_cnn_densa():
    red = Red(PIEZAS["ruteo_adaptativo"], 5, 3, ocultas=8)
    assert red(torch.randn(4, 5)).shape == (4, 3)
    assert "cnn_densa_entrada" in INVESTIGACION["imagen"]
    t = cargar_tarea("digitos")
    m = crear_modelo(PIEZAS["ida"], t, "cnn_densa_entrada", 8)
    assert m(t.x_prueba[:2]).shape == (2, 10)
    assert m.extractor.conv3.in_channels == 16 + 32 + 1  # recibe conv1, conv2 y la imagen


@pytest.mark.parametrize("pieza", NUEVAS)
def test_corren_en_el_banco(pieza):
    corr = correr(pieza, "circulos", config=Config(epocas=150, semillas=(0,), medir=False))
    assert not corr[0].fallo and corr[0].valor > 0.6
