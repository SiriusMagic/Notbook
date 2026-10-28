"""Cuaderno 02, segunda parte: polinómica, resonante, DMNA, GMO, CDM, CCN y rivales (Dropout, BatchNorm, LayerNorm)."""

import pytest

torch = pytest.importorskip("torch")

from laboratorio.banco.evaluar import Config, correr  # noqa: E402
from laboratorio.banco.piezas import PIEZAS, Red  # noqa: E402
from laboratorio.banco.piezas_02b import (  # noqa: E402
    ActivacionResonante,
    CapaPolinomica,
    HiperReLU,
    NormaCaosAdaptativa,
    RedGMO,
)

NUEVAS = ["dropout", "batchnorm", "layernorm", "polinomica", "resonante", "dmna", "gmo", "cdm", "hiper_relu",
          "norma_caos", "ccn"]


def test_registradas_y_marcadas():
    assert set(NUEVAS) <= set(PIEZAS)
    assert {n for n in NUEVAS if PIEZAS[n].industria} == {"dropout", "batchnorm", "layernorm"}


def test_polinomica_capta_un_producto_que_una_neurona_lineal_no_puede():
    """y = x0·x1 es un término cuadrático: una sola capa polinómica lo aprende casi perfecto."""
    torch.manual_seed(0)
    x = torch.randn(256, 2)
    y = (x[:, 0] * x[:, 1]).unsqueeze(1)
    capa = CapaPolinomica(2, 1)
    with torch.no_grad():
        capa.raw_alpha.fill_(-20.0)  # solo tanh, casi lineal cerca de 0
    opt = torch.optim.Adam(capa.parameters(), lr=0.05)
    for _ in range(300):
        perdida = ((torch.atanh(capa(x).clamp(-0.999, 0.999)) - y) ** 2).mean()
        opt.zero_grad()
        perdida.backward()
        opt.step()
    assert perdida.item() < 0.05


def test_polinomica_rango_bajo_con_muchas_entradas():
    capa = CapaPolinomica(200, 4)
    assert not capa.completa and capa(torch.randn(3, 200)).shape == (3, 4)


def test_resonante_y_hiper_relu():
    z = torch.tensor([-1.0, 0.05, 0.4, 0.6, 2.0])
    r = ActivacionResonante()(z)
    assert torch.equal(r[:3], torch.zeros(3)) and torch.allclose(r[3:], torch.sigmoid(z[3:]))
    h = HiperReLU()(z)
    assert torch.allclose(h, torch.tensor([-0.01, 0.05, 4.0, 6.0, 20.0]))


def test_norma_caos_igual_al_entrenar_y_al_evaluar():
    """Arreglo: con las mismas estadísticas, entrenar y evaluar dan la misma escala."""
    torch.manual_seed(0)
    norma = NormaCaosAdaptativa(3, decaimiento=0.0)  # las estadísticas acumuladas = las del último lote
    x = torch.randn(64, 3) * 5 + 2
    entreno = norma(x)
    norma.eval()
    assert torch.allclose(entreno, norma(x), atol=1e-5)
    norma.train()
    assert norma(x[:1]).shape == (1, 3)  # lote de 1: no falla


def test_gmo_compara_con_el_mismo_lote_y_solo_acepta_si_mejora():
    torch.manual_seed(0)
    red = RedGMO(2, 1, 4, probabilidad=1.0)
    antes = {k: v.clone() for k, v in red.state_dict().items()}
    red.despues_del_paso(perdida_de=lambda: 1.0)  # misma pérdida antes y después: se rechaza
    assert all(torch.equal(v, antes[k]) for k, v in red.state_dict().items()) and red.mutaciones == 0
    valores = iter([1.0, 0.5])  # antes 1,0 en este lote; mutada 0,5 → se acepta
    red.despues_del_paso(perdida_de=lambda: next(valores))
    assert red.mutaciones == 1 and any(not torch.equal(v, antes[k]) for k, v in red.state_dict().items())


def test_perdida_de_evalua_el_modelo_entero_sin_dropout():
    from laboratorio.banco.evaluar import _perdida_en_evaluacion
    from laboratorio.banco.tareas import cargar_tarea

    t = cargar_tarea("circulos")
    modelo = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Dropout(0.5), torch.nn.Linear(8, 2))
    modelo.train()
    a = _perdida_en_evaluacion(t, modelo, t.x_prueba, t.y_prueba)
    assert a == _perdida_en_evaluacion(t, modelo, t.x_prueba, t.y_prueba)  # sin azar del dropout
    assert modelo.training  # vuelve al modo en que estaba


def test_cdm_tiene_perdida_del_extractor():
    torch.manual_seed(0)
    red = Red(PIEZAS["cdm"], 3, 2, ocultas=4)
    salida = red(torch.randn(8, 3))
    extra = red.perdida_extra(salida, None)
    assert extra.item() > 0
    extra.backward()
    assert red.topologia.extractor.weight.grad is not None


@pytest.mark.parametrize("pieza", NUEVAS)
def test_corren_en_el_banco(pieza):
    corr = correr(pieza, "circulos", config=Config(epocas=150, semillas=(0,), medir=False))
    assert not corr[0].fallo and corr[0].valor > 0.6
