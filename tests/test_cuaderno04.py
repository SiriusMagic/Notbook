"""Cuaderno 04: decorrelación (AODC), papelito preguntón (ODFE/RDCR/NIDI/RCF) y celda recurrente RCF."""

import pytest

torch = pytest.importorskip("torch")

from laboratorio.banco.arquitecturas import INVESTIGACION, crear_modelo  # noqa: E402
from laboratorio.banco.evaluar import Config, correr  # noqa: E402
from laboratorio.banco.piezas import PIEZAS, Red  # noqa: E402
from laboratorio.banco.piezas_04 import covarianza_fuera_diagonal  # noqa: E402
from laboratorio.banco.tareas import cargar_tarea  # noqa: E402


def test_covarianza_fuera_diagonal():
    torch.manual_seed(0)
    independientes = torch.randn(2000, 3)
    copiadas = independientes[:, :1].repeat(1, 3)
    assert covarianza_fuera_diagonal(independientes) < 0.05 < covarianza_fuera_diagonal(copiadas)


@pytest.mark.parametrize("pieza,con_perdida", [("decorrelacion", True), ("papelito_pregunton", True),
                                               ("papelito_covarianza", True), ("fusion_reciproca", False)])
def test_perdida_extra(pieza, con_perdida):
    torch.manual_seed(0)
    red = Red(PIEZAS[pieza], 4, 2, ocultas=6)
    red.train()
    salida = red(torch.randn(16, 4))
    extra = red.perdida_extra(salida, None)
    assert (extra.item() > 0) == con_perdida
    if con_perdida:
        extra.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in red.parameters())


def test_arquitecturas_de_investigacion():
    assert "cnn_decorrelada" in INVESTIGACION["imagen"] and "rcf_recurrente" in INVESTIGACION["serie"]
    t = cargar_tarea("digitos")
    m = crear_modelo(PIEZAS["estandar"], t, "cnn_decorrelada", 8)
    m.train()
    salida = m(t.x_entreno[:8])
    assert salida.shape == (8, 10) and m.perdida_extra(salida, None).item() > 0
    s = cargar_tarea("serie_seno")
    r = crear_modelo(PIEZAS["ida"], s, "rcf_recurrente", 8)
    assert r(s.x_prueba[:3]).shape == (3, 1)


@pytest.mark.parametrize("pieza", ["decorrelacion", "papelito_pregunton"])
def test_corren_en_el_banco(pieza):
    corr = correr(pieza, "lunas", config=Config(epocas=150, semillas=(0,), medir=False))
    assert not corr[0].fallo and corr[0].valor > 0.6


def test_cnn_decorrelada_mide_despues_de_la_activacion():
    """La covarianza se mide sobre lo que pasa a la capa siguiente (después de BatchNorm y ReLU)."""
    t = cargar_tarea("digitos")
    m = crear_modelo(PIEZAS["estandar"], t, "cnn_decorrelada", 8)
    m.train()
    m(t.x_entreno[:8])
    assert len(m.extractor.mapas) == 2 and all((mapa >= 0).all() for mapa in m.extractor.mapas)  # ya pasaron por ReLU
