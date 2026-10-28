"""Cuaderno 03: clásicos de scikit-learn, neuronas KNN y redes genéticas."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from laboratorio.banco.clasicos import competir_clasicos, correr_clasicos, informe_clasicos  # noqa: E402
from laboratorio.banco.evaluar import Config, correr  # noqa: E402
from laboratorio.banco.piezas import PIEZAS, Red  # noqa: E402
from laboratorio.banco.piezas_03 import CapaGenetica, CapaKNN  # noqa: E402


def test_clasicos_clasificacion_y_regresion():
    cla = correr_clasicos("lunas", semillas=(0,))
    assert "Gradient Boosting" in cla and cla["Gradient Boosting"][0].valor > 0.85
    reg = correr_clasicos("regresion_cubica", semillas=(0,))
    assert "Naive Bayes" not in reg and reg["Random Forest"][0].valor < 0.05  # MSE


def test_informe_clasicos():
    comps = competir_clasicos(["xor"], semillas=(0, 1), aviso=lambda *_: None)
    texto = informe_clasicos(comps, (0, 1))
    assert "## xor" in texto and "Gradient Boosting" in texto and "Red estándar" in texto


def test_knn_escala_y_aprende_prototipos():
    torch.manual_seed(0)
    capa = CapaKNN(64, 4)
    x = torch.randn(32, 64) * 3
    capa.train()
    a = capa(x)
    assert a.shape == (32, 4) and a.std() > 0.02  # con escala, las activaciones no son todas iguales
    antes = capa.patrones.clone()
    capa.despues_del_paso()
    assert not torch.equal(antes, capa.patrones)


def _red_genetica(modo):
    torch.manual_seed(0)
    capa = CapaGenetica(3, 8, inicio=0, umbral=1, modo=modo)
    x = torch.randn(16, 3)
    with torch.no_grad():
        capa.lineal.weight[0].zero_()  # neurona 0 muerta: nunca se activa
        capa.lineal.bias[0] = -10.0
    return capa, x


def test_genetica_original_copia_neuronas_muertas():
    """El fallo de diseño: las «menos ajustadas» son las muertas, y el original las copia en las vivas."""
    capa, x = _red_genetica("original")
    capa(x).sum().backward()
    capa.despues_del_paso(epoca=1)
    assert capa.reemplazos >= 6  # casi todas las vivas reemplazadas de una vez
    assert (capa(x) > 0).float().mean() < 0.2  # ahora casi todo está apagado


def test_genetica_invertida_recicla_la_muerta():
    capa, x = _red_genetica("invertida")
    capa(x).sum().backward()
    capa.despues_del_paso(epoca=1)
    assert capa.reemplazos == 1 and (capa(x)[:, 0] > 0).any()  # la neurona 0 revivió


@pytest.mark.parametrize("pieza", ["knn_prototipos", "genetica_invertida"])
def test_corren_en_el_banco(pieza):
    corr = correr(pieza, "lunas", config=Config(epocas=150, semillas=(0,), medir=False))
    assert not corr[0].fallo and corr[0].valor > 0.6


def test_registradas():
    assert {"knn_prototipos", "genetica", "genetica_arreglada", "genetica_invertida"} <= set(PIEZAS)
    assert Red(PIEZAS["genetica"], 3, 2, 4)(torch.randn(5, 3)).shape == (5, 2)


def test_genetica_borra_la_inercia_de_adam_de_las_neuronas_reemplazadas():
    capa, x = _red_genetica("invertida")
    opt = torch.optim.Adam(capa.parameters(), lr=0.01)
    capa(x).sum().backward()
    opt.step()
    assert opt.state[capa.lineal.weight]["exp_avg"].abs().sum() > 0
    capa.despues_del_paso(epoca=1, optimizador=opt)
    assert capa.reemplazos == 1
    assert opt.state[capa.lineal.weight]["exp_avg"][0].abs().sum() == 0  # la neurona 0 se reemplazó
    assert opt.state[capa.lineal.weight]["exp_avg_sq"][0].abs().sum() == 0
