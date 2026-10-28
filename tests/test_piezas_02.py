"""Piezas del cuaderno 02 (tanda 2) y sus rivales de la industria."""

import pytest

torch = pytest.importorskip("torch")

from laboratorio.banco.evaluar import Config, correr  # noqa: E402
from laboratorio.banco.medicion import capas_observadas  # noqa: E402
from laboratorio.banco.piezas import PIEZAS, Red  # noqa: E402

CON_REGLA = ["memoria_errores", "memoria_gradiente", "lr_por_neurona", "memoria_atencion", "memoria_embedding",
             "memoria_atencion_embedding"]


def test_industria_marcada():
    industria = {n for n, p in PIEZAS.items() if p.industria}
    assert industria == {"gelu", "silu", "mish", "elu", "leaky_relu", "tanh", "residual"}
    assert all(PIEZAS[n].activacion is not None for n in industria - {"residual"})


@pytest.mark.parametrize("pieza", CON_REGLA)
def test_la_regla_extra_cambia_los_pesos(pieza):
    """Después del paso del optimizador, la regla propia de la pieza mueve los pesos (no es decorativa)."""
    torch.manual_seed(0)
    red = Red(PIEZAS[pieza], 4, 2, ocultas=8)
    x, y = torch.randn(32, 4), torch.randn(32, 2)
    for _ in range(2):  # dos pasos: las memorias necesitan algo guardado
        salida = red(x)
        perdida = (salida - y).pow(2).mean()
        red.zero_grad()
        perdida.backward()
        antes = {k: v.clone() for k, v in red.state_dict().items()}
        red.despues_del_paso(perdida=perdida.item(), error=(salida - y).detach(), epoca=0)
    cambiaron = [k for k, v in red.state_dict().items() if not torch.equal(v, antes[k])]
    assert cambiaron, f"{pieza}: la regla extra no cambió nada"


@pytest.mark.parametrize("pieza", ["denso_cruzada", "densa_entrada"])
def test_topologias(pieza):
    red = Red(PIEZAS[pieza], 5, 3, ocultas=6)
    assert red(torch.randn(4, 5)).shape == (4, 3)
    assert set(capas_observadas(red)) == {"capa1", "capa2"}
    # La segunda capa recibe la primera (y, en densa_entrada, también la entrada original).
    assert red.topologia.capa2.in_features == (6 + 5 if pieza == "densa_entrada" else 6)


@pytest.mark.parametrize("pieza", ["gelu", "memoria_errores", "denso_cruzada"])
def test_corren_en_el_banco(pieza):
    corr = correr(pieza, "circulos", config=Config(epocas=150, semillas=(0,), medir=False))
    assert not corr[0].fallo and corr[0].valor > 0.6


def test_grupo_industria_en_la_linea_de_comandos(tmp_path):
    from laboratorio.banco.__main__ import main

    main(["--grupo", "industria", "--tarea", "xor", "--semillas", "2", "--epocas", "2", "--salida", str(tmp_path)])
    texto = (tmp_path / "informe.md").read_text(encoding="utf-8")
    assert "gelu (industria)" in texto and "memoria_errores" not in texto and "| ida |" not in texto


@pytest.mark.parametrize("pieza", ["memoria_embedding", "memoria_atencion_embedding"])
def test_memoria_embedding_no_depende_del_lote(pieza):
    torch.manual_seed(0)
    red = Red(PIEZAS[pieza], 4, 2, ocultas=8)
    x = torch.randn(6, 4)
    red.despues_del_paso(perdida=0.7, error=None, epoca=0)  # algo en la memoria
    red.eval()
    with torch.no_grad():
        juntos = red(x)
        separados = torch.cat([red(x[i : i + 1]) for i in range(len(x))])
    assert torch.allclose(juntos, separados, atol=1e-6)


def test_ruido_perlin_suave_y_acotado():
    import numpy as np

    from laboratorio.inicializadores import pesos_perlin, ruido_perlin

    rng = np.random.default_rng(0)
    r = ruido_perlin(rng.random(5000) * 40, rng.random(5000) * 40)
    assert np.abs(r).max() <= 1.0 + 1e-9 and r.std() > 0.1
    assert np.allclose(ruido_perlin(np.arange(5.0), np.arange(5.0)), 0)  # vale 0 en los puntos de la cuadrícula
    w = pesos_perlin((16, 8))
    assert w.shape == (16, 8) and np.corrcoef(w.ravel()[:-1], w.ravel()[1:])[0, 1] > 0.8  # vecinos parecidos


@pytest.mark.parametrize("pieza", ["inicio_perlin", "inicio_lorenz"])
def test_inicios_especiales_repetibles_y_distintos_por_semilla(pieza):
    def pesos(semilla):
        torch.manual_seed(semilla)
        return Red(PIEZAS[pieza], 4, 2, ocultas=8).capa1.lineal.weight.detach().clone()

    assert torch.equal(pesos(0), pesos(0)) and not torch.equal(pesos(0), pesos(1))
