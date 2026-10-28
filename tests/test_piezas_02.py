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
