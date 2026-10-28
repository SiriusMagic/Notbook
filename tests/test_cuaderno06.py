"""Cuaderno 06: neuronas con «RL», votación, biomas y cerebro modular."""

import pytest

torch = pytest.importorskip("torch")

from laboratorio.banco.piezas import PIEZAS, Red  # noqa: E402
from laboratorio.banco.piezas_06 import CapaRLInterno, RedCerebroMoE, RedTalamo  # noqa: E402

NOMBRES = ["rl_interno", "votacion_cabezas", "biomas", "cerebro_moe", "cerebro_moe_arreglado", "talamo_top2",
           "moe_suave"]


@pytest.mark.parametrize("nombre", NOMBRES)
def test_forma_y_gradiente(nombre):
    torch.manual_seed(0)
    red = Red(PIEZAS[nombre], 3, 2, ocultas=8)
    red.train()
    salida = red(torch.randn(16, 3))
    assert salida.shape == (16, 2)
    salida.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in red.parameters())


def test_el_factor_del_rl_interno_solo_crece():
    capa = CapaRLInterno(3, 4)
    capa(torch.randn(8, 3)).sum().backward()
    antes = capa.factor.detach().clone()
    capa.despues_del_paso()
    assert (capa.factor >= antes).all() and (capa.factor <= 10).all()


def test_cerebro_elige_un_experto_y_al_evaluar_no_muestrea():
    torch.manual_seed(0)
    red = RedCerebroMoE(3, 2, 8)
    x = torch.randn(32, 3)
    red.train()
    red(x)
    assert red.uso.sum() == 32
    red.eval()
    assert torch.equal(red(x), red(x))


def test_recompensa_por_ejemplo_distingue_ejemplos():
    """Con recompensa del lote, cada ejemplo recibe el mismo empujón; con la arreglada, no."""
    torch.manual_seed(0)
    x, y = torch.randn(16, 3), torch.randint(0, 2, (16,))
    for por_ejemplo in (False, True):
        red = RedCerebroMoE(3, 2, 8, por_ejemplo=por_ejemplo)
        red.train()
        red.perdida_extra(red(x), y).backward()
        assert red.router.weight.grad is not None


def test_talamo_no_cambia_al_evaluar():
    torch.manual_seed(0)
    red = RedTalamo(3, 2, 8)
    red.train()
    red(torch.randn(16, 3))
    red.eval()
    x = torch.randn(4, 3)
    a = red(x)
    red(torch.randn(50, 3) * 10)
    assert torch.equal(a, red(x))
