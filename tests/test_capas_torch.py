import pytest

torch = pytest.importorskip("torch")

from laboratorio.capas_torch import CapaInversionGradiente  # noqa: E402


def test_adelante_igual_atras_invertido():
    x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    capa = CapaInversionGradiente(lambd=0.5)
    y = capa(x)
    assert torch.equal(y, x)
    (y * torch.tensor([1.0, 2.0, 3.0])).sum().backward()
    assert torch.allclose(x.grad, torch.tensor([-0.5, -1.0, -1.5]))


def test_solo_afecta_lo_que_esta_antes_de_la_capa():
    torch.manual_seed(0)
    antes, despues = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
    x = torch.randn(4, 2)

    def grads(con_grl):
        for p in (*antes.parameters(), *despues.parameters()):
            p.grad = None
        h = antes(x)
        if con_grl:
            h = CapaInversionGradiente(1.0)(h)
        despues(h).sum().backward()
        return antes.weight.grad.clone(), despues.weight.grad.clone()

    a_normal, d_normal = grads(False)
    a_grl, d_grl = grads(True)
    assert torch.allclose(a_grl, -a_normal)   # la parte de antes recibe el gradiente invertido
    assert torch.allclose(d_grl, d_normal)    # la de después no cambia
    assert "lambd=1.0" in repr(CapaInversionGradiente(1.0))
