"""Cuaderno 16 · RNRA-M: superred anidada, memoria portátil y elección de configuración por RAM."""

import copy

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from laboratorio.rnra import memoria_multiescala as mem  # noqa: E402
from laboratorio.rnra.adaptador_dispositivo import dispositivos, elegir  # noqa: E402
from laboratorio.rnra.superred import (SuperRed, bytes_modelo, cuantizar_modelo, cuantizar_tensor,  # noqa: E402
                                       entrenar, exactitud)

ANCHOS = (4, 8, 16)


@pytest.fixture(scope="module")
def datos():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(400, 6, generator=g)
    y = (x[:, 0] + x[:, 1] * x[:, 2] > 0).long() + (x[:, 3] > 1).long()
    return x, y


@pytest.fixture(scope="module")
def red(datos):
    torch.manual_seed(0)
    return entrenar(SuperRed(6, 3, ANCHOS), *datos, pasos=60, lote=64)


def test_los_anchos_estan_anidados(red, datos):
    """Lo que hay fuera de las primeras w neuronas no cambia la salida de ancho w."""
    x, _ = datos
    antes = red(x[:10], 4).detach()
    r2 = copy.deepcopy(red)
    with torch.no_grad():
        r2.capas[1].weight[4:, :] += 5
        r2.capas[1].weight[:, 4:] += 5
        r2.capas[0].weight[4:] += 5
    assert torch.allclose(r2(x[:10], 4), antes, atol=1e-6)
    assert not torch.allclose(r2(x[:10], 16), red(x[:10], 16))


def test_cuenta_de_parametros_coincide_con_la_red_real(red):
    for w in ANCHOS:
        real = 6 * w + w + (w * w + w) + w * 3 + 3
        assert red.parametros(w) == real
    assert bytes_modelo(red, 16, 8) < bytes_modelo(red, 16, 32) and bytes_modelo(red, 4, 32) < bytes_modelo(red, 8, 32)


def test_cuantizar_tiene_error_acotado():
    t = torch.randn(1000)
    for b in (8, 4):
        q = cuantizar_tensor(t, b)
        escala = t.abs().max() / (2 ** (b - 1) - 1)
        assert (q - t).abs().max() <= escala / 2 + 1e-6
        assert len(torch.unique(q)) <= 2 ** b
    assert torch.equal(cuantizar_tensor(t, 32), t)
    assert cuantizar_modelo(SuperRed(3, 2, (2, 4)), 4) is not None


def test_reducir_y_expandir_con_memoria_recupera_todo(red, datos):
    x, _ = datos
    p = mem.reducir(red, 4, bits_nucleo=8, bits_memoria=32)
    e = mem.expandir(p, red)
    for a, b in zip(e.parameters(), red.parameters()):
        assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(e(x, 16), red(x, 16), atol=1e-4)


def test_sin_memoria_la_capacidad_se_pierde(red, datos):
    x, _ = datos
    p = mem.reducir(red, 4, bits_nucleo=8, bits_memoria=32)
    tirada = mem.expandir(p, red, con_memoria=False)
    assert not torch.allclose(tirada(x, 16), red(x, 16), atol=1e-3)


def test_el_nucleo_pesa_mucho_menos_que_la_red(red):
    p = mem.reducir(red, 4, bits_nucleo=8, bits_memoria=32)
    assert p.bytes_nucleo() < bytes_modelo(red, 4, 32)
    assert p.bytes_nucleo() < bytes_modelo(red, 16, 32) / 4


def test_memoria_comprimida_pesa_menos_con_menos_bits_y_el_error_es_pequeno(red):
    p32, p8 = mem.reducir(red, 4, 8, 32), mem.reducir(red, 4, 8, 8)
    assert p8.bytes_memoria() < p32.bytes_memoria()
    e = mem.expandir(p8, red)
    err = max(float((a - b).detach().abs().max()) for a, b in zip(e.parameters(), red.parameters()))
    assert 0 < err < 0.05


def test_guardar_y_cargar_no_cambia_la_red(red, datos, tmp_path):
    x, _ = datos
    p = mem.reducir(red, 4, bits_nucleo=8, bits_memoria=32)
    ruta = str(tmp_path / "paquete.npz")
    assert mem.guardar(p, ruta) > 0
    e = mem.expandir(mem.cargar(ruta), red)
    assert torch.allclose(e(x, 16), red(x, 16), atol=1e-4)
    z = np.load(ruta, allow_pickle=False)  # sin pickle: el archivo solo trae arreglos
    assert all(z[k].dtype != object for k in z.files)


def test_el_nucleo_funciona_solo_en_el_dispositivo_pequeno(red, datos):
    x, y = datos
    p = mem.reducir(red, 4, bits_nucleo=8, bits_memoria=32)
    chico = mem.nucleo_como_modelo(p, red)
    assert chico.ancho == 4
    assert abs(exactitud(chico, x, y, 4) - exactitud(red, x, y, 4)) < 0.1  # el núcleo cuantizado casi no pierde


def test_elegir_respeta_la_ram_y_toma_el_ancho_mayor(red):
    d = dispositivos(red)
    assert [x.nombre for x in d] == ["celular_basico", "celular_potente", "pc_con_gpu"]
    for disp, w in zip(d, ANCHOS):
        assert elegir(red, disp.ram_bytes, bits=(32,)) == (w, 32)
    w, b = elegir(red, d[0].ram_bytes)
    assert bytes_modelo(red, w, b) <= d[0].ram_bytes and w >= 4
    assert elegir(red, 10) is None
