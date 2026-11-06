"""Cuaderno 15 · banco de dispersión dinámica: presupuesto, máscaras, optimizador, fuga de datos y determinismo."""

from dataclasses import replace

import pytest
import torch

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from laboratorio.banco.tareas import cargar_tarea  # noqa: E402
from laboratorio.rnra import experimentos as ex  # noqa: E402
from laboratorio.rnra import operaciones as op  # noqa: E402
from laboratorio.rnra.controlador import METODOS, Config, Entrenador, entrenar, presupuesto  # noqa: E402
from laboratorio.rnra.modelo import MLPMascarado  # noqa: E402

RAPIDA = Config(pasos=160, calentamiento=40, cada=40, lote=64)


def _coherente(m: MLPMascarado) -> bool:
    """Una neurona apagada no tiene conexiones de entrada ni de salida."""
    return all(not (m.mascara(l)[m.activa(l) == 0].sum() or m.mascara(l + 1)[:, m.activa(l) == 0].sum()) for l in range(m.ocultas))


def test_esparcida_cumple_el_presupuesto():
    m = MLPMascarado(20, 4, 64, 2)
    m.esparcida(640, 32, torch.Generator().manual_seed(0))
    assert m.conexiones() == 640 and m.neuronas_activas(0) == 32 and _coherente(m)


def test_apagada_no_aporta_aunque_se_toquen_sus_pesos():
    m = MLPMascarado(5, 3, 8, 2)
    m.densa(6)
    x = torch.randn(4, 5)
    m.activa(0)[2] = 0
    m.sincronizar()
    m.limpiar_pesos()
    antes = m(x)
    m.capas[0].weight.data[2] += 10
    assert _coherente(m) and torch.allclose(m(x), antes)


def test_gradiente_denso_existe_donde_no_hay_conexion():
    m = MLPMascarado(6, 3, 8, 2)
    m.esparcida(40, 6, torch.Generator().manual_seed(1))
    x, y = torch.randn(32, 6), torch.randint(0, 3, (32,))
    g = op.gradientes_densos(m, x, y, torch.nn.functional.cross_entropy)
    apagadas = (m.mascara(0) == 0) & m.validas(0)
    assert g[0][apagadas].sum() > 0 and g[0].shape == m.mascara(0).shape


def test_reorganizar_conserva_el_presupuesto_y_nace_en_cero():
    m = MLPMascarado(8, 3, 16, 2)
    gen = torch.Generator().manual_seed(2)
    m.esparcida(120, 12, gen)
    punt = [c.weight.detach().abs() * m.mascara(i) for i, c in enumerate(m.capas)]
    antes = [m.mascara(l).clone() for l in range(3)]
    r = op.reorganizar(m, punt, [torch.rand_like(p) for p in punt], 0.3, 120, gen)
    assert m.conexiones() == 120 and r["movidas"] > 0 and _coherente(m)
    for l in range(3):
        nacidas = (m.mascara(l) > 0) & (antes[l] == 0)
        assert nacidas.sum() > 0 and m.capas[l].weight[nacidas].abs().sum() == 0


def test_reorganizar_gasta_el_presupuesto_liberado():
    m = MLPMascarado(8, 3, 16, 2)
    gen = torch.Generator().manual_seed(3)
    m.esparcida(120, 12, gen)
    m.activa(0)[0] = 0
    m.activa(1)[0] = 0
    m.sincronizar()
    assert m.conexiones() < 120
    punt = [c.weight.detach().abs() * m.mascara(i) for i, c in enumerate(m.capas)]
    op.reorganizar(m, punt, [torch.rand_like(p) for p in punt], 0.1, 120, gen)
    assert m.conexiones() == 120


def test_activar_neurona_densa_no_cambia_la_funcion():
    m = MLPMascarado(6, 3, 12, 2)
    gen = torch.Generator().manual_seed(4)
    m.densa(8)
    x = torch.randn(10, 6)
    antes = m(x).detach()
    i = op.activar_neurona_densa(m, 0, gen)
    assert i == 8 and m.activa(0)[i] == 1 and _coherente(m) and torch.allclose(m(x), antes, atol=1e-6)


def test_poda_global_deja_el_objetivo_y_reinicia_el_estado_de_adam():
    m = MLPMascarado(8, 3, 16, 2)
    m.densa(16)
    opt = torch.optim.AdamW(m.parameters())
    x, y = torch.randn(32, 8), torch.randint(0, 3, (32,))
    torch.nn.functional.cross_entropy(m(x), y).backward()
    opt.step()
    antes = op.foto_estructura(m)
    op.poda_global_por_magnitud(m, 200)
    assert abs(m.conexiones() - 200) <= 3
    op.reiniciar_estado_optimizador(opt, m, antes)
    cambio = m.mascara(0) != antes["m"][0]
    assert cambio.any() and opt.state[m.capas[0].weight]["exp_avg"][cambio].abs().sum() == 0


@pytest.mark.parametrize("metodo", METODOS)
def test_todos_los_metodos_corren_y_respetan_el_presupuesto(metodo):
    t = cargar_tarea("lunas")
    r = entrenar(metodo, t, 0, RAPIDA)
    B = presupuesto(t.entradas, t.salidas, RAPIDA)
    assert 0.5 <= r.metrica_prueba <= 1.0 and r.flops > 0
    assert r.conexiones > B if metodo == "mlp_grande" else r.conexiones <= B


def test_determinismo_con_la_misma_semilla():
    t = cargar_tarea("lunas")
    a, b, c = entrenar("rigl", t, 7, RAPIDA), entrenar("rigl", t, 7, RAPIDA), entrenar("rigl", t, 8, RAPIDA)
    assert a.metrica_prueba == b.metrica_prueba and a.flops == b.flops
    assert (a.metrica_prueba, a.curva) != (c.metrica_prueba, c.curva)


@pytest.mark.parametrize("metodo", ["rigl", "set", "poda_magnitud"])
def test_sin_fuga_de_datos_la_prueba_no_influye_en_el_entrenamiento(metodo):
    """Si se cambia por completo la prueba, el entrenamiento y la estructura no cambian."""
    t = cargar_tarea("lunas")
    t2 = replace(t, y_prueba=1 - t.y_prueba, x_prueba=t.x_prueba * 3)
    a, b = entrenar(metodo, t, 3, RAPIDA), entrenar(metodo, t2, 3, RAPIDA)
    assert a.flops == b.flops and a.conexiones == b.conexiones and a.curva == b.curva
    assert a.metrica_prueba != b.metrica_prueba


def test_rigl_mueve_conexiones_y_set_tambien():
    t = cargar_tarea("caotica_20d")
    for metodo in ("rigl", "set"):
        e = Entrenador(metodo, t.entradas, t.salidas, t.tipo, RAPIDA, 0)
        e.fijar_datos(t.x_entreno, t.y_entreno)
        antes = [e.m.mascara(l).clone() for l in range(3)]
        e.entrenar_tarea()
        assert any((e.m.mascara(l) != antes[l]).any() for l in range(3)) and e.m.conexiones() <= e.B


def test_continuo_devuelve_matriz_y_olvido():
    sec = ex.lunas_rotadas((0, 90), n=400)
    r = ex.continuo("rigl", sec, 0, RAPIDA)
    assert len(r.matriz) == 2 and len(r.matriz[0]) == 2
    assert r.perdida_prueba == pytest.approx(r.matriz[0][0] - r.matriz[1][0])
    assert ex.olvido([[0.9, 0.1], [0.5, 0.8]]) == pytest.approx(0.4)


def test_secuencias_tienen_la_forma_esperada():
    p = ex.digitos_partidos()
    assert len(p.tareas) == 5 and p.tareas[1].clases == [2, 3] and set(p.tareas[1].y_entreno.tolist()) == {2, 3}
    q = ex.digitos_permutados(3)
    assert q.tareas[0].x_entreno.shape[1] == 64 and not torch.equal(q.tareas[0].x_entreno, q.tareas[1].x_entreno)


def test_cabeza_por_tarea_en_digitos_partidos():
    r = ex.continuo("mlp_pequeno", ex.digitos_partidos(), 0, replace(RAPIDA, pasos=120))
    assert r.metrica_prueba > 0.5 and len(r.matriz) == 5


def test_holm_y_comparar():
    assert ex.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    c = ex.comparar([0.9, 0.92, 0.91, 0.93], [0.8, 0.82, 0.81, 0.85])
    assert c["dif"] > 0.09 and c["p"] < 0.01 and c["ic"][0] > 0


def test_tablas():
    res = ex.correr(cargar_tarea("lunas"), ["mlp_pequeno", "rigl", "set"], [0, 1], RAPIDA)
    assert "rigl" in ex.tabla(res) and "Holm" in ex.tabla_comparacion(res, "rigl")
