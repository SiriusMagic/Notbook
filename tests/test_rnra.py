"""Cuaderno 15 · RNRA-v0: presupuesto, máscaras, optimizador, fuga de datos y determinismo."""

import copy
from dataclasses import replace

import pytest
import torch

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from laboratorio.banco.tareas import cargar_tarea  # noqa: E402
from laboratorio.rnra import experimentos as ex  # noqa: E402
from laboratorio.rnra import operaciones as op  # noqa: E402
from laboratorio.rnra.controlador import METODOS, Config, Entrenador, entrenar, presupuesto  # noqa: E402
from laboratorio.rnra.estadisticas import Demanda, Estadisticas  # noqa: E402
from laboratorio.rnra.modelo import MLPMascarado  # noqa: E402

RAPIDA = Config(pasos=160, calentamiento=40, cada=40, ventana=10, lote=64, lote_medida=128)


def _coherente(m: MLPMascarado) -> bool:
    """Una neurona apagada no tiene conexiones; los pesos apagados valen cero."""
    for l in range(m.ocultas):
        off = m.activa(l) == 0
        if m.mascara(l)[off].sum() or m.mascara(l + 1)[:, off].sum():
            return False
    return True


def test_esparcida_cumple_el_presupuesto():
    m = MLPMascarado(20, 4, 64, 2)
    m.esparcida(640, 32, torch.Generator().manual_seed(0))
    assert m.conexiones() == 640 and m.neuronas_activas(0) == 32 and _coherente(m)


def test_apagada_no_tiene_conexiones_ni_aporta():
    m = MLPMascarado(5, 3, 8, 2)
    m.densa(6)
    x = torch.randn(4, 5)
    op.podar_neuronas(m, [(0, 2), (1, 1)])
    assert _coherente(m) and m.neuronas_activas(0) == 5
    antes = m(x)
    m.capas[0].weight.data[2] += 10  # tocar los pesos de una neurona apagada no cambia nada
    assert torch.allclose(m(x), antes)


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
    grad = [torch.rand_like(p) for p in punt]
    antes = [m.mascara(l).clone() for l in range(3)]
    r = op.reorganizar(m, punt, grad, 0.3, 120, gen)
    assert m.conexiones() == 120 and r["movidas"] > 0
    for l in range(3):
        nacidas = (m.mascara(l) > 0) & (antes[l] == 0)
        assert nacidas.sum() > 0 and m.capas[l].weight[nacidas].abs().sum() == 0
    assert _coherente(m)


def test_reorganizar_gasta_el_presupuesto_liberado():
    m = MLPMascarado(8, 3, 16, 2)
    gen = torch.Generator().manual_seed(3)
    m.esparcida(120, 12, gen)
    op.podar_neuronas(m, [(0, 0), (1, 0)])
    bajo = m.conexiones()
    assert bajo < 120
    punt = [c.weight.detach().abs() * m.mascara(i) for i, c in enumerate(m.capas)]
    op.reorganizar(m, punt, [torch.rand_like(p) for p in punt], 0.1, 120, gen)
    assert m.conexiones() == 120  # lo liberado por la poda se reparte, no se pierde


def test_activar_neurona_no_cambia_la_funcion():
    m = MLPMascarado(6, 3, 12, 2)
    gen = torch.Generator().manual_seed(4)
    m.esparcida(60, 8, gen)
    x = torch.randn(10, 6)
    antes = m(x).detach()
    usadas, i = op.activar_neurona(m, 0, gen)
    assert usadas > 0 and m.activa(0)[i] == 1 and _coherente(m)
    assert torch.allclose(m(x), antes, atol=1e-6)  # salida con peso cero: la función se conserva


def test_activar_no_reutiliza_la_ranura_recien_podada():
    m = MLPMascarado(6, 3, 10, 2)
    gen = torch.Generator().manual_seed(5)
    m.esparcida(60, 8, gen)
    op.podar_neuronas(m, [(0, 7)])
    _, i = op.activar_neurona(m, 0, gen, excluir=[7])
    assert i not in (7,) and i >= 8


def test_estadisticas_tienen_forma_y_marcan_neuronas_muertas():
    m = MLPMascarado(4, 2, 8, 2)
    m.densa(6)
    with torch.no_grad():
        m.capas[0].weight[0].zero_()
        m.capas[0].bias[0] = -50  # neurona 0 siempre apagada
    e = Estadisticas(m)
    x, y = torch.randn(64, 4), torch.randint(0, 2, (64,))
    e.medir(x, y, torch.nn.functional.cross_entropy)
    assert e.utilidad(0).shape == (8,) and e.red[0][0] == 1.0 and e.utilidad(0)[0] == 0
    assert ((e.estabilidad(0) >= 0) & (e.estabilidad(0) <= 1.0001)).all()
    assert torch.isinf(e.puntaje(0)[7])  # las apagadas no son candidatas


def test_demanda_es_la_perdida_media_por_region():
    d = Demanda(3)
    d.actualizar(torch.tensor([0, 0, 1, 2, 2, 2]), torch.tensor([1.0, 3.0, 5.0, 2.0, 2.0, 2.0]))
    assert torch.allclose(d.valor()[:3], torch.tensor([2.0, 5.0, 2.0]), atol=1e-4)
    assert d.pesos_muestras(torch.tensor([0, 1, 2])).argmax() == 1


@pytest.mark.parametrize("metodo", METODOS)
def test_todos_los_metodos_corren_y_respetan_el_presupuesto(metodo):
    t = cargar_tarea("lunas")
    r = entrenar(metodo, t, 0, RAPIDA)
    B = presupuesto(t.entradas, t.salidas, RAPIDA)
    assert 0.5 <= r.metrica_prueba <= 1.0 and r.flops > 0
    if metodo == "mlp_grande":
        assert r.conexiones > B
    else:
        assert r.conexiones <= B


def test_determinismo_con_la_misma_semilla():
    t = cargar_tarea("lunas")
    a, b = entrenar("rnra", t, 7, RAPIDA), entrenar("rnra", t, 7, RAPIDA)
    assert a.metrica_prueba == b.metrica_prueba and a.flops == b.flops
    assert [e["aceptado"] for e in a.eventos] == [e["aceptado"] for e in b.eventos]
    c = entrenar("rnra", t, 8, RAPIDA)
    assert (a.metrica_prueba, a.flops) != (c.metrica_prueba, c.flops)


def test_sin_fuga_de_datos_la_prueba_no_influye_en_el_entrenamiento():
    """Si se cambian por completo las etiquetas de prueba, las decisiones y el entrenamiento no cambian."""
    t = cargar_tarea("lunas")
    t2 = replace(t, y_prueba=1 - t.y_prueba, x_prueba=t.x_prueba * 3)
    a, b = entrenar("rnra", t, 3, RAPIDA), entrenar("rnra", t2, 3, RAPIDA)
    assert a.flops == b.flops and a.conexiones == b.conexiones
    assert [(e["paso"], e["aceptado"], e["podadas"], e["movidas"]) for e in a.eventos] == \
           [(e["paso"], e["aceptado"], e["podadas"], e["movidas"]) for e in b.eventos]
    assert a.curva == b.curva and a.metrica_prueba != b.metrica_prueba


def test_revertir_restaura_pesos_mascaras_y_optimizador():
    t = cargar_tarea("lunas")
    e = Entrenador("rnra", t.entradas, t.salidas, t.tipo, RAPIDA, 0)
    e.fijar_datos(t.x_entreno, t.y_entreno)
    e._entrenar(30, 0, 160, e.gen)
    foto = e._foto()
    pesos, masc = copy.deepcopy(e.m.state_dict()), e.m.conexiones()
    e._entrenar(20, 30, 160, torch.Generator().manual_seed(0))  # cambia pesos, estado de Adam y flops
    e.m.mascara(0)[:5] = 0
    e._restaurar(foto)
    assert e.m.conexiones() == masc
    assert all(torch.equal(v, pesos[k]) for k, v in e.m.state_dict().items())
    estado = e.opt.state_dict()["state"]
    ref = foto[1]["state"]
    assert estado.keys() == ref.keys() and all(torch.equal(estado[k]["exp_avg"], ref[k]["exp_avg"]) for k in ref)


def test_rechazar_un_cambio_deja_la_red_en_la_continuacion_sin_cambio():
    """Si nunca se acepta nada, la estructura queda como al principio (el presupuesto y las neuronas no cambian)."""
    t = cargar_tarea("lunas")
    cfg = replace(RAPIDA, margen=2.0)  # se exige una pérdida negativa: nunca se acepta
    r = entrenar("rnra", t, 1, cfg)
    assert r.eventos and not any(e["aceptado"] for e in r.eventos)
    assert r.ciclos == 0 and r.conexiones == presupuesto(t.entradas, t.salidas, cfg) and r.neuronas == [cfg.activas_inicial] * 2


def test_historesis_evita_apagar_y_encender_la_misma_ranura_seguido():
    t = cargar_tarea("caotica_20d")
    r = entrenar("rnra", t, 0, replace(RAPIDA, pasos=300))
    assert r.ciclos == 0


def test_continuo_devuelve_matriz_y_olvido():
    sec = ex.lunas_rotadas((0, 90), n=400)
    r = ex.continuo("rnra", sec, 0, RAPIDA)
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
    aj = ex.holm([0.01, 0.04, 0.03])
    assert aj == pytest.approx([0.03, 0.06, 0.06])
    c = ex.comparar([0.9, 0.92, 0.91, 0.93], [0.8, 0.82, 0.81, 0.85])
    assert c["dif"] > 0.09 and c["p"] < 0.01 and c["ic"][0] > 0


def test_criterios_y_tablas():
    t = cargar_tarea("lunas")
    res = ex.correr(t, ["mlp_pequeno", "mlp_grande", "rigl", "rnra"], [0, 1], RAPIDA)
    c = ex.criterios(res)
    assert c["mejor_comparador"] in ("mlp_pequeno", "rigl") and {"precision", "parametros", "estabilidad", "eficiencia"} <= c.keys()
    assert "rnra" in ex.tabla(res) and "Holm" in ex.tabla_comparacion(res)


def test_especializacion_esta_entre_0_y_1():
    t = cargar_tarea("lunas")
    e = Entrenador("rnra", t.entradas, t.salidas, t.tipo, RAPIDA, 0)
    e.fijar_datos(t.x_entreno, t.y_entreno)
    e.entrenar_tarea()
    assert 0 <= ex.especializacion(e, e.x) <= 1
