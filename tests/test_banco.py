import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from laboratorio.banco import PIEZAS, TAREAS, cargar_tarea, comparar  # noqa: E402
from laboratorio.banco.evaluar import Config, Corrida, Registro, entrenar_una, informe  # noqa: E402
from laboratorio.banco.piezas import Red  # noqa: E402
from laboratorio.banco.tareas import TAREAS as FABRICAS  # noqa: E402

RAPIDO = Config(epocas=15)


@pytest.mark.parametrize("nombre", TAREAS)
def test_tareas_fijas_y_sin_mezclar(nombre):
    a, b = FABRICAS[nombre](), FABRICAS[nombre]()
    assert torch.equal(a.x_entreno, b.x_entreno) and torch.equal(a.y_prueba, b.y_prueba)
    assert len(a.x_entreno) > len(a.x_prueba) > 0
    assert a.x_entreno.shape[1] == a.entradas
    # ninguna fila de prueba aparece en entrenamiento
    filas = {tuple(r.tolist()) for r in a.x_entreno}
    assert not any(tuple(r.tolist()) in filas for r in a.x_prueba)


@pytest.mark.parametrize("pieza", PIEZAS)
def test_pieza_forma_gradiente_y_sin_nan(pieza):
    torch.manual_seed(0)
    red = Red(PIEZAS[pieza], entradas=3, salidas=2, ocultas=8)
    x = torch.randn(10, 3)
    y = red(x)
    assert y.shape == (10, 2) and torch.isfinite(y).all()
    perdida = y.pow(2).mean()
    if hasattr(red, "perdida_extra"):  # p. ej. el router con REINFORCE solo aprende por esta vía
        perdida = perdida + red.perdida_extra(y, torch.zeros_like(y))
    perdida.backward()
    sin_grad = [n for n, p in red.named_parameters() if p.grad is None]
    assert not sin_grad, f"parámetros sin gradiente: {sin_grad}"
    c = entrenar_una(pieza, "circulos", 0, RAPIDO)
    assert not c.fallo and 0 <= c.valor <= 1


def test_la_red_estandar_aprende():
    c = entrenar_una("estandar", "circulos", 0, Config(epocas=150))
    assert c.valor > 0.8  # círculos con ruido: ~85 % es el techo práctico
    c = entrenar_una("estandar", "lunas", 0, Config(epocas=150))
    assert c.valor > 0.9
    c = entrenar_una("estandar", "regresion_cubica", 0, Config(epocas=300))
    assert c.valor < 0.1


def test_corridas_repetibles():
    a, b = entrenar_una("sma", "lunas", 3, RAPIDO), entrenar_una("sma", "lunas", 3, RAPIDO)
    assert a.valor == b.valor and a.perdida_final == b.perdida_final


def _corridas(valores, metrica="exactitud", pieza="x"):
    return [Corrida(pieza, "t", s, metrica, v, 0.0, 10, 0.0, valor_entreno=v) for s, v in enumerate(valores)]


def test_veredicto_no_miente():
    base = _corridas([0.80, 0.82, 0.81, 0.79, 0.80], pieza="estandar")
    assert comparar(_corridas([0.80, 0.82, 0.81, 0.79, 0.80]), base).veredicto == "igual"
    assert comparar(_corridas([0.95, 0.96, 0.94, 0.95, 0.97]), base).veredicto == "mejor"
    assert comparar(_corridas([0.50, 0.52, 0.49, 0.51, 0.50]), base).veredicto == "peor"
    # Diferencia pequeña y ruidosa: no se declara ganador.
    assert comparar(_corridas([0.85, 0.75, 0.83, 0.78, 0.81]), base).veredicto == "igual"
    # En MSE menos es mejor.
    base_mse = _corridas([0.10, 0.11, 0.10, 0.12, 0.10], "mse", "estandar")
    assert comparar(_corridas([0.05, 0.05, 0.04, 0.06, 0.05], "mse"), base_mse).veredicto == "mejor"
    # Caso real (RMB en regresión cúbica): peor en todas las semillas y una desastrosa.
    # Con la media simple la varianza lo escondía; en escala logarítmica se ve.
    base_real = _corridas([0.0048, 0.0047, 0.0049, 0.0048, 0.0048], "mse", "estandar")
    assert comparar(_corridas([0.29, 0.43, 12.5, 0.30, 0.31], "mse"), base_real).veredicto == "peor"
    falla = _corridas([0.9, math.nan, 0.9, 0.9, 0.9])
    falla[1].fallo = True
    assert comparar(falla, base).veredicto == "falla"


def test_pieza_inutil_sale_peor_en_una_corrida_real():
    base = [entrenar_una("estandar", "digitos", s, Config(epocas=60)) for s in range(5)]
    inutil = [entrenar_una("estandar", "digitos", s, Config(epocas=1)) for s in range(5)]  # casi sin entrenar
    assert comparar(inutil, base).veredicto == "peor"


def test_registro_jsonl_ida_y_vuelta(tmp_path):
    reg = Registro(tmp_path)
    c = entrenar_una("ida", "xor", 0, RAPIDO)
    reg.guardar(c)
    reg.guardar(c)
    leidas = Registro.leer(reg.ruta)
    assert len(leidas) == 2 and leidas[0] == c


def test_informe_markdown():
    base = _corridas([0.8] * 3, pieza="estandar")
    texto = informe([comparar(_corridas([0.9, 0.91, 0.92]), base)], semillas=3)
    assert "| x | t |" in texto and "🟢 mejor" in texto and "3 semillas" in texto


def test_tarea_desconocida():
    with pytest.raises(KeyError):
        cargar_tarea("no_existe")


def test_una_semilla_o_sin_variacion_no_declara_ganador():
    base = _corridas([0.80], pieza="estandar")
    assert comparar(_corridas([0.99]), base).veredicto == "igual"
    base3 = _corridas([0.80, 0.81, 0.82], pieza="estandar")
    assert comparar(_corridas([0.90, 0.91, 0.92]), base3).veredicto == "igual"  # diferencia idéntica: sin ruido estimable


def test_correccion_por_comparaciones_multiples():
    from laboratorio.banco.evaluar import Comparacion, ajustar_multiples

    def comp(p, veredicto):
        return Comparacion("x", "t", "exactitud", 0.9, 0.01, 0.8, 0.01, 0.1, p, veredicto, 10, 10)

    claro, justo = comp(0.0001, "mejor"), comp(0.03, "mejor")
    ruido = [comp(0.2 + i * 0.03, "igual") for i in range(20)]
    sin_p = comp(math.nan, "igual")
    ajustar_multiples([claro, justo, sin_p, *ruido])
    assert claro.veredicto == "mejor" and claro.p_ajustado == pytest.approx(0.0001 * 22)
    assert justo.veredicto == "igual" and justo.p_ajustado == pytest.approx(0.03 * 22 / 2)
    assert all(c.veredicto == "igual" for c in ruido) and math.isnan(sin_p.p_ajustado)


def test_serie_dividida_en_orden_sin_fuga_de_futuro():
    import numpy as np

    from laboratorio.banco.tareas import _armar

    # Serie = 0, 1, 2, …: cada objetivo es su propia posición en el tiempo.
    serie = np.arange(100, dtype=float)
    x = np.stack([serie[i : i + 5] for i in range(95)])
    t = _armar("prueba", "regresion", x, serie[5:], 10, "", cronologico=True)
    entreno, prueba = t.y_entreno.flatten(), t.y_prueba.flatten()
    assert entreno.max() < prueba.min()  # todo el entrenamiento es anterior a la prueba
    assert prueba.min() == entreno.max() + 1  # y no se pierde ningún punto
    assert "último 30 %" in cargar_tarea("serie_seno").descripcion
