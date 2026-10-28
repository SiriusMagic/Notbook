"""Arquitecturas normales (CNN, TextCNN, Transformer, CNN 1D, LSTM), series reales y competencia, sin internet."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from laboratorio.banco import datos_reales  # noqa: E402
from laboratorio.banco.arquitecturas import ARQUITECTURAS, INVESTIGACION, crear_modelo  # noqa: E402
from laboratorio.banco.competencia import competir, informe_competencia  # noqa: E402
from laboratorio.banco.evaluar import Config, _medir, _predecir, cargar_modelo, entrenar_una  # noqa: E402
from laboratorio.banco.piezas import PIEZAS, ActivacionAdaptativa, ActivacionIDA  # noqa: E402
from laboratorio.banco.tareas import cargar_tarea  # noqa: E402


def _tarea_texto_ids():
    buenas = ["excelente producto me encanta", "muy bueno lo recomiendo", "perfecto llegó rápido y bien"]
    malas = ["pésimo no funciona nada", "muy malo se rompió", "horrible no lo compren"]
    filas = [{"text": (buenas if i % 2 else malas)[i % 3], "label": i % 2} for i in range(60)]
    fuente = datos_reales.FuenteHF("falsa", "x/y", "texto", "text", "prueba", 2, lote=16)
    return datos_reales.armar_tarea(fuente, filas, filas[:20], representacion="ids")


@pytest.mark.parametrize("act", [ActivacionIDA, ActivacionAdaptativa])
def test_activaciones_por_canal_en_cualquier_forma(act):
    a = act(6)
    for forma in [(4, 6), (4, 6, 10), (4, 6, 5, 5)]:
        x = torch.randn(*forma, requires_grad=True)
        y = a(x)
        assert y.shape == x.shape
        y.sum().backward()
        assert all(p.grad is not None for p in a.parameters())


@pytest.mark.parametrize("pieza", PIEZAS)
def test_cada_pieza_funciona_en_cada_arquitectura(pieza):
    texto = _tarea_texto_ids()
    casos = [("cnn", cargar_tarea("digitos")), ("serie_cnn", cargar_tarea("serie_seno")),
             ("lstm", cargar_tarea("serie_seno")), ("texto_cnn", texto), ("transformer", texto)]
    for arq, t in casos:
        torch.manual_seed(0)
        m = crear_modelo(PIEZAS[pieza], t, arq, ocultas=8)
        y = m(t.x_entreno[:5])
        assert y.shape == (5, t.salidas) and torch.isfinite(y).all(), (pieza, arq)
        y.sum().backward()


def test_arquitectura_equivocada_avisa():
    with pytest.raises(ValueError, match="no sirve"):
        crear_modelo(PIEZAS["estandar"], cargar_tarea("circulos"), "cnn")


def test_cnn_le_gana_a_la_densa_en_imagenes():
    densa = entrenar_una("estandar", "digitos", 0, Config(epocas=15, lote=64, lr=0.003))
    cnn = entrenar_una("estandar", "digitos", 0, Config(epocas=15, lote=64, lr=0.003, arquitectura="cnn"))
    assert cnn.valor > densa.valor and cnn.valor > 0.95 and cnn.arquitectura == "cnn"


def test_texto_con_ids_y_transformer_aprenden():
    t = _tarea_texto_ids()
    assert t.x_entreno.dtype == torch.int64 and t.vocabulario > 2 and t.forma == (256,)
    assert (t.x_entreno[:, 0] > 1).all()  # la primera palabra es conocida (no relleno ni desconocida)
    for arq in ["texto_cnn", "transformer"]:
        c = entrenar_una("estandar", t, 0, Config(epocas=30, lote=16, lr=0.003, arquitectura=arq))
        assert c.valor == 1.0, arq


def test_textos_a_ids_vocabulario_solo_de_entrenamiento():
    xe, xp, vocab = datos_reales.textos_a_ids(["hola mundo"], ["hola marte"], largo=4)
    assert xe.tolist() == [[2, 3, 0, 0]] or xe.tolist() == [[3, 2, 0, 0]]
    assert xp[0, 1] == 1  # «marte» no estaba en el entrenamiento → desconocida
    assert vocab == 4


def test_serie_real_en_orden_y_con_referencia_ingenua():
    valores = np.sin(np.arange(3000) / 20) * 10 + 50
    valores[100] = np.nan
    fuente = datos_reales.FuenteHF("s", "x/y", "serie", "id", "prueba", entreno=1000, prueba=300, ventana=24, lote=64)
    t = datos_reales.armar_serie(fuente, valores)
    assert t.x_entreno.shape == (1000, 24) and t.x_prueba.shape == (300, 24)
    assert t.dato == "serie" and t.ingenuo is not None and t.ingenuo > 0
    # normalizada con la parte de entrenamiento
    assert abs(float(torch.cat([t.x_entreno[0], t.x_entreno[:, -1]]).mean())) < 0.2
    # sin fuga: la última ventana de entrenamiento + su objetivo empalman con la primera de prueba
    assert torch.allclose(t.x_prueba[0, :-1], t.x_entreno[-1, 1:]) and torch.isclose(t.x_prueba[0, -1], t.y_entreno[-1, 0])
    c = entrenar_una("estandar", t, 0, Config(epocas=40, lote=64, lr=0.003, arquitectura="serie_cnn"))
    assert c.valor < t.ingenuo  # una onda suave: la CNN 1D debe ganarle a repetir el último valor


def test_huecos_de_la_serie_se_rellenan():
    y = np.array([1.0, np.nan, np.nan, 4.0, np.nan])
    huecos = np.isnan(y)
    idx = np.where(~huecos, np.arange(len(y)), 0)
    np.maximum.accumulate(idx, out=idx)
    assert y[idx].tolist() == [1.0, 1.0, 1.0, 4.0, 4.0]


def test_guardar_y_recargar_modelos_de_cada_arquitectura(tmp_path):
    for arq, t in [("cnn", cargar_tarea("digitos")), ("lstm", cargar_tarea("serie_seno")), ("transformer", _tarea_texto_ids())]:
        c = entrenar_una("ida", t, 0, Config(epocas=3, lote=64, guardar=str(tmp_path), arquitectura=arq))
        assert arq in c.detalle["modelo"]
        m = cargar_modelo(c.detalle["modelo"])
        with torch.no_grad():
            assert _medir(t, _predecir(m, t.x_prueba), t.y_prueba) == pytest.approx(c.valor, abs=1e-6), arq


def test_competencia_e_informe():
    filas = competir(["digitos", "serie_seno"], ["seno"], {"semillas": (0, 1), "epocas": 5, "lote": 128}, aviso=lambda *_: None)
    nombres = {(f.tarea, f.arquitectura, f.pieza) for f in filas}
    assert ("digitos", "cnn", "estandar") in nombres and ("serie_seno", "lstm", "seno") in nombres
    texto = informe_competencia(filas, "2")
    assert "## digitos" in texto and "CNN normal" in texto and "Referencia:" in texto
    assert "LSTM normal" in texto and "| 1 |" in texto
    assert all(not math.isnan(f.media) for f in filas)


def test_todas_las_arquitecturas_registradas():
    assert {"densa", "cnn", "texto_cnn", "transformer", "serie_cnn", "lstm"} <= set(ARQUITECTURAS)
    assert {"transformer_senoidal", "transformer_jer_pares", "transformer_jer_palabra"} <= set(ARQUITECTURAS)


def test_mapa_por_tipo_de_dato():
    from laboratorio.banco.competencia import Fila, _celda, resumen_por_tipo
    from laboratorio.banco.evaluar import Comparacion

    def fila(tarea, arq, veredicto):
        c = Comparacion("ida", tarea, "exactitud", 0.9, 0.0, 0.8, 0.0, 0.1, 0.01, veredicto, 1, 1)
        return Fila(tarea, arq, "ida", "exactitud", 0.9, 0.0, 1, 0.1, c)

    filas = [fila("digitos", "cnn", "mejor"), fila("digitos", "densa", "igual"), fila("serie_seno", "lstm", "peor")]
    mapa = resumen_por_tipo(filas)
    assert mapa["ida"]["imagen"]["mejor"] == 1 and mapa["ida"]["serie"]["peor"] == 1
    assert "prometedora" in _celda(mapa["ida"]["imagen"])
    assert "no sirve aquí" in _celda(mapa["ida"]["serie"])
    texto = informe_competencia(filas, "2")
    assert "¿en qué tipo de dato sirve cada pieza?" in texto and "| ida |" in texto


def test_linea_de_comandos_filtra_tareas_por_arquitectura(tmp_path, capsys):
    from laboratorio.banco.__main__ import main

    main(["--arquitectura", "cnn", "--todo", "--pieza", "seno", "--semillas", "2", "--epocas", "2", "--salida", str(tmp_path)])
    salida = capsys.readouterr().out
    assert "se omiten 5 tareas" in salida
    informe = (tmp_path / "informe.md").read_text(encoding="utf-8")
    assert "| seno | digitos |" in informe and "circulos" not in informe
    with pytest.raises(SystemExit):
        main(["--arquitectura", "cnn", "--tarea", "circulos", "--salida", str(tmp_path)])


@pytest.mark.parametrize("arq", ["fasttext", "atencion_binaria_rl", "atencion_niveles_rl", "atencion_hebbiana"])
def test_agentes_de_atencion_y_fasttext_aprenden(arq):
    t = _tarea_texto_ids()
    c = entrenar_una("estandar", t, 0, Config(epocas=30, lote=16, lr=0.003, arquitectura=arq))
    assert c.valor == 1.0, arq
    if arq.endswith("_rl"):
        assert 0 < c.medidas["fraccion_atendida"] <= 1


def test_reinforce_premia_las_elecciones_que_aciertan():
    from laboratorio.banco.arquitecturas import ExtractorAtencionBinaria

    ext = ExtractorAtencionBinaria(PIEZAS["estandar"], 10, dim=4)
    ext.log_probs = torch.tensor([-1.0, -1.0], requires_grad=True)
    salida = torch.tensor([[2.0, 0.0], [2.0, 0.0]])  # predice clase 0 en ambos
    perdida = ext.perdida_extra(salida, torch.tensor([0, 1]))  # acierta el 1.º, falla el 2.º
    perdida.backward()
    assert ext.log_probs.grad[0] < 0 < ext.log_probs.grad[1]  # sube la probabilidad de lo que acertó


def test_competencia_incluye_tus_arquitecturas():
    from laboratorio.banco import competencia

    ids = _tarea_texto_ids()
    buenas = ["excelente producto me encanta", "muy bueno lo recomiendo", "perfecto llegó rápido y bien"]
    malas = ["pésimo no funciona nada", "muy malo se rompió", "horrible no lo compren"]
    filas_hf = [{"text": (buenas if i % 2 else malas)[i % 3], "label": i % 2} for i in range(60)]
    fuente = datos_reales.FuenteHF("falsa", "x/y", "texto", "text", "prueba", 2, lote=16)
    plano = datos_reales.armar_tarea(fuente, filas_hf, filas_hf[:20], representacion="plano")
    elegir = lambda nombre, rep="plano": ids if rep == "ids" else plano  # noqa: E731
    original = competencia.cargar_tarea
    competencia.cargar_tarea = elegir
    import laboratorio.banco.evaluar as ev
    original_ev = ev.cargar_tarea
    ev.cargar_tarea = elegir
    try:
        filas = competir(["falsa"], [], {"semillas": (0, 1), "epocas": 3, "lote": 16}, aviso=lambda *_: None)
        texto = informe_competencia(filas, "2")
    finally:
        competencia.cargar_tarea, ev.cargar_tarea = original, original_ev
    inv = [f for f in filas if f.investigacion]
    assert {f.arquitectura for f in inv} == set(INVESTIGACION["texto"])
    assert {"atencion_binaria_rl", "atencion_niveles_rl", "atencion_hebbiana"} <= set(INVESTIGACION["texto"])
    assert all(set(f.contra_normales) == {"texto_cnn", "transformer", "fasttext"} for f in inv)
    assert "vs fastText" in texto and "vs TextCNN" in texto
