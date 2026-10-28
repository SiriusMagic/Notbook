"""Datos reales, parada temprana y guardado de modelos (sin internet: filas falsas)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")
Image = pytest.importorskip("PIL.Image")

import numpy as np  # noqa: E402

from laboratorio.banco import datos_reales  # noqa: E402
from laboratorio.banco.__main__ import main  # noqa: E402
from laboratorio.banco.evaluar import Config, cargar_modelo, entrenar_una, _medir, _predecir  # noqa: E402
from laboratorio.banco.tareas import cargar_tarea  # noqa: E402


def _filas_imagen(n, semilla):
    rng = np.random.default_rng(semilla)
    filas = []
    for i in range(n):
        clase = i % 2
        pix = rng.integers(0, 60, (8, 8)).astype(np.uint8)
        if clase:
            pix[:, :4] += 150  # la clase 1 tiene la mitad izquierda clara
        filas.append({"image": Image.fromarray(pix), "label": clase})
    return filas


def _filas_texto(n):
    buenas = ["excelente producto me encanta", "muy bueno lo recomiendo", "perfecto llegó rápido"]
    malas = ["pésimo no funciona", "muy malo se rompió", "horrible no lo compren"]
    return [{"text": (buenas if i % 2 else malas)[i % 3], "label": i % 2} for i in range(n)]


FUENTE_IMG = datos_reales.FuenteHF("falsa_img", "x/y", "imagen", "image", "prueba", 2, epocas=30, lote=16)
FUENTE_TXT = datos_reales.FuenteHF("falsa_txt", "x/z", "texto", "text", "prueba", 2, epocas=30, lote=16)


def test_imagenes_se_aplanan_y_normalizan_con_entreno():
    t = datos_reales.armar_tarea(FUENTE_IMG, _filas_imagen(80, 0), _filas_imagen(20, 1))
    assert t.x_entreno.shape == (80, 64) and t.x_prueba.shape == (20, 64)
    assert t.x_entreno.dtype == torch.float32 and t.y_entreno.dtype == torch.int64
    assert abs(float(t.x_entreno.mean())) < 1e-4  # normalizado con la media de entrenamiento
    assert t.lote == 16 and t.tipo == "clasificacion"


def test_texto_a_vector_sin_mirar_la_prueba():
    t = datos_reales.armar_tarea(FUENTE_TXT, _filas_texto(60), _filas_texto(10))
    assert t.x_entreno.shape == (60, datos_reales.DIM_TEXTO)
    c = entrenar_una("estandar", t, 0, Config(epocas=20, lr=0.01))
    assert c.valor == 1.0  # frases separables: debe acertar todo


def test_cache_evita_descargar_dos_veces(tmp_path, monkeypatch):
    monkeypatch.setenv("LABORATORIO_CACHE", str(tmp_path))
    monkeypatch.setitem(datos_reales.FUENTES, "falsa_img", FUENTE_IMG)
    llamadas = []

    def falso(repo, split, n, semilla):
        llamadas.append(split)
        return _filas_imagen(n, 0 if split == "train" else 1)

    monkeypatch.setattr(datos_reales, "_muestra", falso)
    a = datos_reales.cargar_real("falsa_img", entreno=40, prueba=10)
    b = datos_reales.cargar_real("falsa_img", entreno=40, prueba=10)
    assert llamadas == ["train", "test"]  # la segunda vez viene de la caché
    assert torch.equal(a.x_entreno, b.x_entreno) and torch.equal(a.y_prueba, b.y_prueba)


def test_barajar_antes_de_cortar_da_todas_las_clases(monkeypatch):
    datasets = pytest.importorskip("datasets")
    ordenado = datasets.Dataset.from_dict({"text": ["x"] * 100, "label": [0] * 50 + [1] * 50})  # como IMDB
    monkeypatch.setattr(datasets, "load_dataset", lambda repo, split: ordenado)
    muestra = datos_reales._muestra("x/imdb", "train", 20, 1234)
    assert set(muestra["label"]) == {0, 1}


def test_parada_temprana_corta_y_vuelve_a_la_mejor_epoca():
    c = entrenar_una("estandar", "lunas", 0, Config(epocas=2000, paciencia=10, lr=0.05))
    d = c.detalle
    assert d["epocas_entrenadas"] < 2000
    assert d["epocas_entrenadas"] == d["mejor_epoca"] + 10  # cortó tras 10 épocas sin mejorar
    assert c.valor > 0.9


def test_por_lotes_funciona_con_cada_pieza():
    for pieza in ["estandar", "hmn", "sma", "rmb"]:
        c = entrenar_una(pieza, "digitos", 0, Config(epocas=3, lote=64, lr=0.005))
        assert not c.fallo and c.valor > 0.3, pieza


def test_guardar_y_recargar_da_el_mismo_resultado(tmp_path):
    t = cargar_tarea("lunas")
    for pieza in ["ida", "sma"]:
        c = entrenar_una(pieza, t, 1, Config(epocas=150, paciencia=20, guardar=str(tmp_path)))
        modelo = cargar_modelo(c.detalle["modelo"])
        with torch.no_grad():
            assert _medir(t, _predecir(modelo, t.x_prueba), t.y_prueba) == pytest.approx(c.valor)


def test_linea_de_comandos(tmp_path, capsys):
    main(["--pieza", "ida", "--tarea", "xor", "--semillas", "2", "--epocas", "5", "--paciencia", "2", "--salida", str(tmp_path)])
    assert (tmp_path / "informe.md").read_text(encoding="utf-8").count("| ida | xor |") == 1
    assert "xor" in capsys.readouterr().out
