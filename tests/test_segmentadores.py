"""Competencia de segmentadores de sílabas (cuaderno 01)."""

import pytest

torch = pytest.importorskip("torch")

from laboratorio.banco.segmentadores import (  # noqa: E402
    competir_segmentadores, cortes, entrenar_red, evaluar, informe_segmentadores, palabras_de, unir,
)
from laboratorio.banco.silabas_referencia import REFERENCIA  # noqa: E402
from laboratorio.silabas import silabear  # noqa: E402

PALABRAS = ("casa mesa perro calle noche camino palabra problema hablar ciudad bueno tierra causa aire poeta leer "
            "teatro museo carta campo mundo puerta instante constante hombre siempre producto acción calidad "
            "rápido material precioso pantalla recomiendo funciona tamaño ventana montaña cuaderno").split()


def test_cortes_y_unir_ida_y_vuelta():
    assert cortes("ca-sa") == [0, 1, 0]
    assert cortes("sol") == [0, 0]
    for w, s in list(REFERENCIA.items())[:30]:
        assert unir(w.lower(), cortes(s)) == s.lower()


def test_la_referencia_es_coherente():
    assert len(REFERENCIA) >= 120
    for w, s in REFERENCIA.items():
        assert s.replace("-", "").lower() == w.lower()


def test_evaluar_metricas():
    r = evaluar(silabear)
    assert r["exactas"] == 1.0 and r["f1"] == 1.0 and r["errores"] == []
    r = evaluar(lambda w: w.lower())  # sin cortes: casi todo mal
    assert r["exactas"] < 0.15 and r["f1"] == 0.0


def test_palabras_de_excluye_la_referencia():
    pal = palabras_de(["La casa es bonita y el perro corre"])
    assert "bonita" in pal and "casa" not in pal and "perro" not in pal


@pytest.mark.parametrize("red", ["bilstm", "transformer"])
def test_redes_aprenden(red):
    modelo, info = entrenar_red(red, PALABRAS * 3, 0, epocas=30, paciencia=30, lote=16)
    assert info["parametros"] > 1000
    r = evaluar(lambda w: __import__("laboratorio.banco.segmentadores", fromlist=["_predecir"])._predecir(modelo, [w])[0])
    assert r["f1"] > 0.5


def test_competencia_e_informe():
    r = competir_segmentadores(PALABRAS, semillas=(0,), redes=("bilstm",), aviso=lambda *_: None, epocas=3)
    nombres = {f["nombre"] for f in r["filas"]}
    assert {"reglas", "cuaderno", "bilstm"} <= nombres
    texto = informe_segmentadores(r)
    assert "| reglas | 100.0 %" in texto and "Ejemplos de errores" in texto
