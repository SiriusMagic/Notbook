"""Corrección ortográfica (cuaderno 01): tu GNN contra la industria."""

import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rapidfuzz")

from laboratorio.banco.correccion import (  # noqa: E402
    CorrectorDiccionario, GCNCadena, competir_correctores, entrenar, frases_de, informe_correctores,
    introducir_errores_en_frase, medir, predecir, preparar,
)

FRASES = [
    "me encanto el producto llego rapido", "muy buena calidad lo recomiendo", "el envio tardo mucho",
    "no funciona bien la bateria", "excelente precio y buena calidad", "llego roto y no sirve",
    "la pantalla es muy bonita", "buen producto para el precio", "no lo recomiendo para nada",
    "muy contento con la compra",
] * 6


def test_gcn_igual_a_la_formula():
    torch.manual_seed(0)
    capa = GCNCadena(3, 2)
    x = torch.randn(1, 4, 3)
    mascara = torch.tensor([[True, True, True, False]])  # 3 letras + relleno
    salida = capa(x, mascara)[0, :3]
    # Fórmula de Kipf y Welling: D^-1/2 (A + I) D^-1/2 X W + b, con A = cadena 0-1-2
    a = torch.tensor([[1.0, 1, 0], [1, 1, 1], [0, 1, 1]])
    d = a.sum(1).rsqrt()
    esperado = (d[:, None] * a * d[None, :]) @ capa.lin(x[0, :3]) + capa.bias
    assert torch.allclose(salida, esperado, atol=1e-6)


def test_errores_reproducibles_y_escenarios():
    rng1, rng2 = random.Random(5), random.Random(5)
    f = "la casa es muy bonita"
    assert introducir_errores_en_frase(f, 2, 0.2, rng1) == introducir_errores_en_frase(f, 2, 0.2, rng2)
    for sucia, limpia in preparar(FRASES[:20], "sustituciones"):
        assert len(sucia) == len(limpia) and sucia.count(" ") == limpia.count(" ")


def test_metricas():
    pares = [("la caza", "la casa"), ("el perro", "el perro")]
    m = medir(["la casa", "el perro"], pares)
    assert m["cer"] == 0 and m["perfectas"] == 1 and m["palabras_ok"] == 1
    m = medir(["la caza", "el perro"], pares)
    assert m["cer"] == pytest.approx((1 / 7 + 0) / 2) and m["palabras_ok"] == pytest.approx(3 / 4)
    # Dos palabras pegadas no anulan el resto de la frase.
    m = medir(["elperro come mucho"], [("", "el perro come mucho")])
    assert m["palabras_ok"] == pytest.approx(2 / 4)


def test_diccionario_corrige():
    d = CorrectorDiccionario(FRASES)
    assert d.corregir("muy buena calidda") == "muy buena calidad"  # transposición
    assert d.corregir("el envo tardo") == "el envio tardo"  # letra que falta
    assert d.corregir("xyzw") == "xyzw"  # demasiado lejos: no inventa


def test_frases_de():
    assert frases_de(["¡Hola! Me encantó, de verdad. Sí"], minimo=5) == ["me encantó de verdad"]  # «hola» y «sí» son muy cortas


@pytest.mark.parametrize("red", ["gnn", "gnn_rl", "gnn_residual", "bilstm", "transformer"])
def test_redes_entrenan_y_corrigen(red):
    pares = preparar(FRASES, "sustituciones")
    modelo, info = entrenar(red, pares, 0, epocas=3, lote=16, muestras_rl=2)
    assert info["parametros"] > 1000 and info["epocas"] >= 1
    pred = predecir(modelo, [s for s, _ in pares[:5]])
    assert len(pred) == 5 and all(isinstance(p, str) for p in pred)


def test_competencia_e_informe():
    r = competir_correctores(FRASES[:40], FRASES[40:], escenarios=("sustituciones",), redes=("gnn",),
                             semillas=(0,), aviso=lambda *_: None, epocas=2, lote=16)
    texto = informe_correctores(r)
    assert "Tu GNN tal cual, supervisada" in texto and "Diccionario" in texto and "Sin corregir" in texto
