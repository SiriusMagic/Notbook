import pytest

pytest.importorskip("tokenizers")

from laboratorio.banco.tokenizadores import INICIO, SilabasBPE, SilabasYPalabras  # noqa: E402
from laboratorio.silabas import silabas  # noqa: E402

TEXTOS = ["el producto llegó rápido pero la calidad es pésima", "la calidad del producto es buena",
          "llegó tarde pero la calidad es buena"] * 30


def test_silabas_bpe_corta_en_bordes_de_silaba():
    tok = SilabasBPE(300)
    tok.entrenar(TEXTOS)
    tokens = tok.tokens("la calidad del producto")
    assert "▁calidad" in tokens  # palabra frecuente: entera
    # al juntar los tokens de cada palabra se recupera la palabra
    assert "".join(t.lstrip(INICIO) for t in tokens) == "lacalidaddelproducto"


def test_silabas_bpe_palabra_nueva_no_se_pierde():
    tok = SilabasBPE(300)
    tok.entrenar(TEXTOS)
    tokens = tok.tokens("increíble")
    assert "".join(t.lstrip(INICIO) for t in tokens) == "increíble"


def test_silabas_y_palabras():
    tok = SilabasYPalabras(frecuentes=3)
    tok.entrenar(TEXTOS)
    tokens = tok.tokens("la calidad increíble")
    assert tokens[0] == INICIO + "la"
    assert [t.lstrip(INICIO) for t in tokens[-len(silabas("increíble")):]] == silabas("increíble")
