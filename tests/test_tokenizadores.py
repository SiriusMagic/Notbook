"""Competencia de tokenizadores (cuaderno 01), sin internet."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from laboratorio.banco.arquitecturas import PosicionJerarquica, crear_modelo  # noqa: E402
from laboratorio.banco.piezas import PIEZAS  # noqa: E402
from laboratorio.banco.tokenizadores import (  # noqa: E402
    INICIO, Vocabulario, _silabear_cuaderno, competir_tokenizadores, informe_tokenizadores, metricas_intrinsecas,
    tokenizadores_disponibles,
)

FRASE = "Me encantó el producto, llegó rápido."


def test_silabas_marcan_inicio_de_palabra():
    t = tokenizadores_disponibles()["silabas"]
    assert t.tokens(FRASE) == ["▁me", "▁en", "can", "tó", "▁el", "▁pro", "duc", "to", "▁,", "▁lle", "gó", "▁rá", "pi", "do", "▁."]


def test_heuristica_del_cuaderno_tal_cual():
    # Se conserva sin arreglar para medirla; estos cortes son los que hacía el cuaderno (varios incorrectos).
    assert _silabear_cuaderno("país") == ["país"]  # el hiato no se separa (error del original)
    assert _silabear_cuaderno("construir") == ["con", "struir"]
    assert _silabear_cuaderno("hablar") == ["habl", "ar"]
    assert _silabear_cuaderno("sol") == ["sol"]


def test_caracteres_y_palabras():
    d = tokenizadores_disponibles()
    assert d["palabras"].tokens("Hola, mundo") == ["▁hola", "▁,", "▁mundo"]
    assert d["caracteres"].tokens("sí no") == ["▁s", "í", "▁n", "o"]


@pytest.mark.parametrize("nombre", ["bpe", "unigram"])
def test_tokenizadores_entrenables(nombre):
    pytest.importorskip("tokenizers" if nombre == "bpe" else "sentencepiece")
    t = tokenizadores_disponibles()[nombre]
    t.entrenar([FRASE, "el producto llegó roto y tarde", "muy buen producto, lo recomiendo"] * 30)
    toks = t.tokens(FRASE)
    assert toks and any(x.startswith(INICIO) for x in toks)


def test_vocabulario_y_metricas():
    t = tokenizadores_disponibles()["silabas"]
    listas = [t.tokens("la casa es bonita"), t.tokens("la casa es grande")]
    v = Vocabulario(listas, tamano=50)
    m = v.codificar([t.tokens("la casa es enorme")], largo=8)
    assert m[0, 0] > 1 and (m[0] == 1).sum() == 3  # «▁e», «nor», «me» no estaban en el entrenamiento → desconocidas
    assert v.inicio[v.indice["▁ca"]] and not v.inicio[v.indice["sa"]]
    met = metricas_intrinsecas(t, v, ["la casa es enorme"], largo=8)
    assert met["fertilidad"] == pytest.approx(7 / 4) and met["desconocidas_pct"] == pytest.approx(100 * 3 / 7, abs=0.01)


def test_posicion_jerarquica_palabra_y_pares():
    inicio = torch.tensor([False, True, True, False, False, True])
    ids = torch.tensor([[2, 3, 4, 5, 3, 2]])
    p = PosicionJerarquica(10, 4, "palabra", inicio)
    pos = torch.arange(6).expand_as(ids)
    emp = p.inicio[ids] | (pos == 0)
    g = emp.long().cumsum(1) - 1
    assert g.tolist() == [[0, 0, 0, 1, 1, 2]]
    assert p(ids).shape == (1, 6, 4)
    assert PosicionJerarquica(10, 4, "pares", None)(ids).shape == (1, 6, 4)


def test_competencia_de_tokenizadores_sin_internet():
    buenas = ["excelente producto me encanta", "muy bueno lo recomiendo", "perfecto llegó rápido y bien"]
    malas = ["pésimo no funciona nada", "muy malo se rompió", "horrible no lo compren"]
    textos = [(buenas if i % 2 else malas)[i % 3] for i in range(60)]
    y = [i % 2 for i in range(60)]
    r = competir_tokenizadores("falsa", ["silabas", "palabras", "caracteres"], arquitecturas=("texto_cnn",),
                               cambios={"semillas": (0, 1), "epocas": 15, "lote": 16, "lr": 0.003},
                               aviso=lambda *_: None, textos=(textos, y, textos[:20], y[:20]))
    assert set(r["intrinsecas"]) == {"silabas", "palabras", "caracteres"}
    arqs = {(f["tokenizador"], f["arquitectura"]) for f in r["filas"]}
    assert ("silabas", "transformer_jer_palabra") in arqs and ("silabas", "transformer_jer_pares") in arqs
    texto = informe_tokenizadores(r)
    assert "Tokens por palabra" in texto and "| silabas | texto_cnn |" in texto


def test_transformer_jerarquico_se_guarda_y_recarga():
    from laboratorio.banco.tokenizadores import tarea_tokenizada

    t = tokenizadores_disponibles()["silabas"]
    textos = ["muy bueno", "muy malo", "excelente", "pésimo"] * 10
    tarea, _ = tarea_tokenizada("x", t, textos, [1, 0, 1, 0] * 10, textos, [1, 0, 1, 0] * 10, 2, largo=8)
    m = crear_modelo(PIEZAS["estandar"], tarea, "transformer_jer_palabra", 8)
    assert m(tarea.x_entreno[:3]).shape == (3, 2)
