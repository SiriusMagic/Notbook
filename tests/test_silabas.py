import pytest

from laboratorio.silabas import TokenizadorSilabas, silabas, silabas_a_vector, silabear

CASOS = {
    # sílaba directa, dígrafos y grupos inseparables
    "casa": "ca-sa", "perro": "pe-rro", "chocolate": "cho-co-la-te", "calle": "ca-lle",
    "hablar": "ha-blar", "otro": "o-tro",
    # tres y cuatro consonantes
    "instante": "ins-tan-te", "abstracto": "abs-trac-to", "transporte": "trans-por-te",
    "construir": "cons-truir", "obstruir": "obs-truir",
    # hiato, diptongo y triptongo
    "poeta": "po-e-ta", "leer": "le-er", "país": "pa-ís", "río": "rí-o", "aéreo": "a-é-re-o",
    "ciudad": "ciu-dad", "canción": "can-ción", "buey": "buey", "Uruguay": "U-ru-guay",
    # qu, gu, gü, y, h
    "querer": "que-rer", "guerra": "gue-rra", "pingüino": "pin-güi-no",
    "yema": "ye-ma", "hoy": "hoy", "rey": "rey", "Ahora": "A-ho-ra", "prohibir": "pro-hi-bir",
}


@pytest.mark.parametrize("palabra,esperado", CASOS.items())
def test_silabear(palabra, esperado):
    assert silabear(palabra) == esperado


def test_sin_vocales_y_vacio():
    assert silabas("") == []
    assert silabas("pst") == ["pst"]


def test_silabas_reconstruyen_la_palabra():
    for palabra in CASOS:
        assert "".join(silabas(palabra)) == palabra


def test_vector_de_cortes():
    assert silabas_a_vector("casa", "ca-sa") == ("casa", [0, 1, 0])
    assert silabas_a_vector("sol", "sol") == ("sol", [0, 0])
    palabra = "chocolate"
    assert silabas_a_vector(palabra, silabear(palabra))[1] == [0, 0, 1, 0, 1, 0, 1, 0]


def test_tokenizador_ida_y_vuelta():
    tok = TokenizadorSilabas()
    tok.construir_vocabulario(["Hola, mundo.", "El mundo canta"])
    assert tok.tokenizar("Hola, mundo.") == ["ho", "la", ",", "mun", "do", "."]
    ids = tok.a_ids("mundo hola")
    assert tok.a_silabas(ids) == ["mun", "do", "ho", "la"]
    assert tok.a_ids("xyzzy") == [1] * len(tok.tokenizar("xyzzy"))  # desconocidas → <unk>
    assert tok.vocab["<pad>"] == 0
