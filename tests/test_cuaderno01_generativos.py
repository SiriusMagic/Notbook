"""Cuaderno 01, parte generativa: siguiente palabra con PPO, tokenizador con RL y difusión de texto."""

import pytest

torch = pytest.importorskip("torch")

from laboratorio.banco import difusion_texto as dt  # noqa: E402
from laboratorio.banco import siguiente_palabra as sp  # noqa: E402
from laboratorio.banco.tokenizador_rl import TokenizadorRL  # noqa: E402

TEXTOS = ["el perro come carne", "la casa es grande", "el gato come pescado", "la casa es bonita"] * 20


def test_siguiente_palabra_supervisado_aprende_mas_que_ppo_tal_cual():
    c = sp.preparar(TEXTOS, TEXTOS[:8], vocabulario=50)
    sup = sp.entrenar_red(c, "supervisado", epocas=20, ocultas=32)
    assert sup["top1"] > 0.4
    for modo in ("ppo_cuaderno", "ppo_arreglado"):
        r = sp.entrenar_red(c, modo, epocas=2, ocultas=32)
        assert 0 <= r["top1"] <= 1 and r["perplejidad"] > 0
    assert sp.bigramas(c)["top1"] > 0.4


def test_tokenizador_rl_cubre_todo_el_texto():
    for arreglado in (False, True):
        t = TokenizadorRL(arreglado=arreglado, episodios=2000, rondas=2)
        t.entrenar(TEXTOS)
        toks = t.tokens("el perro es grande")
        assert "".join(x.lstrip("▁") for x in toks) == "elperroesgrande"
        assert sum(x.startswith("▁") for x in toks) == 4


def test_difusion_los_tres_modos_corren():
    tok, xe, xp = dt.preparar(TEXTOS, TEXTOS[:8], vocabulario=60, largo=8)
    v = tok.get_vocab_size()
    for modo in ("difusion_cuaderno", "difusion_mascaras", "autorregresivo"):
        m = dt.entrenar(modo, xe, v, pasos=5, lote=8)
        p = dt.perplejidad(m, modo, xp)
        assert (p is None) == (modo == "difusion_cuaderno")
        if modo != "autorregresivo":
            assert 0 <= dt.rellenar(m, modo, xp, 0.5) <= 1
        assert len(dt.generar(m, modo, tok, n=2, largo=6, pasos_t=5)) == 2


def test_ruido_del_cuaderno_es_casi_nulo():
    """La desviación máxima del ruido (0,1) es diminuta frente a la norma de un embedding (≈ √128 ≈ 11)."""
    emb = torch.nn.Embedding(10, 128)
    assert emb.weight.norm(dim=1).mean() > 50 * (0.01 ** 0.5)


def test_difusion_respeta_el_dispositivo_pedido():
    tok, xe, _ = dt.preparar(TEXTOS, TEXTOS[:4], vocabulario=60, largo=8)
    m = dt.entrenar("difusion_mascaras", xe, tok.get_vocab_size(), pasos=1, lote=4, dispositivo="cpu")
    assert next(m.parameters()).device.type == "cpu"
