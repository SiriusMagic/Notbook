"""Cuaderno 14: maestro (DeepSeek → Hugging Face → reglas), router troncal y parseo de pares."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from laboratorio.banco import expertos_lora as e  # noqa: E402


def test_sin_llave_no_usa_deepseek(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert e.maestro_deepseek() is None
    assert e.maestro_deepseek(llave="") is None


def test_elegir_maestro_cae_a_reglas(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(e, "maestro_hf", lambda modelo=None: None)
    assert e.elegir_maestro().nombre == "reglas"


def test_reglas_etiquetan_las_preguntas_semilla():
    m = e.maestro_reglas()
    textos, etiquetas = e.preguntas_semilla()
    acierto = np.mean([m.etiquetar(t) == d for t, d in zip(textos, etiquetas)])
    assert acierto > 0.8


def test_normalizar_dominio():
    assert e.normalizar_dominio("  Agricultura. ") == "agricultura"
    assert e.normalizar_dominio("Mecánica") == "mecanica"
    assert e.normalizar_dominio("Redes_y_Telecom") == "redes"
    assert e.normalizar_dominio("no sé") == "desconocido"


def test_generar_pares_lee_lineas_json():
    falso = e.Maestro("falso", lambda s, u: 'texto\n{"pregunta": "¿p?", "respuesta": "r"}\n{"mal": 1}\n{roto')
    pares = falso.generar_pares("redes", 2)
    assert pares == [{"pregunta": "¿p?", "respuesta": "r", "dominio": "redes"}]


def test_router_troncal_aprende_los_dominios():
    pytest.importorskip("torch")
    textos, etiquetas = e.preguntas_semilla()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(textos))
    entreno, prueba = idx[:36], idx[36:]
    r = e.comparar_routers([textos[i] for i in entreno], [etiquetas[i] for i in entreno],
                           [textos[i] for i in prueba], [etiquetas[i] for i in prueba], semillas=(0,))
    assert 0 <= r["router_troncal"][0] <= 1 and 0 <= r["logistica"] <= 1
