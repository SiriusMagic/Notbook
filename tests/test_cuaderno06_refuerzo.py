"""Cuaderno 06: entornos de refuerzo (código, cubos y ReroutingNetwork)."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("torch")

from laboratorio.banco import refuerzo_06 as r  # noqa: E402


def test_codigo_se_puede_escribir_hola_pero_no_hola_mundo():
    e = r.entorno_codigo("hola")
    e.reset(seed=0)
    for tok in ["print", "(", '"', "h", "o", "l", "a", '"', ")", "_EOS_"]:
        _, recompensa, terminado, _, info = e.step(e.acciones.index(tok))
    assert terminado and info["exito"] and recompensa > 5
    assert " " not in e.acciones  # «hola mundo» no se puede escribir: no hay espacio


def test_codigo_variables_y_suma():
    e = r.entorno_codigo("12")
    e.reset(seed=0)
    for linea in (["a", "=", "5"], ["b", "=", "7"], ["print", "(", "a", "+", "b", ")"]):
        for tok in linea + ["_EOS_"]:
            *_, info = e.step(e.acciones.index(tok))
    assert e.salida == "12" and info["exito"]


def test_cubos_empujan_juntos_y_terminan():
    e = r.entorno_cubos(normalizar=True, max_pasos=5)
    e.reset(seed=0)
    e.c1, e.c2, e.cubo = np.array([3, 3]), np.array([3, 3]), np.array([4, 3])
    e.step(np.array([3, 3]))  # los dos a la derecha
    assert e.cubo.tolist() == [5, 3]
    e.step(np.array([3, 4]))  # solo uno empuja: no se mueve
    assert e.cubo.tolist() == [5, 3]
    for _ in range(5):
        *_, cortado, _ = e.step(np.array([4, 4]))
    assert cortado  # el límite de pasos corta el episodio


def test_rerouting_optimo_y_rapido():
    assert r.optimo_rerouting() == pytest.approx(35.8, abs=0.1)
    assert r.rerouting("arreglado", episodios=5) >= 0


def test_cubos_guiados_premian_acercarse():
    e = r.entorno_cubos(normalizar=True, guiar=True)
    e.reset(seed=0)
    e.c1, e.c2, e.cubo = np.array([0, 0]), np.array([0, 1]), np.array([5, 5])
    e.cerca = np.linalg.norm(e.c1 - e.cubo) + np.linalg.norm(e.c2 - e.cubo)
    _, recompensa, *_ = e.step(np.array([3, 3]))  # los dos se acercan
    assert recompensa > -0.1


def test_informe_incluye_cubos_guiados():
    cubos = {"al_azar": {"exito": 0, "pasos": 200}, "_pasos": {"ppo_cuaderno": 1, "ppo_arreglado": 1, "ppo_guiado": 3},
             "ppo_cuaderno": [{"exito": 0, "pasos": 200}], "ppo_arreglado": [{"exito": 0, "pasos": 200}],
             "ppo_guiado": [{"exito": 0.6, "pasos": 90}]}
    codigo = {"hola": {"al_azar": {"exito": 0}, "ppo": [{"exito": 0}]}}
    texto = r.informe(codigo, cubos, {"optimo": 35.8, "al_azar": 18, "cuaderno": [1.0], "arreglado": [35.0]}, 1)
    assert "guía hacia el cubo" in texto and "60%" in texto
