import numpy as np
import pytest

from laboratorio.ensamble import voto_ponderado
from laboratorio.entrenamiento import ProgramadorLR
from laboratorio.inicializadores import InicializadorLorenz, fan_in, puntos_lorenz
from laboratorio.series import fechas, serie_ar, serie_combinada, serie_estacional, serie_tendencia


# ---------- Series ----------
def test_series_repetibles_con_semilla():
    assert np.array_equal(serie_combinada(200, semilla=7), serie_combinada(200, semilla=7))
    assert not np.array_equal(serie_combinada(200, semilla=7), serie_combinada(200, semilla=8))


def test_tendencia_sin_ruido_es_recta():
    assert np.allclose(serie_tendencia(5, pendiente=2, ruido=0), [0, 2, 4, 6, 8])


def test_estacional_repite_su_ciclo():
    s = serie_estacional(120, periodo=30, tendencia_final=0, ruido=0)
    assert np.allclose(s[:30], s[30:60])


def test_ar1_recupera_su_coeficiente():
    x = serie_ar(20_000, [0.8], semilla=1)
    estimado = np.dot(x[1:], x[:-1]) / np.dot(x[:-1], x[:-1])
    assert estimado == pytest.approx(0.8, abs=0.02)


def test_fechas_diarias():
    f = fechas(3, "2024-02-28")
    assert [str(d) for d in f] == ["2024-02-28", "2024-02-29", "2024-03-01"]


# ---------- Programador de tasa de aprendizaje ----------
def test_programador_reduce_tras_paciencia_y_respeta_minimo():
    p = ProgramadorLR(lr=1.0, paciencia=2, factor=0.1, lr_minimo=0.05, mayor_es_mejor=True)
    lrs = [p.paso(m) for m in (0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6)]
    assert lrs == [1.0, 1.0, 1.0, 0.1, 0.1, 0.05, 0.05, 0.05]


def test_programador_mejora_reinicia_paciencia():
    p = ProgramadorLR(lr=1.0, paciencia=2, factor=0.5, mayor_es_mejor=False)
    for m in (3.0, 3.1, 2.0, 2.1):
        p.paso(m)
    assert p.lr == 1.0 and p.mejor == 2.0 and p.epocas_sin_mejora == 1


def test_programador_valida_parametros():
    with pytest.raises(ValueError):
        ProgramadorLR(lr=1, factor=1.5)


# ---------- Inicializador de Lorenz ----------
def test_fan_in_usa_entradas():
    assert fan_in((64, 3)) == 3          # Linear(3 → 64)
    assert fan_in((16, 3, 5, 5)) == 75   # Conv2d(3 → 16, 5×5)
    assert fan_in((10,)) == 1


def test_lorenz_es_caotico_pero_acotado():
    p = puntos_lorenz(5000)
    assert np.isfinite(p).all() and np.abs(p).max() < 60
    assert p[1000:, 0].min() < -5 < 5 < p[1000:, 0].max()  # visita ambas «alas»


def test_inicializador_escala_he_y_no_repite():
    init = InicializadorLorenz(total_puntos=50_000)
    w1, w2 = init((256, 64)), init((256, 64))
    assert w1.shape == (256, 64)
    assert w1.std() == pytest.approx(np.sqrt(2 / 64), rel=1e-6)
    assert not np.allclose(w1, w2)
    assert np.array_equal(InicializadorLorenz(total_puntos=50_000)((256, 64)), w1)  # determinista
    with pytest.raises(ValueError):
        init((1000, 1000))


# ---------- Ensamble ----------
class _Fijo:
    def __init__(self, proba):
        self.proba = np.asarray(proba, dtype=float)

    def predict_proba(self, X):
        return np.tile(self.proba, (len(X), 1))


def test_voto_ponderado_respeta_pesos():
    modelos = {"a": _Fijo([0.9, 0.1]), "b": _Fijo([0.2, 0.8]), "sin_proba": object()}
    X = np.zeros((3, 2))
    assert voto_ponderado(X, modelos, {"a": 1, "b": 0.5}).tolist() == [0, 0, 0]
    assert voto_ponderado(X, modelos, {"a": 0.2, "b": 1}).tolist() == [1, 1, 1]


def test_voto_ponderado_con_modelos_reales():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.datasets import make_moons
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    X, y = make_moons(400, noise=0.2, random_state=0)
    modelos = {"lr": LogisticRegression().fit(X, y), "arbol": DecisionTreeClassifier(max_depth=5, random_state=0).fit(X, y)}
    acierto = (voto_ponderado(X, modelos, {"lr": 0.3, "arbol": 0.7}) == y).mean()
    assert acierto > 0.9
    assert sklearn.__version__
