"""Aprendizaje automático clásico (scikit-learn) contra la red estándar del banco.

Viene de la celda «comparar clasificadores» del cuaderno 03 (regresión logística, KNN, SVM, árbol de decisión
y Naive Bayes). Se agregan **Random Forest** y **Gradient Boosting** (``HistGradientBoosting``, el mismo método
que XGBoost y LightGBM), que son lo que la industria usa hoy para datos en tabla.

Mismas tareas, mismos datos de prueba y mismas semillas que el resto del banco; el veredicto usa la misma prueba
t pareada y la corrección por comparaciones múltiples.
"""

from __future__ import annotations

import math
import time

from .evaluar import Comparacion, Config, Corrida, ajustar_multiples, comparar, correr, _ICONO
from .tareas import cargar_tarea


def _modelos(tipo: str, semilla: int) -> dict:
    from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                                  RandomForestClassifier, RandomForestRegressor)
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, SVR
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    escalado = lambda m: make_pipeline(StandardScaler(), m)  # noqa: E731  (KNN y SVM dependen de la escala)
    if tipo == "clasificacion":
        return {
            "Regresión logística": escalado(LogisticRegression(max_iter=2000)),
            "KNN (k=5)": escalado(KNeighborsClassifier(5)),
            "SVM (núcleo RBF)": escalado(SVC()),
            "Árbol de decisión (prof. 5)": DecisionTreeClassifier(max_depth=5, random_state=semilla),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(200, random_state=semilla, n_jobs=-1),
            "Gradient Boosting": HistGradientBoostingClassifier(random_state=semilla),
            "Votación ponderada (cuaderno 08)": _votacion_ponderada(semilla),
        }
    return {
        "Ridge (lineal)": escalado(Ridge()),
        "KNN (k=5)": escalado(KNeighborsRegressor(5)),
        "SVM (núcleo RBF)": escalado(SVR()),
        "Árbol de decisión (prof. 5)": DecisionTreeRegressor(max_depth=5, random_state=semilla),
        "Random Forest": RandomForestRegressor(200, random_state=semilla, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=semilla),
    }


def _votacion_ponderada(semilla: int):
    """«weighted_vote» del cuaderno 08: 5 clasificadores (logística, KNN k=7, SVM lineal, árbol prof. 4, Naive
    Bayes) votan con sus probabilidades, cada uno pesado por su exactitud en validación cruzada de 5 partes.

    El SVM da probabilidades con ``CalibratedClassifierCV``: ``SVC(probability=True)``, que usaba el cuaderno,
    queda obsoleto en scikit-learn 1.11."""
    from sklearn.base import BaseEstimator, ClassifierMixin, clone
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    class VotacionPonderada(ClassifierMixin, BaseEstimator):
        def fit(self, x, y):
            base = [make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                    make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
                    make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(kernel="linear"), ensemble=False)),
                    DecisionTreeClassifier(max_depth=4, random_state=semilla), GaussianNB()]
            pliegues = KFold(5, shuffle=True, random_state=semilla)
            puntajes = [cross_val_score(clone(m), x, y, cv=pliegues).mean() for m in base]
            self.pesos_ = [p / sum(puntajes) for p in puntajes]
            self.modelos_ = [m.fit(x, y) for m in base]
            self.classes_ = self.modelos_[0].classes_
            return self

        def predict(self, x):
            proba = sum(w * m.predict_proba(x) for w, m in zip(self.pesos_, self.modelos_))
            return self.classes_[proba.argmax(1)]

    return VotacionPonderada()


def _medir(tipo: str, y, pred) -> float:
    return float((pred == y).mean()) if tipo == "clasificacion" else float(((pred - y) ** 2).mean())


def correr_clasicos(tarea: str, semillas=(0, 1, 2, 3, 4)) -> dict[str, list[Corrida]]:
    """Entrena cada modelo clásico con cada semilla. Los que no usan azar dan lo mismo en todas."""
    t = cargar_tarea(tarea)
    x_e, x_p = t.x_entreno.flatten(1).numpy(), t.x_prueba.flatten(1).numpy()
    y_e, y_p = t.y_entreno.numpy(), t.y_prueba.numpy()
    if t.tipo != "clasificacion":
        y_e, y_p = y_e.reshape(len(y_e), -1), y_p.reshape(len(y_p), -1)
        if y_e.shape[1] == 1:
            y_e, y_p = y_e.ravel(), y_p.ravel()
    corridas: dict[str, list[Corrida]] = {}
    for s in semillas:
        for nombre, modelo in _modelos(t.tipo, s).items():
            inicio = time.perf_counter()
            try:
                modelo.fit(x_e, y_e)
                pred_p, pred_e = modelo.predict(x_p), modelo.predict(x_e)
                valor, entreno, fallo = _medir(t.tipo, y_p, pred_p), _medir(t.tipo, y_e, pred_e), False
            except ValueError:  # p. ej. un modelo que no admite salidas múltiples
                valor = entreno = math.nan
                fallo = True
            corridas.setdefault(nombre, []).append(Corrida(nombre, tarea, s, t.metrica, valor, math.nan, 0,
                                                           time.perf_counter() - inicio, fallo, entreno))
    return corridas


def competir_clasicos(tareas: list[str], semillas=(0, 1, 2, 3, 4), aviso=print) -> list[Comparacion]:
    """Cada modelo clásico contra la red estándar (2 capas de 32, Adam), tarea por tarea."""
    comparaciones = []
    for tarea in tareas:
        base = correr("estandar", tarea, config=Config(semillas=tuple(semillas), medir=False))
        for nombre, corr in correr_clasicos(tarea, semillas).items():
            c = comparar(corr, base)
            comparaciones.append(c)
            aviso(f"{tarea:16} {nombre:28} {c.pieza_media:.4f}  (red {c.base_media:.4f})  {c.veredicto}")
    if len(comparaciones) > 1:
        ajustar_multiples(comparaciones)
    return comparaciones


def _fmt(metrica: str, media: float, std: float) -> str:
    if math.isnan(media):
        return "—"
    return f"{media * 100:.1f} ± {std * 100:.1f} %" if metrica == "exactitud" else f"{media:.4f} ± {std:.4f}"


def informe_clasicos(comparaciones: list[Comparacion], semillas) -> str:
    lineas = [
        "# Aprendizaje clásico contra la red estándar",
        "",
        f"Media ± desviación en **datos de prueba**, {len(tuple(semillas))} semillas. La red estándar es la del "
        "banco (2 capas de 32 neuronas, ReLU, Adam). Veredicto desde el punto de vista del modelo clásico: "
        "🟢 = el clásico le gana a la red. Prueba t pareada con corrección de Benjamini-Hochberg.",
        "",
    ]
    for tarea in dict.fromkeys(c.tarea for c in comparaciones):
        grupo = [c for c in comparaciones if c.tarea == tarea]
        mayor = grupo[0].metrica == "exactitud"
        lineas += [f"## {tarea}", "", f"Red estándar: **{_fmt(grupo[0].metrica, grupo[0].base_media, grupo[0].base_std)}**", "",
                   "| Modelo clásico | Prueba | Entrenamiento | vs red |", "|---|---|---|---|"]
        for c in sorted(grupo, key=lambda c: (math.isnan(c.pieza_media), -c.pieza_media if mayor else c.pieza_media)):
            entreno = _fmt(c.metrica, c.pieza_entreno, 0).split(" ±")[0]
            lineas.append(f"| {c.pieza} | {_fmt(c.metrica, c.pieza_media, c.pieza_std)} | {entreno} | {_ICONO[c.veredicto]} |")
        lineas.append("")
    gana = [c for c in comparaciones if c.veredicto == "mejor"]
    lineas += ["## Resumen", "",
               f"El modelo clásico le gana a la red en {len(gana)} de {len(comparaciones)} comparaciones."]
    lineas += [f"- **{c.pieza}** en {c.tarea}" for c in gana]
    return "\n".join(lineas) + "\n"


__all__ = ["competir_clasicos", "correr_clasicos", "informe_clasicos"]
