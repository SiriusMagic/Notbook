"""Tareas del banco: las mismas que usan los cuadernos, fijas y con división
entrenamiento / prueba para que todas las piezas se midan igual."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..series import serie_estacional

SEMILLA_DATOS = 1234  # los datos son siempre los mismos; lo que cambia entre corridas es la red


@dataclass
class Tarea:
    nombre: str
    tipo: str  # "regresion" o "clasificacion"
    x_entreno: torch.Tensor
    y_entreno: torch.Tensor
    x_prueba: torch.Tensor
    y_prueba: torch.Tensor
    entradas: int
    salidas: int
    epocas: int
    descripcion: str
    lote: int | None = None  # None = todo el conjunto en cada paso (tareas pequeñas)

    @property
    def metrica(self) -> str:
        return "exactitud" if self.tipo == "clasificacion" else "mse"

    @property
    def mayor_es_mejor(self) -> bool:
        return self.tipo == "clasificacion"


def _dividir(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, prueba: float = 0.3):
    idx = rng.permutation(len(x))
    corte = int(len(x) * (1 - prueba))
    a, b = idx[:corte], idx[corte:]
    return x[a], y[a], x[b], y[b]


def _normalizar(entreno: np.ndarray, prueba: np.ndarray):
    media, std = entreno.mean(0), entreno.std(0) + 1e-8
    return (entreno - media) / std, (prueba - media) / std


def _armar(nombre, tipo, x, y, epocas, descripcion, normalizar_y=False):
    rng = np.random.default_rng(SEMILLA_DATOS)
    xe, ye, xp, yp = _dividir(x.astype(np.float32), y, rng)
    xe, xp = _normalizar(xe, xp)
    if tipo == "regresion":
        ye, yp = ye.astype(np.float32).reshape(-1, 1), yp.astype(np.float32).reshape(-1, 1)
        if normalizar_y:
            ye, yp = _normalizar(ye, yp)
        salidas = 1
        conv = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    else:
        salidas = int(y.max()) + 1
        conv = lambda a: torch.tensor(a, dtype=torch.long)  # noqa: E731
    return Tarea(nombre, tipo, torch.tensor(xe, dtype=torch.float32), conv(ye),
                 torch.tensor(xp, dtype=torch.float32), conv(yp), x.shape[1], salidas, epocas, descripcion)


def regresion_cubica() -> Tarea:
    rng = np.random.default_rng(SEMILLA_DATOS)
    x = np.linspace(-3, 3, 300).reshape(-1, 1)
    y = x[:, 0] ** 3 - x[:, 0] + 0.5 * rng.standard_normal(300)
    return _armar("regresion_cubica", "regresion", x, y, 400, "y = x³ − x + ruido (la más usada en el cuaderno 02)", normalizar_y=True)


def circulos() -> Tarea:
    from sklearn.datasets import make_circles
    x, y = make_circles(600, noise=0.2, factor=0.6, random_state=SEMILLA_DATOS)
    return _armar("circulos", "clasificacion", x, y, 300, "dos círculos concéntricos (cuadernos 02 y 03)")


def lunas() -> Tarea:
    from sklearn.datasets import make_moons
    x, y = make_moons(600, noise=0.2, random_state=SEMILLA_DATOS)
    return _armar("lunas", "clasificacion", x, y, 300, "dos medias lunas entrelazadas")


def xor() -> Tarea:
    rng = np.random.default_rng(SEMILLA_DATOS)
    x = rng.uniform(-1, 1, (400, 2))
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(int)
    return _armar("xor", "clasificacion", x, y, 300, "XOR con puntos en los 4 cuadrantes")


def serie_seno(ventana: int = 12) -> Tarea:
    s = serie_estacional(600, periodo=40, amplitud=1.0, tendencia_final=0.0, ruido=0.1, semilla=SEMILLA_DATOS)
    x = np.stack([s[i : i + ventana] for i in range(len(s) - ventana)])
    y = s[ventana:]
    return _armar("serie_seno", "regresion", x, y, 300, f"predecir el siguiente valor de una onda con ruido ({ventana} pasos previos)", normalizar_y=True)


def digitos() -> Tarea:
    from sklearn.datasets import load_digits
    d = load_digits()
    return _armar("digitos", "clasificacion", d.data, d.target, 200, "dígitos escritos a mano 8×8 (versión pequeña de MNIST)")


TAREAS = {
    "regresion_cubica": regresion_cubica,
    "circulos": circulos,
    "lunas": lunas,
    "xor": xor,
    "serie_seno": serie_seno,
    "digitos": digitos,
}

_cache: dict[str, Tarea] = {}


def cargar_tarea(nombre: str) -> Tarea:
    """Tareas pequeñas (TAREAS) o reales de Hugging Face (datos_reales.FUENTES)."""
    from .datos_reales import FUENTES, cargar_real

    if nombre not in TAREAS and nombre not in FUENTES:
        raise KeyError(f"Tarea desconocida: {nombre}. Opciones: {', '.join([*TAREAS, *FUENTES])}")
    if nombre not in _cache:
        _cache[nombre] = TAREAS[nombre]() if nombre in TAREAS else cargar_real(nombre)
    return _cache[nombre]
