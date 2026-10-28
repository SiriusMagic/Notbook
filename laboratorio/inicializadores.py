"""Pesos iniciales a partir del atractor de Lorenz (un sistema caótico).

Viene de «Inicializador de pesos con ruido Perlin y atractor de Lorenz»
(02_capas_y_activaciones). Arreglo: el cuaderno calculaba el «fan_in» con el
número de salidas de la capa en lugar del de entradas, así que la escala de los
pesos era incorrecta en capas no cuadradas. Aquí se usa la forma de PyTorch:
pesos ``(salidas, entradas)``.

Solo NumPy; para PyTorch: ``capa.weight.data = torch.from_numpy(pesos).float()``.
"""

from __future__ import annotations

import numpy as np


def puntos_lorenz(n: int, sigma: float = 10.0, rho: float = 28.0, beta: float = 8 / 3,
                  dt: float = 0.01, inicio=(0.1, 0.0, 0.0)) -> np.ndarray:
    """Trayectoria de ``n`` pasos del atractor de Lorenz, forma ``(n, 3)``."""
    x, y, z = inicio
    out = np.empty((n, 3))
    for i in range(n):
        x, y, z = (x + sigma * (y - x) * dt,
                   y + (x * (rho - z) - y) * dt,
                   z + (x * y - beta * z) * dt)
        out[i] = (x, y, z)
    return out


def fan_in(forma: tuple[int, ...]) -> int:
    """Entradas por neurona. Lineal: (salidas, entradas). Convolución: (salidas, entradas, kh, kw)."""
    if len(forma) < 2:
        return 1
    return int(np.prod(forma[1:]))


class InicializadorLorenz:
    """Entrega pesos con escala He (√(2/fan_in)) tomados en orden de una trayectoria caótica.

    Cada llamada usa el siguiente tramo de la trayectoria, así dos capas no reciben
    los mismos valores. Es determinista: misma configuración → mismos pesos.
    """

    def __init__(self, total_puntos: int = 200_000, descartar: int = 1000, **parametros) -> None:
        # Se descartan los primeros pasos (transitorio) y se aplanan x, y, z.
        self._datos = puntos_lorenz(total_puntos + descartar, **parametros)[descartar:].T.ravel()
        self._pos = 0

    def __call__(self, forma: tuple[int, ...]) -> np.ndarray:
        n = int(np.prod(forma))
        if self._pos + n > len(self._datos):
            raise ValueError(f"No quedan puntos de Lorenz para una capa de forma {forma}; aumenta total_puntos.")
        tramo = self._datos[self._pos : self._pos + n]
        self._pos += n
        std = tramo.std()
        normal = (tramo - tramo.mean()) / std if std > 0 else np.zeros_like(tramo)
        return (normal * np.sqrt(2.0 / fan_in(forma))).reshape(forma)


# ---------------------------------------------------------------- Ruido Perlin
def ruido_perlin(x: np.ndarray, y: np.ndarray, semilla: int = 0) -> np.ndarray:
    """Ruido Perlin 2D clásico (Ken Perlin, 2002), vectorizado. Valores aproximadamente en [-1, 1].

    Reemplaza a ``opensimplex.noise2`` del cuaderno: es la misma familia de ruido «suave» (valores
    vecinos parecidos) y no necesita instalar nada.
    """
    perm = np.random.default_rng(semilla).permutation(256)
    perm = np.concatenate([perm, perm])
    angulos = np.arange(8) * np.pi / 4
    gradientes = np.stack([np.cos(angulos), np.sin(angulos)], 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    xi, yi = x0 & 255, y0 & 255

    def esquina(dx, dy):
        g = gradientes[perm[perm[xi + dx] + yi + dy] % 8]
        return g[..., 0] * (fx - dx) + g[..., 1] * (fy - dy)

    suave = lambda t: t * t * t * (t * (t * 6 - 15) + 10)  # noqa: E731
    u, v = suave(fx), suave(fy)
    arriba = esquina(0, 0) + u * (esquina(1, 0) - esquina(0, 0))
    abajo = esquina(0, 1) + u * (esquina(1, 1) - esquina(0, 1))
    return (arriba + v * (abajo - arriba)) * np.sqrt(2)


def pesos_perlin(forma: tuple[int, ...], escala: float = 0.05, desfase: float = 0.0, semilla: int = 0) -> np.ndarray:
    """Pesos con ruido Perlin recorrido en diagonal, como la celda «inicializador de pesos» del cuaderno 02
    (``noise2(x=i·escala, y=i·escala·1,5)``), con escala He.

    Arreglo: el cuaderno usaba las salidas como «fan_in»; aquí se usan las entradas (igual que Lorenz).
    """
    i = np.arange(int(np.prod(forma)), dtype=float) + desfase
    ruido = ruido_perlin(i * escala, i * escala * 1.5, semilla)
    return (ruido * np.sqrt(2.0 / fan_in(forma))).reshape(forma)
