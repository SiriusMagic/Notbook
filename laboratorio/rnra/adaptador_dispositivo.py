"""Qué configuración le toca a cada dispositivo según su RAM.

``elegir`` busca el ancho más grande de la superred que cabe y, entre esos, la mayor precisión de pesos. Los presupuestos
de ``dispositivos`` se calculan a partir de los tamaños de la propia superred (celular básico = cabe el ancho menor en
32 bits, celular potente = el ancho medio, PC = el mayor). La RAM es **calculada** (bytes de pesos), no medida en un
teléfono real; la latencia se mide en la CPU de esta máquina. Ninguna de las dos reemplaza una prueba en el aparato.
"""

from __future__ import annotations

from dataclasses import dataclass

from .superred import SuperRed, bytes_modelo


@dataclass(frozen=True)
class Dispositivo:
    nombre: str
    ram_bytes: int


def dispositivos(modelo: SuperRed) -> list[Dispositivo]:
    nombres = ["celular_basico", "celular_potente", "pc_con_gpu"]
    return [Dispositivo(n, bytes_modelo(modelo, w, 32)) for n, w in zip(nombres, modelo.anchos)]


def elegir(modelo: SuperRed, ram_bytes: int, bits=(32, 8, 4, 2)) -> tuple[int, int] | None:
    """(ancho, bits) con el mayor ancho que cabe y, de esos, los bits más altos. ``None`` si nada cabe."""
    for w in sorted(modelo.anchos, reverse=True):
        for b in sorted(bits, reverse=True):
            if bytes_modelo(modelo, w, b) <= ram_bytes:
                return w, b
    return None
