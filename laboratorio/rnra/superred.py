"""Cuaderno 16 · RNRA-M: superred de anchos anidados (como las redes «slimmable» y Once-for-All).

Un solo conjunto de pesos sirve para varios tamaños: el ancho ``w`` usa las primeras ``w`` neuronas de cada capa. Como el
tamaño de las activaciones cambia con el ancho, cada capa oculta normaliza sin parámetros (``layer_norm``).
Se entrena con todos los anchos a la vez y los pequeños copian (destilación) a la salida del más grande.

También: cuantización uniforme simétrica de pesos y cuenta de bytes, para comparar contra «entrenar una red por dispositivo»
y «entrenar una grande y cuantizarla después».
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from torch import nn

ANCHOS = (32, 128, 512)


class SuperRed(nn.Module):
    def __init__(self, entradas: int, salidas: int, anchos=ANCHOS, ocultas: int = 2):
        super().__init__()
        self.entradas, self.salidas, self.anchos, self.ocultas = entradas, salidas, tuple(sorted(anchos)), ocultas
        w = max(anchos)
        dims = [entradas] + [w] * ocultas + [salidas]
        self.capas = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        self.ancho = w

    def forward(self, x, ancho: int | None = None):
        w = ancho or self.ancho
        h = x
        for l, capa in enumerate(self.capas):
            sal = w if l < self.ocultas else self.salidas
            ent = self.entradas if l == 0 else w
            h = F.linear(h, capa.weight[:sal, :ent], capa.bias[:sal])
            if l < self.ocultas:
                h = F.gelu(F.layer_norm(h, (w,)))
        return h

    def parametros(self, ancho: int) -> int:
        w = ancho
        return self.entradas * w + w + (self.ocultas - 1) * (w * w + w) + w * self.salidas + self.salidas


def bytes_modelo(modelo: SuperRed, ancho: int, bits: int = 32) -> int:
    """Memoria de los pesos de un ancho a ``bits`` bits por peso (más 4 bytes de escala por tensor si bits < 32)."""
    extra = 0 if bits >= 32 else 4 * 2 * (modelo.ocultas + 1)
    return math.ceil(modelo.parametros(ancho) * bits / 8) + extra


def cuantizar_tensor(t: torch.Tensor, bits: int) -> torch.Tensor:
    """Cuantiza a ``bits`` bits (simétrica, una escala por tensor) y devuelve los valores ya descuantizados."""
    if bits >= 32:
        return t.clone()
    q = 2 ** (bits - 1) - 1
    escala = t.abs().max().clamp(min=1e-12) / max(q, 1)
    return (t / escala).round().clamp(-q, q) * escala


@torch.no_grad()
def cuantizar_modelo(modelo: SuperRed, bits: int) -> SuperRed:
    """Copia del modelo con todos los pesos cuantizados (cuantización posterior al entrenamiento)."""
    import copy

    m = copy.deepcopy(modelo)
    for p in m.parameters():
        p.copy_(cuantizar_tensor(p, bits))
    return m


def entrenar(modelo: SuperRed, x, y, pasos: int = 600, lote: int = 128, lr: float = 3e-3, semilla: int = 0,
             anchos=None, destilar: float = 0.5, clasificacion: bool = True) -> SuperRed:
    """Entrena todos los ``anchos`` a la vez (pérdida suma). Los más pequeños, además, imitan al más grande."""
    anchos = tuple(anchos or modelo.anchos)
    gen = torch.Generator().manual_seed(semilla)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    for paso in range(pasos):
        for g in opt.param_groups:
            g["lr"] = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * paso / pasos)))
        idx = torch.randint(0, len(x), (lote,), generator=gen)
        xb, yb = x[idx], y[idx]
        opt.zero_grad(set_to_none=True)
        salidas = {w: modelo(xb, w) for w in anchos}
        perdida = sum(F.cross_entropy(s, yb) if clasificacion else F.mse_loss(s, yb) for s in salidas.values())
        if destilar and len(anchos) > 1 and clasificacion:
            grande = F.softmax(salidas[anchos[-1]].detach(), 1)
            perdida = perdida + destilar * sum(F.kl_div(F.log_softmax(salidas[w], 1), grande, reduction="batchmean") for w in anchos[:-1])
        perdida.backward()
        opt.step()
    return modelo


@torch.no_grad()
def exactitud(modelo: SuperRed, x, y, ancho: int | None = None) -> float:
    return (modelo(x, ancho).argmax(1) == y).float().mean().item()


@torch.no_grad()
def latencia_ms(modelo: SuperRed, x, ancho: int, repeticiones: int = 50) -> float:
    """Mínimo del tiempo de un paso hacia adelante (el mínimo resiste mejor el ruido de otros procesos). CPU de esta
    máquina, no del dispositivo real."""
    modelo(x, ancho)
    t = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        modelo(x, ancho)
        t.append((time.perf_counter() - t0) * 1000)
    return float(min(t))
