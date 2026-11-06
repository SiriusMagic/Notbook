"""MLP con máscaras: ancho máximo fijo, ranuras de neuronas activas o de reserva y conexiones activas o apagadas.

Capa ``l``: ``h = GELU((W ⊙ M) h + b) ⊙ a``. ``M`` es la máscara de conexiones y ``a`` marca qué neuronas están
activas. Se mantiene siempre una regla: una neurona apagada no tiene conexiones (``sincronizar``).

Ojo: los pesos se guardan con el tamaño máximo. Apagar conexiones **no** hace la red más rápida ni más pequeña en
memoria; el presupuesto que se mide es el de conexiones activas (FLOPs lógicos).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MLPMascarado(nn.Module):
    def __init__(self, entradas: int, salidas: int, ancho_max: int, capas_ocultas: int = 2):
        super().__init__()
        self.entradas, self.salidas, self.ancho_max, self.ocultas = entradas, salidas, ancho_max, capas_ocultas
        dims = [entradas] + [ancho_max] * capas_ocultas + [salidas]
        self.capas = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        for i, capa in enumerate(self.capas):
            self.register_buffer(f"m{i}", torch.ones_like(capa.weight))
        for i in range(capas_ocultas):
            self.register_buffer(f"a{i}", torch.ones(ancho_max))
        self._h: list[torch.Tensor] = []

    # --- acceso a máscaras -------------------------------------------------------------------------------
    def mascara(self, l: int) -> torch.Tensor:
        return getattr(self, f"m{l}")

    def activa(self, l: int) -> torch.Tensor:
        return getattr(self, f"a{l}")

    def sincronizar(self) -> None:
        """Una neurona apagada pierde sus conexiones de entrada y de salida."""
        for l in range(self.ocultas):
            a = self.activa(l)
            self.mascara(l).mul_(a[:, None])
            self.mascara(l + 1).mul_(a[None, :])

    # --- paso hacia adelante -----------------------------------------------------------------------------
    def forward(self, x, sondas=None, guardar: bool = False):
        """``sondas``: tensores del tamaño de cada W, en cero, que se suman a ``W ⊙ M``: su gradiente es el gradiente
        «denso» (también en las conexiones apagadas), que necesitan RigL y RNRA para decidir dónde crecer."""
        h = x
        self._h = []
        for l, capa in enumerate(self.capas):
            w = capa.weight * self.mascara(l)
            if sondas is not None:
                w = w + sondas[l]
            if l < self.ocultas:
                h = F.gelu(F.linear(h, w, capa.bias * self.activa(l))) * self.activa(l)
                if guardar:
                    if h.requires_grad:
                        h.retain_grad()
                    self._h.append(h)
            else:
                h = F.linear(h, w, capa.bias)
        return h

    # --- conteos -----------------------------------------------------------------------------------------
    def conexiones(self, l: int | None = None) -> int:
        if l is not None:
            return int(self.mascara(l).sum().item())
        return sum(int(self.mascara(i).sum().item()) for i in range(len(self.capas)))

    def neuronas_activas(self, l: int | None = None) -> int:
        if l is not None:
            return int(self.activa(l).sum().item())
        return sum(self.neuronas_activas(i) for i in range(self.ocultas))

    def parametros_activos(self) -> int:
        return self.conexiones() + self.neuronas_activas() + self.salidas

    def conexiones_densas(self) -> int:
        """Conexiones que tendría la red con todas las neuronas activas y densa (coste de una medida densa)."""
        return sum(int((self.activa(l - 1).sum() if l > 0 else self.entradas) * (self.activa(l).sum() if l < self.ocultas else self.salidas))
                   for l in range(len(self.capas)))

    # --- construcción ------------------------------------------------------------------------------------
    def densa(self, ancho: int) -> None:
        """Red densa con ``ancho`` neuronas activas por capa (las demás ranuras quedan apagadas)."""
        for l in range(self.ocultas):
            self.activa(l).zero_()
            self.activa(l)[:ancho] = 1
            self.mascara(l).fill_(1)
            self.mascara(l + 1).fill_(1)
        self.sincronizar()
        self.limpiar_pesos()

    def esparcida(self, presupuesto: int, activas: int, gen: torch.Generator) -> None:
        """``activas`` neuronas por capa y exactamente ``presupuesto`` conexiones repartidas al azar,
        proporcional al tamaño denso de cada capa."""
        for l in range(self.ocultas):
            self.activa(l).zero_()
            self.activa(l)[:activas] = 1
        tam = []
        for l in range(len(self.capas)):
            ent = self.entradas if l == 0 else activas
            sal = self.salidas if l == self.ocultas else activas
            tam.append(ent * sal)
        presupuesto = min(presupuesto, sum(tam))
        cuota = [int(presupuesto * t / sum(tam)) for t in tam]
        cuota[-1] += presupuesto - sum(cuota)
        for l in range(len(self.capas)):
            validas = self._validas(l)
            idx = validas.nonzero()
            n = min(cuota[l], len(idx))
            elegidas = idx[torch.randperm(len(idx), generator=gen)[:n]]
            m = torch.zeros_like(self.mascara(l))
            m[elegidas[:, 0], elegidas[:, 1]] = 1
            self.mascara(l).copy_(m)
        self.sincronizar()
        self.limpiar_pesos()

    def _validas(self, l: int) -> torch.Tensor:
        """Posiciones (salida, entrada) de la capa ``l`` cuyas dos neuronas están activas."""
        sal = self.activa(l) if l < self.ocultas else torch.ones(self.salidas, device=self.mascara(l).device)
        ent = self.activa(l - 1) if l > 0 else torch.ones(self.entradas, device=self.mascara(l).device)
        return (sal[:, None] * ent[None, :]) > 0

    def validas(self, l: int) -> torch.Tensor:
        return self._validas(l)

    @torch.no_grad()
    def limpiar_pesos(self) -> None:
        """Los pesos apagados valen cero (así una conexión que vuelve a nacer empieza en cero)."""
        for l, capa in enumerate(self.capas):
            capa.weight.mul_(self.mascara(l))
            if l < self.ocultas:
                capa.bias.mul_(self.activa(l))
