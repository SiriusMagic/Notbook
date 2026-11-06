"""Memoria portátil de RNRA-M: reducir sin destruir y expandir sin internet.

Al reducir a un ancho ``w``, el **núcleo** (primeras ``w`` neuronas) queda en RAM, cuantizado a ``bits_nucleo``; todo lo
demás (los pesos «extra» y el residuo de la cuantización del núcleo) se guarda **comprimido** en la ``memoria``, que
viaja en el mismo archivo. Al expandir, núcleo + memoria reconstruyen la red grande:

- ``bits_memoria = 32``: recuperación exacta (hasta ~1e-6 por redondeo de coma flotante).
- ``bits_memoria = 8`` o ``4``: la memoria pesa menos, pero se pierde algo de precisión de forma permanente. Se mide.

La RAM activa del dispositivo pequeño es solo el núcleo. El archivo completo (núcleo + memoria) puede pesar tanto como la
red grande: lo que se ahorra es RAM de trabajo, no almacenamiento. Los archivos se guardan con ``np.savez_compressed`` y se
leen con ``allow_pickle=False`` (nada de ``pickle``).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch

from .superred import SuperRed, cuantizar_tensor


@dataclass
class Paquete:
    ancho: int            # ancho del núcleo
    bits_nucleo: int
    bits_memoria: int
    forma: dict           # formas completas {nombre: tupla}
    nucleo: dict          # {nombre: ndarray float32} ya descuantizado para ejecutarlo; en disco va como enteros
    memoria: dict         # {nombre: ndarray} lo que falta para reconstruir
    escalas: dict         # escalas de cuantización {nombre: float}

    def bytes_nucleo(self) -> int:
        """RAM de los pesos del núcleo (enteros de ``bits_nucleo`` bits)."""
        return int(sum(np.ceil(v.size * min(self.bits_nucleo, 32) / 8) for v in self.nucleo.values()) + 4 * len(self.nucleo))

    def bytes_memoria(self) -> int:
        """Tamaño de la memoria comprimida en disco."""
        buf = io.BytesIO()
        np.savez_compressed(buf, **self._codificada()[1])
        return len(buf.getvalue())

    def _codificada(self):
        """(núcleo, memoria) como enteros/flotantes listos para guardar."""
        mem = {}
        for k, v in self.memoria.items():
            if self.bits_memoria >= 32:
                mem[k] = v.astype(np.float32)
            else:
                q = 2 ** (self.bits_memoria - 1) - 1
                esc = max(float(np.abs(v).max()), 1e-12) / q
                mem[k] = np.round(v / esc).astype(np.int8)
                mem[k + "__esc"] = np.float32(esc)
        return self.nucleo, mem


def _region(t: torch.Tensor, w: int, ent_fija: int | None, sal_fija: int | None):
    """Índices de la región núcleo de un tensor de la superred: ``[:sal, :ent]`` o ``[:sal]`` para sesgos."""
    if t.dim() == 1:
        return (slice(0, sal_fija if sal_fija is not None else w),)
    return (slice(0, sal_fija if sal_fija is not None else w), slice(0, ent_fija if ent_fija is not None else w))


@torch.no_grad()
def reducir(modelo: SuperRed, ancho: int, bits_nucleo: int = 8, bits_memoria: int = 32) -> Paquete:
    """Empaqueta ``modelo`` para un dispositivo con poca RAM sin tirar nada: núcleo cuantizado + memoria."""
    nucleo, memoria, forma, escalas = {}, {}, {}, {}
    for l, capa in enumerate(modelo.capas):
        for nombre, t in (("weight", capa.weight), ("bias", capa.bias)):
            k = f"{l}.{nombre}"
            ent = modelo.entradas if l == 0 else ancho
            sal = ancho if l < modelo.ocultas else modelo.salidas
            reg = _region(t, ancho, ent, sal)
            nuc = t[reg].clone()
            deq = cuantizar_tensor(nuc, bits_nucleo)
            resto = t.clone()
            resto[reg] = nuc - deq  # el residuo de cuantizar el núcleo también se guarda: así nada se pierde
            nucleo[k], memoria[k], forma[k] = deq.numpy(), resto.numpy(), tuple(t.shape)
            escalas[k] = float(nuc.abs().max())
    return Paquete(ancho, bits_nucleo, bits_memoria, forma, nucleo, memoria, escalas)


@torch.no_grad()
def nucleo_como_modelo(p: Paquete, modelo_base: SuperRed) -> SuperRed:
    """Red pequeña lista para ejecutar en el dispositivo: solo el núcleo (el resto en cero)."""
    import copy

    m = copy.deepcopy(modelo_base)
    for l, capa in enumerate(m.capas):
        for nombre, t in (("weight", capa.weight), ("bias", capa.bias)):
            k = f"{l}.{nombre}"
            t.zero_()
            nuc = torch.tensor(p.nucleo[k])
            t[tuple(slice(0, s) for s in nuc.shape)] = nuc
    m.ancho = p.ancho
    return m


@torch.no_grad()
def expandir(p: Paquete, modelo_base: SuperRed, con_memoria: bool = True) -> SuperRed:
    """Reconstruye la red grande. ``con_memoria=False`` simula haber tirado lo recortado (los extras quedan en cero)."""
    m = nucleo_como_modelo(p, modelo_base)
    codificado_mem = p._codificada()[1]
    for l, capa in enumerate(m.capas):
        for nombre, t in (("weight", capa.weight), ("bias", capa.bias)):
            k = f"{l}.{nombre}"
            if not con_memoria:
                continue
            if p.bits_memoria >= 32:
                mem = torch.tensor(codificado_mem[k])
            else:
                mem = torch.tensor(codificado_mem[k].astype(np.float32)) * float(codificado_mem[k + "__esc"])
            t.add_(mem)
    m.ancho = max(m.anchos)
    return m


def guardar(p: Paquete, ruta: str) -> int:
    """Escribe el paquete (núcleo + memoria) en un solo archivo ``.npz``; devuelve su tamaño en bytes."""
    nuc, mem = p._codificada()
    datos = {"meta": np.array([p.ancho, p.bits_nucleo, p.bits_memoria])}
    datos.update({f"n|{k}": v for k, v in nuc.items()})
    datos.update({f"m|{k}": v for k, v in mem.items()})
    datos.update({f"f|{k}": np.array(v) for k, v in p.forma.items()})
    with open(ruta, "wb") as f:
        np.savez_compressed(f, **datos)
    import os

    return os.path.getsize(ruta)


def cargar(ruta: str) -> Paquete:
    z = np.load(ruta, allow_pickle=False)
    ancho, bn, bm = (int(v) for v in z["meta"])
    nucleo = {k[2:]: z[k] for k in z.files if k.startswith("n|")}
    forma = {k[2:]: tuple(int(i) for i in z[k]) for k in z.files if k.startswith("f|")}
    crudo = {k[2:]: z[k] for k in z.files if k.startswith("m|")}
    memoria = {}
    for k, v in crudo.items():
        if k.endswith("__esc"):
            continue
        memoria[k] = v.astype(np.float32) * float(crudo[k + "__esc"]) if bm < 32 else v
    return Paquete(ancho, bn, bm, forma, nucleo, memoria, {})
