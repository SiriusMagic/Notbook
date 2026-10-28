"""Mediciones de hardware y de lo que pasa dentro de la red.

**Hardware**
- *Latencia*: milisegundos para responder a un solo ejemplo (mediana y p95), tras
  calentar. Es lo que importa en tiempo real.
- *Throughput*: ejemplos por segundo en lotes de 256 (en texto también tokens/s).
- *Memoria*: huella de los pesos (MB) y, en GPU, el pico de VRAM al entrenar.

**Dentro de la red** (el «por qué» funciona o no)
- *Neuronas muertas*: unidades que dan exactamente 0 para todos los ejemplos.
- *Neuronas inmóviles*: unidades que dan siempre el mismo valor (muertas o saturadas).
- *Norma de activaciones* por capa y *norma del gradiente* por capa: si el gradiente
  de las primeras capas es mucho menor que el de las últimas, se está desvaneciendo.
- *Eficiencia*: métrica por millón de parámetros.
- *CKA lineal* (Kornblith et al., 2019): de 0 a 1, cuánto se parecen las
  representaciones internas de dos redes. 1 = ven lo mismo; cerca de 0 = la pieza
  cambió por completo cómo fluye la información.
"""

from __future__ import annotations

import time
from statistics import median

import torch
from torch import nn

MUESTRA = 1000  # ejemplos usados para medir activaciones y CKA


def capas_observadas(modelo: nn.Module) -> dict[str, nn.Module]:
    """Las capas cuyas salidas se inspeccionan: extractor (si hay) y las dos capas de la pieza."""
    red = getattr(modelo, "cabeza", modelo)
    capas = {}
    if hasattr(modelo, "extractor"):
        capas["extractor"] = modelo.extractor
    for nombre in ("capa1", "capa2"):
        if hasattr(red, nombre):
            capas[nombre] = getattr(red, nombre)
    return capas


@torch.no_grad()
def activaciones(modelo: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Salida de cada capa observada (aplanada a [ejemplos, unidades]) para hasta MUESTRA ejemplos."""
    salidas: dict[str, torch.Tensor] = {}
    ganchos = [m.register_forward_hook(lambda _m, _i, out, n=n: salidas.__setitem__(n, out.detach().flatten(1).float()))
               for n, m in capas_observadas(modelo).items()]
    try:
        modelo.eval()
        modelo(x[:MUESTRA])
    finally:
        for g in ganchos:
            g.remove()
    return salidas


def _sync(disp: torch.device) -> None:
    if disp.type == "cuda":
        torch.cuda.synchronize(disp)


@torch.no_grad()
def medir_hardware(modelo: nn.Module, x: torch.Tensor, tokens_por_ejemplo: float | None = None,
                   repeticiones: int = 30, lote: int = 256) -> dict:
    disp = next(modelo.parameters()).device
    modelo.eval()
    huella = sum(p.numel() * p.element_size() for p in modelo.parameters())
    huella += sum(b.numel() * b.element_size() for b in modelo.buffers())
    uno = x[:1]
    for _ in range(5):  # calentar
        modelo(uno)
    tiempos = []
    for _ in range(repeticiones):
        _sync(disp)
        t0 = time.perf_counter()
        modelo(uno)
        _sync(disp)
        tiempos.append((time.perf_counter() - t0) * 1000)
    tiempos.sort()
    bloque = x[:lote]
    modelo(bloque)
    _sync(disp)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < 0.5 or n < 3:
        modelo(bloque)
        n += 1
    _sync(disp)
    por_seg = n * len(bloque) / (time.perf_counter() - t0)
    medidas = {
        "latencia_ms": round(median(tiempos), 4),
        "latencia_p95_ms": round(tiempos[int(0.95 * (len(tiempos) - 1))], 4),
        "ejemplos_por_s": round(por_seg, 1),
        "huella_mb": round(huella / 2**20, 4),
    }
    if tokens_por_ejemplo:
        medidas["tokens_por_s"] = round(por_seg * tokens_por_ejemplo, 1)
    return medidas


def medir_internas(modelo: nn.Module, x: torch.Tensor, y: torch.Tensor, perdida) -> dict:
    """Neuronas muertas e inmóviles, normas de activación y gradiente por capa."""
    medidas: dict = {}
    for nombre, a in activaciones(modelo, x).items():
        medidas[f"{nombre}_muertas"] = round(float((a.abs().amax(0) < 1e-7).float().mean()), 4)
        medidas[f"{nombre}_inmoviles"] = round(float((a.std(0) < 1e-4).float().mean()), 4)
        medidas[f"{nombre}_norma"] = round(float(a.norm(dim=1).mean()), 4)
    # Gradiente de una pasada (sin cambiar los pesos): ¿llega a las primeras capas?
    modelo.eval()
    modelo.zero_grad(set_to_none=True)
    perdida(modelo(x[:256]), y[:256]).backward()
    for nombre, m in capas_observadas(modelo).items():
        grads = [p.grad.norm() ** 2 for p in m.parameters() if p.grad is not None]
        if grads:
            medidas[f"{nombre}_grad"] = float(torch.stack(grads).sum().sqrt())
    modelo.zero_grad(set_to_none=True)
    return medidas


def cka_lineal(a: torch.Tensor, b: torch.Tensor) -> float:
    """Centered Kernel Alignment lineal entre dos representaciones [n, d1] y [n, d2]."""
    a = a.double() - a.double().mean(0)
    b = b.double() - b.double().mean(0)
    cruz = (b.T @ a).norm() ** 2
    den = (a.T @ a).norm() * (b.T @ b).norm()
    return float(cruz / den) if den > 0 else float("nan")


def cka_entre(modelo_a: nn.Module, modelo_b: nn.Module, x: torch.Tensor) -> dict[str, float]:
    """CKA capa por capa entre dos modelos de la misma arquitectura, con los mismos ejemplos."""
    aa, bb = activaciones(modelo_a, x), activaciones(modelo_b, x)
    return {n: round(cka_lineal(aa[n], bb[n]), 4) for n in aa if n in bb}
