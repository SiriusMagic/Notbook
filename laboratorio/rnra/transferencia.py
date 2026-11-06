"""Aprendizaje bidireccional de RNRA-M: lo que se aprende en el dispositivo pequeño, ¿cómo vuelve a la red grande?

Escenario (H3): la red grande sabe la tarea A. Antes de irse al «celular» deja un **recuerdo** de A (pocos ejemplos con las
respuestas suaves de la red grande, que viajan en el paquete). En el celular se ajusta solo el núcleo con la tarea B. Al volver
a la PC se integra con uno de estos métodos y se mide, en el ancho grande, cuánto vale A (olvido) y cuánto B:

- ``ajuste_convencional``: ajustar la red grande con la tarea B completa (sin repaso de A). Es lo que se haría sin RNRA-M.
- ``ajuste_con_repaso``: igual, pero repasando el recuerdo de A. Referencia con todos los datos de B.
- ``ajuste_buffer``: repaso de A y solo los ``n_buffer`` ejemplos de B que viajan en el paquete (mismos datos que el siguiente).
- ``copiar_nucleo``: se pone el núcleo aprendido en el celular; los extras no se tocan ni se ajustan.
- ``copiar_y_reconciliar``: copiar el núcleo y ajustar la red grande un rato con los mismos recuerdos de A y de B.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from . import memoria_multiescala as mem
from .superred import SuperRed, entrenar, exactitud

METODOS_INTEGRACION = ["ajuste_convencional", "ajuste_con_repaso", "ajuste_buffer", "copiar_nucleo", "copiar_y_reconciliar"]


@torch.no_grad()
def recuerdo(grande: SuperRed, x, n: int, gen) -> tuple[torch.Tensor, torch.Tensor]:
    """Ejemplos de la tarea vieja con las probabilidades de la red grande (memoria que viaja en el paquete)."""
    idx = torch.randperm(len(x), generator=gen)[:n]
    return x[idx], F.softmax(grande(x[idx]), 1)


def aprender_en_pequeno(p: mem.Paquete, base: SuperRed, x, y, pasos: int = 300, semilla: int = 0) -> SuperRed:
    """En el dispositivo: solo existe el núcleo (cuantizado); se ajusta ese ancho con los datos de la tarea nueva."""
    chico = mem.nucleo_como_modelo(p, base)
    chico.anchos = (p.ancho,)
    return entrenar(chico, x, y, pasos=pasos, lr=1e-3, semilla=semilla, anchos=(p.ancho,), destilar=0)


def _ajustar(modelo: SuperRed, xb, yb, xa, pa, pasos: int, semilla: int, repaso: bool, lr: float = 1e-3) -> SuperRed:
    """Entrena todos los anchos con la tarea B (xb, yb) y, si ``repaso``, imitando las salidas guardadas de A (xa, pa)."""
    gen = torch.Generator().manual_seed(semilla)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(pasos):
        ib = torch.randint(0, len(xb), (128,), generator=gen)
        opt.zero_grad(set_to_none=True)
        perdida = 0
        for w in modelo.anchos:
            perdida = perdida + F.cross_entropy(modelo(xb[ib], w), yb[ib])
            if repaso:
                ia = torch.randint(0, len(xa), (64,), generator=gen)
                perdida = perdida + F.kl_div(F.log_softmax(modelo(xa[ia], w), 1), pa[ia], reduction="batchmean")
        perdida.backward()
        opt.step()
    return modelo


@torch.no_grad()
def poner_nucleo(grande: SuperRed, nuevo: SuperRed, ancho: int) -> SuperRed:
    """Copia los pesos del núcleo aprendido en el celular dentro de la red grande (los extras no cambian)."""
    g = copy.deepcopy(grande)
    for l, (cg, cn) in enumerate(zip(g.capas, nuevo.capas)):
        ent = g.entradas if l == 0 else ancho
        sal = ancho if l < g.ocultas else g.salidas
        cg.weight[:sal, :ent] = cn.weight[:sal, :ent]
        cg.bias[:sal] = cn.bias[:sal]
    return g


def integrar(metodo: str, grande: SuperRed, nuevo_nucleo: SuperRed, ancho: int, xb, yb, xa, pa, n_buffer: int = 200,
             pasos: int = 200, semilla: int = 0) -> SuperRed:
    if metodo not in METODOS_INTEGRACION:
        raise KeyError(metodo)
    if metodo == "copiar_nucleo":
        return poner_nucleo(grande, nuevo_nucleo, ancho)
    gen = torch.Generator().manual_seed(semilla + 99)
    ib = torch.randperm(len(xb), generator=gen)[:n_buffer]
    if metodo == "ajuste_convencional":
        return _ajustar(copy.deepcopy(grande), xb, yb, xa, pa, pasos, semilla, repaso=False)
    if metodo == "ajuste_con_repaso":
        return _ajustar(copy.deepcopy(grande), xb, yb, xa, pa, pasos, semilla, repaso=True)
    if metodo == "ajuste_buffer":
        return _ajustar(copy.deepcopy(grande), xb[ib], yb[ib], xa, pa, pasos, semilla, repaso=True)
    return _ajustar(poner_nucleo(grande, nuevo_nucleo, ancho), xb[ib], yb[ib], xa, pa, pasos, semilla, repaso=True)


def medir_integracion(grande, integrada, ancho_chico: int, tA, tB) -> dict:
    """tA, tB: (x_prueba, y_prueba). Olvido de A = lo que valía A antes − lo que vale ahora (ancho grande)."""
    wg = max(grande.anchos)
    a0 = exactitud(grande, *tA, wg)
    return {"A": exactitud(integrada, *tA, wg), "B": exactitud(integrada, *tB, wg), "olvido": a0 - exactitud(integrada, *tA, wg),
            "A_chico": exactitud(integrada, *tA, ancho_chico), "B_chico": exactitud(integrada, *tB, ancho_chico)}
