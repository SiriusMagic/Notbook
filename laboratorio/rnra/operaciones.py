"""Cambios de estructura del banco de dispersión dinámica: reasignar conexiones (SET, RigL), poda global por magnitud,
activar neuronas (crecimiento programado) y rehacer el estado de Adam sin romper el optimizador."""

from __future__ import annotations

import torch


def gradientes_densos(modelo, x, y, perdida) -> list[torch.Tensor]:
    """|∂L/∂(W⊙M)| en **todas** las posiciones, también en las conexiones apagadas (truco de las sondas)."""
    sondas = [torch.zeros_like(c.weight, requires_grad=True) for c in modelo.capas]
    modelo.zero_grad(set_to_none=True)
    perdida(modelo(x, sondas=sondas), y).backward()
    modelo.zero_grad(set_to_none=True)
    return [s.grad.abs().detach() for s in sondas]


def foto_estructura(modelo) -> dict:
    return {"m": [modelo.mascara(l).clone() for l in range(len(modelo.capas))],
            "a": [modelo.activa(l).clone() for l in range(modelo.ocultas)]}


@torch.no_grad()
def reiniciar_estado_optimizador(opt, modelo, antes: dict) -> None:
    """Pone en cero el estado de Adam donde cambió la máscara (una conexión nueva no hereda inercia ajena)."""
    for l, capa in enumerate(modelo.capas):
        cambio = modelo.mascara(l) != antes["m"][l]
        st = opt.state.get(capa.weight, {})
        for k in ("exp_avg", "exp_avg_sq"):
            if k in st:
                st[k][cambio] = 0
        if l < modelo.ocultas:
            cb = modelo.activa(l) != antes["a"][l]
            sb = opt.state.get(capa.bias, {})
            for k in ("exp_avg", "exp_avg_sq"):
                if k in sb:
                    sb[k][cb] = 0


@torch.no_grad()
def _elegir(puntaje: torch.Tensor, candidatas: torch.Tensor, n: int, mayor: bool, aleatorio: bool, gen) -> torch.Tensor:
    """Índices (fila, columna) de ``n`` posiciones entre las ``candidatas``: las de mayor/menor puntaje, o al azar."""
    idx = candidatas.nonzero()
    n = min(n, len(idx))
    if n <= 0:
        return idx[:0]
    if aleatorio:
        return idx[torch.randperm(len(idx), generator=gen)[:n]]
    v = puntaje[idx[:, 0], idx[:, 1]]
    orden = torch.argsort(v, descending=mayor)[:n]
    return idx[orden]


@torch.no_grad()
def activar_neurona_densa(modelo, l: int, gen) -> int:
    """Activa una ranura de reserva de la capa ``l`` con todas sus conexiones (crecimiento programado). Entrada con pesos
    pequeños y salida en cero, así la función de la red casi no cambia (como Net2Net). Devuelve la ranura (-1 si no hay)."""
    libres = (modelo.activa(l) == 0).nonzero().flatten()
    if len(libres) == 0:
        return -1
    i = int(libres[0])
    modelo.activa(l)[i] = 1
    ent = (modelo.activa(l - 1) > 0) if l > 0 else torch.ones(modelo.entradas, dtype=torch.bool)
    sal = (modelo.activa(l + 1) > 0) if l + 1 < modelo.ocultas else torch.ones(modelo.salidas, dtype=torch.bool)
    modelo.mascara(l)[i, ent] = 1
    modelo.mascara(l + 1)[sal, i] = 1
    w = modelo.capas[l].weight
    w[i].zero_()
    w[i, ent] = torch.randn(int(ent.sum()), generator=gen) / max(int(ent.sum()), 1) ** 0.5
    modelo.capas[l + 1].weight[:, i] = 0
    modelo.capas[l].bias[i] = 0
    return i


@torch.no_grad()
def reorganizar(modelo, puntajes: list[torch.Tensor], grad: list[torch.Tensor], fraccion: float, presupuesto: int,
                gen, protegidas: list[torch.Tensor] | None = None, azar_quitar: bool = False, azar_crecer: bool = False) -> dict:
    """Reasigna conexiones sin pasar del ``presupuesto``.

    En cada capa se quitan las ``fraccion`` de conexiones con menor ``puntaje`` (salvo las protegidas) y se crean
    otras tantas donde el gradiente denso ``grad`` es mayor. Después, el presupuesto que haya quedado libre
    (por ejemplo, por neuronas podadas) se gasta en las mejores posiciones de toda la red. Las conexiones nuevas
    nacen con peso cero. Con ``azar_quitar`` / ``azar_crecer`` elige al azar (SET, o el control negativo con los mismos conteos).
    Devuelve ``{"movidas": n, "nuevas": [máscaras de lo nacido por capa]}``."""
    L = len(modelo.capas)
    nuevas, movidas = [torch.zeros_like(modelo.mascara(l)) for l in range(L)], 0
    quitadas = []
    for l in range(L):
        m = modelo.mascara(l)
        cand = (m > 0) & (~protegidas[l] if protegidas is not None else True)
        n = min(int(fraccion * m.sum().item()), int(cand.sum().item()))
        sel = _elegir(puntajes[l], cand, n, mayor=False, aleatorio=azar_quitar, gen=gen)
        m[sel[:, 0], sel[:, 1]] = 0
        quitadas.append(sel)
        # crecer en la misma capa, sin volver a las que acaban de salir
        libre = modelo.validas(l) & (m == 0)
        libre[sel[:, 0], sel[:, 1]] = False
        nuevo = _elegir(grad[l], libre, len(sel), mayor=True, aleatorio=azar_crecer, gen=gen)
        m[nuevo[:, 0], nuevo[:, 1]] = 1
        nuevas[l][nuevo[:, 0], nuevo[:, 1]] = 1
        movidas += len(nuevo)
    sobran = presupuesto - modelo.conexiones()
    if sobran > 0:  # presupuesto libre: va a las posiciones de mayor gradiente de toda la red
        todo = []
        for l in range(L):
            libre = modelo.validas(l) & (modelo.mascara(l) == 0)
            libre[quitadas[l][:, 0], quitadas[l][:, 1]] = False
            idx = libre.nonzero()
            v = torch.rand(len(idx), generator=gen) if azar_crecer else grad[l][idx[:, 0], idx[:, 1]]
            todo.append((l, idx, v))
        vals = torch.cat([v for _, _, v in todo])
        if len(vals):
            k = min(sobran, len(vals))
            corte = torch.sort(vals, descending=True).values[k - 1]
            usadas = 0
            for l, idx, v in todo:
                sel = idx[v >= corte][: max(k - usadas, 0)]
                modelo.mascara(l)[sel[:, 0], sel[:, 1]] = 1
                nuevas[l][sel[:, 0], sel[:, 1]] = 1
                usadas += len(sel)
                movidas += len(sel)
    modelo.limpiar_pesos()
    return {"movidas": movidas, "nuevas": nuevas}


@torch.no_grad()
def poda_global_por_magnitud(modelo, objetivo: int) -> None:
    """Deja ``objetivo`` conexiones: quita las de menor |peso| (misma proporción en cada capa)."""
    total = modelo.conexiones()
    if objetivo >= total:
        return
    for l, capa in enumerate(modelo.capas):
        m = modelo.mascara(l)
        quedan = max(1, int(round(m.sum().item() * objetivo / total)))
        idx = m.nonzero()
        v = capa.weight[idx[:, 0], idx[:, 1]].abs()
        sel = idx[torch.argsort(v)[: len(idx) - quedan]]
        m[sel[:, 0], sel[:, 1]] = 0
    modelo.limpiar_pesos()
