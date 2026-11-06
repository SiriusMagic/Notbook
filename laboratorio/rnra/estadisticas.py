"""Estadísticas por neurona (utilidad, redundancia, plasticidad, estabilidad) y demanda por región.

Utilidad  ``U_i ≈ |E[h_i · ∂L/∂h_i]|``  (primer orden: cuánto cambiaría la pérdida al apagar la neurona).
Redundancia ``R_i`` = mayor correlación (en valor absoluto) con otra neurona activa de la misma capa.
Plasticidad ``P_i`` = norma reciente del gradiente de sus pesos de entrada, dividida entre la media de la capa.
Estabilidad ``S_i = |EMA(u)| / EMA(|u|)``: 1 si su aporte tiene siempre el mismo signo y tamaño, cerca de 0 si fluctúa.
Demanda por región ``D_r = EMA(L_r) / (ε + EMA(n_r))``: pérdida media en cada región de la entrada. Estima dificultad,
**no** prueba por sí sola que falte capacidad.

Las regiones salen de agrupar (k-means) únicamente los datos de **entrenamiento**; las etiquetas de prueba no se usan.
"""

from __future__ import annotations

import numpy as np
import torch

EPS = 1e-8


class Estadisticas:
    def __init__(self, modelo, beta: float = 0.8):
        self.m, self.beta = modelo, beta
        n, k = modelo.ocultas, modelo.ancho_max
        z = lambda: [torch.zeros(k) for _ in range(n)]  # noqa: E731
        self.u, self.u_abs, self.plast, self.red = z(), z(), z(), z()
        self.listo = False

    def medir(self, x, y, perdida) -> None:
        """Un paso hacia adelante y hacia atrás con ``(x, y)``: actualiza U, S, P y R. No cambia la red."""
        m, b = self.m, self.beta
        m.zero_grad(set_to_none=True)
        salida = m(x, guardar=True)
        perdida(salida, y).backward()
        for l, h in enumerate(m._h):
            u = (h.detach() * h.grad).sum(0).cpu()  # E_b[h · ∂l_b/∂h] (la pérdida media ya divide entre el lote)
            self.u[l] = b * self.u[l] + (1 - b) * u if self.listo else u
            self.u_abs[l] = b * self.u_abs[l] + (1 - b) * u.abs() if self.listo else u.abs()
            g = (m.capas[l].weight.grad * m.mascara(l)).norm(dim=1).cpu()
            act = m.activa(l).cpu() > 0
            g = g / (g[act].mean() + EPS) if act.any() else g
            self.plast[l] = b * self.plast[l] + (1 - b) * g if self.listo else g
            self.red[l] = self._redundancia(h.detach().cpu(), act)
        self.listo = True
        m.zero_grad(set_to_none=True)

    @staticmethod
    def _redundancia(h: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        r = torch.zeros(h.shape[1])
        idx = act.nonzero().flatten()
        if len(idx) == 0:
            return r
        a = h[:, idx] - h[:, idx].mean(0)
        sd = a.std(0)
        muerta = sd < 1e-6
        c = (a / (sd + EPS)).T @ (a / (sd + EPS)) / max(len(a) - 1, 1)
        c = c.abs()
        c.fill_diagonal_(0)
        mx = c.max(1).values if len(idx) > 1 else torch.zeros(len(idx))
        mx[muerta] = 1.0  # una neurona que no cambia no aporta información
        r[idx] = mx
        return r

    # --- cantidades derivadas ----------------------------------------------------------------------------
    def utilidad(self, l: int) -> torch.Tensor:
        return self.u[l].abs()

    def estabilidad(self, l: int) -> torch.Tensor:
        return self.u[l].abs() / (self.u_abs[l] + EPS)

    def puntaje(self, l: int, alfa: float = 0.5, beta: float = 0.2, usar_u=True, usar_r=True) -> torch.Tensor:
        """``Q_i = U_i − α R_i + β S_i`` con U normalizada por el máximo de la capa (entre 0 y 1)."""
        act = self.m.activa(l).cpu() > 0
        u = self.utilidad(l)
        u = u / (u[act].max() + EPS) if act.any() else u
        q = (u if usar_u else torch.zeros_like(u)) - (alfa * self.red[l] if usar_r else 0) + beta * self.estabilidad(l)
        return torch.where(act, q, torch.full_like(q, float("inf")))  # las apagadas no son candidatas


class Regiones:
    """Divide el espacio de entrada en ``k`` regiones con k-means sobre los datos de entrenamiento."""

    def __init__(self, x_entreno: torch.Tensor, k: int, semilla: int):
        from sklearn.cluster import KMeans

        n = min(len(x_entreno), 4000)
        x = x_entreno.cpu().numpy()[:n]
        self.k = max(1, min(k, len(x)))
        self.km = KMeans(self.k, n_init=1, random_state=semilla).fit(x)

    def asignar(self, x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.km.predict(x.cpu().numpy()), dtype=torch.long)


class Demanda:
    """``D_r = EMA(L_r) / (ε + EMA(n_r))`` con ``L_r`` la suma de pérdidas de la región y ``n_r`` su frecuencia."""

    def __init__(self, k: int, beta: float = 0.8):
        self.beta, self.suma, self.n, self.listo = beta, torch.zeros(k), torch.zeros(k), False
        self.k = k

    def actualizar(self, region: torch.Tensor, perdidas: torch.Tensor) -> None:
        s = torch.zeros(self.k).index_add_(0, region, perdidas.detach().cpu())
        n = torch.bincount(region, minlength=self.k).float()
        if not self.listo:
            self.suma, self.n, self.listo = s, n, True
        else:
            self.suma = self.beta * self.suma + (1 - self.beta) * s
            self.n = self.beta * self.n + (1 - self.beta) * n

    def valor(self) -> torch.Tensor:
        return self.suma / (EPS + self.n)

    def pesos_muestras(self, region: torch.Tensor) -> torch.Tensor:
        """Probabilidad de elegir cada muestra para medir el gradiente: proporcional a la demanda de su región."""
        d = self.valor()[region] if self.listo else torch.ones(len(region))
        return d / d.sum()
