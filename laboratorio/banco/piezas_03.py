"""Piezas del cuaderno 03 (neuronas alternativas).

- **knn_prototipos**: la «neurona con arreglo KNN». Cada neurona guarda ``m`` patrones (prototipos) y se activa
  según la distancia a los ``k`` más cercanos; los patrones se aprenden sin etiquetas, al estilo Kohonen
  (se acercan a lo que ven o se reemplazan si llega algo muy distinto). Equivale a una red de base radial (RBF)
  con prototipos que se ajustan solos: solo la capa de salida aprende con el gradiente.

Las neuronas con árbol de decisión y con Naive Bayes del mismo cuaderno necesitan una «etiqueta local» para cada
neurona oculta, que el cuaderno no define; meterlas en una red sería inventar. Lo que sí se mide es su poder como
clasificadores completos: ``clasicos.py`` (``--clasicos``).
"""

from __future__ import annotations

import torch
from torch import nn


class CapaKNN(nn.Module):
    """``salidas`` neuronas KNN, cada una con ``memoria`` patrones de ``entradas`` números.

    Activación de cada neurona: promedio de 1/(1 + distancia) a sus ``k`` patrones más cercanos.

    Aprendizaje (``despues_del_paso``, sin gradiente), como ``learn_from_input`` del cuaderno:
    - si un ejemplo está lejos (> ``umbral``) de todos los patrones de una neurona, reemplaza al menos usado;
    - si está cerca, el patrón más cercano se acerca a él (tasa ``lr``).

    Arreglos:
    - El cuaderno recorría los ejemplos de uno en uno en Python. Aquí se hace por lote: cada neurona reemplaza
      como máximo un patrón por paso (con el ejemplo más lejano) y mueve sus patrones hacia el promedio de los
      ejemplos cercanos que les tocaron.
    - Los patrones empiezan como copias de ejemplos reales del primer lote (en el cuaderno, ruido ``randn``).
    - **Escala**: 1/(1 + distancia) no tiene escala. Con 64 entradas todas las distancias rondan 9, todas las
      neuronas dan ≈ 0,1 para cualquier entrada y todo queda «lejos», así que los patrones nunca se mueven
      (en dígitos la red acertaba 7 %, menos que el azar). Ahora la distancia se divide por la distancia típica
      de cada neurona (la mediana en el primer lote), como el ancho de una red de base radial, y el umbral se
      mide en esa misma unidad.
    """

    def __init__(self, entradas: int, salidas: int, memoria: int = 5, k: int = 3, lr: float = 0.01,
                 umbral: float = 0.5) -> None:
        super().__init__()
        self.k, self.lr, self.umbral = min(k, memoria), lr, umbral
        self.register_buffer("patrones", torch.randn(salidas, memoria, entradas))
        self.register_buffer("usos", torch.zeros(salidas, memoria))
        self.register_buffer("escala", torch.ones(salidas))
        self.register_buffer("iniciada", torch.tensor(False))

    def _distancias(self, x):  # [lote, neuronas, memoria], en unidades de la distancia típica de cada neurona
        d = torch.cdist(x.unsqueeze(0).expand(len(self.patrones), -1, -1), self.patrones).permute(1, 0, 2)
        return d / self.escala.view(1, -1, 1)

    def forward(self, x):
        if self.training:
            self.entrada_actual = x.detach()
            if not bool(self.iniciada):
                with torch.no_grad():
                    idx = torch.randint(0, len(x), self.patrones.shape[:2], device=x.device)
                    self.patrones.copy_(x.detach()[idx])
                    self.escala.copy_(self._distancias(x.detach()).median(0).values.median(1).values.clamp(min=1e-6))
                    self.iniciada.fill_(True)
        cercanas = self._distancias(x).topk(self.k, dim=2, largest=False).values
        return (1.0 / (1.0 + cercanas)).mean(2)

    @torch.no_grad()
    def despues_del_paso(self, **_):
        x = getattr(self, "entrada_actual", None)
        if x is None:
            return
        d = self._distancias(x)  # [lote, neuronas, memoria]
        minima, cual = d.min(2)  # [lote, neuronas]
        n_neuronas, memoria, _ = self.patrones.shape
        # Ejemplos cercanos: cada patrón se acerca al promedio de los ejemplos que le tocaron.
        cerca = (minima <= self.umbral).float()  # [lote, neuronas]
        asignacion = torch.nn.functional.one_hot(cual, memoria).float() * cerca.unsqueeze(2)  # [lote, neuronas, memoria]
        cuantos = asignacion.sum(0)  # [neuronas, memoria]
        suma = torch.einsum("bnm,bi->nmi", asignacion, x)
        hay = cuantos > 0
        promedio = suma / cuantos.clamp(min=1).unsqueeze(2)
        self.patrones[hay] = (1 - self.lr) * self.patrones[hay] + self.lr * promedio[hay]
        self.usos += cuantos
        # Ejemplos lejanos: cada neurona reemplaza su patrón menos usado por el ejemplo más lejano.
        lejos = minima.masked_fill(minima <= self.umbral, -1.0)
        mas_lejano, quien = lejos.max(0)  # [neuronas]
        neuronas = torch.nonzero(mas_lejano > 0).flatten()
        if len(neuronas):
            menos_usado = self.usos[neuronas].argmin(1)
            self.patrones[neuronas, menos_usado] = x[quien[neuronas]]
            self.usos[neuronas, menos_usado] = 1


class CapaGenetica(nn.Module):
    """Lineal + ReLU con «evolución» de neuronas (redes genéticas, celdas 79 y 85).

    Cada neurona cuenta cuántos pasos recibió ajuste (gradiente distinto de 0). Desde la época ``inicio``, en
    cada época las neuronas con ``umbral`` ajustes o más se reemplazan por una copia mutada (ruido 0,02) de una de
    las 2 neuronas **menos** ajustadas, y su cuenta vuelve a 0 (la versión vectorizada de la celda 85).

    ``modo``:
    - ``"original"``: tal cual el cuaderno.
    - ``"arreglada"``: corrige que, en una red ReLU, toda neurona viva recibe ajuste en cada paso, así que «las
      menos ajustadas» son las **muertas**: el original reemplaza neuronas vivas por copias de muertas. Aquí se
      mide cuánto se ajustó (suma de |gradiente|), los padres deben estar vivos y se cambia ≤ 10 % cada
      ``umbral`` épocas.
    - ``"invertida"``: el criterio al revés. Con descenso por gradiente, la neurona que más se ajusta es la que
      más importa para el error; las que casi no se ajustan (o no se activan) son las inútiles. Se reemplazan
      esas por mutaciones de las 2 más útiles. Se parece a ReDo («reciclar neuronas dormidas», Sokar et al.,
      2023).
    """

    def __init__(self, entradas: int, salidas: int, inicio: int = 100, umbral: int = 10, ruido: float = 0.02,
                 modo: str = "original") -> None:
        super().__init__()
        self.lineal = nn.Linear(entradas, salidas)
        self.inicio, self.umbral, self.ruido, self.modo = inicio, umbral, ruido, modo
        self.register_buffer("ajustes", torch.zeros(salidas))
        self.ultima_epoca = -1
        self.reemplazos = 0  # cuántas neuronas se reemplazaron en total (el banco lo informa)

    def forward(self, x):
        h = torch.relu(self.lineal(x))
        if self.training:
            self.vivas = (h.detach() > 0).any(0)
        return h

    @torch.no_grad()
    def despues_del_paso(self, epoca: int = 0, **_):
        g, gb = self.lineal.weight.grad, self.lineal.bias.grad
        if g is None:
            return
        if self.modo == "original":
            self.ajustes += ((g.abs().sum(1) > 1e-5) | (gb.abs() > 1e-5)).float()
        else:
            self.ajustes += g.abs().sum(1) + gb.abs()
        if epoca <= self.inicio or epoca == self.ultima_epoca:
            return
        self.ultima_epoca = epoca
        n = len(self.ajustes)
        vivas = getattr(self, "vivas", torch.ones(n, dtype=torch.bool, device=self.ajustes.device))
        if self.modo == "original":
            padres = self.ajustes.argsort()[:2]
            reemplazar = torch.nonzero(self.ajustes >= self.umbral).flatten().tolist()
        else:
            if epoca % self.umbral:
                return
            candidatas = torch.nonzero(vivas).flatten()
            if len(candidatas) < 2:
                return
            orden = self.ajustes[candidatas].argsort(descending=self.modo == "invertida")
            padres = candidatas[orden[:2]]
            cuantas = max(1, n // 10)
            if self.modo == "arreglada":
                peores = self.ajustes.argsort(descending=True)[:cuantas]
            else:  # invertida: primero las muertas, luego las que menos se ajustan
                puntaje = self.ajustes.masked_fill(~vivas, -1.0)
                peores = puntaje.argsort()[:cuantas]
            reemplazar = [i for i in peores.tolist() if i not in padres.tolist()]
        w, b = self.lineal.weight, self.lineal.bias
        for i in reemplazar:
            p = int(padres[torch.randint(0, 2, ()).item()])
            w[i] = w[p] + torch.randn_like(w[p]) * self.ruido
            b[i] = b[p] + torch.randn((), device=b.device) * self.ruido
            self.ajustes[i] = 0
        self.reemplazos += len(reemplazar)


def piezas_03():
    from .piezas import Pieza

    return [
        Pieza("knn_prototipos", "Neuronas KNN: prototipos aprendidos sin etiquetas (estilo Kohonen)",
              "03 · neurona con arreglo KNN", CapaKNN),
        Pieza("genetica", "Neuronas muy ajustadas se reemplazan por mutaciones de las menos ajustadas (tal cual)",
              "03 · redes genéticas (celdas 79 y 85)", CapaGenetica),
        Pieza("genetica_arreglada", "Igual, pero mide cuánto se ajusta, solo copia neuronas vivas y cambia ≤ 10 %",
              "03 · redes genéticas (arreglada)", lambda e, s: CapaGenetica(e, s, modo="arreglada")),
        Pieza("genetica_invertida", "Criterio al revés: recicla las neuronas que no se usan copiando las más útiles",
              "03 · redes genéticas (invertida, como ReDo)", lambda e, s: CapaGenetica(e, s, modo="invertida")),
    ]
