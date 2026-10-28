"""Piezas del cuaderno 06 (refuerzo y cerebro modular).

- **rl_interno** («RlNeuron»): cada neurona multiplica su salida por un factor que, después de cada paso, crece en
  proporción al tamaño de su gradiente (hasta 10). Pese al nombre, no es refuerzo: el factor solo puede crecer.
- **votacion_cabezas** («DecisionVotingNet»): un embedding tanh y 4 cabezas que votan (se promedian sus logits).
  En el cuaderno nunca se entrenaba.
- **biomas** («NeuralBiomesModel»): extractor compartido → «bioma de memoria» (LSTM) → «bioma de procesamiento».
  Ojo: la LSTM recibe secuencias de largo 1, así que no recuerda nada (es una capa con compuertas).
- **cerebro_moe** («CorticalMoE»): un router elige UN experto por ejemplo (memoria/LSTM, lógica/red grande,
  instinto/red chica, contexto/atención) y aprende con REINFORCE. La recompensa es la pérdida del lote entero, igual
  para todos los ejemplos.
- **cerebro_moe_arreglado**: recompensa por ejemplo (su propia pérdida) con línea base, como REINFORCE estándar.
- **talamo_top2** («ThalamusBiomeNet»): los mismos 4 biomas, mezclados con los 2 mejores según el router (como las
  mezclas de expertos dispersas de hoy, p. ej. Mixtral), y una retroalimentación con la salida anterior.
- **moe_suave** («CerebralRouterNet», industria): mezcla suave de 4 expertos iguales, la forma clásica.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .piezas_02 import _pesos_iniciales


# ---------------------------------------------------------------- RlNeuron
class CapaRLInterno(nn.Module):
    """relu((W·x + b) · factor); tras cada paso, factor += 0,005 · ‖∇W‖ (acotado a [0,01; 10]).

    Arreglo: los pesos empiezan como en ``nn.Linear`` (el cuaderno usaba ``randn × 0,1``, que con el factor
    inicial 0,1 deja salidas casi nulas). El factor empieza en 0,1, como en el cuaderno.
    """

    def __init__(self, entradas: int, salidas: int, lr_local: float = 0.005) -> None:
        super().__init__()
        self.pesos, self.bias = _pesos_iniciales(salidas, entradas)
        self.factor = nn.Parameter(torch.full((salidas,), 0.1))
        self.lr_local = lr_local

    def forward(self, x):
        return F.relu(F.linear(x, self.pesos, self.bias) * self.factor)

    @torch.no_grad()
    def despues_del_paso(self, **_):
        if self.pesos.grad is not None:
            self.factor.add_(float(self.pesos.grad.norm()) * self.lr_local).clamp_(0.01, 10.0)


# ---------------------------------------------------------------- DecisionVotingNet
class RedVotacionCabezas(nn.Module):
    """Embedding tanh → 4 cabezas (ReLU de 64 en el cuaderno; aquí ``ocultas``) → promedio de sus logits."""

    def __init__(self, entradas: int, salidas: int, ocultas: int, cabezas: int = 4) -> None:
        super().__init__()
        self.capa1 = nn.Linear(entradas, ocultas)  # el embedding (nombre capa1 para las mediciones)
        self.cabezas = nn.ModuleList(nn.Sequential(nn.Linear(ocultas, ocultas), nn.ReLU(), nn.Linear(ocultas, salidas))
                                     for _ in range(cabezas))

    def forward(self, x):
        z = torch.tanh(self.capa1(x))
        return torch.stack([c(z) for c in self.cabezas]).mean(0)


# ---------------------------------------------------------------- NeuralBiomesModel
class RedBiomas(nn.Module):
    """Extractor → LSTM con secuencia de largo 1 → [rasgos, estado] → procesamiento → salida."""

    def __init__(self, entradas: int, salidas: int, ocultas: int) -> None:
        super().__init__()
        self.capa1 = nn.Sequential(nn.Linear(entradas, 2 * ocultas), nn.ReLU(), nn.Linear(2 * ocultas, 2 * ocultas))
        self.memoria = nn.LSTM(2 * ocultas, ocultas, batch_first=True)
        self.capa2 = nn.Sequential(nn.Linear(3 * ocultas, 4 * ocultas), nn.ReLU(), nn.Linear(4 * ocultas, 2 * ocultas),
                                   nn.ReLU())
        self.salida = nn.Linear(2 * ocultas, salidas)

    def forward(self, x):
        rasgos = self.capa1(x)
        _, (h, _) = self.memoria(rasgos.unsqueeze(1))
        return self.salida(self.capa2(torch.cat([rasgos, h[-1]], 1)))


# ---------------------------------------------------------------- Biomas (expertos) del cerebro modular
class _Biomas(nn.Module):
    """Los 4 expertos del cuaderno: memoria (LSTM, largo 1), lógica (red grande), instinto (red chica) y contexto
    (atención sobre una secuencia de largo 1: el softmax de un solo elemento vale 1, así que es una lineal)."""

    def __init__(self, entradas: int, salidas: int, ocultas: int) -> None:
        super().__init__()
        self.lstm, self.memoria = nn.LSTM(entradas, ocultas, batch_first=True), nn.Linear(ocultas, salidas)
        self.logica = nn.Sequential(nn.Linear(entradas, 4 * ocultas), nn.ReLU(), nn.Linear(4 * ocultas, 2 * ocultas),
                                    nn.ReLU(), nn.Linear(2 * ocultas, salidas))
        self.instinto = nn.Sequential(nn.Linear(entradas, ocultas), nn.ReLU(), nn.Linear(ocultas, salidas))
        self.q, self.k, self.v = (nn.Linear(entradas, 2 * ocultas) for _ in range(3))
        self.contexto = nn.Linear(2 * ocultas, salidas)

    def forward(self, x):  # → [ejemplos, 4, salidas]
        _, (h, _) = self.lstm(x.unsqueeze(1))
        q, k, v = self.q(x).unsqueeze(1), self.k(x).unsqueeze(1), self.v(x).unsqueeze(1)
        atn = F.softmax(q @ k.transpose(1, 2), dim=-1)
        return torch.stack([self.memoria(h[-1]), self.logica(x), self.instinto(x),
                            self.contexto((atn @ v).squeeze(1))], 1)


def _perdida_por_ejemplo(salida, objetivo):
    if objetivo.dtype in (torch.long, torch.int64, torch.int32):
        return F.cross_entropy(salida, objetivo, reduction="none")
    return ((salida - objetivo) ** 2).reshape(len(salida), -1).mean(1)


class RedCerebroMoE(nn.Module):
    """CorticalMoE: el router muestrea UN experto por ejemplo y aprende con REINFORCE (``perdida_extra``).

    ``por_ejemplo=False`` (tal cual): recompensa = −pérdida del lote, la misma para todos; la pérdida del router es
    −media(log p) · recompensa. ``por_ejemplo=True`` (arreglo): recompensa de cada ejemplo = −su pérdida, menos el
    promedio del lote (línea base).

    Al evaluar se usa el experto más probable (el cuaderno muestreaba siempre, también al evaluar).
    """

    def __init__(self, entradas: int, salidas: int, ocultas: int, por_ejemplo: bool = False) -> None:
        super().__init__()
        self.router = nn.Linear(entradas, 4)
        self.biomas = _Biomas(entradas, salidas, ocultas)
        self.capa1 = self.biomas.instinto[0]  # para las mediciones internas
        self.por_ejemplo = por_ejemplo
        self.register_buffer("uso", torch.zeros(4))

    def forward(self, x):
        probs = F.softmax(self.router(x), dim=1)
        if self.training:
            dist = torch.distributions.Categorical(probs)
            idx = dist.sample()
            self.logp = dist.log_prob(idx)
            self.uso += torch.bincount(idx, minlength=4).float()
        else:
            idx = probs.argmax(1)
        salidas = self.biomas(x)
        return salidas[torch.arange(len(x), device=x.device), idx]

    def perdida_extra(self, salida, objetivo):
        if not hasattr(self, "logp"):
            return salida.new_zeros(())
        perdida = _perdida_por_ejemplo(salida, objetivo)
        if self.por_ejemplo:
            ventaja = -(perdida - perdida.mean())
            return -(self.logp * ventaja).mean()
        return -self.logp.mean() * (-perdida.mean())


class RedTalamo(nn.Module):
    """ThalamusBiomeNet: rasgos compartidos → router (rasgos + salida anterior) → mezcla de los 2 mejores biomas.

    Arreglos:
    - La «retroalimentación» (promedio de la salida del lote anterior) solo se actualiza al entrenar; al evaluar
      queda fija. En el cuaderno cambiaba también al evaluar, así que la predicción de un ejemplo dependía de qué
      otros se habían evaluado antes.
    - El cuaderno calculaba la entropía del router («alta = ruteo diverso») pero no la sumaba a la pérdida. La
      nota del cuaderno pide «maximizar entropía para forzar especialización» y «penalizar el uso excesivo de un
      solo bioma»: aquí se resta ``peso_entropia`` × entropía a la pérdida (``perdida_extra``).
    """

    def __init__(self, entradas: int, salidas: int, ocultas: int, k: int = 2, peso_entropia: float = 0.01) -> None:
        super().__init__()
        self.capa1 = nn.Linear(entradas, ocultas)
        self.router = nn.Linear(ocultas + salidas, 4)
        self.biomas = _Biomas(ocultas, salidas, ocultas)
        self.k, self.peso_entropia = k, peso_entropia
        self.register_buffer("anterior", torch.zeros(1, salidas))

    def forward(self, x):
        h = F.relu(self.capa1(x))
        logits = self.router(torch.cat([h, self.anterior.expand(len(x), -1)], 1))
        valores, idx = logits.topk(self.k, dim=1)
        pesos = torch.zeros_like(logits).scatter(1, idx, F.softmax(valores, dim=1))
        salida = (self.biomas(h) * pesos.unsqueeze(-1)).sum(1)
        if self.training:
            self.anterior = salida.detach().mean(0, keepdim=True)
            probs = F.softmax(logits, dim=1)
            self.entropia = -(probs * torch.log(probs + 1e-8)).sum(1).mean()
        return salida

    def perdida_extra(self, salida, objetivo):
        entropia = getattr(self, "entropia", None)
        return salida.new_zeros(()) if entropia is None else -self.peso_entropia * entropia


class RedMoESuave(nn.Module):
    """Industria: mezcla suave clásica (Jacobs et al., 1991). Rasgos compartidos → softmax sobre 4 expertos iguales."""

    def __init__(self, entradas: int, salidas: int, ocultas: int, expertos: int = 4) -> None:
        super().__init__()
        self.capa1 = nn.Linear(entradas, ocultas)
        self.router = nn.Linear(ocultas, expertos)
        self.expertos = nn.ModuleList(nn.Sequential(nn.Linear(ocultas, ocultas), nn.ReLU(), nn.Linear(ocultas, salidas))
                                      for _ in range(expertos))

    def forward(self, x):
        h = F.relu(self.capa1(x))
        pesos = F.softmax(self.router(h), dim=1).unsqueeze(-1)
        return (torch.stack([e(h) for e in self.expertos], 1) * pesos).sum(1)


def piezas_06():
    from .piezas import Pieza

    return [
        Pieza("rl_interno", "Factor de salida que crece con el tamaño del gradiente (RlNeuron)", "06 · neuronas con RL",
              CapaRLInterno),
        Pieza("votacion_cabezas", "Embedding tanh y 4 cabezas que votan (DecisionVotingNet)",
              "06 · votación de decisiones", None, red=lambda e, s, o: RedVotacionCabezas(e, s, o)),
        Pieza("biomas", "Extractor → LSTM de largo 1 → procesamiento (NeuralBiomesModel)", "06 · biomas neuronales",
              None, red=lambda e, s, o: RedBiomas(e, s, o)),
        Pieza("cerebro_moe", "Router con REINFORCE elige 1 de 4 expertos distintos (recompensa del lote)",
              "06 · cerebro modular (CorticalMoE)", None, red=lambda e, s, o: RedCerebroMoE(e, s, o)),
        Pieza("cerebro_moe_arreglado", "Igual, con recompensa por ejemplo y línea base", "06 · cerebro modular",
              None, red=lambda e, s, o: RedCerebroMoE(e, s, o, por_ejemplo=True)),
        Pieza("talamo_top2", "Router con retroalimentación que mezcla los 2 mejores de 4 biomas",
              "06 · tálamo (ThalamusBiomeNet)", None, red=lambda e, s, o: RedTalamo(e, s, o)),
        Pieza("moe_suave", "Mezcla suave de 4 expertos (CerebralRouterNet; industria)", "industria", None,
              industria=True, red=lambda e, s, o: RedMoESuave(e, s, o)),
    ]
