"""Entrenador del banco de dispersión dinámica: el mismo MLP con máscaras y los mismos pasos para cada método.

Todos terminan con el mismo presupuesto de conexiones (las de un MLP denso de ``ancho_pequeno`` neuronas por capa):

- ``mlp_pequeno`` y ``mlp_grande``: densos, sin cambios de estructura.
- ``poda_magnitud``: parte del MLP grande y quita de a poco las conexiones de menor |peso| (calendario cúbico).
- ``crecimiento_programado``: parte angosto y agrega neuronas por calendario hasta el ancho pequeño.
- ``set``: quita las más débiles y crea otras **al azar** (Mocanu 2018). También hace de control aleatorio.
- ``rigl``: quita las más débiles y crea donde el gradiente denso es mayor (Evci 2020).

Historia: aquí vivía RNRA-v0 (utilidad, redundancia, demanda por región, neuronas de reserva y prueba con/sin cambio). No
superó a RigL ni al azar y se quitó; está en el commit ``fd6e8f3`` y su informe es
``laboratorio/banco/informes/15-reorganizacion-autonoma.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from . import operaciones as op
from .modelo import MLPMascarado

METODOS = ["mlp_pequeno", "mlp_grande", "poda_magnitud", "crecimiento_programado", "set", "rigl"]


@dataclass
class Config:
    capas: int = 2
    ancho_pequeno: int = 16       # el MLP fijo pequeño fija el presupuesto de conexiones de todos
    ancho_grande: int = 64
    ancho_max: int = 64           # ranuras de neuronas
    activas_inicial: int = 32     # neuronas activas de SET y RigL
    pasos: int = 800
    lote: int = 128
    lr: float = 3e-3
    decaimiento: float = 1e-4
    calentamiento: int = 100
    cada: int = 50                # pasos entre eventos de cambio de estructura
    fin_reorg: float = 0.75       # no se reorganiza en el último cuarto
    fraccion: float = 0.2         # fracción de conexiones que se mueve (baja con coseno)
    fraccion_val: float = 0.2

    @property
    def ranuras(self) -> int:
        return max(self.ancho_max, self.ancho_grande)


@dataclass
class Resultado:
    metodo: str
    semilla: int
    metrica_prueba: float = float("nan")      # exactitud (clasificación) o mse
    perdida_prueba: float = float("nan")      # en aprendizaje continuo guarda el olvido
    parametros: int = 0
    conexiones: int = 0
    neuronas: list = field(default_factory=list)
    flops: float = 0.0
    curva: list = field(default_factory=list)  # (flops, paso, métrica de validación)
    matriz: list = field(default_factory=list)  # aprendizaje continuo: exactitud[tarea entrenada][tarea evaluada]

    def flops_hasta(self, objetivo: float) -> float:
        for f, _, v in self.curva:
            if v >= objetivo:
                return f
        return float("inf")


def presupuesto(entradas: int, salidas: int, cfg: Config) -> int:
    """Conexiones del MLP denso pequeño."""
    w = cfg.ancho_pequeno
    return entradas * w + (cfg.capas - 1) * w * w + w * salidas


class Entrenador:
    def __init__(self, metodo: str, entradas: int, salidas: int, tipo: str, cfg: Config, semilla: int):
        if metodo not in METODOS:
            raise KeyError(f"Método desconocido: {metodo}. Opciones: {', '.join(METODOS)}")
        self.metodo, self.cfg, self.tipo, self.semilla = metodo, cfg, tipo, semilla
        self.gen = torch.Generator().manual_seed(semilla)
        torch.manual_seed(semilla)
        self.m = MLPMascarado(entradas, salidas, cfg.ranuras, cfg.capas)
        self.B = presupuesto(entradas, salidas, cfg)
        self._armar_estructura()
        self.opt = torch.optim.AdamW(self.m.parameters(), lr=cfg.lr, weight_decay=cfg.decaimiento)
        self.flops, self.curva, self.clases = 0.0, [], None
        self._conn = None

    def _armar_estructura(self) -> None:
        c, m = self.cfg, self.m
        if self.metodo == "mlp_pequeno":
            m.densa(c.ancho_pequeno)
        elif self.metodo in ("mlp_grande", "poda_magnitud"):
            m.densa(c.ancho_grande)
        elif self.metodo == "crecimiento_programado":
            m.densa(max(2, c.ancho_pequeno // 4))
        else:
            m.esparcida(self.B, c.activas_inicial, self.gen)
        self._conn = None

    def fijar_datos(self, x, y, clases=None) -> None:
        """Parte los datos de entrenamiento en entrenamiento y validación (la prueba solo se mide al final)."""
        n = len(x)
        perm = torch.randperm(n, generator=self.gen)
        nv = max(1, int(n * self.cfg.fraccion_val))
        self.xv, self.yv = x[perm[:nv]], y[perm[:nv]]
        self.x, self.y = x[perm[nv:]], y[perm[nv:]]
        self.clases = None if clases is None else torch.as_tensor(clases)
        if self.clases is not None:  # tarea con cabeza propia: se miran solo sus clases
            tabla = torch.full((int(max(self.clases.max(), 1)) + 1,), -1, dtype=torch.long)
            tabla[self.clases] = torch.arange(len(self.clases))
            self.y, self.yv = tabla[self.y], tabla[self.yv]

    # --- pérdida y evaluación ----------------------------------------------------------------------------
    def _salida(self, x):
        s = self.m(x)
        return s if self.clases is None else s[:, self.clases]

    def perdida(self, salida, y):
        return F.cross_entropy(salida, y) if self.tipo == "clasificacion" else F.mse_loss(salida, y)

    def _fn_perdida(self):
        """Pérdida sobre la salida completa del modelo (la que usan los gradientes densos)."""
        return lambda salida, y: self.perdida(salida if self.clases is None else salida[:, self.clases], y)

    @torch.no_grad()
    def evaluar(self, x, y, clases=None, a_mano=False):
        """(pérdida, métrica): métrica = exactitud o mse."""
        cl = self.clases if clases is None else clases
        s = self.m(x)
        if cl is not None:
            s = s[:, cl]
        p = self.perdida(s, y).item()
        met = (s.argmax(1) == y).float().mean().item() if self.tipo == "clasificacion" else p
        if not a_mano:
            self.flops += 2 * self._n_conn() * len(x)
        return p, met

    def evaluar_prueba(self, x, y, clases=None):
        """Para la prueba final: ``clases`` es la lista de clases de la tarea, ``y`` trae las etiquetas originales."""
        if clases is not None:
            tabla = torch.full((int(max(max(clases), y.max().item())) + 1,), -1, dtype=torch.long)
            tabla[torch.as_tensor(clases)] = torch.arange(len(clases))
            y = tabla[y]
        return self.evaluar(x, y, torch.as_tensor(clases) if clases is not None else None, a_mano=True)

    def _n_conn(self) -> int:
        if self._conn is None:
            self._conn = self.m.conexiones()
        return self._conn

    # --- entrenamiento -----------------------------------------------------------------------------------
    def _lr(self, paso: int, total: int) -> float:
        return self.cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(paso / max(total, 1), 1.0))))

    def _entrenar(self, n: int, paso0: int, total: int, gen) -> None:
        c = self.cfg
        for i in range(n):
            for g in self.opt.param_groups:
                g["lr"] = self._lr(paso0 + i, total)
            idx = torch.randint(0, len(self.x), (c.lote,), generator=gen)
            self.opt.zero_grad(set_to_none=True)
            self.perdida(self._salida(self.x[idx]), self.y[idx]).backward()
            self.opt.step()
        self.flops += 6 * self._n_conn() * c.lote * n

    def entrenar_tarea(self, pasos: int | None = None) -> None:
        c = self.cfg
        total = pasos or c.pasos
        paso = 0
        while paso < total:
            n = min(c.cada, total - paso)
            self._entrenar(n, paso, total, self.gen)
            paso += n
            _, v = self.evaluar(self.xv, self.yv)
            self.curva.append((self.flops, paso, v))
            if c.calentamiento <= paso <= c.fin_reorg * total and paso < total and self.metodo not in ("mlp_pequeno", "mlp_grande"):
                self._evento(paso, total)

    # --- eventos de estructura ---------------------------------------------------------------------------
    def _fraccion(self, paso: int, total: int) -> float:
        fin = self.cfg.fin_reorg * total
        return self.cfg.fraccion * 0.5 * (1 + math.cos(math.pi * min(paso / fin, 1.0)))

    def _evento(self, paso: int, total: int) -> None:
        c, m = self.cfg, self.m
        if self.metodo == "poda_magnitud":
            fin = c.fin_reorg * total
            prog = min(max((paso - c.calentamiento) / max(fin - c.calentamiento, 1), 0), 1)
            op.poda_global_por_magnitud(m, int(self.B + (m.conexiones_densas() - self.B) * (1 - prog) ** 3))
        elif self.metodo == "crecimiento_programado":
            prog = min(max(paso / (c.fin_reorg * total), 0), 1)
            w0 = max(2, c.ancho_pequeno // 4)
            meta = int(round(w0 + (c.ancho_pequeno - w0) * prog))
            antes = op.foto_estructura(m)
            while m.neuronas_activas(0) < meta:
                for l in range(c.capas):
                    op.activar_neurona_densa(m, l, self.gen)
            op.reiniciar_estado_optimizador(self.opt, m, antes)
        else:  # set, rigl
            if self.metodo == "rigl":
                idx = torch.randperm(len(self.x), generator=self.gen)[: c.lote]
                self.flops += (4 * self._n_conn() + 2 * m.conexiones_densas()) * len(idx)
                gd = op.gradientes_densos(m, self.x[idx], self.y[idx], self._fn_perdida())
            else:
                gd = [torch.zeros_like(m.mascara(l)) for l in range(c.capas + 1)]
            punt = [m.capas[l].weight.detach().abs() * m.mascara(l) for l in range(c.capas + 1)]
            antes = op.foto_estructura(m)
            op.reorganizar(m, punt, gd, self._fraccion(paso, total), self.B, self.gen, azar_crecer=self.metodo == "set")
            op.reiniciar_estado_optimizador(self.opt, m, antes)
        self._conn = None

    # --- resumen -----------------------------------------------------------------------------------------
    def resultado(self, x_prueba=None, y_prueba=None, clases=None) -> Resultado:
        r = Resultado(self.metodo, self.semilla, parametros=self.m.parametros_activos(), conexiones=self.m.conexiones(),
                      neuronas=[self.m.neuronas_activas(l) for l in range(self.cfg.capas)], flops=self.flops, curva=self.curva)
        if x_prueba is not None:
            r.perdida_prueba, r.metrica_prueba = self.evaluar_prueba(x_prueba, y_prueba, clases)
        return r


def entrenar(metodo: str, tarea, semilla: int, cfg: Config | None = None) -> Resultado:
    """Entrena ``metodo`` en ``tarea`` (una ``Tarea`` del banco) y mide en su prueba al final."""
    e = Entrenador(metodo, tarea.entradas, tarea.salidas, tarea.tipo, cfg or Config(), semilla)
    e.fijar_datos(tarea.x_entreno, tarea.y_entreno)
    e.entrenar_tarea()
    return e.resultado(tarea.x_prueba, tarea.y_prueba)
