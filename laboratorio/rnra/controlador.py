"""El entrenador de RNRA-v0 y de sus comparadores, todos sobre el mismo MLP con máscaras.

Comparadores (mismo optimizador, mismos pasos, mismo presupuesto final de conexiones):
``mlp_pequeno`` y ``mlp_grande`` (densos), ``poda_magnitud`` (parte del grande y poda de a poco), ``crecimiento_programado``
(parte angosto y agrega neuronas por calendario), ``set`` (quita lo más débil, crea al azar), ``rigl`` (quita lo más débil,
crea donde el gradiente es mayor), ``rnra_aleatorio`` (control negativo: mismos conteos que RNRA pero todo al azar).

RNRA-v0 en cada evento: medir → proponer (poda de neuronas redundantes verificada por ablación real, neuronas de reserva si
hay evidencia de falta de capacidad, reasignación de conexiones hacia donde la demanda es alta) → entrenar una ventana corta
**con** el cambio y otra **sin** él, con los mismos lotes → aceptar solo si la validación mejora; si no, revertir pesos,
máscaras y estado del optimizador. El coste de la prueba se cuenta en los FLOPs. Las decisiones usan entrenamiento y una
parte de validación apartada del entrenamiento; la prueba solo se mide al final.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from . import operaciones as op
from .estadisticas import Demanda, Estadisticas, Regiones
from .modelo import MLPMascarado

ABLACIONES = {
    "rnra": {},
    "rnra_sin_utilidad": {"usar_u": False},
    "rnra_sin_redundancia": {"usar_r": False},
    "rnra_sin_local": {"usar_local": False},
    "rnra_sin_histeresis": {"histeresis": False},
    "rnra_sin_crecimiento": {"crecer": False},
    "rnra_sin_poda": {"podar": False},
    "rnra_aleatorio": {"azar": True},
}
COMPARADORES = ["mlp_pequeno", "mlp_grande", "poda_magnitud", "crecimiento_programado", "set", "rigl"]
METODOS = COMPARADORES + list(ABLACIONES)


@dataclass
class Config:
    capas: int = 2
    ancho_pequeno: int = 16       # el MLP fijo pequeño fija el presupuesto de conexiones de todos
    ancho_grande: int = 64
    ancho_max: int = 64           # ranuras de neuronas (activas + reserva)
    activas_inicial: int = 32
    pasos: int = 800
    lote: int = 128
    lr: float = 3e-3
    decaimiento: float = 1e-4
    calentamiento: int = 100
    cada: int = 50                # pasos entre eventos de reorganización
    ventana: int = 25             # pasos de prueba con y sin el cambio
    fin_reorg: float = 0.75       # no se reorganiza en el último cuarto
    fraccion: float = 0.2         # fracción de conexiones que se mueve (baja con coseno)
    regiones: int = 8
    lote_medida: int = 512
    fraccion_val: float = 0.2
    # RNRA
    alfa: float = 0.5             # peso de la redundancia en Q
    beta: float = 0.2             # peso de la estabilidad en Q
    umbral_redundancia: float = 0.9
    tol_poda: float = 0.01        # la ablación real puede subir la pérdida de validación como máximo 1 %
    persistencia: int = 2         # eventos seguidos como candidata antes de podar (histéresis)
    enfriamiento: int = 3         # eventos sin tocar una neurona cambiada (histéresis)
    proteccion: int = 2           # eventos que una conexión nueva no puede quitarse (histéresis)
    max_poda: int = 2
    min_activas: int = 4
    tol_mejora: float = 0.01      # mejora relativa de validación por debajo de la cual se considera estancada
    umbral_perdida: float = 0.05
    margen: float = 0.0           # se acepta si pérdida_con < pérdida_sin · (1 − margen)
    olvido: float = 1.0           # λ de protección de lo aprendido en tareas anteriores (aprendizaje continuo)

    @property
    def ranuras(self) -> int:
        return max(self.ancho_max, self.ancho_grande)


@dataclass
class Resultado:
    metodo: str
    semilla: int
    metrica_prueba: float = float("nan")      # exactitud (clasificación) o mse
    perdida_prueba: float = float("nan")
    parametros: int = 0
    conexiones: int = 0
    neuronas: list = field(default_factory=list)
    flops: float = 0.0
    curva: list = field(default_factory=list)  # (flops, paso, métrica de validación)
    eventos: list = field(default_factory=list)
    ciclos: int = 0                            # ranuras que se apagaron y volvieron a encenderse (oscilación)
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
        self.flag = {"usar_u": True, "usar_r": True, "usar_local": True, "histeresis": True, "crecer": True,
                     "podar": True, "azar": False, **ABLACIONES.get(metodo, {})}
        self.es_rnra = metodo in ABLACIONES
        self.gen = torch.Generator().manual_seed(semilla)
        torch.manual_seed(semilla)
        self.m = MLPMascarado(entradas, salidas, cfg.ranuras, cfg.capas)
        self.B = presupuesto(entradas, salidas, cfg)
        self._armar_estructura()
        self.opt = torch.optim.AdamW(self.m.parameters(), lr=cfg.lr, weight_decay=cfg.decaimiento)
        self.est = Estadisticas(self.m)
        self.imp_previa = [torch.zeros(cfg.ranuras) for _ in range(cfg.capas)]
        self.flops, self.evento, self.curva, self.eventos = 0.0, 0, [], []
        self.hist_val: list[float] = []
        self.bajo = [torch.zeros(cfg.ranuras, dtype=torch.long) for _ in range(cfg.capas)]
        self.ultimo = [torch.full((cfg.ranuras,), -10**6, dtype=torch.long) for _ in range(cfg.capas)]
        self.nacida = [torch.full_like(self.m.mascara(l), -10**6, dtype=torch.long) for l in range(cfg.capas + 1)]
        self.cambios_ranura = [torch.zeros(cfg.ranuras, dtype=torch.long) for _ in range(cfg.capas)]
        self._conn = None

    # --- preparación -------------------------------------------------------------------------------------
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
        """Parte los datos de entrenamiento en entrenamiento y validación (esta última solo decide estructura)."""
        c = self.cfg
        n = len(x)
        perm = torch.randperm(n, generator=self.gen)
        nv = max(1, int(n * c.fraccion_val))
        self.xv, self.yv = x[perm[:nv]], y[perm[:nv]]
        self.x, self.y = x[perm[nv:]], y[perm[nv:]]
        self.clases = None if clases is None else torch.as_tensor(clases)
        if self.clases is not None:  # tarea con cabeza propia: se miran solo sus clases
            tabla = torch.full((int(max(self.clases.max(), 1)) + 1,), -1, dtype=torch.long)
            tabla[self.clases] = torch.arange(len(self.clases))
            self.y, self.yv = tabla[self.y], tabla[self.yv]
        self.reg = Regiones(self.x, c.regiones, self.semilla)
        self.reg_ent = self.reg.asignar(self.x)
        self.dem = Demanda(self.reg.k)
        self.hist_val = []

    # --- pérdida y evaluación ----------------------------------------------------------------------------
    def _salida(self, x, **kw):
        s = self.m(x, **kw)
        return s if self.clases is None else s[:, self.clases]

    def perdida(self, salida, y):
        return F.cross_entropy(salida, y) if self.tipo == "clasificacion" else F.mse_loss(salida, y)

    def _perdida_modelo(self, modelo_salida, y):
        return self.perdida(modelo_salida if self.clases is None else modelo_salida[:, self.clases], y)

    def _fn_perdida(self):
        """Pérdida sobre la salida completa del modelo (la que usan las estadísticas y los gradientes)."""
        return lambda salida, y: self._perdida_modelo(salida, y)

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
        self.curva_tarea = []
        while paso < total:
            n = min(c.cada, total - paso)
            self._entrenar(n, paso, total, self.gen)
            paso += n
            _, v = self.evaluar(self.xv, self.yv)
            self.curva.append((self.flops, paso, v))
            if c.calentamiento <= paso <= c.fin_reorg * total and paso < total and self.metodo not in ("mlp_pequeno", "mlp_grande"):
                paso += self._evento(paso, total)
        self._fin_tarea()

    def _fin_tarea(self) -> None:
        """Guarda qué neuronas fueron importantes en esta tarea (se protegen en las siguientes)."""
        if not self.es_rnra:
            return
        n = min(self.cfg.lote_medida, len(self.x))
        idx = torch.randperm(len(self.x), generator=self.gen)[:n]
        self.est.medir(self.x[idx], self.y[idx], self._fn_perdida())
        for l in range(self.cfg.capas):
            act = self.m.activa(l).cpu() > 0
            u = self.est.utilidad(l)
            u = u / (u[act].max() + 1e-8) if act.any() else u
            self.imp_previa[l] = torch.maximum(self.imp_previa[l], torch.where(act, u, torch.zeros_like(u)))
        self.est.listo = False  # la siguiente tarea empieza sus estadísticas de cero

    # --- eventos -----------------------------------------------------------------------------------------
    def _fraccion(self, paso: int, total: int) -> float:
        fin = self.cfg.fin_reorg * total
        return self.cfg.fraccion * 0.5 * (1 + math.cos(math.pi * min(paso / fin, 1.0)))

    def _evento(self, paso: int, total: int) -> int:
        """Devuelve cuántos pasos extra se avanzaron dentro del evento."""
        self.evento += 1
        m = self.metodo
        if m == "poda_magnitud":
            fin = self.cfg.fin_reorg * total
            prog = min(max((paso - self.cfg.calentamiento) / max(fin - self.cfg.calentamiento, 1), 0), 1)
            c0 = self.m.conexiones_densas()
            op.poda_global_por_magnitud(self.m, int(self.B + (c0 - self.B) * (1 - prog) ** 3))
            self._conn = None
            return 0
        if m == "crecimiento_programado":
            fin = self.cfg.fin_reorg * total
            prog = min(max(paso / fin, 0), 1)
            w0 = max(2, self.cfg.ancho_pequeno // 4)
            meta = int(round(w0 + (self.cfg.ancho_pequeno - w0) * prog))
            antes = op.foto_estructura(self.m)
            while self.m.neuronas_activas(0) < meta:
                for l in range(self.cfg.capas):
                    op.activar_neurona(self.m, l, self.gen, denso=True)
            op.reiniciar_estado_optimizador(self.opt, self.m, antes)
            self._conn = None
            return 0
        if m in ("set", "rigl"):
            self._evento_dinamico(paso, total)
            return 0
        return self._evento_rnra(paso, total)

    def _lote_entreno(self, n: int, pesos: torch.Tensor | None = None):
        n = min(n, len(self.x))
        if pesos is None:
            idx = torch.randperm(len(self.x), generator=self.gen)[:n]
        else:
            idx = torch.multinomial(pesos, n, replacement=True, generator=self.gen)
        return idx

    def _grad_denso(self, idx):
        self.flops += (4 * self._n_conn() + 2 * self.m.conexiones_densas()) * len(idx)
        return op.gradientes_densos(self.m, self.x[idx], self.y[idx], self._fn_perdida())

    def _evento_dinamico(self, paso: int, total: int) -> None:
        """SET y RigL: cambian conexiones directamente, sin probar."""
        idx = self._lote_entreno(self.cfg.lote)
        gd = self._grad_denso(idx) if self.metodo == "rigl" else [torch.zeros_like(self.m.mascara(l)) for l in range(self.cfg.capas + 1)]
        punt = [(self.m.capas[l].weight.detach().abs() * self.m.mascara(l)) for l in range(self.cfg.capas + 1)]
        antes = op.foto_estructura(self.m)
        op.reorganizar(self.m, punt, gd, self._fraccion(paso, total), self.B, self.gen, azar_crecer=self.metodo == "set")
        op.reiniciar_estado_optimizador(self.opt, self.m, antes)
        self._conn = None

    # --- RNRA --------------------------------------------------------------------------------------------
    def _puntajes_conexion(self) -> list[torch.Tensor]:
        """Valor de cada conexión: |w| · valor de su neurona de salida · valor de su neurona de entrada.
        El valor de una neurona es 0,5 + utilidad normalizada (+ λ · importancia en tareas anteriores)."""
        f = self.flag
        valor = []
        for l in range(self.cfg.capas):
            act = self.m.activa(l).cpu() > 0
            u = self.est.utilidad(l)
            u = u / (u[act].max() + 1e-8) if act.any() else u
            v = (0.5 + (u if f["usar_u"] else 0)) + self.cfg.olvido * self.imp_previa[l]
            valor.append(v)
        punt = []
        for l in range(self.cfg.capas + 1):
            p = self.m.capas[l].weight.detach().abs() * self.m.mascara(l)
            if l < self.cfg.capas:
                p = p * valor[l][:, None]
            if l > 0:
                p = p * valor[l - 1][None, :]
            punt.append(p)
        return punt

    def _hay_evidencia(self, perdida_entreno: float, perdida_val: float) -> bool:
        """Falta capacidad si la validación dejó de mejorar y el entrenamiento tampoco está muy por debajo
        (no es sobreajuste) ni ya es casi perfecto."""
        if len(self.hist_val) < 2:
            return False
        ant, ahora = self.hist_val[-2], self.hist_val[-1]
        estancada = (ant - ahora) / (abs(ant) + 1e-8) < self.cfg.tol_mejora
        return estancada and perdida_entreno >= 0.6 * perdida_val and perdida_entreno > self.cfg.umbral_perdida

    def _proponer(self) -> dict | None:
        c, m, f = self.cfg, self.m, self.flag
        idx = self._lote_entreno(c.lote_medida)
        xb, yb = self.x[idx], self.y[idx]
        fn = self._fn_perdida()
        self.flops += 6 * self._n_conn() * len(idx)
        self.est.medir(xb, yb, fn)
        with torch.no_grad():
            s = self._salida(xb)
            por_muestra = (F.cross_entropy(s, yb, reduction="none") if self.tipo == "clasificacion"
                           else F.mse_loss(s, yb, reduction="none").mean(1))
        self.dem.actualizar(self.reg_ent[idx], por_muestra)
        l_ent = por_muestra.mean().item()
        l_val, _ = self.evaluar(self.xv, self.yv)
        self.hist_val.append(l_val)
        pesos = self.dem.pesos_muestras(self.reg_ent) if f["usar_local"] else None
        gd = self._grad_denso(self._lote_entreno(c.lote, pesos))
        prop = {"pares": [], "crecer": [], "grad": gd, "punt": self._puntajes_conexion(), "l_val": l_val}
        # poda de neuronas redundantes, con persistencia y verificada por ablación real
        for l in range(c.capas):
            act = m.activa(l).cpu() > 0
            q = self.est.puntaje(l, c.alfa, c.beta, f["usar_u"], f["usar_r"]) - c.olvido * self.imp_previa[l]
            cand = act & (self.est.red[l] > c.umbral_redundancia if f["usar_r"] else act)
            cand &= q <= torch.quantile(q[act], 0.5) if act.sum() > 2 else cand
            self.bajo[l] = torch.where(cand, self.bajo[l] + 1, torch.zeros_like(self.bajo[l]))
            if f["podar"]:
                persiste = self.bajo[l] >= (c.persistencia if f["histeresis"] else 1)
                libre = (self.evento - self.ultimo[l]) > (c.enfriamiento if f["histeresis"] else 0)
                orden = torch.argsort(q)
                for i in orden.tolist():
                    if len(prop["pares"]) >= c.max_poda:
                        break
                    if persiste[i] and libre[i] and cand[i] and act.sum() - sum(1 for p in prop["pares"] if p[0] == l) > c.min_activas:
                        prop["pares"].append((l, i))
        if prop["pares"]:
            prop["pares"] = self._verificar_ablacion(prop["pares"], l_val)
        # crecer solo con evidencia; en la capa con neuronas más cargadas de utilidad
        if f["crecer"] and self._hay_evidencia(l_ent, l_val) and any((m.activa(l) == 0).any() for l in range(c.capas)):
            con_reserva = [l for l in range(c.capas) if (m.activa(l) == 0).any()]
            act_u = [self.est.utilidad(l)[m.activa(l).cpu() > 0].mean().item() for l in con_reserva]
            prop["crecer"] = [con_reserva[int(torch.tensor(act_u).argmax())]]
        if f["azar"]:  # control negativo: mismos conteos, elección al azar
            todas = [(l, i) for l in range(c.capas) for i in (m.activa(l) > 0).nonzero().flatten().tolist()]
            k = len(prop["pares"])
            prop["pares"] = [todas[j] for j in torch.randperm(len(todas), generator=self.gen)[:k].tolist()] if k else []
        return prop

    @torch.no_grad()
    def _verificar_ablacion(self, pares, l_base):
        """Apaga las candidatas de a una y se queda con las que no suben la pérdida de validación más de ``tol_poda``."""
        buenas = []
        for l, i in pares:
            self.m.activa(l)[i] = 0
            p, _ = self.evaluar(self.xv, self.yv)
            if p <= l_base * (1 + self.cfg.tol_poda) + 1e-4:
                buenas.append((l, i))
            else:
                self.m.activa(l)[i] = 1
        for l, i in buenas:
            self.m.activa(l)[i] = 1  # se devuelve todo: la poda real se hace al aplicar la propuesta
        return buenas

    def _aplicar(self, prop: dict, paso: int, total: int) -> dict:
        c, m, f = self.cfg, self.m, self.flag
        antes = op.foto_estructura(m)
        n0 = m.conexiones()
        tocadas = []
        liberadas = op.podar_neuronas(m, prop["pares"]) if prop["pares"] else 0
        tocadas += prop["pares"]
        nuevas = 0
        for l in prop["crecer"]:
            c_in, c_out = op.costo_activar(m, l)
            libre = self.B - m.conexiones()
            if libre < c_in + c_out:
                op.liberar_conexiones(m, c_in + c_out - libre, prop["punt"], self.gen, aleatorio=f["azar"])
            usadas, ranura = op.activar_neurona(m, l, self.gen, excluir=[i for ll, i in prop["pares"] if ll == l])
            if usadas:
                nuevas += 1
                tocadas.append((l, ranura))
        protegidas = [(self.evento - self.nacida[l]) < (c.proteccion if f["histeresis"] else 0) for l in range(c.capas + 1)]
        r = op.reorganizar(m, prop["punt"], prop["grad"], self._fraccion(paso, total), self.B, self.gen,
                           protegidas=protegidas, azar_quitar=f["azar"], azar_crecer=f["azar"])
        for l in range(c.capas + 1):
            self.nacida[l][r["nuevas"][l] > 0] = self.evento
        for l, i in tocadas:
            self.ultimo[l][i] = self.evento
            self.cambios_ranura[l][i] += 1
        op.reiniciar_estado_optimizador(self.opt, m, antes)
        self._conn = None
        return {"podadas": len(prop["pares"]), "activadas": nuevas, "movidas": r["movidas"],
                "liberadas": liberadas, "conexiones": m.conexiones(), "delta_conexiones": m.conexiones() - n0}

    def _foto(self):
        return copy.deepcopy(self.m.state_dict()), copy.deepcopy(self.opt.state_dict())

    def _restaurar(self, foto) -> None:
        self.m.load_state_dict(foto[0])
        self.opt.load_state_dict(copy.deepcopy(foto[1]))
        self._conn = None

    def _evento_rnra(self, paso: int, total: int) -> int:
        c = self.cfg
        prop = self._proponer()
        ventana = min(c.ventana, total - paso)
        semilla = int(torch.randint(0, 2**31 - 1, (1,), generator=self.gen))
        s0 = self._foto()
        # 1) continuación sin cambios
        self._entrenar(ventana, paso, total, torch.Generator().manual_seed(semilla))
        l_sin, _ = self.evaluar(self.xv, self.yv)
        s_sin = self._foto()
        # 2) lo mismo con el cambio
        self._restaurar(s0)
        info = self._aplicar(prop, paso, total)
        self._entrenar(ventana, paso, total, torch.Generator().manual_seed(semilla))
        l_con, _ = self.evaluar(self.xv, self.yv)
        acepta = l_con < l_sin * (1 - c.margen)
        if not acepta:
            self._restaurar(s_sin)
        info.update(paso=paso, aceptado=bool(acepta), perdida_con=l_con, perdida_sin=l_sin)
        self.eventos.append(info)
        if acepta:
            self.hist_val[-1] = l_con
        return ventana

    # --- resumen -----------------------------------------------------------------------------------------
    def resultado(self, x_prueba=None, y_prueba=None, clases=None) -> Resultado:
        r = Resultado(self.metodo, self.semilla, parametros=self.m.parametros_activos(), conexiones=self.m.conexiones(),
                      neuronas=[self.m.neuronas_activas(l) for l in range(self.cfg.capas)], flops=self.flops,
                      curva=self.curva, eventos=self.eventos,
                      ciclos=int(sum((c >= 3).sum().item() for c in self.cambios_ranura)))
        if x_prueba is not None:
            r.perdida_prueba, r.metrica_prueba = self.evaluar_prueba(x_prueba, y_prueba, clases)
        return r


def entrenar(metodo: str, tarea, semilla: int, cfg: Config | None = None) -> Resultado:
    """Entrena ``metodo`` en ``tarea`` (una ``Tarea`` del banco) y mide en su prueba al final."""
    cfg = cfg or Config()
    e = Entrenador(metodo, tarea.entradas, tarea.salidas, tarea.tipo, cfg, semilla)
    e.fijar_datos(tarea.x_entreno, tarea.y_entreno)
    e.entrenar_tarea()
    return e.resultado(tarea.x_prueba, tarea.y_prueba)
