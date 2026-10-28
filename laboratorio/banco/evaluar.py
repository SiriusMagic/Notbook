"""Entrenar, registrar y comparar (con el mismo diseño que openai/evals:
registro de evaluaciones con nombre, un registrador JSONL y un informe).

Cada corrida entrena la red con una semilla, mide la métrica en datos que la red
no vio (prueba) y guarda una línea JSON. La comparación usa las mismas semillas
para la pieza y para la base, y una prueba t pareada para decir si la diferencia
es real o es ruido.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn.functional as F

from .piezas import PIEZAS, Red, contar_parametros
from .tareas import TAREAS, Tarea, cargar_tarea


@dataclass(frozen=True)
class Config:
    epocas: int | None = None  # None = las que indica la tarea
    lr: float = 0.01
    ocultas: int = 32
    semillas: tuple[int, ...] = (0, 1, 2, 3, 4)


# Registro de evaluaciones «pieza/tarea». Por defecto todas contra todas con la misma config.
EVALUACIONES: dict[str, Config] = {f"{p}/{t}": Config() for p in PIEZAS for t in TAREAS}


@dataclass
class Corrida:
    pieza: str
    tarea: str
    semilla: int
    metrica: str
    valor: float
    perdida_final: float
    parametros: int
    segundos: float
    fallo: bool = False
    valor_entreno: float = math.nan  # misma métrica en los datos de entrenamiento (para ver memorización)
    detalle: dict = field(default_factory=dict)


def _perdida(tarea: Tarea, salida, objetivo):
    if tarea.tipo == "clasificacion":
        return F.cross_entropy(salida, objetivo)
    return F.mse_loss(salida, objetivo)


def _error(tarea: Tarea, salida, objetivo):
    if tarea.tipo == "clasificacion":
        return F.softmax(salida, dim=1) - F.one_hot(objetivo, tarea.salidas).float()
    return salida - objetivo


def _medir(tarea: Tarea, salida, objetivo) -> float:
    if tarea.tipo == "clasificacion":
        return float((salida.argmax(1) == objetivo).float().mean())
    return float(F.mse_loss(salida, objetivo))


def entrenar_una(pieza: str, tarea: str | Tarea, semilla: int, config: Config = Config()) -> Corrida:
    """Entrena una red con la pieza sobre la tarea (lote completo, Adam) y la mide en prueba."""
    t = cargar_tarea(tarea) if isinstance(tarea, str) else tarea
    torch.manual_seed(semilla)
    modelo = Red(PIEZAS[pieza], t.entradas, t.salidas, config.ocultas)
    opt = torch.optim.Adam(modelo.parameters(), lr=config.lr)
    epocas = config.epocas or t.epocas
    inicio = time.perf_counter()
    perdida = math.nan
    fallo = False
    modelo.train()
    for epoca in range(epocas):
        salida = modelo(t.x_entreno)
        loss = _perdida(t, salida, t.y_entreno)
        if not torch.isfinite(loss):
            fallo = True  # la red «explotó»: cuenta como resultado malo, no como error del banco
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        perdida = loss.item()
        modelo.despues_del_paso(perdida=perdida, error=_error(t, salida.detach(), t.y_entreno), epoca=epoca)
    modelo.eval()
    with torch.no_grad():
        salida_e = modelo(t.x_entreno)
        valor_entreno = _medir(t, salida_e, t.y_entreno) if not fallo and torch.isfinite(salida_e).all() else math.nan
        salida = modelo(t.x_prueba)
        valor = _medir(t, salida, t.y_prueba) if not fallo and torch.isfinite(salida).all() else math.nan
    fallo = fallo or math.isnan(valor)
    detalle = {"mutaciones": sum(getattr(m, "mutaciones", 0) for m in modelo.modules() if m is not modelo)}
    return Corrida(pieza, t.nombre, semilla, t.metrica, valor, perdida, contar_parametros(modelo),
                   round(time.perf_counter() - inicio, 3), fallo, valor_entreno, {k: v for k, v in detalle.items() if v})


class Registro:
    """Guarda cada corrida como una línea JSON (como el «recorder» de openai/evals)."""

    def __init__(self, carpeta: str | Path = "resultados") -> None:
        self.ruta = Path(carpeta) / f"{date.today().isoformat()}.jsonl"

    def guardar(self, corrida: Corrida) -> None:
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        with self.ruta.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(corrida), ensure_ascii=False) + "\n")

    @staticmethod
    def leer(ruta: str | Path) -> list[Corrida]:
        with Path(ruta).open(encoding="utf-8") as f:
            return [Corrida(**json.loads(linea)) for linea in f if linea.strip()]


def correr(pieza: str, tarea: str, semillas=None, config: Config | None = None, registro: Registro | None = None) -> list[Corrida]:
    config = config or EVALUACIONES.get(f"{pieza}/{tarea}", Config())
    corridas = [entrenar_una(pieza, tarea, s, config) for s in (semillas or config.semillas)]
    if registro:
        for c in corridas:
            registro.guardar(c)
    return corridas


@dataclass
class Comparacion:
    pieza: str
    tarea: str
    metrica: str
    pieza_media: float
    pieza_std: float
    base_media: float
    base_std: float
    diferencia: float  # positivo = la pieza es mejor
    p_valor: float
    veredicto: str  # "mejor", "igual", "peor" o "falla"
    parametros_pieza: int
    parametros_base: int
    pieza_entreno: float = math.nan  # media en entrenamiento: si es mucho mejor que en prueba, memoriza


def _resumen(valores: list[float]) -> tuple[float, float]:
    return mean(valores), (stdev(valores) if len(valores) > 1 else 0.0)


def _media_entreno(pares) -> float:
    v = [p.valor_entreno for p, _ in pares if not math.isnan(p.valor_entreno)]
    return mean(v) if v else math.nan


def comparar(corridas_pieza: list[Corrida], corridas_base: list[Corrida], alfa: float = 0.05) -> Comparacion:
    """Compara por semilla (pareado). «mejor/peor» solo si la diferencia es significativa."""
    from scipy.stats import ttest_rel

    base = {c.semilla: c for c in corridas_base}
    pares = [(c, base[c.semilla]) for c in corridas_pieza if c.semilla in base]
    if not pares:
        raise ValueError("La pieza y la base no comparten semillas")
    ref = pares[0][0]
    mayor_es_mejor = ref.metrica == "exactitud"
    vb = [b.valor for _, b in pares]
    bm, bs = _resumen(vb)
    if any(p.fallo for p, _ in pares):
        validos = [p.valor for p, _ in pares if not p.fallo] or [math.nan]
        pm, ps = _resumen(validos)
        return Comparacion(ref.pieza, ref.tarea, ref.metrica, pm, ps, bm, bs, math.nan, math.nan, "falla",
                           ref.parametros, pares[0][1].parametros, _media_entreno(pares))
    vp = [p.valor for p, _ in pares]
    pm, ps = _resumen(vp)
    signo = 1 if mayor_es_mejor else -1
    dif = signo * (pm - bm)
    # El error (MSE) crece por multiplicación: se compara su logaritmo para que una
    # sola semilla desastrosa no esconda que la pieza es peor en todas.
    tp, tb = (vp, vb) if mayor_es_mejor else ([math.log(max(v, 1e-12)) for v in vp], [math.log(max(v, 1e-12)) for v in vb])
    diferencias = [signo * (a - b) for a, b in zip(tp, tb)]
    media_d = mean(diferencias)
    if len(pares) < 2 or all(abs(d - diferencias[0]) < 1e-12 for d in diferencias):
        p = 0.0 if abs(media_d) > 1e-12 else 1.0  # sin variación: decide el signo
    else:
        p = float(ttest_rel(tp, tb).pvalue)
    veredicto = "igual" if p >= alfa or abs(media_d) < 1e-12 else ("mejor" if media_d > 0 else "peor")
    return Comparacion(ref.pieza, ref.tarea, ref.metrica, pm, ps, bm, bs, dif, p, veredicto,
                       ref.parametros, pares[0][1].parametros, _media_entreno(pares))


_ICONO = {"mejor": "🟢 mejor", "igual": "⚪ igual", "peor": "🔴 peor", "falla": "💥 falla"}


def _fmt(metrica: str, media: float, std: float) -> str:
    if math.isnan(media):
        return "—"
    return f"{media * 100:.1f} ± {std * 100:.1f} %" if metrica == "exactitud" else f"{media:.4f} ± {std:.4f}"


def informe(comparaciones: list[Comparacion], semillas: int) -> str:
    """Tabla en Markdown para leer en el celular."""
    lineas = [
        "# Informe del banco de evaluación",
        "",
        f"Cada celda: media ± desviación en **datos de prueba**, con {semillas} semillas. "
        "Exactitud: más es mejor. MSE: menos es mejor. Veredicto con prueba t pareada (p < 0,05).",
        "",
        "| Pieza | Tarea | Pieza (prueba) | Estándar (prueba) | Veredicto | Pieza en entreno | Parámetros (pieza / estándar) |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in comparaciones:
        lineas.append(f"| {c.pieza} | {c.tarea} | {_fmt(c.metrica, c.pieza_media, c.pieza_std)} | "
                      f"{_fmt(c.metrica, c.base_media, c.base_std)} | {_ICONO[c.veredicto]} | "
                      f"{_fmt(c.metrica, c.pieza_entreno, 0).split(' ±')[0]} | {c.parametros_pieza:,} / {c.parametros_base:,} |")
    cuenta = {v: sum(c.veredicto == v for c in comparaciones) for v in _ICONO}
    lineas += ["", "**Resumen:** " + ", ".join(f"{_ICONO[v]}: {n}" for v, n in cuenta.items()), ""]
    lineas.append("«Pieza en entreno» mucho mejor que «Pieza (prueba)» = la pieza memoriza en vez de aprender.")
    lineas.append("Ojo: una pieza con muchos más parámetros que la estándar tiene ventaja; compárala también con una estándar más grande.")
    return "\n".join(lineas) + "\n"
