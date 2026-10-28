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

from .arquitecturas import ARQUITECTURAS, crear_modelo
from .piezas import PIEZAS, contar_parametros
from .tareas import SEMILLA_DATOS, TAREAS, Tarea, cargar_tarea


@dataclass(frozen=True)
class Config:
    epocas: int | None = None  # máximo de épocas; None = las que indica la tarea
    lr: float = 0.01
    ocultas: int = 32
    semillas: tuple[int, ...] = (0, 1, 2, 3, 4)
    lote: int | None = None  # tamaño de lote; None = el de la tarea (o todo junto en las pequeñas)
    paciencia: int | None = None  # parada temprana: épocas sin mejorar en validación antes de cortar
    min_mejora: float = 0.0  # cuánto tiene que mejorar la validación para contar como mejora
    validacion: float = 0.15  # parte del entrenamiento que se separa para decidir la parada
    guardar: str | None = None  # carpeta donde guardar el mejor modelo de cada corrida
    arquitectura: str = "densa"  # densa, cnn, texto_cnn, transformer, serie_cnn o lstm (ver arquitecturas.py)
    dispositivo: str = "auto"  # "auto" usa la GPU si hay (Colab), si no la CPU


# Registro de evaluaciones «pieza/tarea» (como el registro de openai/evals).
# Tareas pequeñas: todo junto y épocas fijas. Tareas reales: por lotes, lr más baja,
# parada temprana y 3 semillas (tardan más).
CONFIG_REAL = Config(lr=0.001, paciencia=5, semillas=(0, 1, 2))
EVALUACIONES: dict[str, Config] = {f"{p}/{t}": Config() for p in PIEZAS for t in TAREAS}


def _registrar_reales() -> None:
    from .datos_reales import FUENTES

    EVALUACIONES.update({f"{p}/{t}": CONFIG_REAL for p in PIEZAS for t in FUENTES})


_registrar_reales()


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
    arquitectura: str = "densa"
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


def _predecir(modelo, x: torch.Tensor, lote: int = 4096) -> torch.Tensor:
    """Predice por partes para no llenar la memoria con conjuntos grandes."""
    return torch.cat([modelo(x[i : i + lote]) for i in range(0, len(x), lote)])


def _estado_extra(modelo) -> dict:
    """Lo que no está en state_dict pero define a la red (p. ej. las activaciones mutadas de SMA)."""
    return {n: list(m.current_activations) for n, m in modelo.named_modules() if hasattr(m, "current_activations")}


def _poner_estado_extra(modelo, extra: dict) -> None:
    for n, m in modelo.named_modules():
        if n in extra:
            m.current_activations = list(extra[n])


def ruta_modelo(carpeta: str | Path, pieza: str, tarea: str, semilla: int, arquitectura: str = "densa") -> Path:
    sufijo = "" if arquitectura == "densa" else f"__{arquitectura}"
    return Path(carpeta) / f"{pieza}__{tarea}{sufijo}__s{semilla}.pt"


def _dispositivo(nombre: str) -> torch.device:
    if nombre == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(nombre)


def cargar_modelo(ruta: str | Path):
    """Vuelve a crear una red guardada por el banco (solo pesos: weights_only=True). Queda en CPU."""
    from types import SimpleNamespace

    d = torch.load(ruta, weights_only=True, map_location="cpu")
    forma = d.get("forma")
    meta = SimpleNamespace(nombre=d["tarea"], entradas=d["entradas"], salidas=d["salidas"], dato=d.get("dato", "tabla"),
                           forma=tuple(forma) if forma is not None else None, vocabulario=d.get("vocabulario"))
    modelo = crear_modelo(PIEZAS[d["pieza"]], meta, d.get("arquitectura", "densa"), d["ocultas"])
    modelo.load_state_dict(d["pesos"])
    _poner_estado_extra(modelo, d["extra"])
    modelo.eval()
    return modelo


def entrenar_una(pieza: str, tarea: str | Tarea, semilla: int, config: Config = Config()) -> Corrida:
    """Entrena una red con la pieza sobre la tarea (Adam) y la mide en prueba.

    - Con ``lote`` entrena por lotes (necesario en conjuntos grandes); si no, todo junto.
    - Con ``paciencia`` separa una parte para validación, corta cuando deja de
      mejorar y vuelve a los pesos de la mejor época (parada temprana).
    - Con ``guardar`` escribe el mejor modelo en disco; se recupera con ``cargar_modelo``.
    """
    arq = ARQUITECTURAS[config.arquitectura]
    t = cargar_tarea(tarea, arq.representacion) if isinstance(tarea, str) else tarea
    disp = _dispositivo(config.dispositivo)
    x_e, y_e = t.x_entreno.to(disp), t.y_entreno.to(disp)
    x_prueba, y_prueba = t.x_prueba.to(disp), t.y_prueba.to(disp)
    x_v = y_v = None
    if config.paciencia:
        # La misma división de validación para todas las piezas (depende de los datos, no de la red).
        orden = torch.randperm(len(x_e), generator=torch.Generator().manual_seed(SEMILLA_DATOS)).to(disp)
        n_v = max(1, int(len(x_e) * config.validacion))
        x_v, y_v = x_e[orden[:n_v]], y_e[orden[:n_v]]
        x_e, y_e = x_e[orden[n_v:]], y_e[orden[n_v:]]

    torch.manual_seed(semilla)
    modelo = crear_modelo(PIEZAS[pieza], t, config.arquitectura, config.ocultas).to(disp)
    opt = torch.optim.Adam(modelo.parameters(), lr=config.lr)
    epocas = config.epocas or t.epocas
    lote = config.lote or t.lote
    barajador = torch.Generator().manual_seed(semilla)
    signo = 1 if t.mayor_es_mejor else -1
    mejor, mejor_epoca, sin_mejora = -math.inf, -1, 0
    mejor_estado = None
    inicio = time.perf_counter()
    perdida = math.nan
    fallo = False
    epoca = -1
    for epoca in range(epocas):
        modelo.train()
        indices = torch.randperm(len(x_e), generator=barajador).to(disp) if lote else None
        partes = [indices[i : i + lote] for i in range(0, len(x_e), lote)] if lote else [None]
        for parte in partes:
            xb, yb = (x_e, y_e) if parte is None else (x_e[parte], y_e[parte])
            salida = modelo(xb)
            loss = _perdida(t, salida, yb)
            if not torch.isfinite(loss):
                fallo = True  # la red «explotó»: cuenta como resultado malo, no como error del banco
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
            perdida = loss.item()
            modelo.despues_del_paso(perdida=perdida, error=_error(t, salida.detach(), yb), epoca=epoca)
        if fallo:
            break
        if config.paciencia:
            modelo.eval()
            with torch.no_grad():
                valor_v = _medir(t, _predecir(modelo, x_v), y_v)
            if math.isfinite(valor_v) and signo * valor_v > mejor + config.min_mejora:
                mejor, mejor_epoca, sin_mejora = signo * valor_v, epoca, 0
                mejor_estado = ({k: v.detach().clone() for k, v in modelo.state_dict().items()}, _estado_extra(modelo))
            else:
                sin_mejora += 1
                if sin_mejora >= config.paciencia:
                    break
    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado[0])
        _poner_estado_extra(modelo, mejor_estado[1])
    modelo.eval()
    with torch.no_grad():
        salida_e = _predecir(modelo, x_e)
        valor_entreno = _medir(t, salida_e, y_e) if not fallo and torch.isfinite(salida_e).all() else math.nan
        salida = _predecir(modelo, x_prueba)
        valor = _medir(t, salida, y_prueba) if not fallo and torch.isfinite(salida).all() else math.nan
    fallo = fallo or math.isnan(valor)
    detalle = {"mutaciones": sum(getattr(m, "mutaciones", 0) for m in modelo.modules() if m is not modelo)}
    if config.paciencia:
        detalle.update(epocas_entrenadas=epoca + 1, mejor_epoca=mejor_epoca + 1, validacion=round(signo * mejor, 6))
    if config.guardar and not fallo:
        ruta = ruta_modelo(config.guardar, pieza, t.nombre, semilla, config.arquitectura)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"pieza": pieza, "tarea": t.nombre, "semilla": semilla, "entradas": t.entradas,
                    "salidas": t.salidas, "ocultas": config.ocultas, "arquitectura": config.arquitectura,
                    "dato": t.dato, "forma": t.forma, "vocabulario": t.vocabulario,
                    "pesos": {k: v.cpu() for k, v in modelo.state_dict().items()},
                    "extra": _estado_extra(modelo), "valor_prueba": valor}, ruta)
        detalle["modelo"] = str(ruta)
    return Corrida(pieza, t.nombre, semilla, t.metrica, valor, perdida, contar_parametros(modelo),
                   round(time.perf_counter() - inicio, 3), fallo, valor_entreno, config.arquitectura,
                   {k: v for k, v in detalle.items() if v})


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
    p_ajustado: float = math.nan  # p corregido por comparaciones múltiples (Benjamini-Hochberg)


def _resumen(valores: list[float]) -> tuple[float, float]:
    return mean(valores), (stdev(valores) if len(valores) > 1 else 0.0)


def _media_entreno(pares) -> float:
    v = [p.valor_entreno for p, _ in pares if not math.isnan(p.valor_entreno)]
    return mean(v) if v else math.nan


def _veredicto(p: float, diferencia: float, alfa: float) -> str:
    if math.isnan(p) or p >= alfa or abs(diferencia) < 1e-12:
        return "igual"
    return "mejor" if diferencia > 0 else "peor"


def ajustar_multiples(comparaciones: list[Comparacion], alfa: float = 0.05) -> list[Comparacion]:
    """Corrige por hacer muchas comparaciones a la vez (Benjamini-Hochberg).

    Con 54 comparaciones a p < 0,05, por puro azar saldrían ~3 «ganadores» falsos.
    Este ajuste controla que, de los resultados declarados, como mucho un 5 % sean falsos.
    """
    validas = sorted((c for c in comparaciones if c.veredicto != "falla" and not math.isnan(c.p_valor)),
                     key=lambda c: c.p_valor)
    m = len(validas)
    q_min = 1.0
    for rango in range(m, 0, -1):  # de mayor a menor p
        c = validas[rango - 1]
        q_min = min(q_min, c.p_valor * m / rango)
        c.p_ajustado = q_min
        if q_min >= alfa:  # la corrección solo puede quitar veredictos, nunca crearlos
            c.veredicto = "igual"
    return comparaciones


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
        # Con una semilla, o sin ninguna variación entre semillas, no se puede estimar el ruido:
        # no se declara ganador.
        p = math.nan
    else:
        p = float(ttest_rel(tp, tb).pvalue)
    veredicto = _veredicto(p, media_d, alfa)
    return Comparacion(ref.pieza, ref.tarea, ref.metrica, pm, ps, bm, bs, dif, p, veredicto,
                       ref.parametros, pares[0][1].parametros, _media_entreno(pares))


_ICONO = {"mejor": "🟢 mejor", "igual": "⚪ igual", "peor": "🔴 peor", "falla": "💥 falla"}


def _fmt(metrica: str, media: float, std: float) -> str:
    if math.isnan(media):
        return "—"
    return f"{media * 100:.1f} ± {std * 100:.1f} %" if metrica == "exactitud" else f"{media:.4f} ± {std:.4f}"


def informe(comparaciones: list[Comparacion], semillas) -> str:
    """Tabla en Markdown para leer en el celular."""
    ajustado = any(not math.isnan(c.p_ajustado) for c in comparaciones)
    criterio = ("prueba t pareada corregida por comparaciones múltiples (Benjamini-Hochberg, 5 %)"
                if ajustado else "prueba t pareada (p < 0,05)")
    lineas = [
        "# Informe del banco de evaluación",
        "",
        f"Cada celda: media ± desviación en **datos de prueba**, con {semillas} semillas. "
        f"Exactitud: más es mejor. MSE: menos es mejor. Veredicto con {criterio}. "
        "Es una exploración: un 🟢 indica una pista que vale la pena confirmar, no una prueba definitiva.",
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
