"""Competencia: tus piezas contra la tecnología normal, tarea por tarea.

Para cada tarea compiten:

- la **red densa** (estándar y con cada pieza);
- la **tecnología normal** de ese tipo de dato (CNN para imágenes, TextCNN y
  Transformer para texto, CNN 1D y LSTM para series), en versión estándar y
  con cada pieza dentro;
- en series, además, la **predicción ingenua** (repetir el último valor): un
  modelo que no la supera no sirve para esa serie.

El informe tiene una tabla de posiciones por tarea y dice, para cada pieza, si
mejora la arquitectura donde va (misma arquitectura con ReLU), con la misma
prueba t pareada y la corrección por comparaciones múltiples del resto del banco.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from statistics import mean, stdev

from .arquitecturas import ARQUITECTURAS, NATURALES
from .evaluar import EVALUACIONES, Comparacion, Config, Corrida, Registro, ajustar_multiples, comparar, correr
from .tareas import cargar_tarea


@dataclass
class Fila:
    tarea: str
    arquitectura: str
    pieza: str
    metrica: str
    media: float
    std: float
    parametros: int
    segundos: float
    comparacion: Comparacion | None  # contra la misma arquitectura con la pieza estándar


def _nombre(pieza: str, arq: str) -> str:
    base = {"densa": "Red densa", "cnn": "CNN", "texto_cnn": "TextCNN", "transformer": "Transformer",
            "serie_cnn": "CNN 1D", "lstm": "LSTM"}[arq]
    return f"{base} normal" if pieza == "estandar" else f"{pieza} en {base}"


def _resumen(corridas: list[Corrida]) -> tuple[float, float]:
    v = [c.valor for c in corridas if not c.fallo]
    if not v:
        return math.nan, math.nan
    return mean(v), stdev(v) if len(v) > 1 else 0.0


def competir(tareas: list[str], piezas: list[str], cambios: dict | None = None,
             registro: Registro | None = None, aviso=print) -> list[Fila]:
    """``cambios``: ajustes sobre la configuración de cada tarea (p. ej. {"semillas": (0, 1)})."""
    filas: list[Fila] = []
    for nombre in tareas:
        t = cargar_tarea(nombre)
        arqs = ["densa", *[a for a in NATURALES.get(t.dato, ()) if a != "densa"]]
        for arq in arqs:
            def config(pieza: str, arq=arq) -> Config:
                c = EVALUACIONES.get(f"{pieza}/{nombre}", Config())
                return replace(c, **(cambios or {}), arquitectura=arq)

            base = correr("estandar", nombre, config=config("estandar"), registro=registro)
            m, s = _resumen(base)
            filas.append(Fila(nombre, arq, "estandar", base[0].metrica, m, s, base[0].parametros,
                              mean(c.segundos for c in base), None))
            aviso(f"{nombre:14} {_nombre('estandar', arq):28} {m:.4f}")
            for pieza in piezas:
                corr = correr(pieza, nombre, config=config(pieza), registro=registro)
                cmp = comparar(corr, base)
                m, s = _resumen(corr)
                filas.append(Fila(nombre, arq, pieza, corr[0].metrica, m, s, corr[0].parametros,
                                  mean(c.segundos for c in corr), cmp))
                aviso(f"{nombre:14} {_nombre(pieza, arq):28} {m:.4f}  {cmp.veredicto}")
    comparaciones = [f.comparacion for f in filas if f.comparacion is not None]
    if len(comparaciones) > 1:
        ajustar_multiples(comparaciones)
    return filas


_ICONO = {"mejor": "🟢", "igual": "⚪", "peor": "🔴", "falla": "💥"}


def _fmt(metrica: str, media: float, std: float | None = None) -> str:
    if math.isnan(media):
        return "—"
    if metrica == "exactitud":
        return f"{media * 100:.1f} %" + (f" ± {std * 100:.1f}" if std is not None else "")
    return f"{media:.4f}" + (f" ± {std:.4f}" if std is not None else "")


def informe_competencia(filas: list[Fila], semillas: str) -> str:
    lineas = [
        "# Competencia: tus piezas contra la tecnología normal",
        "",
        f"Media ± desviación en **datos de prueba**, con {semillas} semillas. Exactitud: más es mejor. "
        "MSE (error, en unidades normalizadas): menos es mejor.",
        "",
        "- **vs referencia**: cuánto mejor (+) o peor (−) que la mejor arquitectura normal de esa tarea "
        "(por ejemplo, la CNN normal en imágenes).",
        "- **¿Mejora su arquitectura?**: compara la pieza con la misma arquitectura usando ReLU. Usa una prueba t "
        "pareada con corrección por comparaciones múltiples. 🟢 mejora, ⚪ igual, 🔴 empeora.",
        "",
    ]
    for tarea in dict.fromkeys(f.tarea for f in filas):
        grupo = [f for f in filas if f.tarea == tarea]
        t = cargar_tarea(tarea)
        mayor = grupo[0].metrica == "exactitud"
        normales = [f for f in grupo if f.pieza == "estandar" and f.arquitectura != "densa"] or \
                   [f for f in grupo if f.pieza == "estandar"]
        ref = max(normales, key=lambda f: f.media if mayor else -f.media)
        orden = sorted(grupo, key=lambda f: (math.isnan(f.media), -f.media if mayor else f.media))
        lineas += [f"## {tarea}", "", f"{t.descripcion}. Referencia: **{_nombre(ref.pieza, ref.arquitectura)}**.", "",
                   "| # | Modelo | Prueba | vs referencia | ¿Mejora su arquitectura? | Parámetros | Seg./corrida |",
                   "|---|---|---|---|---|---|---|"]
        for i, f in enumerate(orden, 1):
            if math.isnan(f.media) or math.isnan(ref.media) or ref.media == 0:
                vs = "—"
            else:
                d = (f.media - ref.media) / abs(ref.media) * 100 * (1 if mayor else -1)
                if f is ref:
                    vs = "referencia"
                elif not mayor and f.media > 2 * ref.media:  # error mucho mayor: mejor decir «× veces»
                    vs = f"{f.media / ref.media:.1f}× más error"
                else:
                    vs = f"{d:+.1f} %"
            ver = "—" if f.comparacion is None else _ICONO[f.comparacion.veredicto] + " " + f.comparacion.veredicto
            lineas.append(f"| {i} | {_nombre(f.pieza, f.arquitectura)} | {_fmt(f.metrica, f.media, f.std)} | {vs} | "
                          f"{ver} | {f.parametros:,} | {f.segundos:.1f} |")
        if t.ingenuo is not None:
            gana = sum(1 for f in grupo if not math.isnan(f.media) and f.media < t.ingenuo)
            lineas += ["", f"Predicción ingenua (repetir el último valor): **{t.ingenuo:.4f}**. "
                           f"La superan {gana} de {len(grupo)} modelos."]
        lineas.append("")
    mejoras = [f for f in filas if f.comparacion is not None and f.comparacion.veredicto == "mejor"]
    lineas += ["## Resumen", ""]
    if mejoras:
        lineas += ["Piezas que mejoran de forma clara la arquitectura donde van:", ""]
        lineas += [f"- **{f.pieza}** en {_nombre('estandar', f.arquitectura).replace(' normal', '')} ({f.tarea})" for f in mejoras]
    else:
        lineas.append("Ninguna pieza mejora de forma clara su arquitectura en estas tareas.")
    lineas += ["", "Una 🟢 es una pista para invertir más tiempo (más semillas, más datos), no una prueba definitiva."]
    return "\n".join(lineas) + "\n"


__all__ = ["Fila", "competir", "informe_competencia", "ARQUITECTURAS"]
