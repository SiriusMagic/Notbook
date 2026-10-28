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
from collections import Counter
from dataclasses import dataclass, field, replace
from statistics import mean, stdev

from .arquitecturas import ARQUITECTURAS, INVESTIGACION, NATURALES

INVESTIGACION_TODAS = {a for v in INVESTIGACION.values() for a in v}
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
    medidas: dict = field(default_factory=dict)  # promedio entre semillas (latencia, memoria, muertas…)
    cka: dict = field(default_factory=dict)  # parecido interno con la versión normal (misma semilla)
    investigacion: bool = False  # la arquitectura entera es una idea tuya (compite contra cada arquitectura normal)
    contra_normales: dict = field(default_factory=dict)  # tus arquitecturas: {arquitectura normal: Comparacion}


NOMBRES = {"densa": "Red densa", "cnn": "CNN", "texto_cnn": "TextCNN", "transformer": "Transformer",
           "serie_cnn": "CNN 1D", "lstm": "LSTM", "fasttext": "fastText",
           "atencion_binaria_rl": "Tu atención binaria con RL", "atencion_niveles_rl": "Tu atención por niveles con RL",
           "atencion_hebbiana": "Tu atención hebbiana", "cnn_densa_entrada": "Tu CNN con entrada densa"}


def _promedio_medidas(corridas: list[Corrida]) -> dict:
    claves = {k for c in corridas for k, v in c.medidas.items() if isinstance(v, (int, float))}
    return {k: mean(c.medidas[k] for c in corridas if k in c.medidas) for k in claves}


def _cka_promedio(modelos_a: dict, modelos_b: dict, emparejar=lambda s, semillas: s) -> dict:
    from .medicion import cka_entre

    por_capa: dict[str, list[float]] = {}
    semillas = sorted(modelos_a)
    for s in semillas:
        otra = emparejar(s, semillas)
        if otra is None or otra not in modelos_b:
            continue
        (ma, x), (mb, _) = modelos_a[s], modelos_b[otra]
        for capa, v in cka_entre(ma, mb, x).items():
            if not math.isnan(v):
                por_capa.setdefault(capa, []).append(v)
    return {k: round(mean(v), 3) for k, v in por_capa.items()}


def _nombre(pieza: str, arq: str) -> str:
    from .evaluar import _etiqueta

    base = NOMBRES.get(arq, arq)
    pieza = pieza if pieza == "estandar" else _etiqueta(pieza)
    if arq in INVESTIGACION_TODAS:
        return base if pieza == "estandar" else f"{pieza} en {base}"
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
        bases_tarea: dict = {}
        for arq in arqs:
            def config(pieza: str, arq=arq) -> Config:
                c = EVALUACIONES.get(f"{pieza}/{nombre}", Config())
                return replace(c, **(cambios or {}), arquitectura=arq)

            modelos_base: dict = {}
            base = correr("estandar", nombre, config=config("estandar"), registro=registro, modelos=modelos_base)
            m, s = _resumen(base)
            # Referencia de CKA: la red normal contra sí misma con otra semilla (cuánto varía solo por azar).
            siguiente = lambda sem, sems: sems[(sems.index(sem) + 1) % len(sems)] if len(sems) > 1 else None  # noqa: E731
            filas.append(Fila(nombre, arq, "estandar", base[0].metrica, m, s, base[0].parametros,
                              mean(c.segundos for c in base), None, _promedio_medidas(base),
                              _cka_promedio(modelos_base, modelos_base, siguiente)))
            aviso(f"{nombre:14} {_nombre('estandar', arq):28} {m:.4f}")
            for pieza in piezas:
                modelos: dict = {}
                corr = correr(pieza, nombre, config=config(pieza), registro=registro, modelos=modelos)
                cmp = comparar(corr, base)
                m, s = _resumen(corr)
                filas.append(Fila(nombre, arq, pieza, corr[0].metrica, m, s, corr[0].parametros,
                                  mean(c.segundos for c in corr), cmp, _promedio_medidas(corr),
                                  _cka_promedio(modelos, modelos_base)))
                del modelos
                aviso(f"{nombre:14} {_nombre(pieza, arq):28} {m:.4f}  {cmp.veredicto}")
            bases_tarea[arq] = base
        # Tus arquitecturas (p. ej. los agentes de atención): contra la mejor arquitectura normal de esta tarea.
        investigacion = [a for a in INVESTIGACION.get(t.dato, ()) if a not in arqs]
        if investigacion and bases_tarea:
            # Contra CADA arquitectura normal (no solo «la mejor»: elegirla mirando la prueba sesgaría el veredicto).
            normales = {a: c for a, c in bases_tarea.items() if a != "densa"} or bases_tarea
            for arq in investigacion:
                cfg = replace(EVALUACIONES.get(f"estandar/{nombre}", Config()), **(cambios or {}), arquitectura=arq)
                corr = correr("estandar", nombre, config=cfg, registro=registro)
                contra = {a: comparar(corr, b) for a, b in normales.items()}
                m, s = _resumen(corr)
                filas.append(Fila(nombre, arq, "estandar", corr[0].metrica, m, s, corr[0].parametros,
                                  mean(c.segundos for c in corr), None, _promedio_medidas(corr), {}, True, contra))
                resumen = ", ".join(f"{NOMBRES.get(a, a)}: {c.veredicto}" for a, c in contra.items())
                aviso(f"{nombre:14} {_nombre('estandar', arq):28} {m:.4f}  ({resumen})")
    comparaciones = [f.comparacion for f in filas if f.comparacion is not None]
    comparaciones += [c for f in filas for c in f.contra_normales.values()]
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


def _fila_hardware(f: Fila) -> str:
    m = f.medidas
    if not m:
        return f"| {_nombre(f.pieza, f.arquitectura)} | — | — | — | — | — | — | — |"
    lat = f"{m['latencia_ms']:.3f} ({m['latencia_p95_ms']:.3f})" if "latencia_ms" in m else "—"
    vel = f"{m['tokens_por_s']:,.0f} tokens/s" if "tokens_por_s" in m else (
        f"{m['ejemplos_por_s']:,.0f} ej./s" if "ejemplos_por_s" in m else "—")
    pesos = f"{m['huella_mb']:.2f}" if "huella_mb" in m else "—"
    vram = f"{m['vram_pico_mb']:.0f}" if "vram_pico_mb" in m else "— (CPU)"
    muertas = [m[k] for k in ("capa1_muertas", "capa2_muertas") if k in m]
    muertas_txt = f"{mean(muertas) * 100:.0f} %" if muertas else "—"
    grads = [m[f"{c}_grad"] for c in ("extractor", "capa1", "capa2") if f"{c}_grad" in m]
    grad_txt = f"{grads[0]:.2g} / {grads[-1]:.2g}" if len(grads) > 1 else "—"
    if "fraccion_atendida" in m:
        cka_txt = f"lee el {m['fraccion_atendida'] * 100:.0f} % del texto"
    elif f.cka:
        valor = mean(f.cka.values())
        cka_txt = f"{valor:.2f} (normal vs. normal por azar)" if f.pieza == "estandar" else f"{valor:.2f}"
    else:
        cka_txt = "—"
    return f"| {_nombre(f.pieza, f.arquitectura)} | {lat} | {vel} | {pesos} | {vram} | {muertas_txt} | {grad_txt} | {cka_txt} |"


_TIPOS = {"imagen": "Imágenes", "texto": "Texto", "serie": "Series", "tabla": "Tablas"}


def resumen_por_tipo(filas: list[Fila]) -> dict[str, dict[str, Counter]]:
    """Para cada pieza y tipo de dato: cuántas veces mejora, iguala o empeora su arquitectura."""
    mapa: dict[str, dict[str, Counter]] = {}
    for f in filas:
        comps = list(f.contra_normales.values()) or ([f.comparacion] if f.comparacion is not None else [])
        if not comps:
            continue
        tipo = cargar_tarea(f.tarea).dato
        clave = NOMBRES.get(f.arquitectura, f.arquitectura) if f.investigacion else f.pieza
        for c in comps:
            mapa.setdefault(clave, {}).setdefault(tipo, Counter())[c.veredicto] += 1
    return mapa


def _celda(c: Counter | None) -> str:
    if not c:
        return "—"
    partes = [f"{_ICONO[v]} {c[v]}" for v in ("mejor", "igual", "peor", "falla") if c[v]]
    if c["mejor"] and not (c["peor"] or c["falla"]):
        etiqueta = "**prometedora**"
    elif c["peor"] + c["falla"] > c["mejor"]:
        etiqueta = "no sirve aquí"
    else:
        etiqueta = "neutral"
    return " · ".join(partes) + f" → {etiqueta}"


def _mapa_por_tipo(filas: list[Fila]) -> list[str]:
    mapa = resumen_por_tipo(filas)
    if not mapa:
        return []
    tipos = [t for t in _TIPOS if any(t in m for m in mapa.values())]
    lineas = ["## Mapa: ¿en qué tipo de dato sirve cada pieza?", "",
              "Cuenta, en todas las tareas y arquitecturas de cada tipo, cuántas veces la pieza mejora (🟢), "
              "iguala (⚪) o empeora (🔴) la arquitectura donde va. Una pieza puede ser mala en texto y buena en imágenes.", "",
              "| Pieza | " + " | ".join(_TIPOS[t] for t in tipos) + " |", "|---|" + "---|" * len(tipos)]
    for pieza, por_tipo in mapa.items():
        lineas.append(f"| {pieza} | " + " | ".join(_celda(por_tipo.get(t)) for t in tipos) + " |")
    return lineas + [""]


def informe_competencia(filas: list[Fila], semillas: str) -> str:
    lineas = [
        "# Competencia: tus piezas contra la tecnología normal",
        "",
        f"Media ± desviación en **datos de prueba**, con {semillas} semillas. Exactitud: más es mejor. "
        "MSE (error, en unidades normalizadas): menos es mejor.",
        "",
        "- **vs referencia**: cuánto mejor (+) o peor (−) que la mejor arquitectura normal de esa tarea "
        "(por ejemplo, la CNN normal en imágenes).",
        "- **Eficiencia**: exactitud (en %) por millón de parámetros. Una red pequeña que empata con una grande gana aquí.",
        "- **Hardware**: latencia para un ejemplo (mediana y p95), velocidad en lotes de 256, peso de la red, "
        "y en GPU el pico de memoria al entrenar.",
        "- **Por dentro**: % de neuronas muertas (siempre 0) en las capas de la pieza; gradiente de la primera y "
        "la última capa (si la primera es mucho menor, el aprendizaje no llega al inicio); CKA = cuánto se parecen "
        "las representaciones internas a las de la red normal (1 = iguales). Compáralo con el CKA de la red normal "
        "contra sí misma: eso es lo que cambia solo por azar.",
        "- **Tus arquitecturas** (por ejemplo «Tu atención binaria con RL») se comparan, en esa misma columna, "
        "contra **cada** arquitectura normal de la tarea (todas entran en la corrección por comparaciones múltiples).",
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
        ref = max(normales, key=lambda f: f.media if mayor else -f.media) if normales else None
        orden = sorted(grupo, key=lambda f: (math.isnan(f.media), -f.media if mayor else f.media))
        nombre_ref = _nombre(ref.pieza, ref.arquitectura) if ref else "—"
        lineas += [f"## {tarea}", "", f"{t.descripcion}. Referencia: **{nombre_ref}**.", "",
                   "| # | Modelo | Prueba | vs referencia | ¿Mejora su arquitectura? | Parámetros | Eficiencia | Seg. entrenar |",
                   "|---|---|---|---|---|---|---|---|"]
        for i, f in enumerate(orden, 1):
            if ref is None or math.isnan(f.media) or math.isnan(ref.media) or ref.media == 0:
                vs = "—"
            else:
                d = (f.media - ref.media) / abs(ref.media) * 100 * (1 if mayor else -1)
                if f is ref:
                    vs = "referencia"
                elif not mayor and f.media > 2 * ref.media:  # error mucho mayor: mejor decir «× veces»
                    vs = f"{f.media / ref.media:.1f}× más error"
                else:
                    vs = f"{d:+.1f} %"
            if f.contra_normales:
                ver = " · ".join(f"vs {NOMBRES.get(a, a)} {_ICONO[c.veredicto]}" for a, c in f.contra_normales.items())
            else:
                ver = "—" if f.comparacion is None else _ICONO[f.comparacion.veredicto] + " " + f.comparacion.veredicto
            efi = f"{f.media * 100 / (f.parametros / 1e6):,.0f}" if mayor and not math.isnan(f.media) else "—"
            lineas.append(f"| {i} | {_nombre(f.pieza, f.arquitectura)} | {_fmt(f.metrica, f.media, f.std)} | {vs} | "
                          f"{ver} | {f.parametros:,} | {efi} | {f.segundos:.1f} |")
        lineas += ["", "**Hardware y por dentro**", "",
                   "| Modelo | Latencia (ms, p95) | Velocidad | Pesos (MB) | VRAM pico (MB) | Neuronas muertas | Gradiente 1.ª / última capa | Parecido interno (CKA) |",
                   "|---|---|---|---|---|---|---|---|"]
        for f in orden:
            lineas.append(_fila_hardware(f))
        if t.ingenuo is not None:
            gana = sum(1 for f in grupo if not math.isnan(f.media) and f.media < t.ingenuo)
            lineas += ["", f"Predicción ingenua (repetir el último valor): **{t.ingenuo:.4f}**. "
                           f"La superan {gana} de {len(grupo)} modelos."]
        lineas.append("")
    lineas += _mapa_por_tipo(filas)
    mejoras = [f for f in filas if f.comparacion is not None and f.comparacion.veredicto == "mejor"]
    lineas += ["## Resumen", ""]
    if mejoras:
        lineas += ["Piezas que mejoran de forma clara la arquitectura donde van:", ""]
        lineas += [f"- **{f.pieza}** en {_nombre('estandar', f.arquitectura).replace(' normal', '')} ({f.tarea})" for f in mejoras]
    else:
        lineas.append("Ninguna pieza mejora de forma clara su arquitectura en estas tareas.")
    for f in filas:
        ganadas = [NOMBRES.get(a, a) for a, c in f.contra_normales.items() if c.veredicto == "mejor"]
        if ganadas:
            lineas.append(f"- **{_nombre('estandar', f.arquitectura)}** le gana a {', '.join(ganadas)} ({f.tarea})")
    lineas += ["", "Una 🟢 es una pista para invertir más tiempo (más semillas, más datos), no una prueba definitiva."]
    return "\n".join(lineas) + "\n"


__all__ = ["Fila", "competir", "informe_competencia", "resumen_por_tipo", "ARQUITECTURAS"]
