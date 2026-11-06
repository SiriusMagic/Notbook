"""Uso:
    python -m laboratorio.rnra --estudio E1 --estudio E2 --semillas 10 --salida laboratorio/banco/informes/15-reorganizacion-autonoma.md
    python -m laboratorio.rnra --todo --ablaciones --semillas 10
    python -m laboratorio.rnra --estudio E2 --rapido            # 3 semillas y pasos cortos, para probar
E5 (Fashion-MNIST) baja datos de Hugging Face; E4 completo (Split/Permuted MNIST) está en el cuaderno 15.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from . import experimentos as ex
from .controlador import ABLACIONES, COMPARADORES, Config

BASE = [*COMPARADORES, "rnra", "rnra_aleatorio"]
TITULOS = {"E1": "E1 · Crecimiento necesario", "E2": "E2 · Complejidad localizada", "E3": "E3 · Cambios de distribución",
           "E4": "E4 · Aprendizaje continuo", "E5": "E5 · Generalización"}


def _veredicto(c: dict) -> str:
    marcas = {k: ("✅" if c[k]["ok"] else "❌") for k in ("precision", "parametros", "eficiencia", "estabilidad") if k in c}
    return " · ".join(f"{k} {v}" for k, v in marcas.items())


def main(argv=None) -> str:
    ap = argparse.ArgumentParser(prog="python -m laboratorio.rnra", description="Cuaderno 15: RNRA-v0 contra sus comparadores.")
    ap.add_argument("--estudio", action="append", choices=list(TITULOS), help="se puede repetir")
    ap.add_argument("--todo", action="store_true", help="E1-E4 (E5 baja datos)")
    ap.add_argument("--semillas", type=int, default=10)
    ap.add_argument("--pasos", type=int, default=800)
    ap.add_argument("--ablaciones", action="store_true", help="agrega las 6 ablaciones de RNRA")
    ap.add_argument("--rapido", action="store_true", help="3 semillas y 300 pasos (solo para probar)")
    ap.add_argument("--salida", help="archivo .md donde guardar el informe")
    a = ap.parse_args(argv)
    estudios = a.estudio or (["E1", "E2", "E3", "E4"] if a.todo else ["E2"])
    semillas = range(3 if a.rapido else a.semillas)
    cfg = replace(Config(), pasos=300 if a.rapido else a.pasos)
    metodos = BASE + ([m for m in ABLACIONES if m not in BASE] if a.ablaciones else [])
    partes = [f"Semillas: {len(semillas)} · pasos por tarea: {cfg.pasos} · presupuesto = conexiones del MLP denso de {cfg.ancho_pequeno} neuronas por capa.\n"]
    from ..banco.tareas import cargar_tarea

    for est in estudios:
        partes.append(f"## {TITULOS[est]}\n")
        if est in ("E1", "E2", "E5"):
            for nombre in ex.ESTUDIOS[est]:
                t = cargar_tarea(nombre)
                res = ex.correr(t, metodos, semillas, cfg)
                c = ex.criterios(res, mayor_mejor=t.tipo == "clasificacion")
                partes += [f"### {nombre}\n", ex.tabla(res), "", ex.tabla_comparacion(res), "",
                           f"Criterios (contra `{c['mejor_comparador']}`, el mejor con el mismo presupuesto): {_veredicto(c)} · "
                           f"diferencia de precisión {c['dif_precision'] * 100:+.2f} pp.\n"]
        else:
            secs = [ex.lunas_rotadas()] if est == "E3" else [ex.digitos_permutados(), ex.digitos_partidos()]
            for sec in secs:
                res = ex.correr_continuo(sec, metodos, semillas, cfg)
                partes += [f"### {sec.nombre} ({len(sec.tareas)} tareas)\n", "| Método | Exactitud final media | Olvido |", "|---|---|---|"]
                for m, rs in res.items():
                    partes.append(f"| `{m}` | {ex._ms([r.metrica_prueba for r in rs], True)} | {ex._ms([r.perdida_prueba for r in rs], True)} |")
                partes += ["", ex.tabla_comparacion(res), ""]
    texto = "\n".join(partes)
    print(texto)
    if a.salida:
        with open(a.salida, "w", encoding="utf-8") as f:
            f.write(texto)
    return texto


if __name__ == "__main__":
    main()
