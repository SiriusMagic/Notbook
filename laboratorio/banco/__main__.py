"""Uso:
    python -m laboratorio.banco --pieza ida --tarea circulos
    python -m laboratorio.banco --todo --semillas 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluar import Config, Registro, comparar, correr, informe
from .piezas import PIEZAS
from .tareas import TAREAS


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m laboratorio.banco", description="Compara piezas de investigación con la red estándar.")
    ap.add_argument("--pieza", action="append", choices=[p for p in PIEZAS if p != "estandar"], help="se puede repetir")
    ap.add_argument("--tarea", action="append", choices=list(TAREAS), help="se puede repetir")
    ap.add_argument("--todo", action="store_true", help="todas las piezas en todas las tareas")
    ap.add_argument("--semillas", type=int, default=5)
    ap.add_argument("--epocas", type=int, default=None, help="sobrescribe las épocas de cada tarea")
    ap.add_argument("--salida", default="resultados")
    a = ap.parse_args(argv)

    piezas = [p for p in PIEZAS if p != "estandar"] if a.todo or not a.pieza else a.pieza
    tareas = list(TAREAS) if a.todo or not a.tarea else a.tarea
    config = Config(epocas=a.epocas, semillas=tuple(range(a.semillas)))
    registro = Registro(a.salida)

    comparaciones = []
    for tarea in tareas:
        base = correr("estandar", tarea, config=config, registro=registro)
        for pieza in piezas:
            c = comparar(correr(pieza, tarea, config=config, registro=registro), base)
            comparaciones.append(c)
            print(f"{tarea:17} {pieza:22} {c.veredicto:6} pieza={c.pieza_media:.4f} base={c.base_media:.4f} p={c.p_valor:.3f}", flush=True)

    texto = informe(comparaciones, a.semillas)
    ruta = Path(a.salida) / "informe.md"
    ruta.write_text(texto, encoding="utf-8")
    print(f"\nInforme: {ruta}\nCorridas: {registro.ruta}")


if __name__ == "__main__":
    main()
