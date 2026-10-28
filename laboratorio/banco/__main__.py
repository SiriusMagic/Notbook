"""Uso:
    python -m laboratorio.banco --pieza ida --tarea circulos
    python -m laboratorio.banco --todo --semillas 5           # tareas pequeñas (~5 min)
    python -m laboratorio.banco --reales --guardar modelos    # MNIST, CIFAR-10, reseñas… (descarga de Hugging Face)
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .datos_reales import FUENTES
from .evaluar import EVALUACIONES, Config, Registro, ajustar_multiples, comparar, correr, informe
from .piezas import PIEZAS
from .tareas import TAREAS


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m laboratorio.banco", description="Compara piezas de investigación con la red estándar.")
    ap.add_argument("--pieza", action="append", choices=[p for p in PIEZAS if p != "estandar"], help="se puede repetir")
    ap.add_argument("--tarea", action="append", choices=[*TAREAS, *FUENTES], help="se puede repetir")
    ap.add_argument("--todo", action="store_true", help="todas las piezas en las tareas pequeñas")
    ap.add_argument("--reales", action="store_true", help="todas las piezas en las tareas reales de Hugging Face")
    ap.add_argument("--semillas", type=int, default=None, help="por defecto 5 (pequeñas) o 3 (reales)")
    ap.add_argument("--epocas", type=int, default=None, help="máximo de épocas")
    ap.add_argument("--lote", type=int, default=None, help="tamaño de lote")
    ap.add_argument("--lr", type=float, default=None, help="tasa de aprendizaje")
    ap.add_argument("--paciencia", type=int, default=None, help="parada temprana: épocas sin mejorar (0 = desactivada)")
    ap.add_argument("--guardar", default=None, help="carpeta para guardar el mejor modelo de cada corrida")
    ap.add_argument("--salida", default="resultados")
    a = ap.parse_args(argv)

    piezas = [p for p in PIEZAS if p != "estandar"] if a.todo or a.reales or not a.pieza else a.pieza
    if a.tarea:
        tareas = a.tarea
    else:
        tareas = ([*TAREAS] if a.todo or not a.reales else []) + ([*FUENTES] if a.reales else [])
    registro = Registro(a.salida)

    def config_para(pieza: str, tarea: str) -> Config:
        c = EVALUACIONES.get(f"{pieza}/{tarea}", Config())
        cambios = {k: v for k, v in {"epocas": a.epocas, "lote": a.lote, "lr": a.lr, "guardar": a.guardar}.items() if v is not None}
        if a.paciencia is not None:
            cambios["paciencia"] = a.paciencia or None
        if a.semillas:
            cambios["semillas"] = tuple(range(a.semillas))
        return replace(c, **cambios)

    comparaciones = []
    semillas_usadas = set()
    for tarea in tareas:
        base = correr("estandar", tarea, config=config_para("estandar", tarea), registro=registro)
        for pieza in piezas:
            config = config_para(pieza, tarea)
            semillas_usadas.add(len(config.semillas))
            c = comparar(correr(pieza, tarea, config=config, registro=registro), base)
            comparaciones.append(c)
            print(f"{tarea:17} {pieza:22} {c.veredicto:6} pieza={c.pieza_media:.4f} base={c.base_media:.4f} p={c.p_valor:.3f}", flush=True)

    if len(comparaciones) > 1:
        ajustar_multiples(comparaciones)
        cambiaron = [c for c in comparaciones if c.p_valor < 0.05 and c.veredicto == "igual"]
        if cambiaron:
            print(f"Tras corregir por {len(comparaciones)} comparaciones, {len(cambiaron)} diferencias dejan de ser significativas.")
    texto = informe(comparaciones, "/".join(str(n) for n in sorted(semillas_usadas)))
    ruta = Path(a.salida) / "informe.md"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")
    print(f"\nInforme: {ruta}\nCorridas: {registro.ruta}")


if __name__ == "__main__":
    main()
