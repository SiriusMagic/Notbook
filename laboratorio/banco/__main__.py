"""Uso:
    python -m laboratorio.banco --pieza ida --tarea circulos
    python -m laboratorio.banco --todo --semillas 5            # tareas pequeñas (~5 min en CPU)
    python -m laboratorio.banco --reales --guardar modelos     # tareas reales en la red densa
    python -m laboratorio.banco --competencia --tipo imagen    # tus piezas contra CNN normal (mejor con GPU)
    python -m laboratorio.banco --competencia --tipo texto --tipo serie --rapido
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from .arquitecturas import ARQUITECTURAS
from .datos_reales import FUENTES
from .evaluar import EVALUACIONES, Config, Registro, ajustar_multiples, comparar, correr, informe
from .piezas import PIEZAS
from .tareas import TAREAS


def _tipo_dato(tarea: str) -> str:
    """Tipo de dato de una tarea sin descargar nada (las pequeñas se generan al instante)."""
    if tarea in FUENTES:
        return FUENTES[tarea].tipo
    from .tareas import cargar_tarea

    return cargar_tarea(tarea).dato


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m laboratorio.banco", description="Compara piezas de investigación con la red estándar.")
    ap.add_argument("--pieza", action="append", choices=[p for p in PIEZAS if p != "estandar"], help="se puede repetir")
    ap.add_argument("--grupo", choices=["tuyas", "industria"],
                    help="solo tus piezas o solo las de la industria (GELU, SiLU, Mish, ELU, LeakyReLU, residual)")
    ap.add_argument("--tarea", action="append", choices=[*TAREAS, *FUENTES], help="se puede repetir")
    ap.add_argument("--todo", action="store_true", help="todas las piezas en las tareas pequeñas")
    ap.add_argument("--reales", action="store_true", help="todas las piezas en las tareas reales de Hugging Face")
    ap.add_argument("--competencia", action="store_true",
                    help="tus piezas contra la tecnología normal (CNN, TextCNN, Transformer, CNN 1D, LSTM)")
    ap.add_argument("--tokenizadores", action="store_true",
                    help="cuaderno 01: tus sílabas contra BPE, Unigram y GPT-4o (con --tarea de texto; por defecto resenas_es)")
    ap.add_argument("--segmentadores", action="store_true",
                    help="cuaderno 01: segmentadores de sílabas (reglas, cuaderno, pyphen, BiLSTM, Transformer)")
    ap.add_argument("--clasicos", action="store_true",
                    help="cuaderno 03: regresión logística, KNN, SVM, árbol, Naive Bayes, Random Forest y "
                         "Gradient Boosting contra la red estándar (por defecto, las tareas pequeñas)")
    ap.add_argument("--correccion", action="store_true",
                    help="cuaderno 01: corrección ortográfica, tu GNN contra BiLSTM, Transformer y diccionario")
    ap.add_argument("--siguiente_palabra", action="store_true",
                    help="cuaderno 01: agente PPO que predice la siguiente palabra contra la red supervisada y bigramas")
    ap.add_argument("--difusion", action="store_true",
                    help="cuaderno 01: difusión de texto tal cual, con máscaras (MDLM) y Transformer autorregresivo")
    ap.add_argument("--pasos", type=int, default=None, help="con --difusion: pasos de entrenamiento (3000 por defecto)")
    ap.add_argument("--video", action="store_true", help="cuaderno 04: tripletas y FrameInterpolator contra U-Net")
    ap.add_argument("--refuerzo", action="store_true",
                    help="cuaderno 06: entornos de refuerzo (código, cubos, ReroutingNetwork); necesita stable-baselines3")
    ap.add_argument("--generativos07", action="store_true", help="cuaderno 07: D-ReG y HCMG contra la industria")
    ap.add_argument("--tipo", action="append", choices=["imagen", "texto", "serie"], help="con --competencia: qué datos (se puede repetir)")
    ap.add_argument("--rapido", action="store_true", help="con --competencia: una tarea por tipo, 2 semillas y menos datos")
    ap.add_argument("--arquitectura", default="densa", choices=list(ARQUITECTURAS), help="arquitectura (sin --competencia)")
    ap.add_argument("--semillas", type=int, default=None, help="por defecto 5 (pequeñas) o 3 (reales)")
    ap.add_argument("--epocas", type=int, default=None, help="máximo de épocas")
    ap.add_argument("--lote", type=int, default=None, help="tamaño de lote")
    ap.add_argument("--lr", type=float, default=None, help="tasa de aprendizaje")
    ap.add_argument("--paciencia", type=int, default=None, help="parada temprana: épocas sin mejorar (0 = desactivada)")
    ap.add_argument("--entreno", type=int, default=None, help="tareas reales: cuántos ejemplos para entrenar")
    ap.add_argument("--prueba", type=int, default=None, help="tareas reales: cuántos ejemplos de prueba")
    ap.add_argument("--dispositivo", default="auto", help="auto (GPU si hay), cpu o cuda")
    ap.add_argument("--guardar", default=None, help="carpeta para guardar el mejor modelo de cada corrida")
    ap.add_argument("--salida", default="resultados")
    a = ap.parse_args(argv)

    if a.rapido:
        a.semillas = a.semillas or 2
        a.entreno = a.entreno or 3000
        a.prueba = a.prueba or 1000
    if a.entreno:
        os.environ["LABORATORIO_ENTRENO"] = str(a.entreno)
    if a.prueba:
        os.environ["LABORATORIO_PRUEBA"] = str(a.prueba)

    cambios = {k: v for k, v in {"epocas": a.epocas, "lote": a.lote, "lr": a.lr, "guardar": a.guardar}.items() if v is not None}
    cambios["dispositivo"] = a.dispositivo
    if a.paciencia is not None:
        cambios["paciencia"] = a.paciencia or None
    if a.semillas:
        cambios["semillas"] = tuple(range(a.semillas))

    registro = Registro(a.salida)
    piezas = a.pieza or [p for p in PIEZAS if p != "estandar"]
    if a.grupo:
        piezas = [p for p in piezas if PIEZAS[p].industria == (a.grupo == "industria")]
    Path(a.salida).mkdir(parents=True, exist_ok=True)

    if a.tokenizadores:
        from .tokenizadores import competir_tokenizadores, informe_tokenizadores

        r = competir_tokenizadores((a.tarea or ["resenas_es"])[0], cambios={k: v for k, v in cambios.items() if k != "guardar"})
        ruta = Path(a.salida) / "tokenizadores.md"
        ruta.write_text(informe_tokenizadores(r), encoding="utf-8")
        print(f"\nInforme: {ruta}")
        return

    if a.segmentadores:
        from .datos_reales import textos_reales
        from .segmentadores import competir_segmentadores, informe_segmentadores, palabras_de

        textos, _, _, _ = textos_reales("resenas_es")
        r = competir_segmentadores(palabras_de(textos), semillas=tuple(range(a.semillas or 3)), dispositivo=a.dispositivo)
        ruta = Path(a.salida) / "segmentadores.md"
        ruta.write_text(informe_segmentadores(r), encoding="utf-8")
        print(f"\nInforme: {ruta}")
        return

    if a.clasicos:
        from .clasicos import competir_clasicos, informe_clasicos

        semillas = tuple(range(a.semillas or 5))
        r = competir_clasicos(a.tarea or list(TAREAS), semillas)
        ruta = Path(a.salida) / "clasicos.md"
        ruta.write_text(informe_clasicos(r, semillas), encoding="utf-8")
        print(f"\nInforme: {ruta}")
        return

    if a.correccion:
        from .correccion import competir_correctores, frases_de, informe_correctores
        from .datos_reales import textos_reales

        te, _, tp, _ = textos_reales("resenas_es")
        r = competir_correctores(frases_de(te)[: a.entreno or 8000], frases_de(tp)[: a.prueba or 1000],
                                 semillas=tuple(range(a.semillas or 3)), dispositivo=a.dispositivo)
        ruta = Path(a.salida) / "correccion.md"
        ruta.write_text(informe_correctores(r), encoding="utf-8")
        print(f"\nInforme: {ruta}")
        return

    if a.video or a.refuerzo or a.generativos07:
        semillas = tuple(range(a.semillas or 3))
        if a.video:
            from . import video

            r = video.competir(a.entreno or 2000, a.prueba or 300, semillas, epocas=a.epocas or 20,
                               dispositivo=a.dispositivo)
            (Path(a.salida) / "video.md").write_text(video.informe(r), encoding="utf-8")
        if a.refuerzo:
            from . import refuerzo_06 as rf

            pasos = a.pasos or 100_000
            rer = rf.competir_rerouting(semillas)
            cod = rf.competir_codigo(semillas, pasos=pasos)
            cub = rf.competir_cubos(semillas, pasos=pasos, pasos_arreglo=pasos)
            (Path(a.salida) / "refuerzo.md").write_text(rf.informe(cod, cub, rer, pasos, (pasos, pasos), 50),
                                                        encoding="utf-8")
        if a.generativos07:
            from . import generativos_07 as g

            h = g.competir_hcmg(semillas=semillas, epocas=a.epocas or 20)
            d = g.competir_dreg(semillas=semillas, pasos=a.pasos or 1500)
            (Path(a.salida) / "generativos07.md").write_text(g.informe(d, h), encoding="utf-8")
        print(f"\nInformes en: {a.salida}")
        return

    if a.siguiente_palabra or a.difusion:
        from . import difusion_texto, siguiente_palabra

        semillas = tuple(range(a.semillas or 3))
        if a.siguiente_palabra:
            r = siguiente_palabra.competir(a.entreno or 5000, a.prueba or 1000, semillas, epocas=a.epocas or 30)
            ruta = Path(a.salida) / "siguiente_palabra.md"
            ruta.write_text(siguiente_palabra.informe(r), encoding="utf-8")
            print(f"\nInforme: {ruta}")
        if a.difusion:
            r = difusion_texto.competir(a.entreno or 10000, a.prueba or 1000, semillas, pasos=a.pasos or 3000,
                                        dispositivo=a.dispositivo)
            ruta = Path(a.salida) / "difusion.md"
            ruta.write_text(difusion_texto.informe(r), encoding="utf-8")
            print(f"\nInforme: {ruta}")
        return

    if a.competencia:
        from .competencia import competir, informe_competencia

        tipos = a.tipo or ["imagen", "texto", "serie"]
        if a.tarea:
            tareas = a.tarea
        else:
            tareas = [n for n, f in FUENTES.items() if f.tipo in tipos]
            if a.rapido:  # una por tipo
                tareas = [next(n for n, f in FUENTES.items() if f.tipo == t) for t in tipos]
        filas = competir(tareas, piezas, cambios, registro)
        ruta = Path(a.salida) / "competencia.md"
        ruta.write_text(informe_competencia(filas, str(a.semillas or 3)), encoding="utf-8")
        print(f"\nInforme: {ruta}\nCorridas: {registro.ruta}")
        return

    if a.tarea:
        tareas = a.tarea
    else:
        tareas = ([*TAREAS] if a.todo or not a.reales else []) + ([*FUENTES] if a.reales else [])
    # Solo las tareas cuyo tipo de dato sirve para la arquitectura elegida (una CNN no sirve para texto).
    datos = ARQUITECTURAS[a.arquitectura].datos
    validas = [t for t in tareas if _tipo_dato(t) in datos]
    if a.tarea and len(validas) < len(tareas):
        malas = ", ".join(t for t in tareas if t not in validas)
        ap.error(f"la arquitectura «{a.arquitectura}» no sirve para: {malas} (acepta datos de tipo {', '.join(datos)})")
    if not validas:
        ap.error(f"ninguna tarea elegida sirve para la arquitectura «{a.arquitectura}»")
    if len(validas) < len(tareas):
        print(f"Con «{a.arquitectura}» se omiten {len(tareas) - len(validas)} tareas que no son de tipo {', '.join(datos)}.")
    tareas = validas

    def config_para(pieza: str, tarea: str) -> Config:
        return replace(EVALUACIONES.get(f"{pieza}/{tarea}", Config()), **cambios, arquitectura=a.arquitectura)

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
    ruta.write_text(texto, encoding="utf-8")
    print(f"\nInforme: {ruta}\nCorridas: {registro.ruta}")


if __name__ == "__main__":
    main()
