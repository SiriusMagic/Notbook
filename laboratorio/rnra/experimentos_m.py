"""Cuaderno 16 · Experimentos de RNRA-M, evaluados aparte de RNRA-v0.

Reducir y expandir: para cada dispositivo, red fija por dispositivo · red grande cuantizada después ·
superred de ancho variable · RNRA-M (superred con núcleo cuantizado + memoria portátil). Se mide precisión, RAM
calculada, latencia y **capacidad recuperada** al volver al ancho grande (con memoria, sin memoria y con memoria comprimida).
Historia: un segundo experimento (aprender en el núcleo pequeño y llevarlo a la red grande, H3) no aportó nada medible frente a
repasar un recuerdo de la tarea vieja y se quitó; está en el commit ``fd6e8f3`` (``transferencia.py``).
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import torch

from . import memoria_multiescala as mem
from .adaptador_dispositivo import dispositivos, elegir
from .experimentos import _ms, comparar, holm
from .superred import (ANCHOS, SuperRed, bytes_modelo, cuantizar_modelo, entrenar, exactitud, latencia_ms)

METODOS_1 = ["fija_por_dispositivo", "fija_int8", "cuantizada_despues", "superred_ancho_variable", "rnra_m"]


def reducir_expandir(tarea, semilla: int, anchos=ANCHOS, pasos: int = 600) -> dict:
    """Todo el experimento 1 para una semilla. Devuelve filas por dispositivo y la tabla de recuperación."""
    torch.manual_seed(semilla)
    x, y, xp, yp = tarea.x_entreno, tarea.y_entreno, tarea.x_prueba, tarea.y_prueba
    sup = entrenar(SuperRed(tarea.entradas, tarea.salidas, anchos), x, y, pasos=pasos, semilla=semilla)
    grande = sup.anchos[-1]
    devs = dispositivos(sup)
    ancho_cuant = sup.anchos[1]  # la «red grande» que se cuantiza después: la de ancho medio, entrenada sola
    sola = entrenar(SuperRed(tarea.entradas, tarea.salidas, (ancho_cuant,)), x, y, pasos=pasos, semilla=semilla, anchos=(ancho_cuant,))
    filas = {}
    for d in devs:
        w, _ = elegir(sup, d.ram_bytes, bits=(32,)) or (None, None)  # filas en 32 bits: el ancho mayor que cabe sin cuantizar
        f = {}
        if w is not None:
            fija = entrenar(SuperRed(tarea.entradas, tarea.salidas, (w,)), x, y, pasos=pasos, semilla=semilla, anchos=(w,))
            f["fija_por_dispositivo"] = (exactitud(fija, xp, yp, w), bytes_modelo(sup, w, 32), latencia_ms(fija, xp[:256], w))
            q8 = cuantizar_modelo(fija, 8)  # el comparador justo de RNRA-M en RAM: la misma red fija, también a 8 bits
            f["fija_int8"] = (exactitud(q8, xp, yp, w), bytes_modelo(sup, w, 8), latencia_ms(q8, xp[:256], w))
            f["superred_ancho_variable"] = (exactitud(sup, xp, yp, w), bytes_modelo(sup, w, 32), latencia_ms(sup, xp[:256], w))
            p = mem.reducir(sup, w, bits_nucleo=8, bits_memoria=32)
            chico = mem.nucleo_como_modelo(p, sup)
            f["rnra_m"] = (exactitud(chico, xp, yp, w), p.bytes_nucleo(), latencia_ms(chico, xp[:256], w))
        # cuantizada después: la de ancho medio con los bits más altos que quepan (2, 4, 8)
        bits = next((bb for bb in (8, 4, 2) if bytes_modelo(sup, ancho_cuant, bb) <= d.ram_bytes), None)
        if ancho_cuant and bits is not None and bytes_modelo(sup, ancho_cuant, 32) > d.ram_bytes:
            q = cuantizar_modelo(sola, bits)
            f["cuantizada_despues"] = (exactitud(q, xp, yp, ancho_cuant), bytes_modelo(sup, ancho_cuant, bits), latencia_ms(q, xp[:256], ancho_cuant))
        elif bytes_modelo(sup, ancho_cuant, 32) <= d.ram_bytes:
            f["cuantizada_despues"] = (exactitud(sola, xp, yp, ancho_cuant), bytes_modelo(sup, ancho_cuant, 32), latencia_ms(sola, xp[:256], ancho_cuant))
        filas[d.nombre] = f
    # recuperación: ir al núcleo pequeño y volver al grande
    acc0 = exactitud(sup, xp, yp, grande)
    rec = {"original": acc0}
    for bm in (32, 8, 4):
        p = mem.reducir(sup, sup.anchos[0], bits_nucleo=8, bits_memoria=bm)
        e = mem.expandir(p, sup)
        rec[f"con_memoria_{bm}bits"] = exactitud(e, xp, yp, grande)
        rec[f"bytes_memoria_{bm}bits"] = p.bytes_memoria()
        with tempfile.TemporaryDirectory() as d:  # el viaje por archivo no cambia nada
            ruta = os.path.join(d, "paquete.npz")
            mem.guardar(p, ruta)
            e2 = mem.expandir(mem.cargar(ruta), sup)
            rec[f"archivo_{bm}bits"] = exactitud(e2, xp, yp, grande)
        if bm == 32:
            rec["error_max_peso"] = max(float((a - b).detach().abs().max()) for a, b in zip(e.parameters(), sup.parameters()))
    rec["sin_memoria"] = exactitud(mem.expandir(mem.reducir(sup, sup.anchos[0], 8, 32), sup, con_memoria=False), xp, yp, grande)
    rec["bytes_nucleo"] = mem.reducir(sup, sup.anchos[0], 8, 32).bytes_nucleo()
    rec["bytes_grande_fp32"] = bytes_modelo(sup, grande, 32)
    return {"filas": filas, "rec": rec, "ram": {d.nombre: d.ram_bytes for d in devs}}


def tablas_reducir_expandir(res: list[dict]) -> str:
    sal = []
    nombres = list(res[0]["ram"])
    for nombre in nombres:
        sal += [f"**{nombre}** (RAM disponible {res[0]['ram'][nombre] / 1024:.1f} KB)\n", "| Método | Exactitud | RAM de pesos (KB) | Latencia (ms) |", "|---|---|---|---|"]
        for m in METODOS_1:
            v = [r["filas"][nombre][m] for r in res if m in r["filas"][nombre]]
            if len(v) == len(res):
                sal.append(f"| `{m}` | {_ms([a for a, _, _ in v], True)} | {np.mean([b for _, b, _ in v]) / 1024:.1f} | {np.mean([c for _, _, c in v]):.2f} |")
            else:
                sal.append(f"| `{m}` | no cabe | — | — |")
        sal.append("")
        base = [m for m in METODOS_1 if m != "rnra_m" and all(m in r["filas"][nombre] for r in res)]
        if len(res) > 1 and "rnra_m" in res[0]["filas"][nombre]:
            cs = {m: comparar([r["filas"][nombre]["rnra_m"][0] for r in res], [r["filas"][nombre][m][0] for r in res]) for m in base}
            aj = holm([cs[m]["p"] for m in base])
            sal += ["| `rnra_m` contra | Diferencia | IC 95 % | p (Holm) |", "|---|---|---|---|"]
            sal += [f"| `{m}` | {cs[m]['dif'] * 100:+.2f} pp | [{cs[m]['ic'][0] * 100:+.2f}, {cs[m]['ic'][1] * 100:+.2f}] | {p:.3f} |" for m, p in zip(base, aj)]
            sal.append("")
    rec = {k: [r["rec"][k] for r in res] for k in res[0]["rec"]}
    sal += ["**Recuperar la capacidad al volver al ancho grande** (exactitud de prueba)\n", "| Situación | Exactitud | Memoria comprimida (KB) |", "|---|---|---|",
            f"| Red grande original | {_ms(rec['original'], True)} | — |",
            f"| Reducir y expandir **sin** memoria (se tiró lo recortado) | {_ms(rec['sin_memoria'], True)} | 0 |"]
    for bm in (32, 8, 4):
        sal.append(f"| Reducir y expandir con memoria de {bm} bits | {_ms(rec[f'con_memoria_{bm}bits'], True)} | {np.mean(rec[f'bytes_memoria_{bm}bits']) / 1024:.1f} |")
    sal += ["", f"Núcleo en RAM: {np.mean(rec['bytes_nucleo']) / 1024:.1f} KB (la red grande en 32 bits pesa {np.mean(rec['bytes_grande_fp32']) / 1024:.0f} KB). "
            f"Error máximo por peso al expandir con memoria de 32 bits: {np.max(rec['error_max_peso']):.1e}. "
            f"Pasar por un archivo `.npz` y volver da la misma exactitud ({_ms(rec['archivo_32bits'], True)})."]
    return "\n".join(sal)


def main(argv=None) -> str:
    ap = argparse.ArgumentParser(prog="python -m laboratorio.rnra.experimentos_m", description="Cuaderno 16: RNRA-M.")
    ap.add_argument("--semillas", type=int, default=10)
    ap.add_argument("--pasos", type=int, default=600)
    ap.add_argument("--tarea", action="append", default=None, help="tarea del banco para el experimento 1 (por defecto digitos y caotica_20d)")
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--salida")
    a = ap.parse_args(argv)
    from ..banco.tareas import cargar_tarea

    semillas = range(2 if a.rapido else a.semillas)
    pasos = 200 if a.rapido else a.pasos
    partes = [f"Semillas: {len(semillas)} · pasos: {pasos} · anchos {ANCHOS} · RAM calculada (no medida en un teléfono) y latencia en la CPU de esta máquina.\n", "## Reducir y expandir\n"]
    for nombre in a.tarea or ["digitos", "caotica_20d"]:
        t = cargar_tarea(nombre)
        partes += [f"### {nombre}\n", tablas_reducir_expandir([reducir_expandir(t, s, pasos=pasos) for s in semillas]), ""]
    texto = "\n".join(partes)
    print(texto)
    if a.salida:
        with open(a.salida, "w", encoding="utf-8") as f:
            f.write(texto)
    return texto


if __name__ == "__main__":
    main()
