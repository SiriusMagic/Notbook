# Cuaderno 15: redes de reorganización autónoma (RNRA-v0)

Código: **retirado** tras estos resultados (commit `fd6e8f3`: `git show fd6e8f3:laboratorio/rnra/controlador.py`). Hoy `laboratorio/rnra/` conserva solo el banco de comparación (MLP fijos, poda por magnitud, crecimiento programado, SET y RigL): `python -m laboratorio.rnra --todo --semillas 10`. `rnra_aleatorio` queda cubierto por SET.

## Conclusión

🔴 **RNRA-v0 no cumple ninguno de los umbrales fijados antes de medir.** Con 10 semillas, datos de prueba que ninguna decisión usa y corrección de Holm:

- **H1 (eficiencia estructural) se refuta.** En xor, círculos, lunas y lunas rotadas empata con RigL, SET y el MLP fijo pequeño (diferencias de menos de 1,2 puntos, ninguna significativa). En la tarea caótica 20D le gana al MLP fijo pequeño (+2,3 puntos, p = 0,001), pero RigL y SET hacen lo mismo (77,1 % contra 76,8 %): lo que ayuda es tener conexiones dispersas que se mueven, no la reorganización guiada por utilidad, redundancia y demanda. Y la **poda por magnitud** gradual desde una red de 64 neuronas le gana (78,5 % contra 76,8 %, p = 0,003), aunque gasta 2,7 veces más FLOPs.
- **Contra el control negativo no hay ventaja.** RNRA con todas las decisiones al azar (mismos conteos) da lo mismo en 6 de los 7 conjuntos medidos. La excepción favorece al azar: en dígitos permutados gana por 5,4 puntos (p = 0,038).
- **Las 6 ablaciones no cambian nada** (en la caótica todas quedan entre 76,5 % y 77,0 %): ni la utilidad, ni la redundancia, ni la demanda por región, ni el crecimiento, ni la poda aportan por separado.
- **H2 (aprendizaje continuo) se refuta.** En dígitos permutados todos olvidan mucho (61-74 %) y RNRA no olvida menos (69,9 % contra 64,0 % del MLP pequeño). En dígitos partidos tampoco (16,0 % contra 14,5 %).
- **H3 (especialización) no se observa.** La concentración de la actividad de cada neurona en pocas regiones es 0,022 en RNRA, en el control aleatorio y en RigL (5 semillas, caótica).
- **H4 (estabilidad) se apoya parcialmente.** Sin histéresis hay más neuronas que se apagan y se vuelven a encender (caótica 0,8 contra 0,1 por corrida; círculos 1,9 contra 0,7; lunas 0,8 contra 0,0) sin ganar precisión. No se probó con una prueba estadística.
- **La prueba corta «con y sin cambio» sí hace algo:** en la caótica acepta 4,8 de 7 cambios, contra 1,6 de 7 del control aleatorio. Pero el resultado final es el mismo: los cambios aceptados no mejoran la red más que dejar la red dispersa fija.
- **El crecimiento casi no se activa** (una neurona por corrida en la caótica): `rnra_sin_crecimiento` da igual que `rnra`. No se pudo probar la parte de «crecer solo con evidencia de falta de capacidad».
- **Coste:** la prueba de cada cambio cuesta un 27 % más de FLOPs que RigL en la caótica (526 contra 413 millones).

Es un resultado negativo útil: reasignar conexiones con estas estadísticas no supera a RigL ni al azar en tareas pequeñas.

### Límites de esta v0

- Los pesos se guardan con el tamaño máximo y se enmascaran: apagar conexiones no ahorra tiempo ni memoria reales. Los FLOPs son lógicos (conexiones activas, incluida la prueba de cada cambio).
- Las neuronas nuevas eligen sus conexiones al azar; la utilidad es de primer orden y no se contrastó con ablaciones reales en cada paso (solo al podar).
- Hiperparámetros sin afinar (una sola configuración para todas las tareas). Tareas pequeñas: puede ser que la dispersión dinámica baste y que la reorganización guiada solo importe a mayor escala.
- Faltan E5 (Fashion-MNIST) y Split/Permuted MNIST: necesitan bajar datos de Hugging Face (se corren desde el cuaderno 15 en Colab).
- La novedad frente a Net2Net, RigL y Structured RigL requiere una revisión bibliográfica más exhaustiva; con estos resultados no hay evidencia de que exista.

### Qué sigue

Antes de añadir más mecanismos: (1) probar con más escala (Fashion-MNIST, MLP más anchos) si la reorganización guiada separa de RigL; (2) arreglar el crecimiento para que se active con evidencia real; (3) comparar contra poda gradual desde una red grande con el mismo cómputo, que es el comparador más fuerte que apareció.

## Resultados

## E1 · Crecimiento necesario

### xor

| Método | Prueba | Parámetros activos | Conexiones | FLOPs (×10⁶) | Ciclos |
|---|---|---|---|---|---|
| `mlp_pequeno` | 98.3 ± 0.7 % | 354 | 320 | 197 | 0.0 |
| `mlp_grande` | 97.3 ± 0.7 % | 4482 | 4352 | 2682 | 0.0 |
| `poda_magnitud` | 96.7 ± 1.1 % | 450 | 320 | 977 | 0.0 |
| `crecimiento_programado` | 97.7 ± 0.4 % | 354 | 320 | 114 | 0.0 |
| `set` | 97.2 ± 0.9 % | 386 | 320 | 197 | 0.0 |
| `rigl` | 97.1 ± 0.8 % | 386 | 320 | 202 | 0.0 |
| `rnra` | 97.2 ± 1.1 % | 381 | 320 | 247 | 0.0 |
| `rnra_aleatorio` | 97.2 ± 1.1 % | 386 | 320 | 247 | 0.0 |
| `rnra_sin_utilidad` | 97.2 ± 1.0 % | 382 | 320 | 247 | 0.0 |
| `rnra_sin_redundancia` | 97.2 ± 1.1 % | 381 | 320 | 247 | 0.0 |
| `rnra_sin_local` | 97.0 ± 0.8 % | 382 | 320 | 247 | 0.0 |
| `rnra_sin_histeresis` | 96.9 ± 0.9 % | 380 | 320 | 247 | 0.3 |
| `rnra_sin_crecimiento` | 97.2 ± 1.1 % | 381 | 320 | 247 | 0.0 |
| `rnra_sin_poda` | 97.1 ± 1.2 % | 386 | 320 | 247 | 0.0 |

| `rnra` contra | Diferencia | IC 95 % | p (Holm) |
|---|---|---|---|
| `mlp_pequeno` | -1.17 pp | [-2.02, -0.31] | 0.090 |
| `mlp_grande` | -0.17 pp | [-1.28, +0.95] | 1.000 |
| `poda_magnitud` | +0.42 pp | [-0.72, +1.55] | 1.000 |
| `crecimiento_programado` | -0.50 pp | [-1.30, +0.30] | 1.000 |
| `set` | -0.08 pp | [-0.80, +0.63] | 1.000 |
| `rigl` | +0.08 pp | [-0.57, +0.74] | 1.000 |
| `rnra_aleatorio` | +0.00 pp | [-0.89, +0.89] | 1.000 |

Criterios (contra `mlp_pequeno`, el mejor con el mismo presupuesto): precision ❌ · parametros ❌ · eficiencia ❌ · estabilidad ✅ · diferencia de precisión -1.17 pp.

### circulos

| Método | Prueba | Parámetros activos | Conexiones | FLOPs (×10⁶) | Ciclos |
|---|---|---|---|---|---|
| `mlp_pequeno` | 86.8 ± 1.0 % | 354 | 320 | 197 | 0.0 |
| `mlp_grande` | 86.2 ± 0.9 % | 4482 | 4352 | 2686 | 0.0 |
| `poda_magnitud` | 87.0 ± 0.7 % | 450 | 320 | 979 | 0.0 |
| `crecimiento_programado` | 86.4 ± 1.3 % | 354 | 320 | 114 | 0.0 |
| `set` | 87.1 ± 1.0 % | 386 | 320 | 197 | 0.0 |
| `rigl` | 87.3 ± 1.2 % | 386 | 320 | 203 | 0.0 |
| `rnra` | 87.2 ± 0.9 % | 383 | 320 | 250 | 0.7 |
| `rnra_aleatorio` | 87.2 ± 1.4 % | 385 | 320 | 250 | 0.6 |
| `rnra_sin_utilidad` | 87.2 ± 1.1 % | 384 | 320 | 250 | 0.8 |
| `rnra_sin_redundancia` | 87.2 ± 0.8 % | 383 | 320 | 250 | 0.8 |
| `rnra_sin_local` | 87.1 ± 0.6 % | 384 | 320 | 250 | 0.7 |
| `rnra_sin_histeresis` | 87.2 ± 0.7 % | 382 | 320 | 250 | 1.9 |
| `rnra_sin_crecimiento` | 87.4 ± 0.8 % | 382 | 320 | 250 | 0.0 |
| `rnra_sin_poda` | 87.4 ± 0.9 % | 387 | 320 | 249 | 0.6 |

| `rnra` contra | Diferencia | IC 95 % | p (Holm) |
|---|---|---|---|
| `mlp_pequeno` | +0.39 pp | [-0.81, +1.59] | 1.000 |
| `mlp_grande` | +1.06 pp | [+0.04, +2.07] | 0.304 |
| `poda_magnitud` | +0.22 pp | [-0.77, +1.22] | 1.000 |
| `crecimiento_programado` | +0.78 pp | [-0.40, +1.95] | 1.000 |
| `set` | +0.17 pp | [-0.33, +0.66] | 1.000 |
| `rigl` | -0.06 pp | [-0.57, +0.46] | 1.000 |
| `rnra_aleatorio` | +0.06 pp | [-0.96, +1.07] | 1.000 |

Criterios (contra `rigl`, el mejor con el mismo presupuesto): precision ❌ · parametros ✅ · eficiencia ❌ · estabilidad ❌ · diferencia de precisión -0.06 pp.

### lunas

| Método | Prueba | Parámetros activos | Conexiones | FLOPs (×10⁶) | Ciclos |
|---|---|---|---|---|---|
| `mlp_pequeno` | 96.6 ± 0.8 % | 354 | 320 | 197 | 0.0 |
| `mlp_grande` | 96.1 ± 0.6 % | 4482 | 4352 | 2686 | 0.0 |
| `poda_magnitud` | 96.3 ± 0.5 % | 450 | 320 | 979 | 0.0 |
| `crecimiento_programado` | 96.3 ± 0.3 % | 354 | 320 | 114 | 0.0 |
| `set` | 95.3 ± 2.3 % | 386 | 320 | 197 | 0.0 |
| `rigl` | 96.0 ± 1.7 % | 386 | 320 | 203 | 0.0 |
| `rnra` | 96.3 ± 2.0 % | 380 | 320 | 250 | 0.0 |
| `rnra_aleatorio` | 94.4 ± 1.9 % | 383 | 320 | 250 | 0.2 |
| `rnra_sin_utilidad` | 96.3 ± 1.2 % | 381 | 320 | 250 | 0.0 |
| `rnra_sin_redundancia` | 96.3 ± 2.0 % | 381 | 320 | 250 | 0.0 |
| `rnra_sin_local` | 95.9 ± 2.2 % | 380 | 320 | 250 | 0.1 |
| `rnra_sin_histeresis` | 96.1 ± 1.6 % | 381 | 320 | 250 | 0.8 |
| `rnra_sin_crecimiento` | 96.2 ± 1.4 % | 380 | 320 | 250 | 0.0 |
| `rnra_sin_poda` | 95.7 ± 1.9 % | 386 | 320 | 249 | 0.0 |

| `rnra` contra | Diferencia | IC 95 % | p (Holm) |
|---|---|---|---|
| `mlp_pequeno` | -0.28 pp | [-1.84, +1.28] | 1.000 |
| `mlp_grande` | +0.22 pp | [-1.36, +1.80] | 1.000 |
| `poda_magnitud` | -0.06 pp | [-1.59, +1.48] | 1.000 |
| `crecimiento_programado` | -0.00 pp | [-1.38, +1.38] | 1.000 |
| `set` | +1.00 pp | [+0.03, +1.97] | 0.267 |
| `rigl` | +0.28 pp | [-0.00, +0.56] | 0.267 |
| `rnra_aleatorio` | +1.89 pp | [+0.47, +3.31] | 0.102 |

Criterios (contra `mlp_pequeno`, el mejor con el mismo presupuesto): precision ❌ · parametros ❌ · eficiencia ❌ · estabilidad ✅ · diferencia de precisión -0.28 pp.

## E2 · Complejidad localizada

### caotica_20d

| Método | Prueba | Parámetros activos | Conexiones | FLOPs (×10⁶) | Ciclos |
|---|---|---|---|---|---|
| `mlp_pequeno` | 74.6 ± 0.7 % | 676 | 640 | 405 | 0.0 |
| `mlp_grande` | 74.6 ± 0.8 % | 5764 | 5632 | 3561 | 0.0 |
| `poda_magnitud` | 78.5 ± 0.7 % | 772 | 640 | 1396 | 0.0 |
| `crecimiento_programado` | 75.3 ± 1.0 % | 676 | 640 | 256 | 0.0 |
| `set` | 77.1 ± 1.4 % | 708 | 640 | 405 | 0.0 |
| `rigl` | 77.1 ± 1.1 % | 708 | 640 | 413 | 0.0 |
| `rnra` | 76.8 ± 1.0 % | 707 | 640 | 526 | 0.1 |
| `rnra_aleatorio` | 76.8 ± 1.1 % | 708 | 640 | 525 | 0.1 |
| `rnra_sin_utilidad` | 77.0 ± 0.9 % | 707 | 640 | 529 | 0.2 |
| `rnra_sin_redundancia` | 76.7 ± 0.9 % | 705 | 640 | 531 | 0.1 |
| `rnra_sin_local` | 77.0 ± 1.3 % | 707 | 640 | 526 | 0.0 |
| `rnra_sin_histeresis` | 76.9 ± 1.3 % | 707 | 640 | 527 | 0.8 |
| `rnra_sin_crecimiento` | 76.7 ± 1.0 % | 706 | 640 | 526 | 0.0 |
| `rnra_sin_poda` | 76.5 ± 1.0 % | 708 | 640 | 523 | 0.1 |

| `rnra` contra | Diferencia | IC 95 % | p (Holm) |
|---|---|---|---|
| `mlp_pequeno` | +2.26 pp | [+1.43, +3.08] | 0.001 |
| `mlp_grande` | +2.22 pp | [+1.21, +3.22] | 0.004 |
| `poda_magnitud` | -1.72 pp | [-2.45, -0.99] | 0.003 |
| `crecimiento_programado` | +1.53 pp | [+0.58, +2.49] | 0.022 |
| `set` | -0.26 pp | [-1.00, +0.48] | 1.000 |
| `rigl` | -0.32 pp | [-1.05, +0.41] | 1.000 |
| `rnra_aleatorio` | -0.02 pp | [-0.68, +0.65] | 1.000 |

Criterios (contra `poda_magnitud`, el mejor con el mismo presupuesto): precision ❌ · parametros ✅ · eficiencia ❌ · estabilidad ❌ · diferencia de precisión -1.72 pp.

## E3 · Cambios de distribución

### lunas_rotadas (3 tareas)

| Método | Exactitud final media | Olvido |
|---|---|---|
| `mlp_pequeno` | 80.4 ± 1.1 % | 24.2 ± 1.5 % |
| `mlp_grande` | 78.3 ± 1.5 % | 28.2 ± 2.2 % |
| `poda_magnitud` | 76.8 ± 3.4 % | 27.7 ± 4.7 % |
| `crecimiento_programado` | 80.1 ± 0.8 % | 25.3 ± 1.3 % |
| `set` | 80.1 ± 1.8 % | 23.1 ± 3.6 % |
| `rigl` | 79.7 ± 0.9 % | 24.3 ± 3.1 % |
| `rnra` | 79.3 ± 1.7 % | 24.7 ± 3.3 % |
| `rnra_aleatorio` | 80.0 ± 3.2 % | 20.3 ± 3.8 % |
| `rnra_sin_utilidad` | 79.8 ± 1.7 % | 24.3 ± 3.7 % |
| `rnra_sin_redundancia` | 80.0 ± 1.0 % | 23.8 ± 2.7 % |
| `rnra_sin_local` | 79.3 ± 1.5 % | 24.5 ± 3.5 % |
| `rnra_sin_histeresis` | 80.3 ± 1.1 % | 23.4 ± 3.0 % |
| `rnra_sin_crecimiento` | 79.7 ± 1.1 % | 24.1 ± 2.9 % |
| `rnra_sin_poda` | 79.7 ± 1.7 % | 23.4 ± 4.2 % |

| `rnra` contra | Diferencia | IC 95 % | p (Holm) |
|---|---|---|---|
| `mlp_pequeno` | -1.10 pp | [-2.38, +0.19] | 0.515 |
| `mlp_grande` | +1.04 pp | [-0.84, +2.93] | 0.958 |
| `poda_magnitud` | +2.59 pp | [+0.31, +4.86] | 0.212 |
| `crecimiento_programado` | -0.79 pp | [-2.06, +0.48] | 0.958 |
| `set` | -0.79 pp | [-2.13, +0.56] | 0.958 |
| `rigl` | -0.31 pp | [-1.62, +0.99] | 1.000 |
| `rnra_aleatorio` | -0.63 pp | [-3.49, +2.23] | 1.000 |

## E4 · Aprendizaje continuo

### digitos_permutados (4 tareas)

| Método | Exactitud final media | Olvido |
|---|---|---|
| `mlp_pequeno` | 45.5 ± 2.9 % | 64.0 ± 4.1 % |
| `mlp_grande` | 44.0 ± 3.2 % | 70.1 ± 4.2 % |
| `poda_magnitud` | 46.8 ± 3.8 % | 61.9 ± 5.0 % |
| `crecimiento_programado` | 43.9 ± 3.7 % | 67.3 ± 4.9 % |
| `set` | 40.4 ± 3.5 % | 73.5 ± 4.8 % |
| `rigl` | 41.9 ± 2.7 % | 71.7 ± 3.6 % |
| `rnra` | 43.2 ± 4.5 % | 69.9 ± 6.0 % |
| `rnra_aleatorio` | 48.6 ± 2.5 % | 61.3 ± 3.2 % |
| `rnra_sin_utilidad` | 42.4 ± 1.9 % | 70.9 ± 2.6 % |
| `rnra_sin_redundancia` | 40.7 ± 2.7 % | 73.6 ± 3.7 % |
| `rnra_sin_local` | 43.5 ± 2.6 % | 69.5 ± 3.5 % |
| `rnra_sin_histeresis` | 43.5 ± 1.8 % | 69.6 ± 2.4 % |
| `rnra_sin_crecimiento` | 43.2 ± 4.5 % | 69.9 ± 6.0 % |
| `rnra_sin_poda` | 43.6 ± 4.0 % | 69.3 ± 5.4 % |

| `rnra` contra | Diferencia | IC 95 % | p (Holm) |
|---|---|---|---|
| `mlp_pequeno` | -2.28 pp | [-5.89, +1.34] | 0.939 |
| `mlp_grande` | -0.82 pp | [-4.93, +3.29] | 1.000 |
| `poda_magnitud` | -3.57 pp | [-5.91, -1.24] | 0.043 |
| `crecimiento_programado` | -0.63 pp | [-5.26, +3.99] | 1.000 |
| `set` | +2.81 pp | [-1.73, +7.34] | 0.939 |
| `rigl` | +1.34 pp | [-1.20, +3.88] | 0.939 |
| `rnra_aleatorio` | -5.37 pp | [-8.70, -2.03] | 0.038 |

### digitos_partidos (5 tareas)

| Método | Exactitud final media | Olvido |
|---|---|---|
| `mlp_pequeno` | 87.7 ± 5.8 % | 14.5 ± 7.3 % |
| `mlp_grande` | 88.5 ± 7.9 % | 13.7 ± 9.9 % |
| `poda_magnitud` | 92.4 ± 5.1 % | 8.8 ± 6.4 % |
| `crecimiento_programado` | 89.5 ± 3.9 % | 12.1 ± 4.8 % |
| `set` | 86.1 ± 6.2 % | 16.6 ± 7.8 % |
| `rigl` | 85.7 ± 6.2 % | 17.1 ± 7.8 % |
| `rnra` | 86.8 ± 3.5 % | 16.0 ± 4.5 % |
| `rnra_aleatorio` | 85.2 ± 5.9 % | 17.7 ± 7.4 % |
| `rnra_sin_utilidad` | 86.8 ± 3.6 % | 15.9 ± 4.4 % |
| `rnra_sin_redundancia` | 87.0 ± 3.5 % | 15.5 ± 4.3 % |
| `rnra_sin_local` | 84.2 ± 5.6 % | 19.0 ± 7.1 % |
| `rnra_sin_histeresis` | 86.6 ± 4.4 % | 16.0 ± 5.4 % |
| `rnra_sin_crecimiento` | 86.8 ± 3.5 % | 16.0 ± 4.5 % |
| `rnra_sin_poda` | 86.1 ± 5.0 % | 16.7 ± 6.3 % |

| `rnra` contra | Diferencia | IC 95 % | p (Holm) |
|---|---|---|---|
| `mlp_pequeno` | -0.94 pp | [-4.50, +2.62] | 1.000 |
| `mlp_grande` | -1.71 pp | [-8.16, +4.74] | 1.000 |
| `poda_magnitud` | -5.57 pp | [-7.82, -3.32] | 0.002 |
| `crecimiento_programado` | -2.73 pp | [-6.99, +1.53] | 1.000 |
| `set` | +0.67 pp | [-4.51, +5.84] | 1.000 |
| `rigl` | +1.08 pp | [-3.77, +5.94] | 1.000 |
| `rnra_aleatorio` | +1.56 pp | [-4.42, +7.54] | 1.000 |
