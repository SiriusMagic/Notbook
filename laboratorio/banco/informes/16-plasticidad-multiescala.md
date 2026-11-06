# Cuaderno 16: RNRA-M, plasticidad multiescala

Código: `laboratorio/rnra/` (`superred.py`, `memoria_multiescala.py`, `adaptador_dispositivo.py`, `transferencia.py`, `experimentos_m.py`; `python -m laboratorio.rnra.experimentos_m`). Pruebas: `tests/test_rnra_m.py`. El experimento 2 (aprender en el celular y volver, H3) se **retiró** tras estos resultados; está en el commit `fd6e8f3` (`transferencia.py`). Una sola superred de 32, 128 y 512 neuronas con núcleo cuantizado y una **memoria portátil** comprimida (sin internet, en un `.npz` sin `pickle`).

## Conclusión

⚪ **El mecanismo funciona; la ventaja de precisión no se demuestra.** 5 semillas, 400 pasos, tarea caótica 20D (experimento 1) y dígitos permutados (experimento 2); datos de prueba aparte y Holm.

- **Reducir y expandir sin perder la red (parte de H2): ✅ como mecanismo.** Con memoria de 32 bits la red grande vuelve **idéntica** (error máximo por peso 0,0; también tras pasar por un archivo). Con memoria de 8 bits no se pierde exactitud (75,4 % igual que la original) y la memoria pesa 4,7 veces menos (212 KB contra 1000 KB). Con 4 bits se pierden 0,6 puntos (74,8 %).
- **Pero esta tarea casi no necesita la capacidad extra:** tirar lo recortado y expandir cuesta solo 0,4 puntos (75,0 % contra 75,4 %). Por eso la recuperación **no se nota en la exactitud**. Hace falta una tarea que exija más capacidad (Fashion-MNIST, texto) para ver si la memoria importa.
- **H1 (superred contra una red por dispositivo) no se sostiene.** La superred y RNRA-M empatan con la red fija en los tres dispositivos (+0,3, +1,3 y +0,4 puntos, ninguna significativa; la más grande, en el celular potente, tiene p = 0,11). Contra «entrenar grande y cuantizar después» solo gana en el celular básico (+24 puntos, p = 0,012), porque ese comparador cae a 2 bits; es un comparador débil.
- **Ahorro de RAM: no es mérito de la superred.** RNRA-M usa 1,8 KB en el celular básico, igual que la red fija **cuantizada a 8 bits** (1,8 KB, 75,7 % contra 76,0 %). Lo que da la superred es que **una sola red** sirve para los tres dispositivos y que la memoria permite volver al tamaño grande.
- **H3 (aprender en el celular y volver) no se sostiene.** `copiar_y_reconciliar` olvida 0,6 puntos de la tarea vieja contra 31 del ajuste convencional (p = 0,012), pero eso se debe al **repaso del recuerdo de A**: contra `ajuste_con_repaso` la diferencia es −0,19 puntos (p = 0,53) y contra `ajuste_buffer`, con los mismos datos, −0,26 (p = 0,49). Aprender primero en el celular no aporta nada medible. `copiar_nucleo` no olvida nada pero la tarea nueva queda en 22,5 %.
- En el celular, el núcleo de 32 neuronas aprende B (94,8 %) pero olvida A (98 % → 61 %).

### Qué no se midió

- **RAM real, energía y latencia en un teléfono:** la RAM es calculada (bytes de pesos), la latencia es la de esta CPU (0,07-0,40 ms) y la cuantización es simulada (cálculo en 32 bits con pesos redondeados), así que no hay aceleración real de 8 bits.
- **Memoria remota** (dejar parte en un servidor): se eligió la versión portátil sin internet, que no la necesita.
- **Más de una familia de tareas** y redes grandes: con 262 mil pesos por capa la memoria comprimida casi no comprime (los pesos son casi aleatorios); el archivo completo pesa casi lo mismo que la red grande. **Se ahorra RAM de trabajo, no almacenamiento.**

### Qué sigue

1. Una tarea que exija capacidad (Fashion-MNIST, texto) para que «sin memoria» pierda de verdad.
2. Cuantización real de 8 bits (ONNX Runtime o PyTorch) y medir RAM y energía en un teléfono.
3. Para H3: comparar contra LoRA por dispositivo y contra distilación en vez de solo repaso.

## Resultados

## Experimento 1 · Reducir y expandir

### caotica_20d

**celular_basico** (RAM disponible 7.3 KB)

| Método | Exactitud | RAM de pesos (KB) | Latencia (ms) |
|---|---|---|---|
| `fija_por_dispositivo` | 75.6 ± 0.7 % | 7.3 | 0.07 |
| `fija_int8` | 75.7 ± 0.6 % | 1.8 | 0.07 |
| `cuantizada_despues` | 51.7 ± 7.9 % | 4.8 | 0.11 |
| `superred_ancho_variable` | 75.9 ± 0.8 % | 7.3 | 0.07 |
| `rnra_m` | 76.0 ± 0.8 % | 1.8 | 0.07 |

| `rnra_m` contra | Diferencia | IC 95 % | p (Holm) |
|---|---|---|---|
| `fija_por_dispositivo` | +0.33 pp | [-0.87, +1.54] | 1.000 |
| `fija_int8` | +0.25 pp | [-0.91, +1.41] | 1.000 |
| `cuantizada_despues` | +24.28 pp | [+13.80, +34.77] | 0.012 |
| `superred_ancho_variable` | +0.03 pp | [-0.08, +0.15] | 1.000 |

**celular_potente** (RAM disponible 77.0 KB)

| Método | Exactitud | RAM de pesos (KB) | Latencia (ms) |
|---|---|---|---|
| `fija_por_dispositivo` | 75.2 ± 0.4 % | 77.0 | 0.11 |
| `fija_int8` | 75.2 ± 0.5 % | 19.3 | 0.11 |
| `cuantizada_despues` | 75.1 ± 0.5 % | 77.0 | 0.11 |
| `superred_ancho_variable` | 76.4 ± 1.0 % | 77.0 | 0.11 |
| `rnra_m` | 76.5 ± 0.9 % | 19.3 | 0.11 |

| `rnra_m` contra | Diferencia | IC 95 % | p (Holm) |
|---|---|---|---|
| `fija_por_dispositivo` | +1.25 pp | [+0.17, +2.33] | 0.112 |
| `fija_int8` | +1.28 pp | [+0.11, +2.46] | 0.112 |
| `cuantizada_despues` | +1.38 pp | [+0.24, +2.52] | 0.112 |
| `superred_ancho_variable` | +0.08 pp | [-0.02, +0.19] | 0.112 |

**pc_con_gpu** (RAM disponible 1076.0 KB)

| Método | Exactitud | RAM de pesos (KB) | Latencia (ms) |
|---|---|---|---|
| `fija_por_dispositivo` | 75.0 ± 0.7 % | 1076.0 | 0.40 |
| `fija_int8` | 75.0 ± 0.7 % | 269.0 | 0.40 |
| `cuantizada_despues` | 75.1 ± 0.5 % | 77.0 | 0.11 |
| `superred_ancho_variable` | 75.4 ± 0.7 % | 1076.0 | 0.40 |
| `rnra_m` | 75.4 ± 0.7 % | 269.0 | 0.40 |

| `rnra_m` contra | Diferencia | IC 95 % | p (Holm) |
|---|---|---|---|
| `fija_por_dispositivo` | +0.38 pp | [-1.20, +1.96] | 1.000 |
| `fija_int8` | +0.42 pp | [-1.24, +2.07] | 1.000 |
| `cuantizada_despues` | +0.30 pp | [-0.42, +1.02] | 1.000 |
| `superred_ancho_variable` | -0.02 pp | [-0.15, +0.12] | 1.000 |

**Recuperar la capacidad al volver al ancho grande** (exactitud de prueba)

| Situación | Exactitud | Memoria comprimida (KB) |
|---|---|---|
| Red grande original | 75.4 ± 0.7 % | — |
| Reducir y expandir **sin** memoria (se tiró lo recortado) | 75.0 ± 0.6 % | 0 |
| Reducir y expandir con memoria de 32 bits | 75.4 ± 0.7 % | 1000.3 |
| Reducir y expandir con memoria de 8 bits | 75.4 ± 0.7 % | 212.1 |
| Reducir y expandir con memoria de 4 bits | 74.8 ± 0.9 % | 88.5 |

Núcleo en RAM: 1.8 KB (la red grande en 32 bits pesa 1076 KB). Error máximo por peso al expandir con memoria de 32 bits: 0.0e+00. Pasar por un archivo `.npz` y volver da la misma exactitud (75.4 ± 0.7 %).

## Experimento 2 · Aprender en el celular y volver a la PC (dígitos permutados)

En el celular (núcleo de 32 neuronas): A 60.7 ± 11.4 %, B 94.8 ± 0.7 %.

| Integración en la red grande | Tarea A (vieja) | Tarea B (nueva) | Olvido de A |
|---|---|---|---|
| (antes de integrar) | 98.2 ± 0.2 % | 15.1 ± 3.1 % | 0 |
| `ajuste_convencional` | 67.3 ± 10.7 % | 97.7 ± 0.1 % | 31.0 ± 10.8 % |
| `ajuste_con_repaso` | 97.4 ± 0.7 % | 97.7 ± 0.1 % | 0.8 ± 0.7 % |
| `ajuste_buffer` | 97.4 ± 0.7 % | 92.2 ± 1.1 % | 0.9 ± 0.7 % |
| `copiar_nucleo` | 98.2 ± 0.2 % | 22.5 ± 1.4 % | 0.0 ± 0.0 % |
| `copiar_y_reconciliar` | 97.6 ± 0.6 % | 92.2 ± 1.5 % | 0.6 ± 0.7 % |

Olvido de `copiar_y_reconciliar` menos el de cada método (negativo = olvida menos):

| Contra | Diferencia | IC 95 % | p (Holm) |
|---|---|---|---|
| `ajuste_convencional` | -30.37 pp | [-43.58, -17.16] | 0.012 |
| `ajuste_con_repaso` | -0.19 pp | [-0.93, +0.56] | 0.528 |
| `ajuste_buffer` | -0.26 pp | [-0.79, +0.27] | 0.491 |
| `copiar_nucleo` | +0.59 pp | [-0.30, +1.49] | 0.421 |