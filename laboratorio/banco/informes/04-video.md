# Cuaderno 04: interpolación y predicción de fotogramas

Código: `laboratorio/banco/video.py` (`python -m laboratorio.banco --video`).

## Conclusión

Los dígitos de los clips de prueba salen de un grupo distinto al de entrenamiento (nunca se vieron al entrenar).

- **Tus tripletas son muy eficientes.** Con 11 mil parámetros suben de 16,6 dB (promediar los vecinos) a 18,3 dB; la U-Net de la industria llega a 19,0 dB con 467 mil (42 veces más). Es decir, logran el 71 % de la mejora con el 2 % de los parámetros. **Recomendables donde la memoria o la batería cuentan** (celular, cámaras, video en vivo).
- **Mejoradas, empatan con la U-Net usando 19 veces menos parámetros.** Las **tripletas con flujo** (estiman hacia dónde se movió cada píxel, lo traen a su lugar y mezclan los dos fotogramas, como RIFE pero en miniatura) llegan a **19,22 dB con 24 mil parámetros**, contra 19,13 dB de la U-Net con 467 mil, y entrenan en la mitad del tiempo. Ganan en las 3 semillas, pero por poco (+0,09 dB, p = 0,10), y la U-Net sigue algo más nítida (SSIM 0,738 contra 0,719). Las **tripletas dilatadas** (ven más lejos con convoluciones dilatadas) quedan en 18,95 dB con 17 mil. Ver «Mejoras» abajo.
- Hacer que las tripletas aprendan solo la corrección sobre el promedio (el arreglo) da lo mismo: la CNN ya lo aprende sola.
- **El FrameInterpolator con LSTM no sirve:** usa 66 millones de parámetros, rinde 2,6 dB menos que la U-Net y sus imágenes son más borrosas que copiar el fotograma vecino (SSIM 0,37 contra 0,50). La LSTM recibe la imagen aplanada como una secuencia de un solo paso, así que no recuerda nada ni sabe qué píxeles están cerca. Para video hay que usar convoluciones.
- Predecir el fotograma siguiente desde uno solo es casi imposible: con una sola imagen no se sabe hacia dónde se mueve cada dígito. Hacen falta al menos dos.
- La GAN del cuaderno es la DCGAN estándar de la industria (sin una idea nueva) y no se midió.

Videos de 2 dígitos que se mueven y rebotan (como Moving MNIST), 32×32 a color: 2000 clips para entrenar y 300 de prueba, 20 épocas, 3 semillas. PSNR en dB (más es mejor; +3 dB = la mitad de error) y SSIM (1 = idéntico).

## Dibujar el fotograma del medio a partir del anterior y el siguiente (tripletas)

| Modelo | PSNR | SSIM | Parámetros |
|---|---|---|---|
| copiar el fotograma más cercano (sin red) | 14,44 | 0,496 | 0 |
| promedio de los vecinos (sin red) | 16,57 | 0,535 | 0 |
| tripletas tal cual (CNN de 3 capas) | 18,31 ± 0,07 | 0,653 ± 0,009 | 11 107 |
| tripletas con corrección sobre el promedio (arreglo) | 18,34 ± 0,02 | 0,632 ± 0,020 | 11 107 |
| U-Net (industria) | 19,02 ± 0,03 | 0,710 ± 0,029 | 467 459 |

## Mejoras de las tripletas (2026-10)

Mismo protocolo (2000 clips para entrenar y 300 de prueba con dígitos que no se vieron, 20 épocas, 3 semillas, CPU). Las 4 redes se entrenaron con los mismos clips en cada semilla, así que la comparación es pareada.

| Modelo | PSNR | SSIM | Parámetros | Segundos por entrenamiento (CPU, 1 hilo) |
|---|---|---|---|---|
| tripletas tal cual | 18,32 ± 0,07 | 0,654 ± 0,012 | 11 107 | 29 |
| **tripletas dilatadas** (mejora 1: convoluciones dilatadas 1-2-4-8, ven ~31 píxeles alrededor) | 18,95 ± 0,10 | 0,674 ± 0,012 | 17 019 | 56 |
| **tripletas con flujo** (mejora 2: mueven los píxeles con `grid_sample` y los mezclan, como RIFE) | **19,22 ± 0,10** | 0,719 ± 0,013 | 24 105 | 101 |
| U-Net (industria) | 19,13 ± 0,14 | **0,738 ± 0,008** | 467 459 | 198 |

Por semilla, las tripletas con flujo superan a la U-Net en PSNR las 3 veces (+0,09, +0,14 y +0,04 dB; prueba t pareada p = 0,10) y quedan por debajo en SSIM las 3 veces (p = 0,09). Con 3 semillas no hay diferencia significativa: **empatan, con 19 veces menos parámetros y la mitad del tiempo**. Falta medir con video real (cuaderno 13).

## Dibujar el fotograma del medio a partir de 4 (FrameInterpolator, celda 13)

| Modelo | PSNR | SSIM | Parámetros |
|---|---|---|---|
| copiar el fotograma más cercano (sin red) | 14,44 | 0,496 | 0 |
| promedio de los vecinos (sin red) | 16,02 | 0,427 | 0 |
| FrameInterpolator tal cual (LSTM con la imagen aplanada) | 17,15 ± 0,02 | 0,374 ± 0,008 | 66 079 744 |
| U-Net (industria) | 19,76 ± 0,01 | 0,721 ± 0,021 | 469 187 |

**Reparado con convoluciones (2026-10):** las tripletas con flujo reciben los 4 fotogramas y hacen lo mismo que el FrameInterpolator, pero mirando qué píxeles están cerca. Con las mismas 3 semillas pareadas:

| Modelo | PSNR | SSIM | Parámetros |
|---|---|---|---|
| **tripletas con flujo, 4 fotogramas** | **20,25 ± 0,34** | **0,756 ± 0,028** | 26 703 |
| U-Net (industria), mismas semillas | 19,79 ± 0,04 | 0,726 ± 0,043 | 469 187 |

Le ganan a la U-Net en 2 de 3 semillas (+0,73 y +0,71 dB; en la otra pierden por 0,08), con **17 veces menos parámetros**; en promedio +0,46 dB, sin diferencia significativa con 3 semillas (p = 0,23). Frente al FrameInterpolator original: +3,1 dB con 2 500 veces menos parámetros.

## Predecir el fotograma siguiente a partir de 1 (celdas 42-43)

| Modelo | PSNR | SSIM | Parámetros |
|---|---|---|---|
| copiar el fotograma más cercano (sin red) | 14,54 | 0,506 | 0 |
| FrameInterpolator tal cual (LSTM con la imagen aplanada) | 15,53 ± 0,01 | 0,254 ± 0,007 | 28 331 008 |
| U-Net (industria) | 16,53 ± 0,01 | 0,462 ± 0,033 | 466 595 |
