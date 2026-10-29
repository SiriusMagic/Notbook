# Cuaderno 04: interpolación y predicción de fotogramas

Código: `laboratorio/banco/video.py` (`python -m laboratorio.banco --video`).

## Conclusión

Los dígitos de los clips de prueba salen de un grupo distinto al de entrenamiento (nunca se vieron al entrenar).

- **Tus tripletas son muy eficientes.** Con 11 mil parámetros suben de 16,6 dB (promediar los vecinos) a 18,3 dB; la U-Net de la industria llega a 19,0 dB con 467 mil (42 veces más). Es decir, logran el 71 % de la mejora con el 2 % de los parámetros. **Recomendables donde la memoria o la batería cuentan** (celular, cámaras, video en vivo).
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

## Dibujar el fotograma del medio a partir de 4 (FrameInterpolator, celda 13)

| Modelo | PSNR | SSIM | Parámetros |
|---|---|---|---|
| copiar el fotograma más cercano (sin red) | 14,44 | 0,496 | 0 |
| promedio de los vecinos (sin red) | 16,02 | 0,427 | 0 |
| FrameInterpolator tal cual (LSTM con la imagen aplanada) | 17,15 ± 0,02 | 0,374 ± 0,008 | 66 079 744 |
| U-Net (industria) | 19,76 ± 0,01 | 0,721 ± 0,021 | 469 187 |

## Predecir el fotograma siguiente a partir de 1 (celdas 42-43)

| Modelo | PSNR | SSIM | Parámetros |
|---|---|---|---|
| copiar el fotograma más cercano (sin red) | 14,54 | 0,506 | 0 |
| FrameInterpolator tal cual (LSTM con la imagen aplanada) | 15,53 ± 0,01 | 0,254 ± 0,007 | 28 331 008 |
| U-Net (industria) | 16,53 ± 0,01 | 0,462 ± 0,033 | 466 595 |
