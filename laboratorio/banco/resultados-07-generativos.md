# Cuaderno 07: D-ReG y HCMG

Código: `laboratorio/banco/generativos_07.py` (`python -m laboratorio.banco --generativos07`).

## Conclusión

- **HCMG gana a la U-Net de la industria**, poco pero con constancia: 20,44 dB contra 20,04 (3 semillas, desviación 0,10) y SSIM 0,825 contra 0,804. Sin condiciones cae a 15,9 dB: **la forma en que mete las dos condiciones (normalizar y reescalar según cada una, como AdaIN o SPADE) funciona mejor que pasarlas como canales extra**. Ojo: usa 3 veces más parámetros que la U-Net. Es la pista más interesante del cuaderno 07; vale la pena probarla con imágenes reales y texto real (por ejemplo, embeddings de CLIP) en GPU.
- **D-ReG no le gana a una GRU del mismo tamaño.** Con poco entrenamiento (600 pasos) parecía el mejor (161 contra 186 de la GRU chica), pero con 1 500 pasos una GRU con los mismos parámetros queda en 146 contra 176. La ventaja venía del tamaño. Su atención sobre un solo elemento no hace nada; el arreglo (mirar las últimas 16 palabras) no mejora con este entrenamiento.
- El Transformer empeora con más entrenamiento (169 a 292): con 3 000 reseñas memoriza. Para texto hacen falta muchos más datos.

## Más entrenamiento (1 500 pasos, 3 semillas)

| Modelo | Perplejidad | Parámetros |
|---|---|---|
| GRU del mismo tamaño que D-ReG (industria) | 146,4 ± 2,4 | 1 043 200 |
| D-ReG tal cual | 175,7 ± 2,9 | 1 043 024 |
| Transformer causal (como GPT) | 292,1 ± 0,8 | 918 736 |


## D-ReG: predecir la palabra siguiente en reseñas reales (3000 para entrenar, 500 de prueba, BPE de 2000, 600 pasos, 2 semillas)

| Modelo | Perplejidad (menos es mejor) | Parámetros |
|---|---|---|
| D-ReG tal cual (atención sobre 1 elemento) | 160,9 ± 1,8 | 1 043 024 |
| D-ReG con atención a las últimas 16 palabras (arreglo) | 230,7 ± 0,5 | 1 043 024 |
| GRU (industria clásica) | 185,7 ± 0,5 | 613 072 |
| LSTM (industria clásica) | 230,2 ± 2,4 | 646 096 |
| Transformer causal, como GPT (industria) | 168,5 ± 3,1 | 918 736 |

## HCMG: dibujar el dígito nítido de la clase y el color pedidos a partir de un borrón (3000 para entrenar, 500 de prueba, 20 épocas, 3 semillas)

| Modelo | PSNR (más es mejor) | SSIM | Parámetros |
|---|---|---|---|
| HCMG tal cual (dos normalizaciones condicionadas) | 20,44 ± 0,10 | 0,825 ± 0,005 | 1 373 731 |
| HCMG sin condiciones (control) | 15,92 ± 0,06 | 0,602 ± 0,007 | 1 373 731 |
| U-Net con las condiciones como canales (industria) | 20,04 ± 0,10 | 0,804 ± 0,006 | 470 339 |
