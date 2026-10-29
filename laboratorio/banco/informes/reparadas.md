# Piezas reparadas: la lista «no están listas», arreglada

Tomé las piezas que perdían por un error de diseño (sección 3 de [`RECOMENDACIONES.md`](../../../RECOMENDACIONES.md)), corregí ese error sin cambiar la idea y las medí otra vez **junto a su versión original**, en las mismas 7 tareas, con 5 semillas y datos de prueba que la red no vio. El código está en [`laboratorio/banco/piezas_reparadas.py`](../piezas_reparadas.py).

## Conclusión

**Las 8 mejoran: 5 ya no pierden en ninguna tarea y 3 pierden en una sola** (antes perdían en 1 a 7). Además, la neurona KNN convertida en **base radial entrenable** pasa a ser prometedora: en la tarea caótica llega a **76,9 %** (red estándar: 70,5 %; la estándar de su mejor tamaño: 75,6 %; Dropout: 77,4 %).

| Pieza | Error de diseño | Arreglo | Antes (🔴 pierde en…) | Después |
|---|---|---|---|---|
| AttentiveLinear (05) | La «atención» no dependía de la entrada | Compuerta calculada desde la propia salida (como Squeeze-and-Excitation) | 6 de 7 tareas (azar en círculos y lunas) | ⚪ empata en las 7 |
| Neuronas KNN (03) | Solo aprendía la capa de salida | **Base radial**: centros y anchos aprendidos con gradiente | 7 de 7 | 🟢 **gana** en caótica (76,9 % contra 70,5 %) y serie de seno; 🔴 pierde por poco en regresión (0,0051 contra 0,0048) |
| HMN (02) | Memorizaba la posición de cada fila | La mini-LSTM de cada neurona arranca de cero en cada ejemplo | 6 de 7 | 🟢 gana en la serie de seno; 🔴 pierde en círculos; empata en el resto |
| RMB (02) | Algunas semillas se disparaban | Inicio de `nn.Linear` y memoria acotada con tanh | 1 de 7 (regresión: 2,76 ± 5,4) | ⚪ empata en las 7 (regresión: 0,0058) |
| Resonante (02) | El umbral duro apagaba el gradiente | Umbral suave y aprendido | 2 de 7 (regresión: 95 veces más error) | ⚪ empata en 6; 🔴 pierde en caótica (68,9 % contra 70,5 %) |
| HiperReLU (02) | ×10 fijo desestabilizaba | Amplificación aprendida que empieza en ×1 | 3 de 7 | ⚪ empata en las 7 (aprende a no amplificar) |
| Polinómica (02) | Memorizaba (matriz cuadrática completa) | Rango 4, entrada acotada con tanh y penalización | 4 de 7 | ⚪ empata en las 7, con 3 a 6 veces menos parámetros |
| Anti-gradiente (07) | Invertía el gradiente en el camino principal | **DANN**: la inversión va en una rama aparte que adivina el dominio | 5 de 6 | ⚪ +3 puntos en el dominio nuevo, sin diferencia significativa (ver abajo) |

Qué aprendí: casi todas fallaban por el mismo tipo de error, algo **fijo o duro** donde hacía falta algo **aprendido y suave** (umbral, amplificación, atención, centros). Una vez arreglado, la pieza se comporta como la red estándar: no daña, pero solo la base radial aporta algo propio.

## Anti-gradiente bien usado: DANN

La inversión de gradiente solo sirve cuando hay **dos dominios**: datos con etiqueta de uno (por ejemplo, una cámara) y datos sin etiqueta de otro (otra cámara). La rama aparte intenta adivinar de qué dominio viene cada ejemplo, y el gradiente invertido obliga a la red a ignorar esa diferencia.

Prueba: dígitos 8×8 limpios con etiqueta (origen) y, sin etiquetas, dígitos movidos 1 píxel, con otro contraste y con ruido (destino). Se mide la exactitud en dígitos del destino que no se vieron. 5 semillas, 30 épocas.

| Red | Destino (nuevo) | Origen | p frente a la normal |
|---|---|---|---|
| Red normal | 39,1 ± 2,8 % | 93,9 ± 0,7 % | — |
| DANN, λ = 0,03 | 42,2 ± 4,1 % | 94,0 ± 0,9 % | 0,182 |
| DANN, λ = 0,1 | 40,4 ± 5,0 % | 94,0 ± 0,9 % | 0,294 |

Con λ = 0,03 la red DANN acierta 3 puntos más en el destino sin perder nada en el origen, pero la diferencia no es significativa con 5 semillas. Con λ = 1, como en el cuaderno, empeoró en una prueba rápida (22 % contra 38 %): la rama del dominio domina. Para estos datos, normalizar cada imagen resolvería casi todo el cambio de contraste; DANN vale la pena cuando el cambio entre dominios no se arregla con una regla simple (reseñas de otra tienda, otra cámara, otro idioma).

## Tablas completas

Cada celda: media ± desviación en **datos de prueba**, con 5 semillas. Exactitud: más es mejor. MSE: menos es mejor. Veredicto con prueba t pareada corregida por comparaciones múltiples (Benjamini-Hochberg, 5 %). Es una exploración: un 🟢 indica una pista que vale la pena confirmar, no una prueba definitiva.

| Pieza | Tarea | Pieza (prueba) | Estándar (prueba) | Veredicto | Pieza en entreno | Parámetros (pieza / estándar) |
|---|---|---|---|---|---|---|
| atencion_dinamica | regresion_cubica | 0.0050 ± 0.0007 | 0.0048 ± 0.0002 | ⚪ igual | 0.0033 | 3,265 / 1,153 |
| atencion_salida | regresion_cubica | 1.3182 ± 0.0000 | 0.0048 ± 0.0002 | 🔴 peor | 1.0000 | 3,201 / 1,153 |
| rbf_entrenable | regresion_cubica | 0.0051 ± 0.0001 | 0.0048 ± 0.0002 | 🔴 peor | 0.0034 | 2,273 / 1,153 |
| knn_prototipos | regresion_cubica | 1.0803 ± 0.0821 | 0.0048 ± 0.0002 | 🔴 peor | 0.7792 | 33 / 1,153 |
| hmn_sin_memoria | regresion_cubica | 0.0048 ± 0.0001 | 0.0048 ± 0.0002 | ⚪ igual | 0.0033 | 1,537 / 1,153 |
| hmn | regresion_cubica | 0.6934 ± 0.1247 | 0.0048 ± 0.0002 | 🔴 peor | 0.0349 | 2,177 / 1,153 |
| rmb_estable | regresion_cubica | 0.0058 ± 0.0013 | 0.0048 ± 0.0002 | ⚪ igual | 0.0043 | 3,265 / 1,153 |
| rmb | regresion_cubica | 2.7640 ± 5.4354 | 0.0048 ± 0.0002 | 🔴 peor | 2.6951 | 3,265 / 1,153 |
| resonante_suave | regresion_cubica | 0.0046 ± 0.0002 | 0.0048 ± 0.0002 | ⚪ igual | 0.0034 | 1,157 / 1,153 |
| resonante | regresion_cubica | 0.4564 ± 0.2568 | 0.0048 ± 0.0002 | 🔴 peor | 0.3752 | 1,153 / 1,153 |
| hiper_relu_suave | regresion_cubica | 0.0048 ± 0.0002 | 0.0048 ± 0.0002 | ⚪ igual | 0.0032 | 1,155 / 1,153 |
| hiper_relu | regresion_cubica | 0.0135 ± 0.0023 | 0.0048 ± 0.0002 | 🔴 peor | 0.0099 | 1,153 / 1,153 |
| polinomica_regularizada | regresion_cubica | 0.0048 ± 0.0002 | 0.0048 ± 0.0002 | ⚪ igual | 0.0031 | 9,601 / 1,153 |
| polinomica | regresion_cubica | 0.0046 ± 0.0001 | 0.0048 ± 0.0002 | ⚪ igual | 0.0031 | 34,145 / 1,153 |
| gelu (industria) | regresion_cubica | 0.0046 ± 0.0001 | 0.0048 ± 0.0002 | ⚪ igual | 0.0035 | 1,153 / 1,153 |
| atencion_dinamica | circulos | 83.3 ± 2.2 % | 84.8 ± 0.6 % | ⚪ igual | 85.8 | 3,330 / 1,218 |
| atencion_salida | circulos | 50.0 ± 0.0 % | 84.8 ± 0.6 % | 🔴 peor | 50.0 | 3,266 / 1,218 |
| rbf_entrenable | circulos | 85.7 ± 1.2 % | 84.8 ± 0.6 % | ⚪ igual | 84.2 | 2,370 / 1,218 |
| knn_prototipos | circulos | 77.8 ± 2.2 % | 84.8 ± 0.6 % | 🔴 peor | 76.7 | 66 / 1,218 |
| hmn_sin_memoria | circulos | 80.9 ± 0.6 % | 84.8 ± 0.6 % | 🔴 peor | 93.0 | 1,602 / 1,218 |
| hmn | circulos | 79.9 ± 1.4 % | 84.8 ± 0.6 % | 🔴 peor | 83.6 | 2,242 / 1,218 |
| rmb_estable | circulos | 83.3 ± 1.3 % | 84.8 ± 0.6 % | ⚪ igual | 84.7 | 3,330 / 1,218 |
| rmb | circulos | 85.7 ± 1.0 % | 84.8 ± 0.6 % | ⚪ igual | 84.0 | 3,330 / 1,218 |
| resonante_suave | circulos | 81.0 ± 1.9 % | 84.8 ± 0.6 % | ⚪ igual | 91.5 | 1,222 / 1,218 |
| resonante | circulos | 82.7 ± 1.2 % | 84.8 ± 0.6 % | ⚪ igual | 84.0 | 1,218 / 1,218 |
| hiper_relu_suave | circulos | 84.3 ± 1.0 % | 84.8 ± 0.6 % | ⚪ igual | 85.2 | 1,220 / 1,218 |
| hiper_relu | circulos | 83.6 ± 2.4 % | 84.8 ± 0.6 % | ⚪ igual | 85.6 | 1,218 / 1,218 |
| polinomica_regularizada | circulos | 83.3 ± 1.6 % | 84.8 ± 0.6 % | ⚪ igual | 87.4 | 9,922 / 1,218 |
| polinomica | circulos | 78.8 ± 1.2 % | 84.8 ± 0.6 % | 🔴 peor | 93.3 | 34,306 / 1,218 |
| gelu (industria) | circulos | 83.8 ± 1.3 % | 84.8 ± 0.6 % | ⚪ igual | 84.5 | 1,218 / 1,218 |
| atencion_dinamica | lunas | 96.7 ± 0.0 % | 96.3 ± 0.3 % | ⚪ igual | 98.2 | 3,330 / 1,218 |
| atencion_salida | lunas | 50.0 ± 0.0 % | 96.3 ± 0.3 % | 🔴 peor | 50.0 | 3,266 / 1,218 |
| rbf_entrenable | lunas | 96.3 ± 0.3 % | 96.3 ± 0.3 % | ⚪ igual | 98.0 | 2,370 / 1,218 |
| knn_prototipos | lunas | 90.7 ± 1.8 % | 96.3 ± 0.3 % | 🔴 peor | 87.3 | 66 / 1,218 |
| hmn_sin_memoria | lunas | 95.9 ± 0.3 % | 96.3 ± 0.3 % | ⚪ igual | 98.1 | 1,602 / 1,218 |
| hmn | lunas | 92.4 ± 2.2 % | 96.3 ± 0.3 % | 🔴 peor | 96.9 | 2,242 / 1,218 |
| rmb_estable | lunas | 96.2 ± 0.2 % | 96.3 ± 0.3 % | ⚪ igual | 98.0 | 3,330 / 1,218 |
| rmb | lunas | 86.4 ± 18.5 % | 96.3 ± 0.3 % | ⚪ igual | 85.3 | 3,330 / 1,218 |
| resonante_suave | lunas | 96.2 ± 0.2 % | 96.3 ± 0.3 % | ⚪ igual | 98.9 | 1,222 / 1,218 |
| resonante | lunas | 96.9 ± 0.3 % | 96.3 ± 0.3 % | ⚪ igual | 97.5 | 1,218 / 1,218 |
| hiper_relu_suave | lunas | 96.3 ± 0.3 % | 96.3 ± 0.3 % | ⚪ igual | 97.8 | 1,220 / 1,218 |
| hiper_relu | lunas | 96.3 ± 0.5 % | 96.3 ± 0.3 % | ⚪ igual | 98.3 | 1,218 / 1,218 |
| polinomica_regularizada | lunas | 96.0 ± 0.5 % | 96.3 ± 0.3 % | ⚪ igual | 98.4 | 9,922 / 1,218 |
| polinomica | lunas | 96.4 ± 0.3 % | 96.3 ± 0.3 % | ⚪ igual | 98.3 | 34,306 / 1,218 |
| gelu (industria) | lunas | 96.2 ± 0.2 % | 96.3 ± 0.3 % | ⚪ igual | 97.9 | 1,218 / 1,218 |
| atencion_dinamica | xor | 97.7 ± 0.4 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 3,330 / 1,218 |
| atencion_salida | xor | 48.3 ± 0.0 % | 98.2 ± 0.4 % | 🔴 peor | 52.9 | 3,266 / 1,218 |
| rbf_entrenable | xor | 97.7 ± 1.1 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 2,370 / 1,218 |
| knn_prototipos | xor | 64.7 ± 3.0 % | 98.2 ± 0.4 % | 🔴 peor | 67.3 | 66 / 1,218 |
| hmn_sin_memoria | xor | 96.7 ± 0.8 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,602 / 1,218 |
| hmn | xor | 90.8 ± 2.8 % | 98.2 ± 0.4 % | 🔴 peor | 99.6 | 2,242 / 1,218 |
| rmb_estable | xor | 98.2 ± 0.4 % | 98.2 ± 0.4 % | ⚪ igual | 99.9 | 3,330 / 1,218 |
| rmb | xor | 92.3 ± 5.2 % | 98.2 ± 0.4 % | ⚪ igual | 93.9 | 3,330 / 1,218 |
| resonante_suave | xor | 97.2 ± 0.5 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,222 / 1,218 |
| resonante | xor | 95.5 ± 1.7 % | 98.2 ± 0.4 % | ⚪ igual | 99.6 | 1,218 / 1,218 |
| hiper_relu_suave | xor | 97.8 ± 0.5 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,220 / 1,218 |
| hiper_relu | xor | 97.2 ± 0.5 % | 98.2 ± 0.4 % | 🔴 peor | 100.0 | 1,218 / 1,218 |
| polinomica_regularizada | xor | 97.7 ± 0.4 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 9,922 / 1,218 |
| polinomica | xor | 97.8 ± 0.5 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 34,306 / 1,218 |
| gelu (industria) | xor | 97.2 ± 0.5 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,218 / 1,218 |
| atencion_dinamica | serie_seno | 0.0261 ± 0.0011 | 0.0261 ± 0.0007 | ⚪ igual | 0.0173 | 3,617 / 1,505 |
| atencion_salida | serie_seno | 1.0124 ± 0.0011 | 0.0261 ± 0.0007 | 🔴 peor | 1.0000 | 3,553 / 1,505 |
| rbf_entrenable | serie_seno | 0.0246 ± 0.0002 | 0.0261 ± 0.0007 | 🟢 mejor | 0.0215 | 2,977 / 1,505 |
| knn_prototipos | serie_seno | 0.8168 ± 0.1425 | 0.0261 ± 0.0007 | 🔴 peor | 0.8101 | 33 / 1,505 |
| hmn_sin_memoria | serie_seno | 0.0244 ± 0.0003 | 0.0261 ± 0.0007 | 🟢 mejor | 0.0198 | 1,889 / 1,505 |
| hmn | serie_seno | 0.3737 ± 0.0365 | 0.0261 ± 0.0007 | 🔴 peor | 0.0220 | 2,529 / 1,505 |
| rmb_estable | serie_seno | 0.0253 ± 0.0009 | 0.0261 ± 0.0007 | ⚪ igual | 0.0195 | 3,617 / 1,505 |
| rmb | serie_seno | 0.0254 ± 0.0006 | 0.0261 ± 0.0007 | ⚪ igual | 0.0217 | 3,617 / 1,505 |
| resonante_suave | serie_seno | 0.0259 ± 0.0007 | 0.0261 ± 0.0007 | ⚪ igual | 0.0188 | 1,509 / 1,505 |
| resonante | serie_seno | 0.2800 ± 0.4110 | 0.0261 ± 0.0007 | ⚪ igual | 0.2836 | 1,505 / 1,505 |
| hiper_relu_suave | serie_seno | 0.0261 ± 0.0011 | 0.0261 ± 0.0007 | ⚪ igual | 0.0187 | 1,507 / 1,505 |
| hiper_relu | serie_seno | 0.0612 ± 0.0136 | 0.0261 ± 0.0007 | 🔴 peor | 0.0428 | 1,505 / 1,505 |
| polinomica_regularizada | serie_seno | 0.0299 ± 0.0018 | 0.0261 ± 0.0007 | ⚪ igual | 0.0131 | 12,769 / 1,505 |
| polinomica | serie_seno | 0.0293 ± 0.0015 | 0.0261 ± 0.0007 | 🔴 peor | 0.0140 | 39,073 / 1,505 |
| gelu (industria) | serie_seno | 0.0241 ± 0.0005 | 0.0261 ± 0.0007 | 🟢 mejor | 0.0199 | 1,505 / 1,505 |
| atencion_dinamica | digitos | 97.0 ± 0.4 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 5,578 / 3,466 |
| atencion_salida | digitos | 26.1 ± 8.0 % | 97.1 ± 0.5 % | 🔴 peor | 32.5 | 5,514 / 3,466 |
| rbf_entrenable | digitos | 96.9 ± 0.5 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 6,602 / 3,466 |
| knn_prototipos | digitos | 51.1 ± 5.1 % | 97.1 ± 0.5 % | 🔴 peor | 54.7 | 330 / 3,466 |
| hmn_sin_memoria | digitos | 95.8 ± 0.7 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,850 / 3,466 |
| hmn | digitos | 92.6 ± 1.7 % | 97.1 ± 0.5 % | 🔴 peor | 100.0 | 4,490 / 3,466 |
| rmb_estable | digitos | 97.3 ± 0.2 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 5,578 / 3,466 |
| rmb | digitos | 96.8 ± 0.8 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 5,578 / 3,466 |
| resonante_suave | digitos | 96.6 ± 0.5 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,470 / 3,466 |
| resonante | digitos | 94.0 ± 1.2 % | 97.1 ± 0.5 % | 🔴 peor | 99.9 | 3,466 / 3,466 |
| hiper_relu_suave | digitos | 96.9 ± 0.5 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,468 / 3,466 |
| hiper_relu | digitos | 95.1 ± 0.8 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |
| polinomica_regularizada | digitos | 97.1 ± 0.4 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 28,042 / 3,466 |
| polinomica | digitos | 91.9 ± 1.6 % | 97.1 ± 0.5 % | 🔴 peor | 97.9 | 167,498 / 3,466 |
| gelu (industria) | digitos | 96.6 ± 0.4 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |
| atencion_dinamica | caotica_20d | 69.7 ± 0.7 % | 70.5 ± 1.1 % | ⚪ igual | 100.0 | 3,972 / 1,860 |
| atencion_salida | caotica_20d | 71.6 ± 1.7 % | 70.5 ± 1.1 % | ⚪ igual | 80.2 | 3,908 / 1,860 |
| rbf_entrenable | caotica_20d | 76.9 ± 0.7 % | 70.5 ± 1.1 % | 🟢 mejor | 86.3 | 3,588 / 1,860 |
| knn_prototipos | caotica_20d | 60.4 ± 0.0 % | 70.5 ± 1.1 % | 🔴 peor | 63.4 | 132 / 1,860 |
| hmn_sin_memoria | caotica_20d | 67.8 ± 1.7 % | 70.5 ± 1.1 % | ⚪ igual | 99.7 | 2,244 / 1,860 |
| hmn | caotica_20d | 67.9 ± 3.4 % | 70.5 ± 1.1 % | ⚪ igual | 89.6 | 2,884 / 1,860 |
| rmb_estable | caotica_20d | 71.7 ± 1.7 % | 70.5 ± 1.1 % | ⚪ igual | 96.0 | 3,972 / 1,860 |
| rmb | caotica_20d | 71.4 ± 3.0 % | 70.5 ± 1.1 % | ⚪ igual | 88.0 | 3,972 / 1,860 |
| resonante_suave | caotica_20d | 68.9 ± 1.4 % | 70.5 ± 1.1 % | 🔴 peor | 100.0 | 1,864 / 1,860 |
| resonante | caotica_20d | 70.0 ± 1.1 % | 70.5 ± 1.1 % | ⚪ igual | 84.4 | 1,860 / 1,860 |
| hiper_relu_suave | caotica_20d | 69.2 ± 3.5 % | 70.5 ± 1.1 % | ⚪ igual | 95.8 | 1,862 / 1,860 |
| hiper_relu | caotica_20d | 68.8 ± 1.0 % | 70.5 ± 1.1 % | ⚪ igual | 97.0 | 1,860 / 1,860 |
| polinomica_regularizada | caotica_20d | 72.3 ± 0.6 % | 70.5 ± 1.1 % | ⚪ igual | 100.0 | 15,172 / 1,860 |
| polinomica | caotica_20d | 66.9 ± 1.0 % | 70.5 ± 1.1 % | 🔴 peor | 100.0 | 47,620 / 1,860 |
| gelu (industria) | caotica_20d | 70.7 ± 1.0 % | 70.5 ± 1.1 % | ⚪ igual | 100.0 | 1,860 / 1,860 |

**Resumen:** 🟢 mejor: 4, ⚪ igual: 69, 🔴 peor: 32, 💥 falla: 0

«Pieza en entreno» mucho mejor que «Pieza (prueba)» = la pieza memoriza en vez de aprender.
Las piezas marcadas «(industria)» (GELU, SiLU, Mish…) son la tecnología de hoy: si tu pieza no les gana a ellas, no conviene invertir en ella aunque le gane a ReLU.
Ojo: una pieza con muchos más parámetros que la estándar tiene ventaja; compárala también con una estándar más grande.

Para repetirlo:

```bash
python -m laboratorio.banco --pieza atencion_dinamica --pieza atencion_salida --pieza rbf_entrenable --pieza knn_prototipos \
  --pieza hmn_sin_memoria --pieza hmn --pieza rmb_estable --pieza rmb --pieza resonante_suave --pieza resonante \
  --pieza hiper_relu_suave --pieza hiper_relu --pieza polinomica_regularizada --pieza polinomica --pieza gelu --semillas 5
python -c "from laboratorio.banco.piezas_reparadas import competir_dann; print(competir_dann(5))"
```
