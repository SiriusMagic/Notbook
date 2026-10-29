# Informe del banco de evaluación

Cada celda: media ± desviación en **datos de prueba**, con 5 semillas. Exactitud: más es mejor. MSE: menos es mejor. Veredicto con prueba t pareada corregida por comparaciones múltiples (Benjamini-Hochberg, 5 %). Es una exploración: un 🟢 indica una pista que vale la pena confirmar, no una prueba definitiva.

| Pieza | Tarea | Pieza (prueba) | Estándar (prueba) | Veredicto | Pieza en entreno | Parámetros (pieza / estándar) |
|---|---|---|---|---|---|---|
| prelu_adaptativa | caotica_20d | 70.2 ± 0.9 % | 70.0 ± 0.9 % | ⚪ igual | 100.0 | 1,924 / 1,860 |
| integracion_no_lineal | caotica_20d | 75.7 ± 1.0 % | 70.0 ± 0.9 % | 🟢 mejor | 100.0 | 16,260 / 1,860 |
| activacion_adaptativa | caotica_20d | 69.6 ± 1.3 % | 70.0 ± 0.9 % | ⚪ igual | 96.8 | 1,988 / 1,860 |
| ida | caotica_20d | 69.0 ± 1.0 % | 70.0 ± 0.9 % | ⚪ igual | 100.0 | 2,052 / 1,860 |
| sma | caotica_20d | 69.4 ± 0.9 % | 70.0 ± 0.9 % | ⚪ igual | 98.2 | 1,860 / 1,860 |
| umbral_dinamico | caotica_20d | 67.0 ± 0.6 % | 70.0 ± 0.9 % | 🔴 peor | 99.9 | 1,860 / 1,860 |
| rmb | caotica_20d | 71.9 ± 2.6 % | 70.0 ± 0.9 % | ⚪ igual | 90.3 | 3,972 / 1,860 |
| hmn | caotica_20d | 67.3 ± 3.7 % | 70.0 ± 0.9 % | ⚪ igual | 89.2 | 2,884 / 1,860 |
| seno | caotica_20d | 64.5 ± 1.2 % | 70.0 ± 0.9 % | 🔴 peor | 100.0 | 1,860 / 1,860 |
| gelu (industria) | caotica_20d | 70.7 ± 1.0 % | 70.0 ± 0.9 % | ⚪ igual | 100.0 | 1,860 / 1,860 |
| silu (industria) | caotica_20d | 70.2 ± 0.6 % | 70.0 ± 0.9 % | ⚪ igual | 100.0 | 1,860 / 1,860 |
| mish (industria) | caotica_20d | 69.8 ± 0.8 % | 70.0 ± 0.9 % | ⚪ igual | 100.0 | 1,860 / 1,860 |
| elu (industria) | caotica_20d | 69.7 ± 0.9 % | 70.0 ± 0.9 % | ⚪ igual | 100.0 | 1,860 / 1,860 |
| leaky_relu (industria) | caotica_20d | 70.4 ± 0.9 % | 70.0 ± 0.9 % | ⚪ igual | 98.6 | 1,860 / 1,860 |
| tanh (industria) | caotica_20d | 65.7 ± 1.3 % | 70.0 ± 0.9 % | 🔴 peor | 99.9 | 1,860 / 1,860 |
| residual (industria) | caotica_20d | 70.1 ± 1.3 % | 70.0 ± 0.9 % | ⚪ igual | 100.0 | 2,500 / 1,860 |
| memoria_errores | caotica_20d | 66.3 ± 0.8 % | 70.0 ± 0.9 % | 🔴 peor | 100.0 | 1,860 / 1,860 |
| memoria_gradiente | caotica_20d | 66.5 ± 1.3 % | 70.0 ± 0.9 % | 🔴 peor | 99.9 | 1,860 / 1,860 |
| lr_por_neurona | caotica_20d | 67.2 ± 1.1 % | 70.0 ± 0.9 % | ⚪ igual | 99.7 | 1,860 / 1,860 |
| memoria_atencion | caotica_20d | 65.7 ± 1.5 % | 70.0 ± 0.9 % | 🔴 peor | 99.9 | 1,860 / 1,860 |
| memoria_embedding | caotica_20d | 71.7 ± 1.6 % | 70.0 ± 0.9 % | ⚪ igual | 94.0 | 4,100 / 1,860 |
| memoria_atencion_embedding | caotica_20d | 71.1 ± 2.0 % | 70.0 ± 0.9 % | ⚪ igual | 93.6 | 10,246 / 1,860 |
| inicio_perlin | caotica_20d | 69.7 ± 1.2 % | 70.0 ± 0.9 % | ⚪ igual | 97.5 | 1,860 / 1,860 |
| inicio_lorenz | caotica_20d | 71.7 ± 1.5 % | 70.0 ± 0.9 % | ⚪ igual | 94.5 | 1,860 / 1,860 |
| denso_cruzada | caotica_20d | 69.9 ± 0.6 % | 70.0 ± 0.9 % | ⚪ igual | 100.0 | 4,196 / 1,860 |
| densa_entrada | caotica_20d | 71.2 ± 0.2 % | 70.0 ± 0.9 % | ⚪ igual | 100.0 | 5,556 / 1,860 |
| dropout (industria) | caotica_20d | 77.4 ± 1.2 % | 70.0 ± 0.9 % | 🟢 mejor | 90.9 | 1,860 / 1,860 |
| batchnorm (industria) | caotica_20d | 69.6 ± 1.1 % | 70.0 ± 0.9 % | ⚪ igual | 100.0 | 1,988 / 1,860 |
| layernorm (industria) | caotica_20d | 69.3 ± 1.3 % | 70.0 ± 0.9 % | ⚪ igual | 100.0 | 1,988 / 1,860 |
| polinomica | caotica_20d | 66.9 ± 1.0 % | 70.0 ± 0.9 % | 🔴 peor | 100.0 | 47,620 / 1,860 |
| resonante | caotica_20d | 70.4 ± 1.2 % | 70.0 ± 0.9 % | ⚪ igual | 85.0 | 1,860 / 1,860 |
| dmna | caotica_20d | 72.2 ± 0.8 % | 70.0 ± 0.9 % | 🟢 mejor | 97.6 | 1,860 / 1,860 |
| gmo | caotica_20d | 70.0 ± 0.9 % | 70.0 ± 0.9 % | ⚪ igual | 98.4 | 1,860 / 1,860 |
| cdm | caotica_20d | 70.3 ± 1.0 % | 70.0 ± 0.9 % | ⚪ igual | 97.7 | 2,280 / 1,860 |
| hiper_relu | caotica_20d | 69.0 ± 0.8 % | 70.0 ± 0.9 % | ⚪ igual | 97.2 | 1,860 / 1,860 |
| norma_caos | caotica_20d | 70.0 ± 0.9 % | 70.0 ± 0.9 % | ⚪ igual | 100.0 | 2,052 / 1,860 |
| ccn | caotica_20d | 68.3 ± 1.5 % | 70.0 ± 0.9 % | ⚪ igual | 99.9 | 2,052 / 1,860 |
| dendritica | caotica_20d | 76.1 ± 0.9 % | 70.0 ± 0.9 % | 🟢 mejor | 82.0 | 1,860 / 1,860 |
| synaptic_tagging | caotica_20d | 65.7 ± 1.3 % | 70.0 ± 0.9 % | 🔴 peor | 99.9 | 1,860 / 1,860 |
| estado_interno | caotica_20d | 67.3 ± 1.1 % | 70.0 ± 0.9 % | ⚪ igual | 99.9 | 3,972 / 1,860 |
| ruteo_adaptativo | caotica_20d | 69.0 ± 1.0 % | 70.0 ± 0.9 % | ⚪ igual | 100.0 | 2,326 / 1,860 |
| knn_prototipos | caotica_20d | 60.4 ± 0.0 % | 70.0 ± 0.9 % | 🔴 peor | 63.4 | 132 / 1,860 |
| genetica | caotica_20d | 60.4 ± 0.0 % | 70.0 ± 0.9 % | 🔴 peor | 63.4 | 1,860 / 1,860 |
| genetica_arreglada | caotica_20d | 70.7 ± 1.5 % | 70.0 ± 0.9 % | ⚪ igual | 77.3 | 1,860 / 1,860 |
| genetica_invertida | caotica_20d | 72.2 ± 0.4 % | 70.0 ± 0.9 % | 🟢 mejor | 94.5 | 1,860 / 1,860 |
| decorrelacion | caotica_20d | 71.0 ± 1.5 % | 70.0 ± 0.9 % | ⚪ igual | 98.6 | 1,860 / 1,860 |
| papelito_pregunton | caotica_20d | 70.3 ± 1.1 % | 70.0 ± 0.9 % | ⚪ igual | 100.0 | 7,876 / 1,860 |
| papelito_covarianza | caotica_20d | 70.8 ± 0.6 % | 70.0 ± 0.9 % | ⚪ igual | 98.8 | 7,876 / 1,860 |
| fusion_reciproca | caotica_20d | 70.5 ± 0.8 % | 70.0 ± 0.9 % | ⚪ igual | 100.0 | 7,876 / 1,860 |
| atencion_salida | caotica_20d | 71.7 ± 1.6 % | 70.0 ± 0.9 % | ⚪ igual | 80.3 | 3,908 / 1,860 |
| atencion_local | caotica_20d | 72.4 ± 1.1 % | 70.0 ± 0.9 % | ⚪ igual | 94.1 | 3,524 / 1,860 |
| anti_gradiente | caotica_20d | 60.7 ± 0.3 % | 70.0 ± 0.9 % | 🔴 peor | 63.5 | 1,860 / 1,860 |

**Resumen:** 🟢 mejor: 5, ⚪ igual: 36, 🔴 peor: 11, 💥 falla: 0

«Pieza en entreno» mucho mejor que «Pieza (prueba)» = la pieza memoriza en vez de aprender.
Las piezas marcadas «(industria)» (GELU, SiLU, Mish…) son la tecnología de hoy: si tu pieza no les gana a ellas, no conviene invertir en ella aunque le gane a ReLU.
Ojo: una pieza con muchos más parámetros que la estándar tiene ventaja; compárala también con una estándar más grande.

---

# Aprendizaje clásico contra la red estándar

Media ± desviación en **datos de prueba**, 5 semillas. La red estándar es la del banco (2 capas de 32 neuronas, ReLU, Adam). Veredicto desde el punto de vista del modelo clásico: 🟢 = el clásico le gana a la red. Prueba t pareada con corrección de Benjamini-Hochberg.

## regresion_cubica

Red estándar: **0.0048 ± 0.0002**

| Modelo clásico | Prueba | Entrenamiento | vs red |
|---|---|---|---|
| Random Forest | 0.0074 ± 0.0001 | 0.0008 | 🔴 peor |
| Árbol de decisión (prof. 5) | 0.0091 ± 0.0000 | 0.0035 | 🔴 peor |
| KNN (k=5) | 0.0104 ± 0.0000 | 0.0033 | 🔴 peor |
| SVM (núcleo RBF) | 0.0182 ± 0.0000 | 0.0081 | 🔴 peor |
| Gradient Boosting | 0.0591 ± 0.0000 | 0.0286 | 🔴 peor |
| Ridge (lineal) | 0.3057 ± 0.0000 | 0.2300 | 🔴 peor |

## circulos

Red estándar: **84.8 ± 0.6 %**

| Modelo clásico | Prueba | Entrenamiento | vs red |
|---|---|---|---|
| SVM (núcleo RBF) | 86.1 ± 0.0 % | 81.4 | 🟢 mejor |
| Árbol de decisión (prof. 5) | 85.0 ± 0.0 % | 85.5 | ⚪ igual |
| Naive Bayes | 83.9 ± 0.0 % | 81.4 | 🔴 peor |
| Gradient Boosting | 83.9 ± 0.0 % | 95.2 | 🔴 peor |
| Votación ponderada (cuaderno 08) | 82.8 ± 0.0 % | 85.7 | 🔴 peor |
| Random Forest | 82.3 ± 0.6 % | 100.0 | 🔴 peor |
| KNN (k=5) | 81.1 ± 0.0 % | 85.5 | 🔴 peor |
| Regresión logística | 48.3 ± 0.0 % | 52.6 | 🔴 peor |

## lunas

Red estándar: **96.3 ± 0.3 %**

| Modelo clásico | Prueba | Entrenamiento | vs red |
|---|---|---|---|
| SVM (núcleo RBF) | 97.2 ± 0.0 % | 96.7 | 🟢 mejor |
| Random Forest | 96.1 ± 0.4 % | 100.0 | ⚪ igual |
| KNN (k=5) | 95.6 ± 0.0 % | 97.6 | 🔴 peor |
| Árbol de decisión (prof. 5) | 95.6 ± 0.0 % | 97.4 | 🔴 peor |
| Votación ponderada (cuaderno 08) | 95.0 ± 0.0 % | 93.1 | 🔴 peor |
| Gradient Boosting | 94.4 ± 0.0 % | 100.0 | 🔴 peor |
| Regresión logística | 90.0 ± 0.0 % | 85.0 | 🔴 peor |
| Naive Bayes | 88.9 ± 0.0 % | 84.8 | 🔴 peor |

## xor

Red estándar: **98.2 ± 0.4 %**

| Modelo clásico | Prueba | Entrenamiento | vs red |
|---|---|---|---|
| Gradient Boosting | 99.2 ± 0.0 % | 100.0 | 🟢 mejor |
| Random Forest | 98.3 ± 0.0 % | 100.0 | ⚪ igual |
| Árbol de decisión (prof. 5) | 97.5 ± 0.0 % | 99.6 | 🔴 peor |
| KNN (k=5) | 95.0 ± 0.0 % | 98.9 | 🔴 peor |
| SVM (núcleo RBF) | 95.0 ± 0.0 % | 98.2 | 🔴 peor |
| Votación ponderada (cuaderno 08) | 91.0 ± 0.4 % | 96.9 | 🔴 peor |
| Naive Bayes | 75.8 ± 0.0 % | 69.3 | 🔴 peor |
| Regresión logística | 66.7 ± 0.0 % | 67.5 | 🔴 peor |

## serie_seno

Red estándar: **0.0261 ± 0.0007**

| Modelo clásico | Prueba | Entrenamiento | vs red |
|---|---|---|---|
| SVM (núcleo RBF) | 0.0243 ± 0.0000 | 0.0200 | 🟢 mejor |
| Ridge (lineal) | 0.0252 ± 0.0000 | 0.0249 | 🟢 mejor |
| KNN (k=5) | 0.0267 ± 0.0000 | 0.0187 | ⚪ igual |
| Random Forest | 0.0324 ± 0.0004 | 0.0040 | 🔴 peor |
| Gradient Boosting | 0.0338 ± 0.0000 | 0.0042 | 🔴 peor |
| Árbol de decisión (prof. 5) | 0.0537 ± 0.0017 | 0.0196 | 🔴 peor |

## digitos

Red estándar: **97.1 ± 0.5 %**

| Modelo clásico | Prueba | Entrenamiento | vs red |
|---|---|---|---|
| Votación ponderada (cuaderno 08) | 98.3 ± 0.0 % | 99.8 | 🟢 mejor |
| Random Forest | 97.5 ± 0.1 % | 100.0 | ⚪ igual |
| Gradient Boosting | 97.2 ± 0.0 % | 100.0 | ⚪ igual |
| SVM (núcleo RBF) | 97.0 ± 0.0 % | 99.7 | ⚪ igual |
| Regresión logística | 96.9 ± 0.0 % | 99.8 | ⚪ igual |
| KNN (k=5) | 96.5 ± 0.0 % | 98.6 | 🔴 peor |
| Naive Bayes | 80.0 ± 0.0 % | 80.5 | 🔴 peor |
| Árbol de decisión (prof. 5) | 64.0 ± 0.4 % | 69.1 | 🔴 peor |

## caotica_20d

Red estándar: **70.0 ± 0.9 %**

| Modelo clásico | Prueba | Entrenamiento | vs red |
|---|---|---|---|
| Gradient Boosting | 94.2 ± 0.0 % | 100.0 | 🟢 mejor |
| Árbol de decisión (prof. 5) | 81.8 ± 0.1 % | 86.0 | 🟢 mejor |
| Random Forest | 81.7 ± 0.2 % | 100.0 | 🟢 mejor |
| Naive Bayes | 76.4 ± 0.0 % | 79.6 | 🟢 mejor |
| Votación ponderada (cuaderno 08) | 76.3 ± 0.0 % | 80.1 | 🟢 mejor |
| Regresión logística | 74.8 ± 0.0 % | 76.8 | 🟢 mejor |
| SVM (núcleo RBF) | 73.7 ± 0.0 % | 83.2 | 🟢 mejor |
| KNN (k=5) | 67.2 ± 0.0 % | 77.3 | 🔴 peor |

## Resumen

El modelo clásico le gana a la red en 13 de 52 comparaciones.
- **SVM (núcleo RBF)** en circulos
- **SVM (núcleo RBF)** en lunas
- **Gradient Boosting** en xor
- **Ridge (lineal)** en serie_seno
- **SVM (núcleo RBF)** en serie_seno
- **Votación ponderada (cuaderno 08)** en digitos
- **Regresión logística** en caotica_20d
- **SVM (núcleo RBF)** en caotica_20d
- **Árbol de decisión (prof. 5)** en caotica_20d
- **Naive Bayes** en caotica_20d
- **Random Forest** en caotica_20d
- **Gradient Boosting** en caotica_20d
- **Votación ponderada (cuaderno 08)** en caotica_20d

## Stacking (recuperado del cuaderno completo de Drive, celda 397)

Una regresión logística aprende a combinar las probabilidades de 5 modelos (logística, KNN, árbol, Random Forest y SVM lineal), entrenada con validación cruzada. 5 semillas, datos de prueba.

| Tarea | Red estándar | Votación ponderada (08) | Stacking | Gradient Boosting |
|---|---|---|---|---|
| dígitos | 97,1 % | 98,3 % | **98,5 %** | 97,2 % |
| caótica 20D | 70,0 % | 76,3 % | 83,6 % | **94,3 %** |
| XOR | 98,2 % | 91,0 % | 97,8 % | **99,2 %** |
| lunas | 96,3 % | 95,0 % | **96,3 %** | 94,4 % |
| círculos | **84,8 %** | 82,8 % | 82,9 % | 83,9 % |

- Aprender a combinar los votos funciona mejor que pesarlos por exactitud: el stacking supera a la votación ponderada en 4 de 5 tareas (en la caótica, 83,6 % contra 76,3 %).
- En dígitos es lo mejor de todo lo medido (98,5 %).
- En datos en tabla con reglas enredadas (caótica), Gradient Boosting sigue arriba.
