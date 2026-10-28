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
