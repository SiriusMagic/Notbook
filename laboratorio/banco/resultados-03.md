# Informe del banco de evaluación

Cada celda: media ± desviación en **datos de prueba**, con 5 semillas. Exactitud: más es mejor. MSE: menos es mejor. Veredicto con prueba t pareada corregida por comparaciones múltiples (Benjamini-Hochberg, 5 %). Es una exploración: un 🟢 indica una pista que vale la pena confirmar, no una prueba definitiva.

| Pieza | Tarea | Pieza (prueba) | Estándar (prueba) | Veredicto | Pieza en entreno | Parámetros (pieza / estándar) |
|---|---|---|---|---|---|---|
| knn_prototipos | regresion_cubica | 1.0803 ± 0.0821 | 0.0048 ± 0.0002 | 🔴 peor | 0.7792 | 33 / 1,153 |
| genetica | regresion_cubica | 1.3182 ± 0.0000 | 0.0048 ± 0.0002 | 🔴 peor | 1.0000 | 1,153 / 1,153 |
| genetica_arreglada | regresion_cubica | 0.8164 ± 0.3134 | 0.0048 ± 0.0002 | 🔴 peor | 0.5891 | 1,153 / 1,153 |
| genetica_invertida | regresion_cubica | 0.0052 ± 0.0008 | 0.0048 ± 0.0002 | ⚪ igual | 0.0035 | 1,153 / 1,153 |
| knn_prototipos | circulos | 77.8 ± 2.2 % | 84.8 ± 0.6 % | 🔴 peor | 76.7 | 66 / 1,218 |
| genetica | circulos | 50.0 ± 0.0 % | 84.8 ± 0.6 % | 🔴 peor | 50.0 | 1,218 / 1,218 |
| genetica_arreglada | circulos | 61.4 ± 11.4 % | 84.8 ± 0.6 % | 🔴 peor | 62.0 | 1,218 / 1,218 |
| genetica_invertida | circulos | 85.2 ± 0.7 % | 84.8 ± 0.6 % | ⚪ igual | 84.4 | 1,218 / 1,218 |
| knn_prototipos | lunas | 90.7 ± 1.8 % | 96.3 ± 0.3 % | 🔴 peor | 87.3 | 66 / 1,218 |
| genetica | lunas | 50.0 ± 0.0 % | 96.3 ± 0.3 % | 🔴 peor | 50.0 | 1,218 / 1,218 |
| genetica_arreglada | lunas | 86.2 ± 5.1 % | 96.3 ± 0.3 % | 🔴 peor | 83.8 | 1,218 / 1,218 |
| genetica_invertida | lunas | 96.1 ± 0.0 % | 96.3 ± 0.3 % | ⚪ igual | 97.9 | 1,218 / 1,218 |
| knn_prototipos | xor | 64.7 ± 3.0 % | 98.2 ± 0.4 % | 🔴 peor | 67.3 | 66 / 1,218 |
| genetica | xor | 48.3 ± 0.0 % | 98.2 ± 0.4 % | 🔴 peor | 52.9 | 1,218 / 1,218 |
| genetica_arreglada | xor | 67.7 ± 4.8 % | 98.2 ± 0.4 % | 🔴 peor | 66.0 | 1,218 / 1,218 |
| genetica_invertida | xor | 98.3 ± 0.6 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,218 / 1,218 |
| knn_prototipos | serie_seno | 0.8168 ± 0.1425 | 0.0261 ± 0.0007 | 🔴 peor | 0.8101 | 33 / 1,505 |
| genetica | serie_seno | 1.0119 ± 0.0000 | 0.0261 ± 0.0007 | 🔴 peor | 1.0000 | 1,505 / 1,505 |
| genetica_arreglada | serie_seno | 0.2535 ± 0.1735 | 0.0261 ± 0.0007 | 🔴 peor | 0.2372 | 1,505 / 1,505 |
| genetica_invertida | serie_seno | 0.0258 ± 0.0004 | 0.0261 ± 0.0007 | ⚪ igual | 0.0193 | 1,505 / 1,505 |
| knn_prototipos | digitos | 51.1 ± 5.1 % | 97.1 ± 0.5 % | 🔴 peor | 54.7 | 330 / 3,466 |
| genetica | digitos | 19.4 ± 18.1 % | 97.1 ± 0.5 % | 🔴 peor | 22.4 | 3,466 / 3,466 |
| genetica_arreglada | digitos | 86.7 ± 7.3 % | 97.1 ± 0.5 % | 🔴 peor | 90.4 | 3,466 / 3,466 |
| genetica_invertida | digitos | 97.1 ± 0.3 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |

**Resumen:** 🟢 mejor: 0, ⚪ igual: 6, 🔴 peor: 18, 💥 falla: 0

«Pieza en entreno» mucho mejor que «Pieza (prueba)» = la pieza memoriza en vez de aprender.
Las piezas marcadas «(industria)» (GELU, SiLU, Mish…) son la tecnología de hoy: si tu pieza no les gana a ellas, no conviene invertir en ella aunque le gane a ReLU.
Ojo: una pieza con muchos más parámetros que la estándar tiene ventaja; compárala también con una estándar más grande.
