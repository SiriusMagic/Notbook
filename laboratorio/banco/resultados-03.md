# Informe del banco de evaluación

Cada celda: media ± desviación en **datos de prueba**, con 5 semillas. Exactitud: más es mejor. MSE: menos es mejor. Veredicto con prueba t pareada corregida por comparaciones múltiples (Benjamini-Hochberg, 5 %). Es una exploración: un 🟢 indica una pista que vale la pena confirmar, no una prueba definitiva.

| Pieza | Tarea | Pieza (prueba) | Estándar (prueba) | Veredicto | Pieza en entreno | Parámetros (pieza / estándar) |
|---|---|---|---|---|---|---|
| knn_prototipos | regresion_cubica | 1.0803 ± 0.0821 | 0.0048 ± 0.0002 | 🔴 peor | 0.7792 | 33 / 1,153 |
| genetica | regresion_cubica | 1.3182 ± 0.0000 | 0.0048 ± 0.0002 | 🔴 peor | 1.0000 | 1,153 / 1,153 |
| genetica_arreglada | regresion_cubica | 0.6877 ± 0.0590 | 0.0048 ± 0.0002 | 🔴 peor | 0.5083 | 1,153 / 1,153 |
| genetica_invertida | regresion_cubica | 0.0049 ± 0.0004 | 0.0048 ± 0.0002 | ⚪ igual | 0.0033 | 1,153 / 1,153 |
| knn_prototipos | circulos | 77.8 ± 2.2 % | 84.8 ± 0.6 % | 🔴 peor | 76.7 | 66 / 1,218 |
| genetica | circulos | 50.0 ± 0.0 % | 84.8 ± 0.6 % | 🔴 peor | 50.0 | 1,218 / 1,218 |
| genetica_arreglada | circulos | 72.3 ± 6.8 % | 84.8 ± 0.6 % | 🔴 peor | 71.9 | 1,218 / 1,218 |
| genetica_invertida | circulos | 85.1 ± 0.9 % | 84.8 ± 0.6 % | ⚪ igual | 84.1 | 1,218 / 1,218 |
| knn_prototipos | lunas | 90.7 ± 1.8 % | 96.3 ± 0.3 % | 🔴 peor | 87.3 | 66 / 1,218 |
| genetica | lunas | 50.0 ± 0.0 % | 96.3 ± 0.3 % | 🔴 peor | 50.0 | 1,218 / 1,218 |
| genetica_arreglada | lunas | 88.0 ± 4.8 % | 96.3 ± 0.3 % | 🔴 peor | 86.9 | 1,218 / 1,218 |
| genetica_invertida | lunas | 96.4 ± 0.3 % | 96.3 ± 0.3 % | ⚪ igual | 97.8 | 1,218 / 1,218 |
| knn_prototipos | xor | 64.7 ± 3.0 % | 98.2 ± 0.4 % | 🔴 peor | 67.3 | 66 / 1,218 |
| genetica | xor | 48.3 ± 0.0 % | 98.2 ± 0.4 % | 🔴 peor | 52.9 | 1,218 / 1,218 |
| genetica_arreglada | xor | 62.8 ± 9.5 % | 98.2 ± 0.4 % | 🔴 peor | 64.2 | 1,218 / 1,218 |
| genetica_invertida | xor | 98.3 ± 0.6 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,218 / 1,218 |
| knn_prototipos | serie_seno | 0.8168 ± 0.1425 | 0.0261 ± 0.0007 | 🔴 peor | 0.8101 | 33 / 1,505 |
| genetica | serie_seno | 1.0118 ± 0.0000 | 0.0261 ± 0.0007 | 🔴 peor | 1.0000 | 1,505 / 1,505 |
| genetica_arreglada | serie_seno | 0.4654 ± 0.2111 | 0.0261 ± 0.0007 | 🔴 peor | 0.4796 | 1,505 / 1,505 |
| genetica_invertida | serie_seno | 0.0257 ± 0.0007 | 0.0261 ± 0.0007 | ⚪ igual | 0.0191 | 1,505 / 1,505 |
| knn_prototipos | digitos | 51.1 ± 5.1 % | 97.1 ± 0.5 % | 🔴 peor | 54.7 | 330 / 3,466 |
| genetica | digitos | 21.1 ± 18.5 % | 97.1 ± 0.5 % | 🔴 peor | 23.4 | 3,466 / 3,466 |
| genetica_arreglada | digitos | 92.8 ± 3.7 % | 97.1 ± 0.5 % | ⚪ igual | 96.4 | 3,466 / 3,466 |
| genetica_invertida | digitos | 97.1 ± 0.2 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |

**Resumen:** 🟢 mejor: 0, ⚪ igual: 7, 🔴 peor: 17, 💥 falla: 0

«Pieza en entreno» mucho mejor que «Pieza (prueba)» = la pieza memoriza en vez de aprender.
Las piezas marcadas «(industria)» (GELU, SiLU, Mish…) son la tecnología de hoy: si tu pieza no les gana a ellas, no conviene invertir en ella aunque le gane a ReLU.
Ojo: una pieza con muchos más parámetros que la estándar tiene ventaja; compárala también con una estándar más grande.
