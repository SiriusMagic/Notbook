# Informe del banco de evaluación

Cada celda: media ± desviación en **datos de prueba**, con 5 semillas. Exactitud: más es mejor. MSE: menos es mejor. Veredicto con prueba t pareada corregida por comparaciones múltiples (Benjamini-Hochberg, 5 %). Es una exploración: un 🟢 indica una pista que vale la pena confirmar, no una prueba definitiva.

| Pieza | Tarea | Pieza (prueba) | Estándar (prueba) | Veredicto | Pieza en entreno | Parámetros (pieza / estándar) |
|---|---|---|---|---|---|---|
| anti_gradiente | regresion_cubica | 1.8133 ± 1.5726 | 0.0048 ± 0.0002 | 🔴 peor | 1.7369 | 1,153 / 1,153 |
| anti_gradiente | circulos | 61.2 ± 7.2 % | 84.8 ± 0.6 % | 🔴 peor | 62.0 | 1,218 / 1,218 |
| anti_gradiente | lunas | 67.0 ± 32.1 % | 96.3 ± 0.3 % | ⚪ igual | 64.8 | 1,218 / 1,218 |
| anti_gradiente | xor | 52.8 ± 26.5 % | 98.2 ± 0.4 % | 🔴 peor | 49.4 | 1,218 / 1,218 |
| anti_gradiente | serie_seno | 0.0499 ± 0.0191 | 0.0261 ± 0.0007 | 🔴 peor | 0.0522 | 1,505 / 1,505 |
| anti_gradiente | digitos | 37.3 ± 10.0 % | 97.1 ± 0.5 % | 🔴 peor | 39.3 | 3,466 / 3,466 |

**Resumen:** 🟢 mejor: 0, ⚪ igual: 1, 🔴 peor: 5, 💥 falla: 0

«Pieza en entreno» mucho mejor que «Pieza (prueba)» = la pieza memoriza en vez de aprender.
Las piezas marcadas «(industria)» (GELU, SiLU, Mish…) son la tecnología de hoy: si tu pieza no les gana a ellas, no conviene invertir en ella aunque le gane a ReLU.
Ojo: una pieza con muchos más parámetros que la estándar tiene ventaja; compárala también con una estándar más grande.
