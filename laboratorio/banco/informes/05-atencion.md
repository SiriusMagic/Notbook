# Informe del banco de evaluación

Cada celda: media ± desviación en **datos de prueba**, con 5 semillas. Exactitud: más es mejor. MSE: menos es mejor. Veredicto con prueba t pareada corregida por comparaciones múltiples (Benjamini-Hochberg, 5 %). Es una exploración: un 🟢 indica una pista que vale la pena confirmar, no una prueba definitiva.

| Pieza | Tarea | Pieza (prueba) | Estándar (prueba) | Veredicto | Pieza en entreno | Parámetros (pieza / estándar) |
|---|---|---|---|---|---|---|
| atencion_salida | regresion_cubica | 1.3182 ± 0.0000 | 0.0048 ± 0.0002 | 🔴 peor | 1.0000 | 3,201 / 1,153 |
| atencion_local | regresion_cubica | 0.0050 ± 0.0003 | 0.0048 ± 0.0002 | ⚪ igual | 0.0034 | 2,209 / 1,153 |
| atencion_salida | circulos | 50.0 ± 0.0 % | 84.8 ± 0.6 % | 🔴 peor | 50.0 | 3,266 / 1,218 |
| atencion_local | circulos | 84.7 ± 1.0 % | 84.8 ± 0.6 % | ⚪ igual | 84.6 | 2,306 / 1,218 |
| atencion_salida | lunas | 50.0 ± 0.0 % | 96.3 ± 0.3 % | 🔴 peor | 50.0 | 3,266 / 1,218 |
| atencion_local | lunas | 96.2 ± 0.2 % | 96.3 ± 0.3 % | ⚪ igual | 97.8 | 2,306 / 1,218 |
| atencion_salida | xor | 48.3 ± 0.0 % | 98.2 ± 0.4 % | 🔴 peor | 52.9 | 3,266 / 1,218 |
| atencion_local | xor | 97.5 ± 0.0 % | 98.2 ± 0.4 % | 🔴 peor | 100.0 | 2,306 / 1,218 |
| atencion_salida | serie_seno | 1.0124 ± 0.0011 | 0.0261 ± 0.0007 | 🔴 peor | 1.0000 | 3,553 / 1,505 |
| atencion_local | serie_seno | 0.0260 ± 0.0008 | 0.0261 ± 0.0007 | ⚪ igual | 0.0202 | 2,913 / 1,505 |
| atencion_salida | digitos | 26.1 ± 8.0 % | 97.1 ± 0.5 % | 🔴 peor | 32.5 | 5,514 / 3,466 |
| atencion_local | digitos | 97.0 ± 0.5 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 6,538 / 3,466 |

**Resumen:** 🟢 mejor: 0, ⚪ igual: 5, 🔴 peor: 7, 💥 falla: 0

«Pieza en entreno» mucho mejor que «Pieza (prueba)» = la pieza memoriza en vez de aprender.
Las piezas marcadas «(industria)» (GELU, SiLU, Mish…) son la tecnología de hoy: si tu pieza no les gana a ellas, no conviene invertir en ella aunque le gane a ReLU.
Ojo: una pieza con muchos más parámetros que la estándar tiene ventaja; compárala también con una estándar más grande.

## Transformer con tu máscara de atención (reseñas en español, 3 000 ejemplos, 3 semillas)

```
transformer [0.388, 0.347, 0.374] 617957
transformer_mascara_rl: 0.3787 ± 0.0106 (1142245 p.) vs transformer 0.3697 ± 0.0208 (617957 p.) → igual (p = 0.489)
transformer_mascara_rl_log: 0.3863 ± 0.0153 (1142245 p.) vs transformer 0.3697 ± 0.0208 (617957 p.) → igual (p = 0.352)
```
