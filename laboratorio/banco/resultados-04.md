# Informe del banco de evaluación

Cada celda: media ± desviación en **datos de prueba**, con 5 semillas. Exactitud: más es mejor. MSE: menos es mejor. Veredicto con prueba t pareada corregida por comparaciones múltiples (Benjamini-Hochberg, 5 %). Es una exploración: un 🟢 indica una pista que vale la pena confirmar, no una prueba definitiva.

| Pieza | Tarea | Pieza (prueba) | Estándar (prueba) | Veredicto | Pieza en entreno | Parámetros (pieza / estándar) |
|---|---|---|---|---|---|---|
| decorrelacion | regresion_cubica | 0.0049 ± 0.0002 | 0.0048 ± 0.0002 | ⚪ igual | 0.0033 | 1,153 / 1,153 |
| papelito_pregunton | regresion_cubica | 0.0051 ± 0.0005 | 0.0048 ± 0.0002 | ⚪ igual | 0.0033 | 6,465 / 1,153 |
| papelito_covarianza | regresion_cubica | 0.0048 ± 0.0001 | 0.0048 ± 0.0002 | ⚪ igual | 0.0031 | 6,465 / 1,153 |
| fusion_reciproca | regresion_cubica | 0.0053 ± 0.0007 | 0.0048 ± 0.0002 | ⚪ igual | 0.0033 | 6,465 / 1,153 |
| decorrelacion | circulos | 84.2 ± 0.5 % | 84.8 ± 0.6 % | ⚪ igual | 84.5 | 1,218 / 1,218 |
| papelito_pregunton | circulos | 79.4 ± 3.6 % | 84.8 ± 0.6 % | ⚪ igual | 88.0 | 6,594 / 1,218 |
| papelito_covarianza | circulos | 80.0 ± 1.0 % | 84.8 ± 0.6 % | 🔴 peor | 87.6 | 6,594 / 1,218 |
| fusion_reciproca | circulos | 79.1 ± 1.3 % | 84.8 ± 0.6 % | 🔴 peor | 89.4 | 6,594 / 1,218 |
| decorrelacion | lunas | 96.3 ± 0.3 % | 96.3 ± 0.3 % | ⚪ igual | 97.9 | 1,218 / 1,218 |
| papelito_pregunton | lunas | 96.1 ± 0.6 % | 96.3 ± 0.3 % | ⚪ igual | 98.1 | 6,594 / 1,218 |
| papelito_covarianza | lunas | 96.4 ± 0.3 % | 96.3 ± 0.3 % | ⚪ igual | 98.8 | 6,594 / 1,218 |
| fusion_reciproca | lunas | 96.1 ± 0.4 % | 96.3 ± 0.3 % | ⚪ igual | 98.1 | 6,594 / 1,218 |
| decorrelacion | xor | 97.8 ± 0.5 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,218 / 1,218 |
| papelito_pregunton | xor | 97.7 ± 0.4 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 6,594 / 1,218 |
| papelito_covarianza | xor | 97.3 ± 0.7 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 6,594 / 1,218 |
| fusion_reciproca | xor | 97.7 ± 0.4 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 6,594 / 1,218 |
| decorrelacion | serie_seno | 0.0274 ± 0.0009 | 0.0261 ± 0.0007 | ⚪ igual | 0.0164 | 1,505 / 1,505 |
| papelito_pregunton | serie_seno | 0.0269 ± 0.0013 | 0.0261 ± 0.0007 | ⚪ igual | 0.0156 | 7,169 / 1,505 |
| papelito_covarianza | serie_seno | 0.0293 ± 0.0025 | 0.0261 ± 0.0007 | ⚪ igual | 0.0126 | 7,169 / 1,505 |
| fusion_reciproca | serie_seno | 0.0282 ± 0.0024 | 0.0261 ± 0.0007 | ⚪ igual | 0.0149 | 7,169 / 1,505 |
| decorrelacion | digitos | 97.2 ± 0.5 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |
| papelito_pregunton | digitos | 97.0 ± 0.5 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 11,082 / 3,466 |
| papelito_covarianza | digitos | 97.5 ± 0.3 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 11,082 / 3,466 |
| fusion_reciproca | digitos | 97.1 ± 0.6 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 11,082 / 3,466 |

**Resumen:** 🟢 mejor: 0, ⚪ igual: 22, 🔴 peor: 2, 💥 falla: 0

«Pieza en entreno» mucho mejor que «Pieza (prueba)» = la pieza memoriza en vez de aprender.
Las piezas marcadas «(industria)» (GELU, SiLU, Mish…) son la tecnología de hoy: si tu pieza no les gana a ellas, no conviene invertir en ella aunque le gane a ReLU.
Ojo: una pieza con muchos más parámetros que la estándar tiene ventaja; compárala también con una estándar más grande.

## Arquitecturas de investigación contra las normales (5 semillas)

```
digitos: cnn_decorrelada 0.9707 ± 0.0092 (28618 p.) vs cnn 0.9837 ± 0.0040 (28618 p.) → peor (p = 0.029)
serie_seno: rcf_recurrente 0.0282 ± 0.0013 (13729 p.) vs lstm 0.0256 ± 0.0004 (20321 p.) → peor (p = 0.010)
serie_seno: rcf_recurrente 0.0282 ± 0.0013 (13729 p.) vs serie_cnn 0.0266 ± 0.0009 (18753 p.) → igual (p = 0.124)
```
