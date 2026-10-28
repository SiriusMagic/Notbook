# Informe del banco de evaluación

Cada celda: media ± desviación en **datos de prueba**, con 5 semillas. Exactitud: más es mejor. MSE: menos es mejor. Veredicto con prueba t pareada corregida por comparaciones múltiples (Benjamini-Hochberg, 5 %). Es una exploración: un 🟢 indica una pista que vale la pena confirmar, no una prueba definitiva.

| Pieza | Tarea | Pieza (prueba) | Estándar (prueba) | Veredicto | Pieza en entreno | Parámetros (pieza / estándar) |
|---|---|---|---|---|---|---|
| dropout (industria) | regresion_cubica | 0.0147 ± 0.0065 | 0.0048 ± 0.0002 | 🔴 peor | 0.0093 | 1,153 / 1,153 |
| batchnorm (industria) | regresion_cubica | 0.0049 ± 0.0003 | 0.0048 ± 0.0002 | ⚪ igual | 0.0032 | 1,281 / 1,153 |
| layernorm (industria) | regresion_cubica | 0.0050 ± 0.0001 | 0.0048 ± 0.0002 | ⚪ igual | 0.0031 | 1,281 / 1,153 |
| polinomica | regresion_cubica | 0.0046 ± 0.0001 | 0.0048 ± 0.0002 | ⚪ igual | 0.0031 | 34,145 / 1,153 |
| resonante | regresion_cubica | 0.4564 ± 0.2568 | 0.0048 ± 0.0002 | 🔴 peor | 0.3752 | 1,153 / 1,153 |
| dmna | regresion_cubica | 0.0046 ± 0.0001 | 0.0048 ± 0.0002 | ⚪ igual | 0.0031 | 1,153 / 1,153 |
| gmo | regresion_cubica | 0.0048 ± 0.0002 | 0.0048 ± 0.0002 | ⚪ igual | 0.0032 | 1,153 / 1,153 |
| cdm | regresion_cubica | 0.0048 ± 0.0001 | 0.0048 ± 0.0002 | ⚪ igual | 0.0032 | 1,155 / 1,153 |
| hiper_relu | regresion_cubica | 0.0135 ± 0.0023 | 0.0048 ± 0.0002 | 🔴 peor | 0.0099 | 1,153 / 1,153 |
| norma_caos | regresion_cubica | 0.0056 ± 0.0017 | 0.0048 ± 0.0002 | ⚪ igual | 0.0032 | 1,345 / 1,153 |
| ccn | regresion_cubica | 0.0188 ± 0.0155 | 0.0048 ± 0.0002 | ⚪ igual | 0.0138 | 1,345 / 1,153 |
| dropout (industria) | circulos | 85.2 ± 1.0 % | 84.8 ± 0.6 % | ⚪ igual | 84.4 | 1,218 / 1,218 |
| batchnorm (industria) | circulos | 79.6 ± 2.4 % | 84.8 ± 0.6 % | 🔴 peor | 90.0 | 1,346 / 1,218 |
| layernorm (industria) | circulos | 82.6 ± 0.9 % | 84.8 ± 0.6 % | ⚪ igual | 86.4 | 1,346 / 1,218 |
| polinomica | circulos | 78.8 ± 1.2 % | 84.8 ± 0.6 % | 🔴 peor | 93.3 | 34,306 / 1,218 |
| resonante | circulos | 82.7 ± 1.2 % | 84.8 ± 0.6 % | ⚪ igual | 84.0 | 1,218 / 1,218 |
| dmna | circulos | 86.2 ± 0.5 % | 84.8 ± 0.6 % | ⚪ igual | 83.2 | 1,218 / 1,218 |
| gmo | circulos | 84.8 ± 0.6 % | 84.8 ± 0.6 % | ⚪ igual | 84.4 | 1,218 / 1,218 |
| cdm | circulos | 84.6 ± 1.1 % | 84.8 ± 0.6 % | ⚪ igual | 84.8 | 1,224 / 1,218 |
| hiper_relu | circulos | 83.6 ± 2.4 % | 84.8 ± 0.6 % | ⚪ igual | 85.6 | 1,218 / 1,218 |
| norma_caos | circulos | 79.2 ± 1.9 % | 84.8 ± 0.6 % | 🔴 peor | 89.5 | 1,410 / 1,218 |
| ccn | circulos | 81.9 ± 2.0 % | 84.8 ± 0.6 % | ⚪ igual | 87.2 | 1,410 / 1,218 |
| dropout (industria) | lunas | 96.7 ± 0.0 % | 96.3 ± 0.3 % | ⚪ igual | 98.4 | 1,218 / 1,218 |
| batchnorm (industria) | lunas | 96.0 ± 1.2 % | 96.3 ± 0.3 % | ⚪ igual | 97.9 | 1,346 / 1,218 |
| layernorm (industria) | lunas | 96.3 ± 0.3 % | 96.3 ± 0.3 % | ⚪ igual | 98.3 | 1,346 / 1,218 |
| polinomica | lunas | 96.4 ± 0.3 % | 96.3 ± 0.3 % | ⚪ igual | 98.3 | 34,306 / 1,218 |
| resonante | lunas | 96.9 ± 0.3 % | 96.3 ± 0.3 % | ⚪ igual | 97.5 | 1,218 / 1,218 |
| dmna | lunas | 96.3 ± 0.3 % | 96.3 ± 0.3 % | ⚪ igual | 97.9 | 1,218 / 1,218 |
| gmo | lunas | 96.3 ± 0.3 % | 96.3 ± 0.3 % | ⚪ igual | 97.7 | 1,218 / 1,218 |
| cdm | lunas | 96.2 ± 0.2 % | 96.3 ± 0.3 % | ⚪ igual | 97.9 | 1,224 / 1,218 |
| hiper_relu | lunas | 96.3 ± 0.5 % | 96.3 ± 0.3 % | ⚪ igual | 98.3 | 1,218 / 1,218 |
| norma_caos | lunas | 95.9 ± 1.2 % | 96.3 ± 0.3 % | ⚪ igual | 97.9 | 1,410 / 1,218 |
| ccn | lunas | 96.4 ± 0.5 % | 96.3 ± 0.3 % | ⚪ igual | 98.2 | 1,410 / 1,218 |
| dropout (industria) | xor | 97.3 ± 1.2 % | 98.2 ± 0.4 % | ⚪ igual | 99.8 | 1,218 / 1,218 |
| batchnorm (industria) | xor | 97.3 ± 0.7 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,346 / 1,218 |
| layernorm (industria) | xor | 97.5 ± 0.6 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,346 / 1,218 |
| polinomica | xor | 97.8 ± 0.5 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 34,306 / 1,218 |
| resonante | xor | 95.5 ± 1.7 % | 98.2 ± 0.4 % | ⚪ igual | 99.6 | 1,218 / 1,218 |
| dmna | xor | 97.7 ± 0.4 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,218 / 1,218 |
| gmo | xor | 98.2 ± 0.4 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,218 / 1,218 |
| cdm | xor | 97.8 ± 0.5 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,224 / 1,218 |
| hiper_relu | xor | 97.2 ± 0.5 % | 98.2 ± 0.4 % | 🔴 peor | 100.0 | 1,218 / 1,218 |
| norma_caos | xor | 97.2 ± 0.7 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,410 / 1,218 |
| ccn | xor | 96.3 ± 1.0 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,410 / 1,218 |
| dropout (industria) | serie_seno | 0.0277 ± 0.0013 | 0.0261 ± 0.0007 | ⚪ igual | 0.0216 | 1,505 / 1,505 |
| batchnorm (industria) | serie_seno | 0.0311 ± 0.0042 | 0.0261 ± 0.0007 | ⚪ igual | 0.0142 | 1,633 / 1,505 |
| layernorm (industria) | serie_seno | 0.0250 ± 0.0010 | 0.0261 ± 0.0007 | ⚪ igual | 0.0183 | 1,633 / 1,505 |
| polinomica | serie_seno | 0.0293 ± 0.0015 | 0.0261 ± 0.0007 | 🔴 peor | 0.0140 | 39,073 / 1,505 |
| resonante | serie_seno | 0.2800 ± 0.4110 | 0.0261 ± 0.0007 | ⚪ igual | 0.2836 | 1,505 / 1,505 |
| dmna | serie_seno | 0.0253 ± 0.0006 | 0.0261 ± 0.0007 | ⚪ igual | 0.0188 | 1,505 / 1,505 |
| gmo | serie_seno | 0.0261 ± 0.0007 | 0.0261 ± 0.0007 | ⚪ igual | 0.0186 | 1,505 / 1,505 |
| cdm | serie_seno | 0.0258 ± 0.0011 | 0.0261 ± 0.0007 | ⚪ igual | 0.0190 | 1,661 / 1,505 |
| hiper_relu | serie_seno | 0.0612 ± 0.0136 | 0.0261 ± 0.0007 | 🔴 peor | 0.0428 | 1,505 / 1,505 |
| norma_caos | serie_seno | 0.0311 ± 0.0044 | 0.0261 ± 0.0007 | ⚪ igual | 0.0135 | 1,697 / 1,505 |
| ccn | serie_seno | 0.0314 ± 0.0027 | 0.0261 ± 0.0007 | 🔴 peor | 0.0138 | 1,697 / 1,505 |
| dropout (industria) | digitos | 97.1 ± 0.5 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |
| batchnorm (industria) | digitos | 97.4 ± 0.4 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,594 / 3,466 |
| layernorm (industria) | digitos | 96.7 ± 0.4 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,594 / 3,466 |
| polinomica | digitos | 90.3 ± 1.0 % | 97.1 ± 0.5 % | 🔴 peor | 97.6 | 167,498 / 3,466 |
| resonante | digitos | 93.5 ± 1.4 % | 97.1 ± 0.5 % | 🔴 peor | 99.8 | 3,466 / 3,466 |
| dmna | digitos | 97.0 ± 0.5 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |
| gmo | digitos | 97.1 ± 0.5 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |
| cdm | digitos | 96.4 ± 0.2 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 7,626 / 3,466 |
| hiper_relu | digitos | 95.1 ± 0.8 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |
| norma_caos | digitos | 97.4 ± 0.3 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,658 / 3,466 |
| ccn | digitos | 97.4 ± 0.5 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,658 / 3,466 |

**Resumen:** 🟢 mejor: 0, ⚪ igual: 54, 🔴 peor: 12, 💥 falla: 0

«Pieza en entreno» mucho mejor que «Pieza (prueba)» = la pieza memoriza en vez de aprender.
Las piezas marcadas «(industria)» (GELU, SiLU, Mish…) son la tecnología de hoy: si tu pieza no les gana a ellas, no conviene invertir en ella aunque le gane a ReLU.
Ojo: una pieza con muchos más parámetros que la estándar tiene ventaja; compárala también con una estándar más grande.
