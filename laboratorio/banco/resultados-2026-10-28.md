# Resultados del banco — 28 de octubre de 2026 (5 semillas, CPU)

Cada celda: media ± desviación en **datos de prueba**, con 5 semillas. Exactitud: más es mejor. MSE: menos es mejor. Veredicto con prueba t pareada (p < 0,05).

| Pieza | Tarea | Pieza (prueba) | Estándar (prueba) | Veredicto | Pieza en entreno | Parámetros (pieza / estándar) |
|---|---|---|---|---|---|---|
| prelu_adaptativa | regresion_cubica | 0.0047 ± 0.0001 | 0.0048 ± 0.0002 | ⚪ igual | 0.0031 | 1,217 / 1,153 |
| integracion_no_lineal | regresion_cubica | 0.0057 ± 0.0013 | 0.0048 ± 0.0002 | ⚪ igual | 0.0041 | 10,689 / 1,153 |
| activacion_adaptativa | regresion_cubica | 0.0050 ± 0.0002 | 0.0048 ± 0.0002 | ⚪ igual | 0.0033 | 1,281 / 1,153 |
| ida | regresion_cubica | 0.0054 ± 0.0011 | 0.0048 ± 0.0002 | ⚪ igual | 0.0040 | 1,345 / 1,153 |
| sma | regresion_cubica | 0.0049 ± 0.0002 | 0.0048 ± 0.0002 | ⚪ igual | 0.0032 | 1,153 / 1,153 |
| umbral_dinamico | regresion_cubica | 0.0069 ± 0.0007 | 0.0048 ± 0.0002 | 🔴 peor | 0.0042 | 1,153 / 1,153 |
| rmb | regresion_cubica | 2.7640 ± 5.4354 | 0.0048 ± 0.0002 | 🔴 peor | 2.6951 | 3,265 / 1,153 |
| hmn | regresion_cubica | 0.6934 ± 0.1247 | 0.0048 ± 0.0002 | 🔴 peor | 0.0349 | 2,177 / 1,153 |
| seno | regresion_cubica | 0.0048 ± 0.0003 | 0.0048 ± 0.0002 | ⚪ igual | 0.0037 | 1,153 / 1,153 |
| prelu_adaptativa | circulos | 83.7 ± 1.8 % | 84.8 ± 0.6 % | ⚪ igual | 85.2 | 1,282 / 1,218 |
| integracion_no_lineal | circulos | 81.7 ± 1.1 % | 84.8 ± 0.6 % | 🔴 peor | 87.6 | 11,010 / 1,218 |
| activacion_adaptativa | circulos | 84.8 ± 0.9 % | 84.8 ± 0.6 % | ⚪ igual | 83.5 | 1,346 / 1,218 |
| ida | circulos | 85.1 ± 0.7 % | 84.8 ± 0.6 % | ⚪ igual | 84.3 | 1,410 / 1,218 |
| sma | circulos | 85.2 ± 0.8 % | 84.8 ± 0.6 % | ⚪ igual | 84.2 | 1,218 / 1,218 |
| umbral_dinamico | circulos | 85.8 ± 1.3 % | 84.8 ± 0.6 % | ⚪ igual | 83.7 | 1,218 / 1,218 |
| rmb | circulos | 85.7 ± 1.0 % | 84.8 ± 0.6 % | ⚪ igual | 84.0 | 3,330 / 1,218 |
| hmn | circulos | 79.9 ± 1.4 % | 84.8 ± 0.6 % | 🔴 peor | 83.6 | 2,242 / 1,218 |
| seno | circulos | 81.2 ± 1.7 % | 84.8 ± 0.6 % | 🔴 peor | 85.7 | 1,218 / 1,218 |
| prelu_adaptativa | lunas | 96.2 ± 0.2 % | 96.3 ± 0.3 % | ⚪ igual | 98.0 | 1,282 / 1,218 |
| integracion_no_lineal | lunas | 96.7 ± 0.0 % | 96.3 ± 0.3 % | ⚪ igual | 98.0 | 11,010 / 1,218 |
| activacion_adaptativa | lunas | 96.3 ± 0.3 % | 96.3 ± 0.3 % | ⚪ igual | 97.6 | 1,346 / 1,218 |
| ida | lunas | 96.1 ± 0.4 % | 96.3 ± 0.3 % | ⚪ igual | 98.0 | 1,410 / 1,218 |
| sma | lunas | 96.7 ± 0.4 % | 96.3 ± 0.3 % | ⚪ igual | 97.2 | 1,218 / 1,218 |
| umbral_dinamico | lunas | 96.6 ± 0.2 % | 96.3 ± 0.3 % | ⚪ igual | 97.5 | 1,218 / 1,218 |
| rmb | lunas | 86.4 ± 18.5 % | 96.3 ± 0.3 % | ⚪ igual | 85.3 | 3,330 / 1,218 |
| hmn | lunas | 92.4 ± 2.2 % | 96.3 ± 0.3 % | 🔴 peor | 96.9 | 2,242 / 1,218 |
| seno | lunas | 96.1 ± 0.0 % | 96.3 ± 0.3 % | ⚪ igual | 97.9 | 1,218 / 1,218 |
| prelu_adaptativa | xor | 97.7 ± 0.4 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,282 / 1,218 |
| integracion_no_lineal | xor | 97.7 ± 0.4 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 11,010 / 1,218 |
| activacion_adaptativa | xor | 97.8 ± 0.5 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,346 / 1,218 |
| ida | xor | 97.5 ± 0.0 % | 98.2 ± 0.4 % | 🔴 peor | 100.0 | 1,410 / 1,218 |
| sma | xor | 97.7 ± 0.4 % | 98.2 ± 0.4 % | ⚪ igual | 99.9 | 1,218 / 1,218 |
| umbral_dinamico | xor | 96.8 ± 0.4 % | 98.2 ± 0.4 % | 🔴 peor | 100.0 | 1,218 / 1,218 |
| rmb | xor | 92.3 ± 5.2 % | 98.2 ± 0.4 % | ⚪ igual | 93.9 | 3,330 / 1,218 |
| hmn | xor | 90.8 ± 2.8 % | 98.2 ± 0.4 % | 🔴 peor | 99.6 | 2,242 / 1,218 |
| seno | xor | 95.7 ± 0.9 % | 98.2 ± 0.4 % | 🔴 peor | 100.0 | 1,218 / 1,218 |
| prelu_adaptativa | serie_seno | 0.0310 ± 0.0016 | 0.0332 ± 0.0011 | 🟢 mejor | 0.0182 | 1,569 / 1,505 |
| integracion_no_lineal | serie_seno | 0.0308 ± 0.0010 | 0.0332 ± 0.0011 | ⚪ igual | 0.0178 | 13,857 / 1,505 |
| activacion_adaptativa | serie_seno | 0.0290 ± 0.0006 | 0.0332 ± 0.0011 | 🟢 mejor | 0.0212 | 1,633 / 1,505 |
| ida | serie_seno | 0.0287 ± 0.0005 | 0.0332 ± 0.0011 | 🟢 mejor | 0.0204 | 1,697 / 1,505 |
| sma | serie_seno | 0.0307 ± 0.0038 | 0.0332 ± 0.0011 | ⚪ igual | 0.0229 | 1,505 / 1,505 |
| umbral_dinamico | serie_seno | 0.0280 ± 0.0003 | 0.0332 ± 0.0011 | 🟢 mejor | 0.0208 | 1,505 / 1,505 |
| rmb | serie_seno | 0.0290 ± 0.0007 | 0.0332 ± 0.0011 | 🟢 mejor | 0.0209 | 3,617 / 1,505 |
| hmn | serie_seno | 0.4152 ± 0.0436 | 0.0332 ± 0.0011 | 🔴 peor | 0.0215 | 2,529 / 1,505 |
| seno | serie_seno | 0.0286 ± 0.0009 | 0.0332 ± 0.0011 | 🟢 mejor | 0.0194 | 1,505 / 1,505 |
| prelu_adaptativa | digitos | 96.9 ± 0.3 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,530 / 3,466 |
| integracion_no_lineal | digitos | 97.3 ± 0.4 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 29,130 / 3,466 |
| activacion_adaptativa | digitos | 96.8 ± 0.7 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,594 / 3,466 |
| ida | digitos | 97.0 ± 0.5 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,658 / 3,466 |
| sma | digitos | 97.1 ± 0.5 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |
| umbral_dinamico | digitos | 97.3 ± 0.4 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |
| rmb | digitos | 96.8 ± 0.8 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 5,578 / 3,466 |
| hmn | digitos | 92.6 ± 1.7 % | 97.1 ± 0.5 % | 🔴 peor | 100.0 | 4,490 / 3,466 |
| seno | digitos | 97.1 ± 0.2 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |

**Resumen:** 🟢 mejor: 6, ⚪ igual: 35, 🔴 peor: 13, 💥 falla: 0

«Pieza en entreno» mucho mejor que «Pieza (prueba)» = la pieza memoriza en vez de aprender.
Ojo: una pieza con muchos más parámetros que la estándar tiene ventaja; compárala también con una estándar más grande.
