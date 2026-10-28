# Informe del banco de evaluación

Cada celda: media ± desviación en **datos de prueba**, con 5 semillas. Exactitud: más es mejor. MSE: menos es mejor. Veredicto con prueba t pareada corregida por comparaciones múltiples (Benjamini-Hochberg, 5 %). Es una exploración: un 🟢 indica una pista que vale la pena confirmar, no una prueba definitiva.

| Pieza | Tarea | Pieza (prueba) | Estándar (prueba) | Veredicto | Pieza en entreno | Parámetros (pieza / estándar) |
|---|---|---|---|---|---|---|
| gelu (industria) | regresion_cubica | 0.0046 ± 0.0001 | 0.0048 ± 0.0002 | ⚪ igual | 0.0035 | 1,153 / 1,153 |
| silu (industria) | regresion_cubica | 0.0045 ± 0.0001 | 0.0048 ± 0.0002 | ⚪ igual | 0.0034 | 1,153 / 1,153 |
| mish (industria) | regresion_cubica | 0.0046 ± 0.0001 | 0.0048 ± 0.0002 | ⚪ igual | 0.0035 | 1,153 / 1,153 |
| elu (industria) | regresion_cubica | 0.0055 ± 0.0012 | 0.0048 ± 0.0002 | ⚪ igual | 0.0044 | 1,153 / 1,153 |
| leaky_relu (industria) | regresion_cubica | 0.0048 ± 0.0004 | 0.0048 ± 0.0002 | ⚪ igual | 0.0033 | 1,153 / 1,153 |
| tanh (industria) | regresion_cubica | 0.0048 ± 0.0000 | 0.0048 ± 0.0002 | ⚪ igual | 0.0034 | 1,153 / 1,153 |
| residual (industria) | regresion_cubica | 0.0047 ± 0.0002 | 0.0048 ± 0.0002 | ⚪ igual | 0.0033 | 1,185 / 1,153 |
| memoria_errores | regresion_cubica | 0.0152 ± 0.0052 | 0.0048 ± 0.0002 | 🔴 peor | 0.0085 | 1,153 / 1,153 |
| memoria_gradiente | regresion_cubica | 0.0048 ± 0.0001 | 0.0048 ± 0.0002 | ⚪ igual | 0.0034 | 1,153 / 1,153 |
| lr_por_neurona | regresion_cubica | 0.0121 ± 0.0012 | 0.0048 ± 0.0002 | 🔴 peor | 0.0083 | 1,153 / 1,153 |
| memoria_atencion | regresion_cubica | 0.0050 ± 0.0001 | 0.0048 ± 0.0002 | ⚪ igual | 0.0035 | 1,153 / 1,153 |
| memoria_embedding | regresion_cubica | 0.0051 ± 0.0003 | 0.0048 ± 0.0002 | ⚪ igual | 0.0034 | 3,393 / 1,153 |
| memoria_atencion_embedding | regresion_cubica | 0.0048 ± 0.0002 | 0.0048 ± 0.0002 | ⚪ igual | 0.0033 | 9,539 / 1,153 |
| denso_cruzada | regresion_cubica | 0.0051 ± 0.0007 | 0.0048 ± 0.0002 | ⚪ igual | 0.0035 | 3,297 / 1,153 |
| densa_entrada | regresion_cubica | 0.0051 ± 0.0009 | 0.0048 ± 0.0002 | ⚪ igual | 0.0033 | 3,362 / 1,153 |
| gelu (industria) | circulos | 83.8 ± 1.3 % | 84.8 ± 0.6 % | ⚪ igual | 84.5 | 1,218 / 1,218 |
| silu (industria) | circulos | 83.7 ± 0.8 % | 84.8 ± 0.6 % | ⚪ igual | 84.2 | 1,218 / 1,218 |
| mish (industria) | circulos | 83.7 ± 0.9 % | 84.8 ± 0.6 % | ⚪ igual | 84.0 | 1,218 / 1,218 |
| elu (industria) | circulos | 86.1 ± 0.8 % | 84.8 ± 0.6 % | ⚪ igual | 83.7 | 1,218 / 1,218 |
| leaky_relu (industria) | circulos | 84.6 ± 1.6 % | 84.8 ± 0.6 % | ⚪ igual | 84.6 | 1,218 / 1,218 |
| tanh (industria) | circulos | 83.9 ± 1.9 % | 84.8 ± 0.6 % | ⚪ igual | 85.0 | 1,218 / 1,218 |
| residual (industria) | circulos | 84.4 ± 0.7 % | 84.8 ± 0.6 % | ⚪ igual | 84.9 | 1,282 / 1,218 |
| memoria_errores | circulos | 85.1 ± 0.8 % | 84.8 ± 0.6 % | ⚪ igual | 83.4 | 1,218 / 1,218 |
| memoria_gradiente | circulos | 84.4 ± 1.8 % | 84.8 ± 0.6 % | ⚪ igual | 85.3 | 1,218 / 1,218 |
| lr_por_neurona | circulos | 84.6 ± 1.6 % | 84.8 ± 0.6 % | ⚪ igual | 84.3 | 1,218 / 1,218 |
| memoria_atencion | circulos | 84.0 ± 1.6 % | 84.8 ± 0.6 % | ⚪ igual | 84.9 | 1,218 / 1,218 |
| memoria_embedding | circulos | 84.2 ± 1.8 % | 84.8 ± 0.6 % | ⚪ igual | 84.2 | 3,458 / 1,218 |
| memoria_atencion_embedding | circulos | 83.9 ± 0.9 % | 84.8 ± 0.6 % | ⚪ igual | 85.1 | 9,604 / 1,218 |
| denso_cruzada | circulos | 84.0 ± 2.5 % | 84.8 ± 0.6 % | ⚪ igual | 85.3 | 3,426 / 1,218 |
| densa_entrada | circulos | 85.1 ± 1.6 % | 84.8 ± 0.6 % | ⚪ igual | 85.1 | 3,558 / 1,218 |
| gelu (industria) | lunas | 96.2 ± 0.2 % | 96.3 ± 0.3 % | ⚪ igual | 97.9 | 1,218 / 1,218 |
| silu (industria) | lunas | 96.2 ± 0.2 % | 96.3 ± 0.3 % | ⚪ igual | 97.8 | 1,218 / 1,218 |
| mish (industria) | lunas | 96.1 ± 0.0 % | 96.3 ± 0.3 % | ⚪ igual | 97.8 | 1,218 / 1,218 |
| elu (industria) | lunas | 96.1 ± 0.0 % | 96.3 ± 0.3 % | ⚪ igual | 97.8 | 1,218 / 1,218 |
| leaky_relu (industria) | lunas | 96.1 ± 0.6 % | 96.3 ± 0.3 % | ⚪ igual | 97.8 | 1,218 / 1,218 |
| tanh (industria) | lunas | 96.1 ± 0.0 % | 96.3 ± 0.3 % | ⚪ igual | 97.8 | 1,218 / 1,218 |
| residual (industria) | lunas | 96.1 ± 0.0 % | 96.3 ± 0.3 % | ⚪ igual | 97.9 | 1,282 / 1,218 |
| memoria_errores | lunas | 96.1 ± 0.0 % | 96.3 ± 0.3 % | ⚪ igual | 97.9 | 1,218 / 1,218 |
| memoria_gradiente | lunas | 96.1 ± 0.0 % | 96.3 ± 0.3 % | ⚪ igual | 97.8 | 1,218 / 1,218 |
| lr_por_neurona | lunas | 96.4 ± 0.5 % | 96.3 ± 0.3 % | ⚪ igual | 97.3 | 1,218 / 1,218 |
| memoria_atencion | lunas | 96.1 ± 0.0 % | 96.3 ± 0.3 % | ⚪ igual | 97.8 | 1,218 / 1,218 |
| memoria_embedding | lunas | 96.4 ± 0.5 % | 96.3 ± 0.3 % | ⚪ igual | 97.7 | 3,458 / 1,218 |
| memoria_atencion_embedding | lunas | 96.1 ± 0.4 % | 96.3 ± 0.3 % | ⚪ igual | 97.6 | 9,604 / 1,218 |
| denso_cruzada | lunas | 96.7 ± 0.0 % | 96.3 ± 0.3 % | ⚪ igual | 98.4 | 3,426 / 1,218 |
| densa_entrada | lunas | 96.4 ± 0.3 % | 96.3 ± 0.3 % | ⚪ igual | 98.2 | 3,558 / 1,218 |
| gelu (industria) | xor | 97.2 ± 0.5 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,218 / 1,218 |
| silu (industria) | xor | 97.5 ± 0.0 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,218 / 1,218 |
| mish (industria) | xor | 97.5 ± 0.6 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,218 / 1,218 |
| elu (industria) | xor | 97.5 ± 0.0 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,218 / 1,218 |
| leaky_relu (industria) | xor | 97.8 ± 0.5 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,218 / 1,218 |
| tanh (industria) | xor | 97.0 ± 0.5 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,218 / 1,218 |
| residual (industria) | xor | 97.8 ± 0.5 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,282 / 1,218 |
| memoria_errores | xor | 96.8 ± 0.4 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,218 / 1,218 |
| memoria_gradiente | xor | 97.0 ± 0.5 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,218 / 1,218 |
| lr_por_neurona | xor | 96.7 ± 0.0 % | 98.2 ± 0.4 % | 🔴 peor | 100.0 | 1,218 / 1,218 |
| memoria_atencion | xor | 97.0 ± 0.5 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,218 / 1,218 |
| memoria_embedding | xor | 97.7 ± 0.4 % | 98.2 ± 0.4 % | ⚪ igual | 99.9 | 3,458 / 1,218 |
| memoria_atencion_embedding | xor | 97.3 ± 0.7 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 9,604 / 1,218 |
| denso_cruzada | xor | 97.3 ± 0.4 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 3,426 / 1,218 |
| densa_entrada | xor | 97.5 ± 0.0 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 3,558 / 1,218 |
| gelu (industria) | serie_seno | 0.0241 ± 0.0005 | 0.0261 ± 0.0007 | 🟢 mejor | 0.0199 | 1,505 / 1,505 |
| silu (industria) | serie_seno | 0.0245 ± 0.0005 | 0.0261 ± 0.0007 | ⚪ igual | 0.0217 | 1,505 / 1,505 |
| mish (industria) | serie_seno | 0.0246 ± 0.0004 | 0.0261 ± 0.0007 | ⚪ igual | 0.0211 | 1,505 / 1,505 |
| elu (industria) | serie_seno | 0.0249 ± 0.0004 | 0.0261 ± 0.0007 | ⚪ igual | 0.0220 | 1,505 / 1,505 |
| leaky_relu (industria) | serie_seno | 0.0260 ± 0.0008 | 0.0261 ± 0.0007 | ⚪ igual | 0.0184 | 1,505 / 1,505 |
| tanh (industria) | serie_seno | 0.0243 ± 0.0002 | 0.0261 ± 0.0007 | ⚪ igual | 0.0208 | 1,505 / 1,505 |
| residual (industria) | serie_seno | 0.0264 ± 0.0003 | 0.0261 ± 0.0007 | ⚪ igual | 0.0205 | 1,889 / 1,505 |
| memoria_errores | serie_seno | 0.0251 ± 0.0001 | 0.0261 ± 0.0007 | ⚪ igual | 0.0238 | 1,505 / 1,505 |
| memoria_gradiente | serie_seno | 0.0243 ± 0.0002 | 0.0261 ± 0.0007 | ⚪ igual | 0.0208 | 1,505 / 1,505 |
| lr_por_neurona | serie_seno | 0.0269 ± 0.0016 | 0.0261 ± 0.0007 | ⚪ igual | 0.0214 | 1,505 / 1,505 |
| memoria_atencion | serie_seno | 0.0244 ± 0.0002 | 0.0261 ± 0.0007 | ⚪ igual | 0.0211 | 1,505 / 1,505 |
| memoria_embedding | serie_seno | 0.0257 ± 0.0005 | 0.0261 ± 0.0007 | ⚪ igual | 0.0201 | 3,745 / 1,505 |
| memoria_atencion_embedding | serie_seno | 0.0256 ± 0.0008 | 0.0261 ± 0.0007 | ⚪ igual | 0.0189 | 9,891 / 1,505 |
| denso_cruzada | serie_seno | 0.0270 ± 0.0012 | 0.0261 ± 0.0007 | ⚪ igual | 0.0176 | 3,649 / 1,505 |
| densa_entrada | serie_seno | 0.0282 ± 0.0016 | 0.0261 ± 0.0007 | ⚪ igual | 0.0164 | 4,429 / 1,505 |
| gelu (industria) | digitos | 96.6 ± 0.4 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |
| silu (industria) | digitos | 96.9 ± 0.4 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |
| mish (industria) | digitos | 96.7 ± 0.5 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |
| elu (industria) | digitos | 97.3 ± 0.5 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |
| leaky_relu (industria) | digitos | 96.9 ± 0.6 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |
| tanh (industria) | digitos | 97.0 ± 0.3 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |
| residual (industria) | digitos | 97.1 ± 0.5 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 5,514 / 3,466 |
| memoria_errores | digitos | 96.6 ± 0.4 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |
| memoria_gradiente | digitos | 97.0 ± 0.3 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |
| lr_por_neurona | digitos | 97.2 ± 0.4 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |
| memoria_atencion | digitos | 97.0 ± 0.4 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |
| memoria_embedding | digitos | 96.9 ± 0.4 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 5,706 / 3,466 |
| memoria_atencion_embedding | digitos | 97.0 ± 0.3 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 11,852 / 3,466 |
| denso_cruzada | digitos | 97.0 ± 0.3 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 6,186 / 3,466 |
| densa_entrada | digitos | 97.6 ± 0.5 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 10,922 / 3,466 |

**Resumen:** 🟢 mejor: 1, ⚪ igual: 86, 🔴 peor: 3, 💥 falla: 0

«Pieza en entreno» mucho mejor que «Pieza (prueba)» = la pieza memoriza en vez de aprender.
Las piezas marcadas «(industria)» (GELU, SiLU, Mish…) son la tecnología de hoy: si tu pieza no les gana a ellas, no conviene invertir en ella aunque le gane a ReLU.
Ojo: una pieza con muchos más parámetros que la estándar tiene ventaja; compárala también con una estándar más grande.
