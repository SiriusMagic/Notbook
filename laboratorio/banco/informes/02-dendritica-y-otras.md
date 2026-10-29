# Informe del banco de evaluación

Cada celda: media ± desviación en **datos de prueba**, con 5 semillas. Exactitud: más es mejor. MSE: menos es mejor. Veredicto con prueba t pareada corregida por comparaciones múltiples (Benjamini-Hochberg, 5 %). Es una exploración: un 🟢 indica una pista que vale la pena confirmar, no una prueba definitiva.

| Pieza | Tarea | Pieza (prueba) | Estándar (prueba) | Veredicto | Pieza en entreno | Parámetros (pieza / estándar) |
|---|---|---|---|---|---|---|
| dendritica | regresion_cubica | 0.0133 ± 0.0059 | 0.0048 ± 0.0002 | 🔴 peor | 0.0065 | 1,185 / 1,153 |
| synaptic_tagging | regresion_cubica | 0.0048 ± 0.0000 | 0.0048 ± 0.0002 | ⚪ igual | 0.0034 | 1,153 / 1,153 |
| estado_interno | regresion_cubica | 0.0047 ± 0.0001 | 0.0048 ± 0.0002 | ⚪ igual | 0.0034 | 3,265 / 1,153 |
| ruteo_adaptativo | regresion_cubica | 0.0047 ± 0.0001 | 0.0048 ± 0.0002 | ⚪ igual | 0.0031 | 1,208 / 1,153 |
| dendritica | circulos | 83.1 ± 5.8 % | 84.8 ± 0.6 % | ⚪ igual | 79.4 | 1,218 / 1,218 |
| synaptic_tagging | circulos | 84.2 ± 2.0 % | 84.8 ± 0.6 % | ⚪ igual | 85.4 | 1,218 / 1,218 |
| estado_interno | circulos | 85.6 ± 0.9 % | 84.8 ± 0.6 % | ⚪ igual | 84.4 | 3,330 / 1,218 |
| ruteo_adaptativo | circulos | 84.2 ± 1.4 % | 84.8 ± 0.6 % | ⚪ igual | 84.4 | 1,312 / 1,218 |
| dendritica | lunas | 96.7 ± 0.6 % | 96.3 ± 0.3 % | ⚪ igual | 96.4 | 1,218 / 1,218 |
| synaptic_tagging | lunas | 96.1 ± 0.0 % | 96.3 ± 0.3 % | ⚪ igual | 97.8 | 1,218 / 1,218 |
| estado_interno | lunas | 96.1 ± 0.0 % | 96.3 ± 0.3 % | ⚪ igual | 97.9 | 3,330 / 1,218 |
| ruteo_adaptativo | lunas | 96.4 ± 0.3 % | 96.3 ± 0.3 % | ⚪ igual | 97.9 | 1,312 / 1,218 |
| dendritica | xor | 93.2 ± 5.6 % | 98.2 ± 0.4 % | ⚪ igual | 95.1 | 1,218 / 1,218 |
| synaptic_tagging | xor | 97.0 ± 0.5 % | 98.2 ± 0.4 % | 🔴 peor | 100.0 | 1,218 / 1,218 |
| estado_interno | xor | 97.2 ± 0.5 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 3,330 / 1,218 |
| ruteo_adaptativo | xor | 97.5 ± 0.0 % | 98.2 ± 0.4 % | ⚪ igual | 100.0 | 1,312 / 1,218 |
| dendritica | serie_seno | 0.0273 ± 0.0005 | 0.0261 ± 0.0007 | 🔴 peor | 0.0234 | 1,505 / 1,505 |
| synaptic_tagging | serie_seno | 0.0243 ± 0.0002 | 0.0261 ± 0.0007 | 🟢 mejor | 0.0208 | 1,505 / 1,505 |
| estado_interno | serie_seno | 0.0246 ± 0.0002 | 0.0261 ± 0.0007 | 🟢 mejor | 0.0207 | 3,617 / 1,505 |
| ruteo_adaptativo | serie_seno | 0.0255 ± 0.0011 | 0.0261 ± 0.0007 | ⚪ igual | 0.0194 | 1,758 / 1,505 |
| dendritica | digitos | 94.8 ± 0.7 % | 97.1 ± 0.5 % | 🔴 peor | 99.7 | 3,466 / 3,466 |
| synaptic_tagging | digitos | 97.0 ± 0.3 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 3,466 / 3,466 |
| estado_interno | digitos | 97.5 ± 0.6 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 5,578 / 3,466 |
| ruteo_adaptativo | digitos | 96.0 ± 0.7 % | 97.1 ± 0.5 % | ⚪ igual | 100.0 | 4,916 / 3,466 |

**Resumen:** 🟢 mejor: 2, ⚪ igual: 18, 🔴 peor: 4, 💥 falla: 0

«Pieza en entreno» mucho mejor que «Pieza (prueba)» = la pieza memoriza en vez de aprender.
Las piezas marcadas «(industria)» (GELU, SiLU, Mish…) son la tecnología de hoy: si tu pieza no les gana a ellas, no conviene invertir en ella aunque le gane a ReLU.
Ojo: una pieza con muchos más parámetros que la estándar tiene ventaja; compárala también con una estándar más grande.
