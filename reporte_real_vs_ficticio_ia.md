# Reporte IA: Real vs Ficticia

- Generado (UTC): 2026-02-15T22:45:20.458922+00:00
- Señales cerradas: 82

## Orientación
- Estado: ok
- AUC: 0.539039
- AUC si invertimos p→1-p: 0.460961

## Umbrales (predicho vs real)
| Umbral | n | Pred media | Real | Error (pred-real) | LB |
|---:|---:|---:|---:|---:|---:|
| 60% | 82 | 80.47% | 45.12% | +35.35% | 34.81% |
| 65% | 82 | 80.47% | 45.12% | +35.35% | 34.81% |
| 70% | 82 | 80.47% | 45.12% | +35.35% | 34.81% |
| 75% | 69 | 82.22% | 43.48% | +38.74% | 32.43% |
| 80% | 38 | 86.47% | 50.00% | +36.47% | 34.85% |

## Bins
| Bin | n | Pred media | Real | Gap |
|---|---:|---:|---:|---:|
| [0.60,0.70) | 0 | 0.00% | 0.00% | +0.00% |
| [0.70,0.80) | 44 | 75.29% | 40.91% | +34.38% |
| [0.80,0.90) | 29 | 82.27% | 48.28% | +33.99% |
| [0.90,1.00) | 9 | 100.00% | 55.56% | +44.44% |

## Por bot
| Bot | n | Pred media | Real | Gap | LB |
|---|---:|---:|---:|---:|---:|
| fulll45 | 17 | 80.27% | 52.94% | +27.33% | 30.96% |
| fulll46 | 11 | 82.01% | 27.27% | +54.74% | 9.75% |
| fulll47 | 9 | 83.04% | 66.67% | +16.37% | 35.42% |
| fulll48 | 8 | 76.65% | 50.00% | +26.65% | 21.52% |
| fulll49 | 20 | 80.90% | 30.00% | +50.90% | 14.55% |
| fulll50 | 17 | 79.62% | 52.94% | +26.68% | 30.96% |

## ¿Vamos en buen camino?
- Orientación ok: ✅
- Gap bins altos <10pp: ❌
- Evidencia n>=200 en >=70: ❌
- Real >=70% arriba de 60%: ❌

## Resumen ejecutivo
- Estado critico: orientación=OK, gap_max=44.4pp, n>=70=82, real>=70=45.1%
