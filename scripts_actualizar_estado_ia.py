#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('.')
BOTS = [f'fulll{n}' for n in range(45, 51)]
TARGET = 0.70
THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def _wilson_interval(successes: int, n: int, z: float = 1.96):
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + (z * z / n)
    center = (p + (z * z / (2 * n))) / denom
    margin = (z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def read_bot_stats(bot: str):
    path = ROOT / f'registro_enriquecido_{bot}.csv'
    stats = {
        'bot': bot,
        'rows_total': 0,
        'closed': 0,
        'wins': 0,
        'losses': 0,
        'pending': 0,
        'last_ts': '',
        'win_rate_closed': 0.0,
    }
    if not path.exists():
        return stats

    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats['rows_total'] += 1
            status = (row.get('trade_status') or '').strip().upper()
            res = (row.get('resultado') or '').strip().upper()
            stats['last_ts'] = row.get('ts') or stats['last_ts']
            if status == 'CERRADO':
                stats['closed'] += 1
                if res == 'GANANCIA':
                    stats['wins'] += 1
                elif res in ('PÉRDIDA', 'PERDIDA'):
                    stats['losses'] += 1
            elif status == 'PRE_TRADE' or res == 'PENDIENTE':
                stats['pending'] += 1

    stats['win_rate_closed'] = (stats['wins'] / stats['closed']) if stats['closed'] else 0.0
    return stats


def read_model_meta():
    path = ROOT / 'model_meta.json'
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def read_ia_signals():
    path = ROOT / 'ia_signals_log.csv'
    rows = []
    if not path.exists():
        return rows

    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not any((v or '').strip() for v in row.values()):
                continue
            prob_raw = (row.get('prob') or '').strip()
            y_raw = (row.get('y') or '').strip()
            try:
                prob = float(prob_raw)
            except Exception:
                continue
            if y_raw == '':
                y = None
            else:
                try:
                    y = float(y_raw)
                except Exception:
                    y = None
            rows.append({
                'bot': (row.get('bot') or '').strip(),
                'prob': prob,
                'y': y,
                'modo': (row.get('modo') or '').strip(),
                'epoch': (row.get('epoch') or '').strip(),
                'ts': (row.get('ts') or '').strip(),
            })
    return rows


def summarize_ia(rows):
    out = {
        'signals_total': len(rows),
        'signals_closed': 0,
        'signals_ge70': 0,
        'hits_ge70': 0,
        'hit_rate_ge70': 0.0,
        'wilson_low_ge70': 0.0,
        'wilson_high_ge70': 0.0,
    }
    closed = [r for r in rows if r['y'] is not None]
    out['signals_closed'] = len(closed)

    ge70 = [r for r in closed if r['prob'] >= TARGET]
    out['signals_ge70'] = len(ge70)
    out['hits_ge70'] = sum(1 for r in ge70 if r['y'] >= 0.5)

    if out['signals_ge70']:
        out['hit_rate_ge70'] = out['hits_ge70'] / out['signals_ge70']
        lo, hi = _wilson_interval(out['hits_ge70'], out['signals_ge70'])
        out['wilson_low_ge70'] = lo
        out['wilson_high_ge70'] = hi

    return out, closed


def threshold_table(closed_rows):
    table = []
    for thr in THRESHOLDS:
        grp = [r for r in closed_rows if r['prob'] >= thr]
        n = len(grp)
        hits = sum(1 for r in grp if r['y'] >= 0.5)
        hr = (hits / n) if n else 0.0
        lo, hi = _wilson_interval(hits, n)
        table.append({
            'threshold': thr,
            'n': n,
            'hits': hits,
            'hit_rate': hr,
            'wilson_low': lo,
            'wilson_high': hi,
            'enough_sample': n >= 60,
        })
    return table


def calibration_bins(closed_rows):
    bins = [
        (0.50, 0.60),
        (0.60, 0.70),
        (0.70, 0.80),
        (0.80, 0.90),
        (0.90, 1.01),
    ]
    out = []
    for lo, hi in bins:
        grp = [r for r in closed_rows if lo <= r['prob'] < hi]
        n = len(grp)
        if n:
            avg_pred = sum(r['prob'] for r in grp) / n
            avg_real = sum(1 if r['y'] >= 0.5 else 0 for r in grp) / n
            gap = avg_pred - avg_real
        else:
            avg_pred = 0.0
            avg_real = 0.0
            gap = 0.0
        out.append({
            'bin': f'[{lo:.2f},{min(hi,1.0):.2f})',
            'n': n,
            'avg_pred': avg_pred,
            'avg_real': avg_real,
            'gap': gap,
        })
    return out


def build_recommendations(real_rate, ia_summary, thr_rows, calib_rows, model_meta):
    recs = []

    # 1) Gap principal
    if real_rate < TARGET:
        recs.append(
            f"Brecha principal: estás en {real_rate:.2%} global vs objetivo {TARGET:.0%}. "
            "En corto plazo, prioriza reducir exposición REAL y subir filtro de calidad antes de aumentar volumen."
        )

    # 2) Muestra estadística
    n70 = ia_summary['signals_ge70']
    if n70 < 200:
        recs.append(
            f"Muestra IA >=70% insuficiente (n={n70}). "
            "No tomes decisiones estructurales hasta llegar al menos a n>=200 cierres IA >=70%."
        )

    # 3) Threshold recomendado por límite inferior Wilson
    viable = [r for r in thr_rows if r['n'] >= 30]
    if viable:
        best = max(viable, key=lambda x: x['wilson_low'])
        recs.append(
            f"Umbral operativo sugerido temporal: >= {best['threshold']:.0%} "
            f"(hit={best['hit_rate']:.2%}, IC95%=[{best['wilson_low']:.2%},{best['wilson_high']:.2%}], n={best['n']})."
        )

    # 4) Inflación de probabilidad
    inflation_bins = [b for b in calib_rows if b['n'] >= 5 and b['gap'] > 0.10]
    if inflation_bins:
        recs.append(
            "Hay sobreconfianza en bins de probabilidad (gap pred-real >10 pts). "
            "Aplicar shrinkage recomendado: p_ajustada = 0.6*p_calibrada + 0.4*tasa_base_rolling."
        )

    # 5) Confiabilidad meta actual
    if isinstance(model_meta, dict):
        reliable = model_meta.get('reliable')
        auc = model_meta.get('auc')
        brier = model_meta.get('brier')
        recs.append(
            f"Meta de modelo actual: reliable={reliable}, auc={auc}, brier={brier}. "
            "Monitorear semanalmente ECE/Brier y recalibrar más frecuente que reentrenar base."
        )

    return recs


def main():
    bot_stats = [read_bot_stats(b) for b in BOTS]
    ia_rows = read_ia_signals()
    ia_summary, ia_closed_rows = summarize_ia(ia_rows)
    thr_rows = threshold_table(ia_closed_rows)
    calib_rows = calibration_bins(ia_closed_rows)
    model_meta = read_model_meta()

    closed_total = sum(r['closed'] for r in bot_stats)
    wins_total = sum(r['wins'] for r in bot_stats)
    real_rate = (wins_total / closed_total) if closed_total else 0.0

    # CSV por bot
    with (ROOT / 'status_objetivo_ia.csv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['bot', 'rows_total', 'closed', 'wins', 'losses', 'pending', 'win_rate_closed', 'last_ts']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in bot_stats:
            w.writerow(row)

    # CSV thresholds
    with (ROOT / 'status_objetivo_ia_thresholds.csv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['threshold', 'n', 'hits', 'hit_rate', 'wilson_low', 'wilson_high', 'enough_sample']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in thr_rows:
            w.writerow(row)

    # CSV calibración por bins
    with (ROOT / 'status_objetivo_ia_bins.csv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['bin', 'n', 'avg_pred', 'avg_real', 'gap']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in calib_rows:
            w.writerow(row)

    recs = build_recommendations(real_rate, ia_summary, thr_rows, calib_rows, model_meta)

    md = []
    md.append('# Estado IA y avance al objetivo\n')
    md.append(f'- Actualizado (UTC): {datetime.now(timezone.utc).isoformat()}')
    md.append(f'- Objetivo principal (Prob IA real): {TARGET:.0%}')
    md.append(f'- Efectividad real global de cierres (bots 45-50): {real_rate:.2%} ({wins_total}/{closed_total})')
    md.append(f'- Brecha vs objetivo: {(real_rate - TARGET):+.2%}')
    md.append('')

    md.append('## Señales IA cerradas (log)')
    md.append(f"- Total señales registradas: {ia_summary['signals_total']}")
    md.append(f"- Total señales cerradas: {ia_summary['signals_closed']}")
    md.append(f"- Señales cerradas con prob >=70%: {ia_summary['signals_ge70']}")
    md.append(
        f"- Acierto real en señales >=70%: {ia_summary['hit_rate_ge70']:.2%} "
        f"({ia_summary['hits_ge70']}/{ia_summary['signals_ge70']}) | "
        f"IC95%=[{ia_summary['wilson_low_ge70']:.2%},{ia_summary['wilson_high_ge70']:.2%}]"
    )
    md.append(f"- Estado semáforo objetivo 70%: {'🟢 OK' if ia_summary['hit_rate_ge70'] >= TARGET and ia_summary['signals_ge70'] > 0 else '🔴 Aún no'}")
    md.append('')

    md.append('## Recomendaciones priorizadas para subir Prob IA real')
    for i, rec in enumerate(recs, start=1):
        md.append(f'{i}. {rec}')
    md.append('')

    md.append('## Resumen por bot (cierres)')
    md.append('| Bot | Cerrados | Ganancias | Pérdidas | % Éxito |')
    md.append('|---|---:|---:|---:|---:|')
    for r in bot_stats:
        md.append(f"| {r['bot']} | {r['closed']} | {r['wins']} | {r['losses']} | {r['win_rate_closed']:.2%} |")
    md.append('')

    md.append('## Sensibilidad por umbral (señales IA cerradas)')
    md.append('| Umbral | n | hit rate | IC95% | Muestra suficiente |')
    md.append('|---:|---:|---:|---:|:---:|')
    for r in thr_rows:
        md.append(
            f"| {r['threshold']:.0%} | {r['n']} | {r['hit_rate']:.2%} | "
            f"[{r['wilson_low']:.2%},{r['wilson_high']:.2%}] | {'✅' if r['enough_sample'] else '⚠️'} |"
        )

    (ROOT / 'status_objetivo_ia.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('Archivos actualizados: status_objetivo_ia.csv, status_objetivo_ia_thresholds.csv, status_objetivo_ia_bins.csv, status_objetivo_ia.md')


if __name__ == '__main__':
    main()
