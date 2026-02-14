#!/usr/bin/env python3
import csv
import json
import math
import glob
from collections import defaultdict, Counter
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
            'bin': f'[{lo:.2f},{min(hi, 1.0):.2f})',
            'n': n,
            'avg_pred': avg_pred,
            'avg_real': avg_real,
            'gap': gap,
        })
    return out


def classify_bot_risk(n, inflation):
    if n < 15:
        maturity = 'BAJA_MUESTRA'
    elif n < 30:
        maturity = 'MEDIA_MUESTRA'
    else:
        maturity = 'ALTA_MUESTRA'

    if inflation >= 0.25:
        semaforo = 'CRITICO'
        action = 'Reducir stake 50% y aplicar beta_bot completo'
    elif inflation >= 0.15:
        semaforo = 'ALERTA'
        action = 'Reducir stake 25% y aplicar beta_bot parcial'
    else:
        semaforo = 'OK'
        action = 'Mantener stake, monitoreo semanal'

    return maturity, semaforo, action


def per_bot_signal_table(closed_rows):
    grouped = defaultdict(list)
    for row in closed_rows:
        grouped[row['bot']].append(row)

    out = []
    for bot in sorted(grouped.keys()):
        grp = grouped[bot]
        n = len(grp)
        hits = sum(1 for r in grp if r['y'] >= 0.5)
        avg_pred = sum(r['prob'] for r in grp) / n if n else 0.0
        avg_real = hits / n if n else 0.0
        inflation = avg_pred - avg_real
        lo, hi = _wilson_interval(hits, n)

        beta_bot = max(0.0, inflation - 0.05)
        maturity, semaforo, action = classify_bot_risk(n, inflation)

        # Prioriza por inflación ponderada por madurez de muestra (evita sobrecastigar n muy chico)
        weight = min(1.0, n / 30)
        priority_score = inflation * weight

        out.append({
            'bot': bot,
            'n': n,
            'hits': hits,
            'hit_rate': avg_real,
            'avg_pred': avg_pred,
            'inflation': inflation,
            'wilson_low': lo,
            'wilson_high': hi,
            'beta_bot': beta_bot,
            'maturity': maturity,
            'semaforo': semaforo,
            'action': action,
            'priority_score': priority_score,
        })

    return sorted(out, key=lambda x: x['priority_score'], reverse=True)




def _safe_float(v, default=0.0):
    try:
        if v is None:
            return default
        t = str(v).strip()
        if t == '':
            return default
        return float(t)
    except Exception:
        return default


def _auc_rank(scores, labels):
    n = len(scores)
    if n <= 1:
        return 0.5
    n1 = sum(1 for y in labels if y == 1)
    n0 = n - n1
    if n0 == 0 or n1 == 0:
        return 0.5

    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    rank = 1.0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (rank + (rank + (j - i))) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        rank += (j - i + 1)
        i = j + 1

    s1 = sum(ranks[i] for i, y in enumerate(labels) if y == 1)
    auc = (s1 - n1 * (n1 + 1) / 2.0) / (n0 * n1)
    return max(auc, 1.0 - auc)


def read_feature_diagnostics():
    feats = [
        'rsi_9','rsi_14','sma_5','sma_20','cruce_sma','breakout',
        'rsi_reversion','racha_actual','payout','puntaje_estrategia',
        'volatilidad','es_rebote','hora_bucket',
    ]

    rows = []
    for path in sorted(glob.glob(str(ROOT / 'registro_enriquecido_fulll*.csv'))):
        fh = None
        reader = None
        for enc in ('utf-8', 'latin-1', 'windows-1252'):
            try:
                fh = open(path, encoding=enc, newline='')
                reader = csv.DictReader(fh)
                break
            except Exception:
                reader = None
                if fh is not None:
                    try:
                        fh.close()
                    except Exception:
                        pass
                    fh = None
                continue

        if reader is None:
            continue

        try:
            for row in reader:
                y_raw = (row.get('result_bin') or '').strip()
                if y_raw not in ('0', '1'):
                    continue

                payout = None
                p_raw = row.get('payout')
                if p_raw not in (None, ''):
                    p = _safe_float(p_raw, None)
                    if p is not None:
                        if p <= 1.5:
                            payout = max(0.0, min(1.5, p))
                        elif p <= 3.5:
                            payout = max(0.0, min(1.5, p - 1.0))
                if payout is None:
                    pm = _safe_float(row.get('payout_multiplier'), 0.0)
                    payout = max(0.0, min(1.5, pm - 1.0)) if pm > 0 else 0.0

                sma5 = _safe_float(row.get('sma_5'), 0.0)
                sma20 = _safe_float(row.get('sma_20'), 0.0)

                vol = _safe_float(row.get('volatilidad'), None)
                if vol is None:
                    base = abs(sma20) if abs(sma20) > 1e-9 else 1.0
                    vol = abs(sma5 - sma20) / base
                vol = max(0.0, min(1.0, float(vol)))

                hb = _safe_float(row.get('hora_bucket'), None)
                if hb is None:
                    ts_raw = (row.get('ts') or '').strip()
                    if ts_raw:
                        try:
                            dt = datetime.fromisoformat(ts_raw.replace('Z', '+00:00'))
                            h = dt.hour
                            b = 0 if h < 6 else (1 if h < 12 else (2 if h < 18 else 3))
                            hb = b / 3.0
                        except Exception:
                            hb = 0.5
                    else:
                        hb = 0.5

                vals = {
                    'rsi_9': _safe_float(row.get('rsi_9'), 0.0),
                    'rsi_14': _safe_float(row.get('rsi_14'), 0.0),
                    'sma_5': sma5,
                    'sma_20': sma20,
                    'cruce_sma': _safe_float(row.get('cruce_sma'), 0.0),
                    'breakout': _safe_float(row.get('breakout'), 0.0),
                    'rsi_reversion': _safe_float(row.get('rsi_reversion'), 0.0),
                    'racha_actual': _safe_float(row.get('racha_actual'), 0.0),
                    'payout': payout,
                    'puntaje_estrategia': _safe_float(row.get('puntaje_estrategia'), 0.0),
                    'volatilidad': vol,
                    'es_rebote': _safe_float(row.get('es_rebote'), 0.0),
                    'hora_bucket': float(hb),
                    'y': int(y_raw),
                }
                rows.append(vals)
        except Exception:
            continue
        finally:
            try:
                if fh is not None:
                    fh.close()
            except Exception:
                pass

    if not rows:
        return {'rows': 0, 'features': []}

    ys = [r['y'] for r in rows]
    y_mean = sum(ys) / len(ys)
    y_var = sum((y - y_mean) ** 2 for y in ys)

    out = []
    for f in feats:
        xs = [r[f] for r in rows]
        x_mean = sum(xs) / len(xs)
        x_var = sum((x - x_mean) ** 2 for x in xs)
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den = (x_var * y_var) ** 0.5
        corr = (num / den) if den else 0.0

        c = Counter(xs)
        top_ratio = (c.most_common(1)[0][1] / len(xs)) if c else 1.0
        auc = _auc_rank(xs, ys)
        strength = 'fuerte' if auc >= 0.57 else ('media' if auc >= 0.53 else 'débil')

        out.append({
            'feature': f,
            'auc': auc,
            'corr': corr,
            'top_ratio': top_ratio,
            'unique_values': len(c),
            'strength': strength,
        })

    out.sort(key=lambda r: r['auc'], reverse=True)
    weak = [r for r in out if r['strength'] == 'débil' and r['top_ratio'] >= 0.90]

    return {'rows': len(rows), 'features': out, 'weak_high_constant': weak}


def build_recommendations(real_rate, ia_summary, thr_rows, calib_rows, bot_signal_rows, model_meta, feat_diag):
    recs = []

    if real_rate < TARGET:
        recs.append(
            f"Brecha principal: estás en {real_rate:.2%} global vs objetivo {TARGET:.0%}. "
            "En corto plazo, prioriza reducir exposición REAL y subir filtro de calidad antes de aumentar volumen."
        )

    n70 = ia_summary['signals_ge70']
    if n70 < 200:
        recs.append(
            f"Muestra IA >=70% insuficiente (n={n70}). "
            "No tomes decisiones estructurales hasta llegar al menos a n>=200 cierres IA >=70%."
        )

    viable = [r for r in thr_rows if r['n'] >= 30]
    if viable:
        best = max(viable, key=lambda x: x['wilson_low'])
        recs.append(
            f"Umbral operativo sugerido temporal: >= {best['threshold']:.0%} "
            f"(hit={best['hit_rate']:.2%}, IC95%=[{best['wilson_low']:.2%},{best['wilson_high']:.2%}], n={best['n']})."
        )

    inflation_bins = [b for b in calib_rows if b['n'] >= 5 and b['gap'] > 0.10]
    if inflation_bins:
        recs.append(
            "Hay sobreconfianza en bins de probabilidad (gap pred-real >10 pts). "
            "Aplicar shrinkage recomendado: p_ajustada = 0.6*p_calibrada + 0.4*tasa_base_rolling."
        )

    if bot_signal_rows:
        top2 = bot_signal_rows[:2]
        bots_txt = ', '.join(
            f"{r['bot']}({r['inflation']:+.1%}, n={r['n']}, prioridad={r['priority_score']:.2f})"
            for r in top2
        )
        recs.append(
            f"Bots a intervenir primero (impacto ponderado): {bots_txt}. "
            "Aplicar beta_bot y reducción de stake según semáforo."
        )

    if isinstance(model_meta, dict):
        reliable = model_meta.get('reliable')
        auc = model_meta.get('auc')
        brier = model_meta.get('brier')
        recs.append(
            f"Meta de modelo actual: reliable={reliable}, auc={auc}, brier={brier}. "
            "Monitorear semanalmente ECE/Brier y recalibrar más frecuente que reentrenar base."
        )

    weak = (feat_diag or {}).get('weak_high_constant', [])
    if weak:
        names = ', '.join(r['feature'] for r in weak[:6])
        recs.append(
            f"Variables con señal débil + alta constancia detectadas: {names}. "
            "Reingenierizar umbrales/estados para que no queden casi constantes."
        )

    return recs


def main():
    bot_stats = [read_bot_stats(b) for b in BOTS]
    ia_rows = read_ia_signals()
    ia_summary, ia_closed_rows = summarize_ia(ia_rows)
    thr_rows = threshold_table(ia_closed_rows)
    calib_rows = calibration_bins(ia_closed_rows)
    bot_signal_rows = per_bot_signal_table(ia_closed_rows)
    model_meta = read_model_meta()
    feat_diag = read_feature_diagnostics()

    closed_total = sum(r['closed'] for r in bot_stats)
    wins_total = sum(r['wins'] for r in bot_stats)
    real_rate = (wins_total / closed_total) if closed_total else 0.0

    with (ROOT / 'status_objetivo_ia.csv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['bot', 'rows_total', 'closed', 'wins', 'losses', 'pending', 'win_rate_closed', 'last_ts']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in bot_stats:
            w.writerow(row)

    with (ROOT / 'status_objetivo_ia_thresholds.csv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['threshold', 'n', 'hits', 'hit_rate', 'wilson_low', 'wilson_high', 'enough_sample']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in thr_rows:
            w.writerow(row)

    with (ROOT / 'status_objetivo_ia_bins.csv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['bin', 'n', 'avg_pred', 'avg_real', 'gap']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in calib_rows:
            w.writerow(row)

    with (ROOT / 'status_objetivo_ia_por_bot.csv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['bot', 'n', 'hits', 'hit_rate', 'avg_pred', 'inflation', 'wilson_low', 'wilson_high', 'beta_bot', 'maturity', 'semaforo', 'action', 'priority_score']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in bot_signal_rows:
            w.writerow(row)

    recs = build_recommendations(real_rate, ia_summary, thr_rows, calib_rows, bot_signal_rows, model_meta, feat_diag)

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

    md.append('## Riesgo de calibración por bot (log IA)')
    md.append('| Bot | n | Madurez | %Real | %Pred media | Inflación | beta_bot | Prioridad | Semáforo | Acción sugerida |')
    md.append('|---|---:|:---:|---:|---:|---:|---:|---:|:---:|---|')
    for r in bot_signal_rows:
        md.append(
            f"| {r['bot']} | {r['n']} | {r['maturity']} | {r['hit_rate']:.2%} | {r['avg_pred']:.2%} | {r['inflation']:+.2%} | {r['beta_bot']:.2%} | {r['priority_score']:.2f} | {r['semaforo']} | {r['action']} |"
        )
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


    md.append('')
    md.append('## Diagnóstico rápido de las 13 variables (cierres reales)')
    rows_diag = int((feat_diag or {}).get('rows', 0) or 0)
    md.append(f'- Muestra usada: {rows_diag} filas cerradas con `result_bin`.')
    md.append('| Feature | AUC univariado | Corr(y) | % valor dominante | Únicos | Señal |')
    md.append('|---|---:|---:|---:|---:|:---:|')
    for r in (feat_diag or {}).get('features', []):
        md.append(
            f"| {r['feature']} | {r['auc']:.4f} | {r['corr']:+.4f} | {r['top_ratio']:.2%} | {r['unique_values']} | {r['strength']} |"
        )

    weak = (feat_diag or {}).get('weak_high_constant', [])
    if weak:
        md.append('')
        md.append('**Variables candidatas a reingeniería (señal débil + constancia alta):** ' + ', '.join(w['feature'] for w in weak[:10]))

    (ROOT / 'status_objetivo_ia.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('Archivos actualizados: status_objetivo_ia.csv, status_objetivo_ia_thresholds.csv, status_objetivo_ia_bins.csv, status_objetivo_ia_por_bot.csv, status_objetivo_ia.md')


if __name__ == '__main__':
    main()
