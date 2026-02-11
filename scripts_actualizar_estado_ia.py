#!/usr/bin/env python3
import csv
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('.')
BOTS = [f'fulll{n}' for n in range(45, 51)]
TARGET = 0.70


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
                elif res == 'PÉRDIDA' or res == 'PERDIDA':
                    stats['losses'] += 1
            elif status == 'PRE_TRADE' or res == 'PENDIENTE':
                stats['pending'] += 1

    stats['win_rate_closed'] = (stats['wins'] / stats['closed']) if stats['closed'] else 0.0
    return stats


def read_ia_stats():
    path = ROOT / 'ia_signals_log.csv'
    out = {
        'signals_total': 0,
        'signals_closed': 0,
        'signals_ge70': 0,
        'hits_ge70': 0,
        'hit_rate_ge70': 0.0,
    }
    if not path.exists():
        return out

    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # saltar filas vacías
            if not any((v or '').strip() for v in row.values()):
                continue
            out['signals_total'] += 1
            prob_raw = (row.get('prob') or '').strip()
            y_raw = (row.get('y') or '').strip()
            try:
                prob = float(prob_raw)
            except Exception:
                continue
            if y_raw == '':
                continue
            try:
                y = float(y_raw)
            except Exception:
                continue

            out['signals_closed'] += 1
            if prob >= TARGET:
                out['signals_ge70'] += 1
                if y >= 0.5:
                    out['hits_ge70'] += 1

    if out['signals_ge70']:
        out['hit_rate_ge70'] = out['hits_ge70'] / out['signals_ge70']
    return out


bot_stats = [read_bot_stats(b) for b in BOTS]
ia_stats = read_ia_stats()

with (ROOT / 'status_objetivo_ia.csv').open('w', encoding='utf-8', newline='') as f:
    fieldnames = ['bot', 'rows_total', 'closed', 'wins', 'losses', 'pending', 'win_rate_closed', 'last_ts']
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in bot_stats:
        w.writerow(row)

closed_total = sum(r['closed'] for r in bot_stats)
wins_total = sum(r['wins'] for r in bot_stats)
real_rate = (wins_total / closed_total) if closed_total else 0.0

md = []
md.append('# Estado IA y avance al objetivo\n')
md.append(f'- Actualizado (UTC): {datetime.now(timezone.utc).isoformat()}')
md.append(f'- Objetivo principal (Prob IA real): {TARGET:.0%}')
md.append(f'- Efectividad real global de cierres (bots 45-50): {real_rate:.2%} ({wins_total}/{closed_total})')
md.append(f'- Brecha vs objetivo: {(real_rate - TARGET):+.2%}')
md.append('')
md.append('## Señales IA cerradas (log)')
md.append(f"- Total señales registradas: {ia_stats['signals_total']}")
md.append(f"- Señales cerradas con prob >=70%: {ia_stats['signals_ge70']}")
md.append(f"- Acierto real en esas señales >=70%: {ia_stats['hit_rate_ge70']:.2%} ({ia_stats['hits_ge70']}/{ia_stats['signals_ge70']})")
md.append(f"- Estado semáforo objetivo 70%: {'🟢 OK' if ia_stats['hit_rate_ge70'] >= TARGET and ia_stats['signals_ge70'] > 0 else '🔴 Aún no'}")
md.append('')
md.append('## Resumen por bot (cierres)')
md.append('| Bot | Cerrados | Ganancias | Pérdidas | % Éxito |')
md.append('|---|---:|---:|---:|---:|')
for r in bot_stats:
    md.append(f"| {r['bot']} | {r['closed']} | {r['wins']} | {r['losses']} | {r['win_rate_closed']:.2%} |")

(ROOT / 'status_objetivo_ia.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
print('Archivos actualizados: status_objetivo_ia.csv, status_objetivo_ia.md')
