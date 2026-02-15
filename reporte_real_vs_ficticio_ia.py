#!/usr/bin/env python3
"""
Reporte standalone de calibración IA (Real vs Ficticia) para operación.
Genera:
- reporte_real_vs_ficticio_ia.json
- reporte_real_vs_ficticio_ia.md
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path('.')
LOG = ROOT / 'ia_signals_log.csv'
DIAG = ROOT / 'diagnostico_pipeline_ia.json'
OUT_JSON = ROOT / 'reporte_real_vs_ficticio_ia.json'
OUT_MD = ROOT / 'reporte_real_vs_ficticio_ia.md'
THRESHOLDS = (0.60, 0.65, 0.70, 0.75, 0.80)
BINS = ((0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01))


def _safe_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    for enc in ('utf-8', 'utf-8-sig', 'latin-1', 'windows-1252'):
        try:
            with path.open('r', encoding=enc, newline='') as f:
                return list(csv.DictReader(f))
        except Exception:
            continue
    return []


def _to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        s = str(v).strip().replace(',', '.')
        if s == '':
            return None
        return float(s)
    except Exception:
        return None


def _wilson_lb(hits: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = hits / n
    denom = 1 + z * z / n
    center = (p + (z * z) / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + (z * z) / (4 * n * n)) ** 0.5) / denom
    return max(0.0, center - margin)


def _auc(y_true: list[int], y_score: list[float]) -> float | None:
    if len(y_true) != len(y_score) or not y_true:
        return None
    pos = sum(y_true)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return None
    pairs = sorted(zip(y_score, y_true), key=lambda t: t[0])
    rank = 1
    pos_rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        pos_in_tie = sum(1 for k in range(i, j) if pairs[k][1] == 1)
        pos_rank_sum += pos_in_tie * avg_rank
        rank += (j - i)
        i = j
    return (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


@dataclass
class ClosedSignal:
    bot: str
    prob: float
    y: int


def _load_closed() -> list[ClosedSignal]:
    out: list[ClosedSignal] = []
    for r in _safe_csv(LOG):
        p = _to_float(r.get('prob'))
        y = _to_float(r.get('y'))
        if p is None or y is None:
            continue
        out.append(ClosedSignal(bot=str(r.get('bot') or '').strip(), prob=max(0.0, min(1.0, p)), y=1 if y >= 0.5 else 0))
    return out



def _diag_quality_snapshot() -> dict[str, Any]:
    out = {
        'available': DIAG.exists(),
        'incremental_rows': 0,
        'duplicates_exact': 0,
        'duplicates_ratio': 0.0,
        'dead_features_incremental': [],
        'dead_features_bots': {},
    }
    if not DIAG.exists():
        return out
    try:
        d = json.loads(DIAG.read_text(encoding='utf-8'))
        inc = d.get('incremental', {}) if isinstance(d, dict) else {}
        out['incremental_rows'] = int(inc.get('rows', 0) or 0)
        out['duplicates_exact'] = int(inc.get('duplicates_exact', 0) or 0)
        out['duplicates_ratio'] = float(inc.get('duplicates_ratio', 0.0) or 0.0)

        feat_inc = inc.get('features', {}) if isinstance(inc.get('features', {}), dict) else {}
        for f in ('volatilidad', 'hora_bucket'):
            st = feat_inc.get(f, {}) if isinstance(feat_inc.get(f, {}), dict) else {}
            if int(st.get('uniq', 0) or 0) <= 1:
                out['dead_features_incremental'].append(f)

        bots = d.get('bots', []) if isinstance(d, dict) else []
        for b in bots:
            bot = str(b.get('bot', '') or '')
            dom = b.get('feature_dominance', {}) if isinstance(b.get('feature_dominance', {}), dict) else {}
            dead = []
            for f in ('volatilidad', 'hora_bucket'):
                st = dom.get(f, {}) if isinstance(dom.get(f, {}), dict) else {}
                if int(st.get('uniq', 0) or 0) <= 1:
                    dead.append(f)
            if dead:
                out['dead_features_bots'][bot] = dead

        return out
    except Exception:
        return out


def build_report() -> dict[str, Any]:
    closed = _load_closed()
    report: dict[str, Any] = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'source': str(LOG),
        'closed_total': len(closed),
        'thresholds': {},
        'bins': {},
        'per_bot': {},
        'orientation': {},
        'summary': {},
        'roadmap_status': {},
        'data_quality': {},
    }
    if not closed:
        report['summary'] = {'status': 'sin_datos', 'message': 'No hay señales cerradas en ia_signals_log.csv'}
        return report

    report['data_quality'] = _diag_quality_snapshot()

    y_true = [r.y for r in closed]
    y_pred = [r.prob for r in closed]

    auc = _auc(y_true, y_pred)
    auc_flip = None if auc is None else (1.0 - auc)
    orientation = 'unknown'
    if auc is not None:
        orientation = 'ok' if auc >= 0.5 else 'possible_inversion'
    report['orientation'] = {
        'status': orientation,
        'auc': round(auc, 6) if auc is not None else None,
        'auc_if_flipped': round(auc_flip, 6) if auc_flip is not None else None,
    }

    max_gap = 0.0
    for thr in THRESHOLDS:
        g = [r for r in closed if r.prob >= thr]
        n = len(g)
        hits = sum(r.y for r in g)
        pred_mean = (sum(r.prob for r in g) / n) if n else 0.0
        real = (hits / n) if n else 0.0
        gap = pred_mean - real
        max_gap = max(max_gap, abs(gap)) if n else max_gap
        report['thresholds'][f'{thr:.2f}'] = {
            'n': n,
            'hits': hits,
            'real_rate': round(real, 6),
            'pred_mean': round(pred_mean, 6),
            'gap': round(gap, 6),
            'wilson_lb': round(_wilson_lb(hits, n), 6),
        }

    for lo, hi in BINS:
        g = [r for r in closed if lo <= r.prob < hi]
        n = len(g)
        pred_mean = (sum(r.prob for r in g) / n) if n else 0.0
        real = (sum(r.y for r in g) / n) if n else 0.0
        gap = pred_mean - real
        if n:
            max_gap = max(max_gap, abs(gap))
        report['bins'][f'[{lo:.2f},{min(1.0, hi):.2f})'] = {
            'n': n,
            'pred_mean': round(pred_mean, 6),
            'real_rate': round(real, 6),
            'gap': round(gap, 6),
        }

    by_bot: dict[str, list[ClosedSignal]] = defaultdict(list)
    for r in closed:
        by_bot[r.bot].append(r)
    for bot, rows in sorted(by_bot.items()):
        n = len(rows)
        hits = sum(r.y for r in rows)
        pred_mean = sum(r.prob for r in rows) / n
        real = hits / n
        report['per_bot'][bot] = {
            'n': n,
            'pred_mean': round(pred_mean, 6),
            'real_rate': round(real, 6),
            'gap': round(pred_mean - real, 6),
            'wilson_lb': round(_wilson_lb(hits, n), 6),
        }

    t70 = report['thresholds'].get('0.70', {})
    n70 = int(t70.get('n', 0) or 0)
    wr70 = float(t70.get('real_rate', 0.0) or 0.0)

    report['roadmap_status'] = {
        'orientation_ok': orientation == 'ok',
        'gap_high_bins_ok': max_gap < 0.10,
        'n70_enough': n70 >= 200,
        'wr70_progress': wr70 >= 0.60,
        'max_abs_gap': round(max_gap, 6),
    }

    q = report.get('data_quality', {}) or {}
    dup_ratio = float(q.get('duplicates_ratio', 0.0) or 0.0)
    dead_inc = list(q.get('dead_features_incremental', []) or [])
    report['roadmap_status']['duplicates_ok'] = dup_ratio < 0.02
    report['roadmap_status']['core_features_alive'] = len(dead_inc) == 0
    flags = report['roadmap_status']
    score = sum(1 for k in ('orientation_ok', 'gap_high_bins_ok', 'n70_enough', 'wr70_progress', 'duplicates_ok', 'core_features_alive') if flags.get(k))
    status = 'bien_encaminado' if score >= 5 else ('en_riesgo' if score >= 3 else 'critico')
    report['summary'] = {
        'status': status,
        'score_ok_6': score,
        'headline': (
            f"Estado {status}: orientación={'OK' if flags['orientation_ok'] else 'REVISAR'}, "
            f"gap_max={flags['max_abs_gap']*100:.1f}pp, n>=70={n70}, real>=70={wr70*100:.1f}%"
        ),
    }

    return report


def render_md(rep: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append('# Reporte IA: Real vs Ficticia\n')
    lines.append(f"- Generado (UTC): {rep.get('generated_at')}")
    lines.append(f"- Señales cerradas: {rep.get('closed_total', 0)}")

    ori = rep.get('orientation', {})
    lines.append('\n## Orientación')
    lines.append(f"- Estado: {ori.get('status')}")
    lines.append(f"- AUC: {ori.get('auc')}")
    lines.append(f"- AUC si invertimos p→1-p: {ori.get('auc_if_flipped')}")

    lines.append('\n## Umbrales (predicho vs real)')
    lines.append('| Umbral | n | Pred media | Real | Error (pred-real) | LB |')
    lines.append('|---:|---:|---:|---:|---:|---:|')
    for k, v in rep.get('thresholds', {}).items():
        lines.append(
            f"| {float(k):.0%} | {v.get('n',0)} | {v.get('pred_mean',0.0):.2%} | {v.get('real_rate',0.0):.2%} | {v.get('gap',0.0):+.2%} | {v.get('wilson_lb',0.0):.2%} |"
        )

    lines.append('\n## Bins')
    lines.append('| Bin | n | Pred media | Real | Gap |')
    lines.append('|---|---:|---:|---:|---:|')
    for b, v in rep.get('bins', {}).items():
        lines.append(
            f"| {b} | {v.get('n',0)} | {v.get('pred_mean',0.0):.2%} | {v.get('real_rate',0.0):.2%} | {v.get('gap',0.0):+.2%} |"
        )

    lines.append('\n## Por bot')
    lines.append('| Bot | n | Pred media | Real | Gap | LB |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for bot, v in rep.get('per_bot', {}).items():
        lines.append(
            f"| {bot} | {v.get('n',0)} | {v.get('pred_mean',0.0):.2%} | {v.get('real_rate',0.0):.2%} | {v.get('gap',0.0):+.2%} | {v.get('wilson_lb',0.0):.2%} |"
        )

    r = rep.get('roadmap_status', {})
    q = rep.get('data_quality', {})
    lines.append('\n## Calidad de datos (incremental/bots)')
    lines.append(f"- Filas incremental: {q.get('incremental_rows', 0)}")
    lines.append(f"- Duplicados exactos: {q.get('duplicates_exact', 0)} ({float(q.get('duplicates_ratio', 0.0) or 0.0):.2%})")
    dead_inc = q.get('dead_features_incremental', []) or []
    lines.append(f"- Features rotas incremental: {', '.join(dead_inc) if dead_inc else 'ninguna'}")
    dead_bots = q.get('dead_features_bots', {}) or {}
    if dead_bots:
        bots_txt = '; '.join([f"{b}:{','.join(v)}" for b, v in sorted(dead_bots.items())])
        lines.append(f"- Features rotas en bots: {bots_txt}")

    lines.append('\n## ¿Vamos en buen camino?')
    lines.append(f"- Orientación ok: {'✅' if r.get('orientation_ok') else '❌'}")
    lines.append(f"- Gap bins altos <10pp: {'✅' if r.get('gap_high_bins_ok') else '❌'}")
    lines.append(f"- Evidencia n>=200 en >=70: {'✅' if r.get('n70_enough') else '❌'}")
    lines.append(f"- Real >=70% arriba de 60%: {'✅' if r.get('wr70_progress') else '❌'}")
    lines.append(f"- Duplicados <2%: {'✅' if r.get('duplicates_ok') else '❌'}")
    lines.append(f"- Volatilidad/hora_bucket vivas: {'✅' if r.get('core_features_alive') else '❌'}")

    sm = rep.get('summary', {})
    lines.append('\n## Resumen ejecutivo')
    lines.append(f"- {sm.get('headline','sin datos')}")
    return '\n'.join(lines) + '\n'


def main() -> int:
    rep = build_report()
    OUT_JSON.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    md = render_md(rep)
    OUT_MD.write_text(md, encoding='utf-8')
    print(f'Generado: {OUT_JSON}')
    print(f'Generado: {OUT_MD}')
    print(rep.get('summary', {}).get('headline', 'sin resumen'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
