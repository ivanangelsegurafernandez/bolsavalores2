#!/usr/bin/env python3
"""Diagnóstico de aporte de variables CORE-13 sobre registros cerrados."""

from __future__ import annotations

import csv
import glob
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone

FEATURES = [
    "rsi_9",
    "rsi_14",
    "sma_5",
    "sma_20",
    "cruce_sma",
    "breakout",
    "rsi_reversion",
    "racha_actual",
    "payout",
    "puntaje_estrategia",
    "volatilidad",
    "es_rebote",
    "hora_bucket",
]

EVENT_FEATURES = ["cruce_sma", "breakout", "rsi_reversion", "puntaje_estrategia", "es_rebote"]
REDESIGNED_FEATURES = [
    "red_cruce_intensidad",
    "red_breakout_soft",
    "red_rsi_extremo",
    "red_rsi_spread",
    "red_rebote_contexto",
    "red_score_soft",
]


def auc_from_scores(scores: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    n = len(pairs)
    if n == 0:
        return 0.5

    n_pos = sum(lbl for _, lbl in pairs)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    sum_pos_ranks = sum(rank for rank, (_, lbl) in zip(ranks, pairs) if lbl == 1)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pearson_corr(x: list[float], y: list[int]) -> float:
    n = len(x)
    if n == 0:
        return 0.0

    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in x))
    den_y = math.sqrt(sum((b - my) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def parse_trade_dt(row: dict[str, str]) -> datetime | None:
    ts_raw = (row.get("ts") or "").strip()
    if ts_raw:
        try:
            return datetime.fromisoformat(ts_raw)
        except ValueError:
            pass

    fecha_raw = (row.get("fecha") or "").strip()
    if fecha_raw:
        try:
            return datetime.strptime(fecha_raw, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    return None


def derive_feature(row: dict[str, str], feature: str) -> float:
    if feature in row and row[feature] not in ("", None):
        return float(row[feature])

    if feature == "payout":
        monto = float(row.get("monto") or 0.0)
        payout_total = float(row.get("payout_total") or 0.0)
        return (payout_total / monto - 1.0) if monto > 0 else 0.0

    if feature == "volatilidad":
        sma_5 = float(row.get("sma_5") or 0.0)
        sma_20 = float(row.get("sma_20") or 0.0)
        return abs(sma_5 - sma_20) / max(abs(sma_20), 1e-9)

    if feature == "hora_bucket":
        raw = row.get("hora_bucket")
        if raw not in ("", None):
            return float(raw)

        dt = parse_trade_dt(row)
        if dt is not None:
            return dt.hour / 23.0

        return 0.5

    raise KeyError(feature)


def redesigned_features(item: dict[str, float | int | str]) -> dict[str, float]:
    sma_5 = float(item["sma_5"])
    sma_20 = float(item["sma_20"])
    vol = float(item["volatilidad"])
    rsi_9 = float(item["rsi_9"])
    rsi_14 = float(item["rsi_14"])

    out = {
        "red_cruce_intensidad": abs(sma_5 - sma_20) / max(abs(sma_20), 1e-9),
        "red_breakout_soft": float(item["breakout"]),
        "red_rsi_extremo": abs(rsi_14 - 50.0) / 50.0,
        "red_rsi_spread": abs(rsi_9 - rsi_14) / 100.0,
        "red_rebote_contexto": float(item["es_rebote"]) * (1.0 + min(vol * 100.0, 5.0)),
        "red_score_soft": float(item["puntaje_estrategia"]),
    }
    return out


def load_closed_rows() -> tuple[list[dict[str, float | int | str]], dict[str, int]]:
    rows: list[dict[str, float | int | str]] = []
    invariant_stats = {
        "rows_closed": 0,
        "rows_invalid_result_bin": 0,
        "rows_skipped_feature_error": 0,
        "rows_hora_bucket_neutral_fallback": 0,
    }

    for path in sorted(glob.glob("registro_enriquecido_fulll*.csv")):
        with open(path, encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            for row in rd:
                if (row.get("trade_status") or "").upper() != "CERRADO":
                    continue
                invariant_stats["rows_closed"] += 1

                y_raw = (row.get("result_bin") or "").strip()
                if y_raw == "":
                    invariant_stats["rows_invalid_result_bin"] += 1
                    continue
                try:
                    y = int(float(y_raw))
                except ValueError:
                    invariant_stats["rows_invalid_result_bin"] += 1
                    continue
                if y not in (0, 1):
                    invariant_stats["rows_invalid_result_bin"] += 1
                    continue

                item: dict[str, float | int | str] = {
                    "y": y,
                    "activo": (row.get("activo") or "").strip(),
                    "ts": (row.get("ts") or "").strip(),
                }
                ok = True
                for feat in FEATURES:
                    try:
                        item[feat] = derive_feature(row, feat)
                    except Exception:
                        ok = False
                        invariant_stats["rows_skipped_feature_error"] += 1
                        break

                if ok and float(item["hora_bucket"]) == 0.5 and not (row.get("hora_bucket") or "").strip():
                    dt = parse_trade_dt(row)
                    if dt is None:
                        invariant_stats["rows_hora_bucket_neutral_fallback"] += 1

                if ok:
                    item.update(redesigned_features(item))
                    rows.append(item)

    return rows, invariant_stats


def assert_invariants(rows: list[dict[str, float | int | str]], invariant_stats: dict[str, int]) -> None:
    closed = invariant_stats["rows_closed"]
    invalid = invariant_stats["rows_invalid_result_bin"]
    feature_error = invariant_stats["rows_skipped_feature_error"]
    expected_loaded = closed - invalid - feature_error

    assert closed >= 0, "rows_closed no puede ser negativo"
    assert expected_loaded >= 0, "conteo esperado de filas cargadas no puede ser negativo"
    assert len(rows) == expected_loaded, (
        f"inconsistencia contable: len(rows)={len(rows)} vs esperado={expected_loaded} "
        f"(cerradas={closed}, invalid={invalid}, feature_error={feature_error})"
    )

    invalid_labels = [r["y"] for r in rows if r["y"] not in (0, 1)]
    assert not invalid_labels, "se detectaron labels fuera de {0,1} en rows cargadas"


def classify_signal(auc: float, dominant_ratio: float, uniq_count: int) -> str:
    if uniq_count <= 1 or dominant_ratio >= 0.99:
        return "NULA"
    if auc >= 0.57:
        return "UTIL"
    if auc >= 0.53:
        return "DEBIL"
    if dominant_ratio >= 0.90:
        return "CASI_CONSTANTE"
    return "BAJA"


def quantile_cuts(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    xs = sorted(values)
    i1 = int(0.33 * (len(xs) - 1))
    i2 = int(0.66 * (len(xs) - 1))
    return xs[i1], xs[i2]


def bucket_tercile(value: float, q1: float, q2: float) -> str:
    if value <= q1:
        return "bajo"
    if value <= q2:
        return "medio"
    return "alto"


def build_segment_rows(rows: list[dict[str, float | int | str]]) -> dict[str, list[dict[str, float | int | str]]]:
    payout_vals = [float(r["payout"]) for r in rows]
    vol_vals = [float(r["volatilidad"]) for r in rows]
    p_q1, p_q2 = quantile_cuts(payout_vals)
    v_q1, v_q2 = quantile_cuts(vol_vals)

    segments: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for r in rows:
        activo = str(r.get("activo") or "") or "NA"
        segments[f"activo:{activo}"].append(r)

        pb = bucket_tercile(float(r["payout"]), p_q1, p_q2)
        vb = bucket_tercile(float(r["volatilidad"]), v_q1, v_q2)
        hb = int(round(float(r["hora_bucket"]) * 23))
        hseg = f"h{(hb // 6) * 6:02d}-{min(((hb // 6) * 6) + 5, 23):02d}"

        segments[f"payout:{pb}"].append(r)
        segments[f"vol:{vb}"].append(r)
        segments[f"hora:{hseg}"].append(r)

    return segments


def segment_auc_table(rows: list[dict[str, float | int | str]], feature: str, min_n: int = 300) -> list[tuple[str, int, float, float]]:
    segments = build_segment_rows(rows)
    out: list[tuple[str, int, float, float]] = []
    for seg, grp in segments.items():
        if len(grp) < min_n:
            continue
        labels = [int(r["y"]) for r in grp]
        values = [float(r[feature]) for r in grp]
        out.append((seg, len(grp), auc_from_scores(values, labels), sum(labels) / len(labels)))
    out.sort(key=lambda t: t[2], reverse=True)
    return out[:8]


def temporal_walkforward(rows: list[dict[str, float | int | str]], feature: str) -> list[tuple[str, int, float]]:
    dated = []
    for r in rows:
        dt = parse_trade_dt({"ts": str(r.get("ts") or "")})
        if dt is None:
            continue
        dated.append((dt, r))
    dated.sort(key=lambda t: t[0])
    if len(dated) < 500:
        return []

    fold_size = max(200, len(dated) // 5)
    folds: list[tuple[str, int, float]] = []
    for i in range(0, len(dated), fold_size):
        fold = dated[i : i + fold_size]
        if len(fold) < 200:
            continue
        labels = [int(r["y"]) for _, r in fold]
        values = [float(r[feature]) for _, r in fold]
        stamp = fold[0][0].strftime("%Y-%m-%d")
        folds.append((stamp, len(fold), auc_from_scores(values, labels)))
    return folds


def fit_standardizer(rows: list[dict[str, float | int | str]], fields: list[str]) -> dict[str, tuple[float, float]]:
    params: dict[str, tuple[float, float]] = {}
    for f in fields:
        vals = [float(r[f]) for r in rows]
        if not vals:
            params[f] = (0.0, 1.0)
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(len(vals), 1)
        std = math.sqrt(var)
        if std < 1e-9:
            std = 1.0
        params[f] = (mean, std)
    return params


def zscore(value: float, mean: float, std: float) -> float:
    return (value - mean) / std


def ablation_score(row: dict[str, float | int | str], params: dict[str, tuple[float, float]], mode: str) -> float:
    r_mean, r_std = params["racha_actual"]
    racha_z = zscore(float(row["racha_actual"]), r_mean, r_std)

    if mode == "solo_racha":
        return racha_z

    red_z = []
    for feat in REDESIGNED_FEATURES:
        mean, std = params[feat]
        red_z.append(zscore(float(row[feat]), mean, std))
    red_avg = sum(red_z) / len(red_z) if red_z else 0.0
    return 0.55 * racha_z + 0.45 * red_avg


def ablation_walkforward(rows: list[dict[str, float | int | str]]) -> tuple[float, float, list[tuple[str, int, float, float]]]:
    dated = []
    for r in rows:
        dt = parse_trade_dt({"ts": str(r.get("ts") or "")})
        if dt is None:
            continue
        dated.append((dt, r))
    dated.sort(key=lambda t: t[0])

    if len(dated) < 2000:
        return 0.5, 0.5, []

    fold_size = max(500, len(dated) // 6)
    folds = [dated[i : i + fold_size] for i in range(0, len(dated), fold_size)]
    folds = [f for f in folds if len(f) >= 400]
    if len(folds) < 3:
        return 0.5, 0.5, []

    global_labels: list[int] = []
    global_solo: list[float] = []
    global_combo: list[float] = []
    table: list[tuple[str, int, float, float]] = []

    used_fields = ["racha_actual", *REDESIGNED_FEATURES]

    for i in range(1, len(folds)):
        train = [r for fold in folds[:i] for _, r in fold]
        test = [r for _, r in folds[i]]
        params = fit_standardizer(train, used_fields)

        labels = [int(r["y"]) for r in test]
        solo_scores = [ablation_score(r, params, "solo_racha") for r in test]
        combo_scores = [ablation_score(r, params, "combo") for r in test]

        auc_solo = auc_from_scores(solo_scores, labels)
        auc_combo = auc_from_scores(combo_scores, labels)
        stamp = folds[i][0][0].strftime("%Y-%m-%d")

        table.append((stamp, len(test), auc_solo, auc_combo))
        global_labels.extend(labels)
        global_solo.extend(solo_scores)
        global_combo.extend(combo_scores)

    return auc_from_scores(global_solo, global_labels), auc_from_scores(global_combo, global_labels), table


def build_report(rows: list[dict[str, float | int | str]], invariant_stats: dict[str, int]) -> str:
    labels = [int(r["y"]) for r in rows]
    n = len(rows)
    winrate = sum(labels) / n if n else 0.0

    lines: list[str] = []
    lines.append("# Veredicto CORE-13 (actualizado)")
    lines.append("")
    lines.append(f"Generado: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Muestra cerrada analizada: **{n}** trades (winrate base **{winrate:.2%}**).")
    lines.append("")

    lines.append("## Invariantes / sanidad del dataset")
    lines.append("")
    lines.append(f"- Filas cerradas detectadas: **{invariant_stats['rows_closed']}**.")
    lines.append(
        f"- Filas cerradas sin `result_bin` válido descartadas: **{invariant_stats['rows_invalid_result_bin']}**."
    )
    lines.append(
        f"- Filas descartadas por error al derivar features: **{invariant_stats['rows_skipped_feature_error']}**."
    )
    lines.append(
        "- Filas con `hora_bucket` en fallback neutro (0.5) por falta total de timestamp: "
        f"**{invariant_stats['rows_hora_bucket_neutral_fallback']}**."
    )
    lines.append(
        "- Chequeo de consistencia contable: **OK** "
        "(`rows_loaded = rows_closed - invalid_result_bin - feature_error`)."
    )
    lines.append("")

    lines.append("## Señal por variable")
    lines.append("")
    lines.append("| feature | corr | auc_uni | uniq | dominancia | veredicto |")
    lines.append("|---|---:|---:|---:|---:|---|")

    summary = []
    for feat in FEATURES:
        x = [float(r[feat]) for r in rows]
        c = pearson_corr(x, labels)
        a = auc_from_scores(x, labels)
        uniq_count = len(set(x))
        counts = Counter(x)
        dominant_ratio = (max(counts.values()) / n) if n else 1.0
        verdict = classify_signal(a, dominant_ratio, uniq_count)
        summary.append((feat, c, a, uniq_count, dominant_ratio, verdict))
        lines.append(
            f"| {feat} | {c:+.4f} | {a:.4f} | {uniq_count} | {dominant_ratio:.3f} | {verdict} |"
        )

    lines.append("")
    lines.append("## Diagnóstico de eventos raros (objetivo: bajar dominancia)")
    lines.append("")
    lines.append("| feature | pct_no_cero | dominancia | uniq |")
    lines.append("|---|---:|---:|---:|")

    feat_map = {s[0]: s for s in summary}
    for feat in EVENT_FEATURES:
        vals = [float(r[feat]) for r in rows]
        nz = sum(1 for v in vals if abs(v) > 1e-12)
        nonzero_ratio = nz / n if n else 0.0
        dom = feat_map[feat][4]
        uniq = feat_map[feat][3]
        lines.append(f"| {feat} | {nonzero_ratio:.2%} | {dom:.3f} | {uniq} |")

    lines.append("")
    lines.append("## Segmentación (AUC univariado de `racha_actual`) ")
    lines.append("")
    seg_rows = segment_auc_table(rows, "racha_actual", min_n=max(300, n // 60))
    if seg_rows:
        lines.append("| segmento | n | winrate | auc_racha_actual |")
        lines.append("|---|---:|---:|---:|")
        for seg, sn, aucv, wr in seg_rows:
            lines.append(f"| {seg} | {sn} | {wr:.2%} | {aucv:.4f} |")
    else:
        lines.append("- Muestra insuficiente para segmentación robusta.")

    lines.append("")
    lines.append("## Estabilidad temporal (walk-forward simple, AUC de `racha_actual`) ")
    lines.append("")
    folds = temporal_walkforward(rows, "racha_actual")
    if folds:
        lines.append("| fold_inicio | n | auc_racha_actual |")
        lines.append("|---|---:|---:|")
        for stamp, fn, aucv in folds:
            lines.append(f"| {stamp} | {fn} | {aucv:.4f} |")
    else:
        lines.append("- Muestra temporal insuficiente o timestamps no parseables para walk-forward.")

    lines.append("")
    lines.append("## Ablation automático (Objetivo 3)")
    lines.append("")
    auc_solo, auc_combo, ablation_rows = ablation_walkforward(rows)
    lines.append(
        f"- AUC walk-forward agregado (`solo racha_actual`): **{auc_solo:.4f}**."
    )
    lines.append(
        f"- AUC walk-forward agregado (`racha_actual + rediseñadas`): **{auc_combo:.4f}**."
    )
    lines.append(
        f"- Delta (`combo - solo`): **{(auc_combo - auc_solo):+.4f}**."
    )
    if ablation_rows:
        lines.append("")
        lines.append("| fold_inicio | n_test | auc_solo_racha | auc_combo_rediseñado |")
        lines.append("|---|---:|---:|---:|")
        for stamp, fn, a_solo, a_combo in ablation_rows:
            lines.append(f"| {stamp} | {fn} | {a_solo:.4f} | {a_combo:.4f} |")
    else:
        lines.append("- Muestra temporal insuficiente para ablation walk-forward robusto.")

    lines.append("")
    utiles = [f for f in summary if f[5] == "UTIL"]
    bajas = [f for f in summary if f[5] in {"NULA", "CASI_CONSTANTE", "BAJA"}]
    hora = feat_map.get("hora_bucket")

    lines.append("## Veredicto")
    lines.append("")
    if utiles:
        lines.append(
            "- Variables con aporte real hoy: "
            + ", ".join(f"`{f[0]}`" for f in utiles)
            + "."
        )
    if bajas:
        lines.append(
            "- Variables con aporte bajo o nulo hoy: "
            + ", ".join(f"`{f[0]}`" for f in bajas)
            + "."
        )

    lines.append("- Sí: **aún existen variables que prácticamente no ponderan** en el estado actual.")

    recommendation_parts = [
        "mantener `racha_actual` como baseline",
        "rediseñar eventos binarios a señales continuas de intensidad/proximidad a umbral",
        "comparar siempre con ablation walk-forward (solo racha vs combo) para evitar maquillaje",
        "validar por segmentación (activo/payout/volatilidad/hora) y por tiempo (walk-forward)",
    ]
    if hora and (hora[3] <= 1 or hora[4] >= 0.99):
        recommendation_parts.append("corregir `hora_bucket` para que no quede fijo")
    else:
        recommendation_parts.append("`hora_bucket` ya no está fijo, pero su aporte predictivo actual sigue bajo")

    lines.append("- Recomendación: " + "; ".join(recommendation_parts) + ".")

    return "\n".join(lines) + "\n"


def main() -> None:
    rows, invariant_stats = load_closed_rows()
    assert_invariants(rows, invariant_stats)
    report = build_report(rows, invariant_stats)
    out_path = "veredicto_variables_core13.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Reporte generado: {out_path}")


if __name__ == "__main__":
    main()
