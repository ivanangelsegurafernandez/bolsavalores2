#!/usr/bin/env python3
"""Diagnóstico rápido de aporte de variables CORE-13 sobre registros cerrados."""

from __future__ import annotations

import csv
import glob
import math
from collections import Counter
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
        # Preferencia: columna explícita si existe.
        raw = row.get("hora_bucket")
        if raw not in ("", None):
            return float(raw)

        # Fallback 1: timestamp ISO en ts.
        ts_raw = (row.get("ts") or "").strip()
        if ts_raw:
            try:
                dt = datetime.fromisoformat(ts_raw)
                return dt.hour / 23.0
            except ValueError:
                pass

        # Fallback 2: fecha local en formato "%Y-%m-%d %H:%M:%S".
        fecha_raw = (row.get("fecha") or "").strip()
        if fecha_raw:
            try:
                dt = datetime.strptime(fecha_raw, "%Y-%m-%d %H:%M:%S")
                return dt.hour / 23.0
            except ValueError:
                pass

        # Fallback final neutro.
        return 0.5

    raise KeyError(feature)


def load_closed_rows() -> tuple[list[dict[str, float]], dict[str, int]]:
    rows = []
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

                item = {"y": int(float(y_raw))}
                ok = True
                for feat in FEATURES:
                    try:
                        item[feat] = derive_feature(row, feat)
                    except Exception:
                        ok = False
                        invariant_stats["rows_skipped_feature_error"] += 1
                        break

                if ok and item["hora_bucket"] == 0.5 and not (row.get("hora_bucket") or "").strip():
                    ts_raw = (row.get("ts") or "").strip()
                    fecha_raw = (row.get("fecha") or "").strip()
                    if not ts_raw and not fecha_raw:
                        invariant_stats["rows_hora_bucket_neutral_fallback"] += 1

                if ok:
                    rows.append(item)
    return rows, invariant_stats


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


def build_report(rows: list[dict[str, float]], invariant_stats: dict[str, int]) -> str:
    labels = [r["y"] for r in rows]
    n = len(rows)
    winrate = sum(labels) / n if n else 0.0

    lines = []
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
    lines.append("")
    lines.append("## Señal por variable")
    lines.append("")
    lines.append("| feature | corr | auc_uni | uniq | dominancia | veredicto |")
    lines.append("|---|---:|---:|---:|---:|---|")

    summary = []
    for feat in FEATURES:
        x = [r[feat] for r in rows]
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
    utiles = [f for f in summary if f[5] == "UTIL"]
    bajas = [f for f in summary if f[5] in {"NULA", "CASI_CONSTANTE", "BAJA"}]

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
    lines.append(
        "- Recomendación: mantener `racha_actual`, revisar/rediseñar features casi constantes "
        "(`cruce_sma`, `breakout`, `rsi_reversion`, `es_rebote`, `puntaje_estrategia`) "
        "y corregir `hora_bucket` para que no quede fijo."
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    rows, invariant_stats = load_closed_rows()
    report = build_report(rows, invariant_stats)
    out_path = "veredicto_variables_core13.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Reporte generado: {out_path}")


if __name__ == "__main__":
    main()
