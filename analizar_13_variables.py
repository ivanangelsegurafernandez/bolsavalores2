#!/usr/bin/env python3
"""Analiza calidad básica de un CSV con 13 variables + result_bin (sin dependencias externas)."""

from __future__ import annotations

import argparse
import csv
import io
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path
from typing import Dict, List

EXPECTED_COLUMNS = [
    "rsi_9", "rsi_14", "sma_5", "sma_20", "cruce_sma", "breakout",
    "rsi_reversion", "racha_actual", "payout", "puntaje_estrategia",
    "volatilidad", "es_rebote", "hora_bucket", "result_bin",
]


def fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analiza dataset de 13 variables + result_bin")
    p.add_argument("csv", nargs="?", type=Path, help="Ruta al CSV. Si se omite, lee desde STDIN")
    return p.parse_args()


def try_float(value: str):
    try:
        return float(value)
    except Exception:
        return None


def percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = q * (len(sorted_values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _read_text(args: argparse.Namespace) -> str:
    if args.csv is None:
        text = io.TextIOWrapper(getattr(__import__("sys"), "stdin").buffer, encoding="utf-8").read()
        if not text.strip():
            raise ValueError("Entrada STDIN vacía")
        return text

    if not args.csv.exists():
        raise FileNotFoundError(f"No existe el archivo: {args.csv}")
    return args.csv.read_text(encoding="utf-8", errors="replace")


def _normalizar_csv_escapado(text: str) -> str:
    """Corrige casos donde el CSV llega con saltos escapados como \"\\n\" literal."""
    stripped = text.strip()
    if "\\n" in stripped and "\n" not in stripped:
        return stripped.replace("\\n", "\n")

    first_line = stripped.splitlines()[0] if stripped else ""
    if "\\n" in first_line and first_line.count(",") >= 5:
        return stripped.replace("\\n", "\n")
    return text


def _crear_dict_reader(text: str) -> csv.DictReader:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except Exception:
        dialect = csv.get_dialect("excel")
    return csv.DictReader(io.StringIO(text), dialect=dialect)


def main() -> int:
    args = parse_args()
    try:
        raw_text = _read_text(args)
    except Exception as exc:
        print(f"❌ {exc}")
        return 1

    text = _normalizar_csv_escapado(raw_text)

    reader = _crear_dict_reader(text)
    cols = reader.fieldnames or []

    if not cols:
        print("❌ No se detectó cabecera CSV válida.")
        return 1

    missing_expected = [c for c in EXPECTED_COLUMNS if c not in cols]
    extras = [c for c in cols if c not in EXPECTED_COLUMNS]

    numeric = {c: [] for c in EXPECTED_COLUMNS if c in cols and c != "result_bin"}
    null_count = Counter()
    exact_dup_counter = Counter()
    signature_labels: Dict[str, set] = defaultdict(set)
    label_counter = Counter()
    invalid_label = 0
    rows = 0

    for row in reader:
        rows += 1
        exact_dup_counter[tuple((c, row.get(c, "")) for c in cols)] += 1

        for c in numeric:
            val = try_float(row.get(c, ""))
            if val is None:
                null_count[c] += 1
            else:
                numeric[c].append(val)

        if "result_bin" in cols:
            raw = (row.get("result_bin") or "").strip()
            label = try_float(raw)
            if label in (0.0, 1.0):
                label_int = int(label)
                label_counter[label_int] += 1
                feat_sig = "|".join(str(row.get(c, "")).strip() for c in EXPECTED_COLUMNS if c not in {"result_bin"} and c in cols)
                signature_labels[feat_sig].add(label_int)
            elif raw != "":
                invalid_label += 1

    print("=" * 80)
    origen = str(args.csv) if args.csv is not None else "STDIN"
    print(f"📄 Archivo: {origen}")
    print(f"📦 Filas: {rows:,}")
    print(f"🧱 Columnas: {len(cols)}")
    print("=" * 80)

    if missing_expected:
        print(f"⚠️ Faltan columnas esperadas: {missing_expected}")
    else:
        print("✅ Están presentes las 14 columnas esperadas.")
    if extras:
        print(f"ℹ️ Columnas extra detectadas: {extras}")

    print("\n--- Calidad de datos ---")
    if not null_count:
        print("✅ Sin NaN/no numéricos en columnas esperadas.")
    else:
        for c in sorted(null_count):
            print(f"  - {c}: {null_count[c]} no numéricos/vacíos")

    dup_exact = sum(v - 1 for v in exact_dup_counter.values() if v > 1)
    print(f"🔁 Filas duplicadas exactas: {dup_exact}")

    if "result_bin" in cols:
        conflicts = sum(1 for _, labels in signature_labels.items() if len(labels) > 1)
        print(f"⚖️ Firmas de features con labels conflictivos: {conflicts}")
        if invalid_label:
            print(f"⚠️ result_bin con valores no binarios: {invalid_label}")
        total_valid = label_counter[0] + label_counter[1]
        if total_valid:
            print(
                f"🎯 Balance clases: 1={fmt_pct(label_counter[1]/total_valid)} | "
                f"0={fmt_pct(label_counter[0]/total_valid)}"
            )

    print("\n--- Resumen estadístico rápido ---")
    print("columna | mean | std | min | p1 | p50 | p99 | max")
    for c in numeric:
        vals = numeric[c]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        n = len(vals)
        mean = sum(vals) / n
        var = sum((x - mean) ** 2 for x in vals) / n
        std = sqrt(var)
        p1 = percentile(vals_sorted, 0.01)
        p50 = percentile(vals_sorted, 0.50)
        p99 = percentile(vals_sorted, 0.99)
        print(
            f"{c} | {mean:.6f} | {std:.6f} | {vals_sorted[0]:.6f} | "
            f"{p1:.6f} | {p50:.6f} | {p99:.6f} | {vals_sorted[-1]:.6f}"
        )

    print("\n✅ Análisis completado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
