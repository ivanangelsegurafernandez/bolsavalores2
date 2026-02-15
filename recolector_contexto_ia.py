#!/usr/bin/env python3
"""
Recolector integral de contexto IA para acelerar iteraciones hacia meta real.

Lee y consolida:
- registro_enriquecido_fulll45..50.csv
- dataset_incremental.csv
- ia_signals_log.csv
- model_meta.json

Genera:
- diagnostico_pipeline_ia.json
- diagnostico_pipeline_ia.md
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(".")
BOTS = [f"fulll{i}" for i in range(45, 51)]
INCREMENTAL = ROOT / "dataset_incremental.csv"
IA_SIGNALS = ROOT / "ia_signals_log.csv"
MODEL_META = ROOT / "model_meta.json"

FEATURES_CORE = [
    "rsi_9", "rsi_14", "sma_5", "sma_spread", "cruce_sma", "breakout",
    "rsi_reversion", "racha_actual", "payout", "puntaje_estrategia",
    "volatilidad", "es_rebote", "hora_bucket",
]


def _safe_read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    for enc in ("utf-8", "utf-8-sig", "latin-1", "windows-1252"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except Exception:
            continue
    return []


def _to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        s = str(v).strip().replace(",", ".")
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _dominance(values: list[Any]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "uniq": 0, "dominance": 0.0, "top": None}
    c = Counter(values)
    top_v, top_n = c.most_common(1)[0]
    n = len(values)
    return {
        "n": n,
        "uniq": len(c),
        "dominance": round(top_n / max(1, n), 6),
        "top": top_v,
    }


def summarize_incremental() -> dict[str, Any]:
    rows = _safe_read_csv(INCREMENTAL)
    out: dict[str, Any] = {
        "path": str(INCREMENTAL),
        "exists": INCREMENTAL.exists(),
        "rows": len(rows),
        "features": {},
        "duplicates_exact": 0,
        "class_balance": {},
    }
    if not rows:
        return out

    # duplicados exactos (firma de core+label)
    sigs = []
    for r in rows:
        sig = tuple((k, r.get(k, "")) for k in FEATURES_CORE + ["result_bin"])
        sigs.append(sig)
    c = Counter(sigs)
    out["duplicates_exact"] = int(sum(v - 1 for v in c.values() if v > 1))

    # dominancia por feature
    for feat in FEATURES_CORE:
        vals = [r.get(feat, "") for r in rows]
        out["features"][feat] = _dominance(vals)

    y = [r.get("result_bin", "") for r in rows]
    yc = Counter(y)
    n = max(1, len(rows))
    out["class_balance"] = {
        "0": int(yc.get("0", 0) + yc.get("0.0", 0)),
        "1": int(yc.get("1", 0) + yc.get("1.0", 0)),
        "p0": round((yc.get("0", 0) + yc.get("0.0", 0)) / n, 4),
        "p1": round((yc.get("1", 0) + yc.get("1.0", 0)) / n, 4),
    }
    return out


def summarize_bot(bot: str) -> dict[str, Any]:
    path = ROOT / f"registro_enriquecido_{bot}.csv"
    rows = _safe_read_csv(path)
    out: dict[str, Any] = {
        "bot": bot,
        "path": str(path),
        "exists": path.exists(),
        "rows": len(rows),
        "closed": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": None,
        "last_ts": "",
        "feature_dominance": {},
    }
    if not rows:
        return out

    closed = []
    for r in rows:
        out["last_ts"] = str(r.get("ts") or out["last_ts"])
        st = str(r.get("trade_status") or "").strip().upper()
        y = str(r.get("result_bin") or "").strip()
        if st == "CERRADO" and y in ("0", "1", "0.0", "1.0"):
            closed.append(r)

    out["closed"] = len(closed)
    out["wins"] = sum(1 for r in closed if str(r.get("result_bin")) in ("1", "1.0"))
    out["losses"] = sum(1 for r in closed if str(r.get("result_bin")) in ("0", "0.0"))
    out["win_rate"] = round(out["wins"] / out["closed"], 6) if out["closed"] else None

    for feat in ["cruce_sma", "breakout", "rsi_reversion", "puntaje_estrategia", "es_rebote", "volatilidad", "hora_bucket"]:
        vals = [r.get(feat, "") for r in closed]
        out["feature_dominance"][feat] = _dominance(vals)

    return out


def summarize_signals() -> dict[str, Any]:
    rows = _safe_read_csv(IA_SIGNALS)
    out: dict[str, Any] = {
        "path": str(IA_SIGNALS),
        "exists": IA_SIGNALS.exists(),
        "rows": len(rows),
        "closed": 0,
        "by_threshold": {},
        "bins": {},
    }
    if not rows:
        return out

    parsed = []
    for r in rows:
        p = _to_float(r.get("prob"))
        y = _to_float(r.get("y"))
        if p is None:
            continue
        if y is None:
            continue
        yb = 1 if y >= 0.5 else 0
        parsed.append((p, yb))

    out["closed"] = len(parsed)

    for thr in (0.65, 0.70, 0.75, 0.80):
        grp = [(p, y) for p, y in parsed if p >= thr]
        n = len(grp)
        hits = sum(y for _, y in grp)
        wr = (hits / n) if n else None
        out["by_threshold"][str(thr)] = {"n": n, "hits": hits, "win_rate": wr}

    bins = [(0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    for lo, hi in bins:
        grp = [(p, y) for p, y in parsed if lo <= p < hi]
        n = len(grp)
        if n:
            avg_pred = sum(p for p, _ in grp) / n
            avg_real = sum(y for _, y in grp) / n
        else:
            avg_pred = 0.0
            avg_real = 0.0
        out["bins"][f"[{lo:.2f},{min(1.0,hi):.2f})"] = {
            "n": n,
            "avg_pred": round(avg_pred, 6),
            "avg_real": round(avg_real, 6),
            "gap": round(avg_pred - avg_real, 6),
        }

    return out


def summarize_model_meta() -> dict[str, Any]:
    if not MODEL_META.exists():
        return {"exists": False}
    try:
        m = json.loads(MODEL_META.read_text(encoding="utf-8"))
    except Exception:
        return {"exists": True, "error": "json_parse_error"}
    return {
        "exists": True,
        "trained_at": m.get("trained_at"),
        "n": m.get("n"),
        "auc": m.get("auc"),
        "brier": m.get("brier"),
        "reliable": m.get("reliable"),
        "features": m.get("feature_names", []),
        "dropped": m.get("dropped_features", []),
    }




def _auc_orientation_check(auc_val: Any) -> dict[str, Any]:
    a = _to_float(auc_val)
    if a is None:
        return {"status": "unknown", "auc": None, "auc_if_flipped": None, "message": "AUC no disponible."}
    if a < 0.50:
        return {
            "status": "possible_inversion",
            "auc": round(a, 6),
            "auc_if_flipped": round(1.0 - a, 6),
            "message": "AUC<0.50: revisar posible inversión de orientación (p -> 1-p).",
        }
    return {
        "status": "ok",
        "auc": round(a, 6),
        "auc_if_flipped": round(1.0 - a, 6),
        "message": "Orientación AUC aparentemente correcta.",
    }

def build_actions(diag: dict[str, Any]) -> list[str]:
    actions: list[str] = []

    inc = diag["incremental"]
    if inc.get("rows", 0) < 400:
        actions.append("Subir muestra incremental a >=400 filas cerradas antes de cambios estructurales.")

    auc_check = diag.get("auc_orientation", {})
    if str(auc_check.get("status")) == "possible_inversion":
        actions.append(
            f"Revisar orientación del modelo: AUC={auc_check.get('auc')} (<0.50), AUC invertida estimada={auc_check.get('auc_if_flipped')}."
        )

    fvol = inc.get("features", {}).get("volatilidad", {})
    fhb = inc.get("features", {}).get("hora_bucket", {})
    if fvol.get("uniq", 0) <= 1:
        actions.append("Corregir generación de volatilidad en origen bot: actualmente está plana en incremental.")
    if fhb.get("uniq", 0) <= 1:
        actions.append("Corregir generación de hora_bucket en origen bot: actualmente está plano en incremental.")

    sig = diag["signals"]
    t70 = sig.get("by_threshold", {}).get("0.7", {})
    if t70.get("n", 0) > 0:
        wr70 = t70.get("win_rate", 0.0) or 0.0
        if wr70 < 0.60:
            actions.append("Mantener gate conservador (LB + evidencia). No forzar gatillo duro 70% todavía.")

    b = sig.get("bins", {})
    gaps = [abs(float(v.get("gap", 0.0))) for v in b.values() if int(v.get("n", 0)) > 0]
    if gaps and max(gaps) > 0.10:
        actions.append("Priorizar recalibración + shrink dinámico hasta reducir gap de bins altos a <10pp.")

    bot45 = next((x for x in diag["bots"] if x.get("bot") == "fulll45"), None)
    if bot45:
        dom = bot45.get("feature_dominance", {})
        high = [k for k, v in dom.items() if float(v.get("dominance", 0.0)) >= 0.90]
        if high:
            actions.append(f"Rediseñar features binarias dominantes a intensidades continuas: {', '.join(high)}.")

    if not actions:
        actions.append("Data y calibración en zona estable: continuar monitoreo semanal.")

    return actions


def build_markdown(diag: dict[str, Any]) -> str:
    md: list[str] = []
    md.append("# Diagnóstico unificado IA (bots 45-50 + incremental + señales)\n")
    md.append(f"- Generado (UTC): {diag['generated_at']}")

    meta = diag["model_meta"]
    auc_check = diag.get("auc_orientation", {})
    md.append("\n## Modelo actual")
    md.append(f"- reliable: {meta.get('reliable')}")
    md.append(f"- auc: {meta.get('auc')}")
    md.append(f"- brier: {meta.get('brier')}")
    md.append(f"- features activas: {meta.get('features')}")
    md.append(f"- chequeo orientación AUC: {auc_check.get('status')} | auc={auc_check.get('auc')} | auc_invertida={auc_check.get('auc_if_flipped')}")
    md.append(f"- nota AUC: {auc_check.get('message')}")

    inc = diag["incremental"]
    md.append("\n## Incremental")
    md.append(f"- filas: {inc.get('rows')}")
    md.append(f"- duplicados exactos: {inc.get('duplicates_exact')}")
    for k in ("volatilidad", "hora_bucket", "payout", "racha_actual"):
        d = inc.get("features", {}).get(k, {})
        md.append(f"- {k}: uniq={d.get('uniq')} dom={d.get('dominance')} top={d.get('top')}")

    md.append("\n## Bots (cierres y WR)")
    md.append("| Bot | Rows | Cerrados | Win | Loss | WR | Last TS |")
    md.append("|---|---:|---:|---:|---:|---:|---|")
    for b in diag["bots"]:
        wr = b.get("win_rate")
        wr_txt = "--" if wr is None else f"{wr*100:.2f}%"
        md.append(f"| {b['bot']} | {b['rows']} | {b['closed']} | {b['wins']} | {b['losses']} | {wr_txt} | {b.get('last_ts','')} |")

    sig = diag["signals"]
    md.append("\n## Señales IA cerradas")
    md.append(f"- total cerradas: {sig.get('closed')} (de {sig.get('rows')} filas de log)")
    for thr in ("0.65", "0.7", "0.75", "0.8"):
        r = sig.get("by_threshold", {}).get(thr, {})
        wr = r.get("win_rate")
        wr_txt = "--" if wr is None else f"{wr*100:.2f}%"
        md.append(f"- >= {float(thr)*100:.0f}%: n={r.get('n',0)}, hits={r.get('hits',0)}, wr={wr_txt}")

    md.append("\n## Bins altos (overconfidence)")
    md.append("| Bin | n | avg_pred | avg_real | gap |")
    md.append("|---|---:|---:|---:|---:|")
    for k, v in sig.get("bins", {}).items():
        md.append(f"| {k} | {v.get('n',0)} | {v.get('avg_pred',0):.4f} | {v.get('avg_real',0):.4f} | {v.get('gap',0):+.4f} |")

    md.append("\n## Acciones sugeridas")
    for i, a in enumerate(diag["actions"], start=1):
        md.append(f"{i}. {a}")

    return "\n".join(md) + "\n"


def main() -> int:
    diag: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_meta": summarize_model_meta(),
        "incremental": summarize_incremental(),
        "bots": [summarize_bot(b) for b in BOTS],
        "signals": summarize_signals(),
    }
    diag["auc_orientation"] = _auc_orientation_check(diag.get("model_meta", {}).get("auc"))
    diag["actions"] = build_actions(diag)

    (ROOT / "diagnostico_pipeline_ia.json").write_text(
        json.dumps(diag, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (ROOT / "diagnostico_pipeline_ia.md").write_text(build_markdown(diag), encoding="utf-8")

    print("Generado: diagnostico_pipeline_ia.json")
    print("Generado: diagnostico_pipeline_ia.md")
    print("Acciones sugeridas:")
    for i, a in enumerate(diag["actions"], start=1):
        print(f"  {i}. {a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
