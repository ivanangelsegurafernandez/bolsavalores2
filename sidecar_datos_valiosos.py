#!/usr/bin/env python3
"""
Sidecar de auditoría para IA/REAL_SIM.
No opera ni toca tokens: solo lee logs y produce reportes.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_TICK_LOG = ROOT / "ia_signal_ticks_log.csv"
DEFAULT_REAL_SIM_STATE = ROOT / "real_sim_state.json"
DEFAULT_PROMOS = ROOT / "registro_promociones.txt"


@dataclass
class TickRow:
    ts: datetime
    bot: str
    prob: float
    piso: float
    roof_eff: float
    confirm_streak: int
    block_reason: str
    modo_exec: str
    modo_saldo: str


def _parse_ts(value: str) -> datetime | None:
    if not value:
        return None
    txt = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(txt)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def load_tick_rows(path: Path) -> list[TickRow]:
    out: list[TickRow] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            ts = _parse_ts(r.get("ts", ""))
            if ts is None:
                continue
            try:
                out.append(
                    TickRow(
                        ts=ts,
                        bot=str(r.get("bot", "")).strip(),
                        prob=float(r.get("prob", 0.0) or 0.0),
                        piso=float(r.get("piso", 0.0) or 0.0),
                        roof_eff=float(r.get("roof_eff", 0.0) or 0.0),
                        confirm_streak=int(float(r.get("confirm_streak", 0) or 0)),
                        block_reason=str(r.get("block_reason", "--") or "--").strip() or "--",
                        modo_exec=str(r.get("modo_exec", "") or "").strip().upper() or "UNKNOWN",
                        modo_saldo=str(r.get("modo_saldo", "") or "").strip().upper() or "UNKNOWN",
                    )
                )
            except Exception:
                continue
    return out


def load_bot_closes() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(glob.glob(str(ROOT / "registro_enriquecido_fulll*.csv"))):
        bot = Path(p).stem.replace("registro_enriquecido_", "")
        total = wins = losses = 0
        first_ts = last_ts = None
        with open(p, "r", encoding="utf-8", errors="replace", newline="") as f:
            rd = csv.DictReader(f)
            for r in rd:
                ts = _parse_ts(r.get("ts", ""))
                if ts is not None:
                    first_ts = ts if first_ts is None else min(first_ts, ts)
                    last_ts = ts if last_ts is None else max(last_ts, ts)
                if str(r.get("trade_status", "")).upper() != "CERRADO":
                    continue
                total += 1
                res = str(r.get("resultado", "")).upper()
                if "GANANCIA" in res:
                    wins += 1
                elif "PÉRDIDA" in res or "PERDIDA" in res:
                    losses += 1
        wr = (wins / total) if total > 0 else 0.0
        out[bot] = {
            "closed": total,
            "wins": wins,
            "losses": losses,
            "win_rate_closed": wr,
            "first_ts": first_ts.isoformat() if first_ts else None,
            "last_ts": last_ts.isoformat() if last_ts else None,
        }
    return out


def load_promotions(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Token REAL (inmediato) asignado" in line:
                n += 1
    return n


def load_real_sim_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"exists": True, "parse_error": True}
    return {
        "exists": True,
        "balance": float(d.get("balance", 0.0) or 0.0),
        "reserved": float(d.get("reserved", 0.0) or 0.0),
        "available": float(d.get("available", (float(d.get("balance", 0.0) or 0.0) - float(d.get("reserved", 0.0) or 0.0))) or 0.0),
        "updated_at": d.get("updated_at"),
    }


def build_report(threshold: float, tick_rows: list[TickRow]) -> dict[str, Any]:
    rows_thr = [r for r in tick_rows if r.prob >= threshold]
    by_bot: dict[str, dict[str, Any]] = {}
    by_hour = Counter()
    by_bot_hour: dict[str, Counter] = defaultdict(Counter)
    block_reasons = Counter(r.block_reason for r in tick_rows)
    modes = Counter(f"{r.modo_exec}|{r.modo_saldo}" for r in tick_rows)

    bot_groups: dict[str, list[TickRow]] = defaultdict(list)
    for r in rows_thr:
        bot_groups[r.bot].append(r)
        h = r.ts.astimezone(timezone.utc).strftime("%H:00")
        by_hour[h] += 1
        by_bot_hour[r.bot][h] += 1

    for bot, rr in sorted(bot_groups.items()):
        probs = [x.prob for x in rr]
        by_bot[bot] = {
            "n_ge_thr": len(rr),
            "prob_avg_ge_thr": sum(probs) / len(probs),
            "prob_max_ge_thr": max(probs),
            "first_ts": min(x.ts for x in rr).isoformat(),
            "last_ts": max(x.ts for x in rr).isoformat(),
            "top_hours_utc": by_bot_hour[bot].most_common(6),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "tick_rows_total": len(tick_rows),
        "tick_rows_ge_threshold": len(rows_thr),
        "by_bot": by_bot,
        "by_hour_utc": dict(sorted(by_hour.items())),
        "block_reasons_all_ticks": dict(block_reasons.most_common()),
        "modes_all_ticks": dict(modes.most_common()),
    }


def render_md(rep: dict[str, Any], bot_closes: dict[str, dict[str, Any]], real_sim: dict[str, Any], promo_count: int) -> str:
    lines: list[str] = []
    lines.append("# Sidecar Reporte Datos Valiosos")
    lines.append("")
    lines.append(f"- Generado UTC: {rep.get('generated_at')}")
    lines.append(f"- Umbral analizado: >= {float(rep.get('threshold', 0.65))*100:.0f}%")
    lines.append(f"- Filas tick totales: {rep.get('tick_rows_total', 0)}")
    lines.append(f"- Filas tick >= umbral: {rep.get('tick_rows_ge_threshold', 0)}")
    lines.append(f"- Promociones registradas (histórico): {promo_count}")
    lines.append("")

    lines.append("## Estado REAL_SIM")
    if not real_sim.get("exists"):
        lines.append("- real_sim_state.json: no existe todavía.")
    else:
        lines.append(f"- Balance: {float(real_sim.get('balance', 0.0)):.2f}")
        lines.append(f"- Reservado: {float(real_sim.get('reserved', 0.0)):.2f}")
        lines.append(f"- Disponible: {float(real_sim.get('available', 0.0)):.2f}")
        lines.append(f"- Updated at: {real_sim.get('updated_at')}")
    lines.append("")

    lines.append("## Prob IA >= umbral por bot")
    lines.append("| Bot | n>=thr | Prob media | Prob max | Primera | Última |")
    lines.append("|---|---:|---:|---:|---|---|")
    for bot, d in sorted((rep.get("by_bot") or {}).items()):
        lines.append(
            f"| {bot} | {int(d.get('n_ge_thr',0))} | {float(d.get('prob_avg_ge_thr',0.0))*100:.2f}% | "
            f"{float(d.get('prob_max_ge_thr',0.0))*100:.2f}% | {d.get('first_ts','--')} | {d.get('last_ts','--')} |"
        )
    lines.append("")

    lines.append("## Frecuencia por hora UTC (prob>=umbral)")
    lines.append("| Hora UTC | Frecuencia |")
    lines.append("|---|---:|")
    for h, n in sorted((rep.get("by_hour_utc") or {}).items()):
        lines.append(f"| {h} | {int(n)} |")
    lines.append("")

    lines.append("## Bloqueos más frecuentes (todos los ticks)")
    lines.append("| Block reason | n |")
    lines.append("|---|---:|")
    for k, v in (rep.get("block_reasons_all_ticks") or {}).items():
        lines.append(f"| {k} | {int(v)} |")
    lines.append("")

    lines.append("## Modo de ejecución observado (todos los ticks)")
    lines.append("| Modo exec|saldo | n |")
    lines.append("|---|---:|")
    for k, v in (rep.get("modes_all_ticks") or {}).items():
        lines.append(f"| {k} | {int(v)} |")
    lines.append("")

    lines.append("## Contexto cierres por bot (registro_enriquecido)")
    lines.append("| Bot | Cerrados | Wins | Losses | WinRate | Último ts |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for bot, d in sorted(bot_closes.items()):
        lines.append(
            f"| {bot} | {int(d.get('closed',0))} | {int(d.get('wins',0))} | {int(d.get('losses',0))} | "
            f"{float(d.get('win_rate_closed',0.0))*100:.2f}% | {d.get('last_ts','--')} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Sidecar auditoría IA/REAL_SIM")
    ap.add_argument("--threshold", type=float, default=0.65, help="Umbral de probabilidad (0..1)")
    ap.add_argument("--tick-log", type=Path, default=DEFAULT_TICK_LOG, help="Ruta de ia_signal_ticks_log.csv")
    ap.add_argument("--out-json", type=Path, default=ROOT / "sidecar_datos_valiosos_report.json")
    ap.add_argument("--out-md", type=Path, default=ROOT / "sidecar_datos_valiosos_report.md")
    args = ap.parse_args()

    thr = max(0.0, min(1.0, float(args.threshold)))
    ticks = load_tick_rows(args.tick_log)
    rep = build_report(thr, ticks)
    rep["real_sim_state"] = load_real_sim_state(DEFAULT_REAL_SIM_STATE)
    rep["promotions_total"] = load_promotions(DEFAULT_PROMOS)
    rep["bot_closes"] = load_bot_closes()

    args.out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render_md(rep, rep["bot_closes"], rep["real_sim_state"], int(rep["promotions_total"]))
    args.out_md.write_text(md, encoding="utf-8")

    print(f"OK JSON: {args.out_json}")
    print(f"OK MD:   {args.out_md}")
    print(f"Ticks leídos: {len(ticks)} | >=thr: {int(rep.get('tick_rows_ge_threshold', 0))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
