"""
metrics.py - turn a SimulationResult into the numbers judges care about.

DEFINITIONS
-----------
wait time        minutes in a queue (ambulance dispatch queue + hospital queue).
                 Travel time is not waiting. Patients still waiting when the
                 simulation ends are CENSORED: their wait-so-far is counted, so a
                 policy cannot look good simply by never treating people.
utilization(r)   = mean_t( in_use_r(t) ) / nominal_capacity_r
                 i.e. the fraction of the resource's normal capacity that was busy.
SLA compliance   share of patients seen within their triage target
                 (SLA_TARGET_MIN). Patients still waiting are counted as a
                 breach only once they have already exceeded their target.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .engine import SimulationResult
from .models import ALL_RESOURCES, CARE_RESOURCES, SLA_TARGET_MIN, AMBULANCE_OFFLOAD_MIN

_NUMERIC = ["hospital_arrival", "dispatch_time", "start", "end"]


def patient_frame(res: SimulationResult) -> pd.DataFrame:
    """One row per patient with derived wait / SLA columns."""
    h = res.cfg.horizon_min
    cols = ["pid", "arrival", "triage", "by_ambulance", "needs_icu", "hospital_arrival",
            "dispatch_time", "start", "end", "stay", "wait", "static_score"]
    rows = [
        (p.pid, p.arrival, p.triage, p.by_ambulance, p.needs_icu, p.hospital_arrival,
         p.dispatch_time, p.start, p.end, p.stay, p.wait_minutes(h), res.policy.static_score(p))
        for p in res.patients
    ]
    df = pd.DataFrame(rows, columns=cols)
    for c in _NUMERIC:
        df[c] = pd.to_numeric(df[c])
    df["admitted"] = df["start"].notna()
    df["sla_target"] = df["triage"].map(SLA_TARGET_MIN)
    within = df["wait"] <= df["sla_target"]
    # 1.0 = met, 0.0 = breached, NaN = not yet decidable (still waiting, target not reached)
    df["sla_met"] = np.where(df["admitted"], within.astype(float), np.where(~within, 0.0, np.nan))
    return df


def kpis(res: SimulationResult, df: pd.DataFrame | None = None) -> Dict[str, float]:
    df = patient_frame(res) if df is None else df
    tl, cap = res.timeline, res.cfg.capacity
    n = len(df)

    def mean(s: pd.Series) -> float:
        return float(s.mean()) if len(s) else 0.0

    out: Dict[str, float] = {
        "arrivals": n,
        "admitted": int(df["admitted"].sum()),
        "still_waiting": int((~df["admitted"]).sum()),
        "throughput_per_hour": float(df["admitted"].sum()) / (res.cfg.horizon_min / 60.0),
        "avg_wait": mean(df["wait"]),
        "avg_wait_admitted": mean(df.loc[df["admitted"], "wait"]),
        "p90_wait": float(df["wait"].quantile(0.9)) if n else 0.0,
        "max_wait": float(df["wait"].max()) if n else 0.0,
        "avg_wait_critical": mean(df.loc[df["triage"] <= 2, "wait"]),
        "avg_wait_low": mean(df.loc[df["triage"] >= 4, "wait"]),
        "sla_pct": 100.0 * mean(df["sla_met"].dropna()),
        "peak_queue": float(tl["queue"].max()),
        "avg_queue": float(tl["queue"].mean()),
    }
    for r in ALL_RESOURCES:
        out[f"util_{r}"] = float(tl[f"use_{r}"].mean() / cap[r]) if cap[r] else 0.0
    out["util_mean"] = float(np.mean([out[f"util_{r}"] for r in CARE_RESOURCES]))
    return out


def per_triage_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("triage")
    t = pd.DataFrame({
        "patients": g.size(),
        "avg_wait": g["wait"].mean(),
        "p90_wait": g["wait"].quantile(0.9),
        "max_wait": g["wait"].max(),
        "sla_target": g["sla_target"].first(),
        "sla_pct": g["sla_met"].apply(lambda s: 100.0 * s.dropna().mean() if s.notna().any() else np.nan),
    })
    return t.reset_index()


def verify_no_violations(res: SimulationResult) -> Tuple[bool, Dict[str, int]]:
    """Independent audit: rebuild resource usage from the patient records alone and
    check it never exceeds nominal capacity. (The engine also guards every
    allocation; this is a second, separate proof.)"""
    size = res.cfg.horizon_min + 3200
    diff = {r: np.zeros(size, dtype=int) for r in ALL_RESOURCES}
    for p in res.patients:
        if p.start is not None:
            for r, n in p.stay_req.items():
                diff[r][p.start] += n
                diff[r][p.end] -= n
            for r, n in p.consult_req.items():
                diff[r][p.start] += n
                diff[r][p.start + p.consult] -= n
        if p.dispatch_time is not None:
            diff["ambulance"][p.dispatch_time] += 1
            diff["ambulance"][p.dispatch_time + 2 * p.pickup + AMBULANCE_OFFLOAD_MIN] -= 1
    peak = {r: int(np.cumsum(diff[r]).max()) for r in ALL_RESOURCES}
    ok = all(peak[r] <= res.cfg.capacity[r] for r in ALL_RESOURCES)
    return ok, peak
