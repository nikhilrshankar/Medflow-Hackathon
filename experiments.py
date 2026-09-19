"""
experiments.py - compare scheduling strategies fairly.

For each replication (a different random seed) we generate ONE stream of patients
and replay it under every policy ("common random numbers"). Differences between
policies are therefore caused by the policy, not by luck. Results are averaged
over replications.
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from .engine import run_simulation
from .generator import generate_patients
from .metrics import kpis, patient_frame, per_triage_table
from .models import SimConfig
from .policies import Policy


def compare_policies(cfg: SimConfig, policies: List[Policy], n_reps: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (summary per policy, wait-by-triage per policy), averaged over n_reps."""
    kpi_rows, tri_rows = [], []
    for rep in range(n_reps):
        patients = generate_patients(cfg, seed=cfg.seed + rep)
        for pol in policies:
            res = run_simulation(cfg, pol, patients)
            df = patient_frame(res)
            kpi_rows.append({"policy": pol.name, "rep": rep, **kpis(res, df)})
            tri = per_triage_table(df)
            tri.insert(0, "policy", pol.name)
            tri.insert(1, "rep", rep)
            tri_rows.append(tri)
    order = [p.name for p in policies]
    summary = (pd.DataFrame(kpi_rows).drop(columns="rep")
               .groupby("policy").mean(numeric_only=True).loc[order].reset_index())
    tri = (pd.concat(tri_rows).drop(columns="rep")
           .groupby(["policy", "triage"], sort=False).mean(numeric_only=True).reset_index())
    return summary, tri
