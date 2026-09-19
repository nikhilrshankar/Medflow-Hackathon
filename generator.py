"""
generator.py - synthetic patient arrivals.

ARRIVAL PROCESS
---------------
Non-homogeneous Poisson process: in every simulated minute the number of new
patients is  N(t) ~ Poisson(lambda(t))  with

    lambda(t) = base_rate * surge(t) * (1 + A * sin(2*pi*(hour - 8)/24))

so we get a daily rhythm (busiest mid-afternoon) plus optional emergency surges.

SERVICE TIMES
-------------
Log-normal (positive, right-skewed: most stays are near the mean, a few are very
long), the standard choice for length-of-stay data. We parametrise by the MEAN:
    mu = ln(mean) - sigma^2 / 2   =>   E[X] = mean.

Patients are generated ONCE per seed and then replayed against every scheduling
policy ("common random numbers"), so policy comparisons are apples-to-apples.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from .models import (
    AMBULANCE_SHARE,
    ICU_NURSES,
    ICU_STAY_MULTIPLIER,
    TRIAGE_PROFILE,
    Patient,
    SimConfig,
)


def _lognormal(rng: np.random.Generator, mean: float, sigma: float) -> float:
    mu = math.log(mean) - sigma ** 2 / 2.0
    return float(rng.lognormal(mu, sigma))


def generate_patients(cfg: SimConfig, seed: Optional[int] = None) -> List[Patient]:
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    levels = sorted(TRIAGE_PROFILE)
    mix = np.array([TRIAGE_PROFILE[k]["mix"] for k in levels], dtype=float)
    mix /= mix.sum()

    patients: List[Patient] = []
    pid = 0
    for t in range(cfg.horizon_min):
        for _ in range(rng.poisson(cfg.arrival_rate_per_min(t))):
            triage = int(rng.choice(levels, p=mix))
            prof = TRIAGE_PROFILE[triage]
            by_ambulance = bool(rng.random() < AMBULANCE_SHARE[triage])
            needs_icu = bool(rng.random() < prof["icu_p"])

            stay_mean = prof["stay"] * (ICU_STAY_MULTIPLIER if needs_icu else 1.0)
            stay = int(np.clip(round(_lognormal(rng, stay_mean, 0.5)), 10, 1440))
            consult = int(np.clip(round(_lognormal(rng, prof["consult"], 0.4)), 5, stay))
            pickup = int(rng.integers(6, 21)) if by_ambulance else 0

            bed = "icu_bed" if needs_icu else "general_bed"
            stay_req = {bed: 1, "nurse": ICU_NURSES if needs_icu else 1}
            consult_req = {"doctor": prof["doctors"]}

            patients.append(
                Patient(
                    pid=pid, arrival=t, triage=triage, by_ambulance=by_ambulance,
                    needs_icu=needs_icu, stay=stay, consult=consult, pickup=pickup,
                    stay_req=stay_req, consult_req=consult_req,
                )
            )
            pid += 1
    return patients
