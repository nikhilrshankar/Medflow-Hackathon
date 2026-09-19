"""Run with:  pytest -q   (from the project root)"""
import random
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from medflow.engine import run_simulation
from medflow.generator import generate_patients
from medflow.metrics import patient_frame, verify_no_violations
from medflow.models import Disruption, SimConfig, Surge
from medflow.policies import PRESET_POLICIES

STRESS = SimConfig(
    seed=7,
    surges=[Surge(6 * 60, 10 * 60, 3.0)],
    disruptions=[Disruption("doctor", 8 * 60, 14 * 60, 0.4), Disruption("nurse", 8 * 60, 14 * 60, 0.3),
                 Disruption("icu_bed", 12 * 60, 16 * 60, 0.4)],
)


@pytest.mark.parametrize("policy", PRESET_POLICIES, ids=lambda p: p.name)
@pytest.mark.parametrize("cfg", [SimConfig(seed=1), STRESS], ids=["normal", "stress"])
def test_no_capacity_violations(cfg, policy):
    res = run_simulation(cfg, policy)               # engine raises CapacityViolation on over-booking
    ok, peak = verify_no_violations(res)            # independent re-derivation from patient records
    assert ok, peak


def test_deterministic():
    a = run_simulation(SimConfig(seed=3), PRESET_POLICIES[2])
    b = run_simulation(SimConfig(seed=3), PRESET_POLICIES[2])
    assert patient_frame(a).equals(patient_frame(b))


def test_heap_order_equals_live_score_order_at_any_time():
    """The 'aging' trick: ordering by the static key == ordering by the live score."""
    pats = generate_patients(SimConfig(seed=5))[:60]
    pol = PRESET_POLICIES[3]
    by_static = sorted(pats, key=lambda p: (-pol.static_score(p), p.arrival, p.pid))
    for t in (0, 500, 1400):
        by_live = sorted(pats, key=lambda p: (-round(pol.score(p, t), 9), p.arrival, p.pid))
        assert [p.pid for p in by_static] == [p.pid for p in by_live]


def test_queue_bookkeeping_identity():
    """Sum of queue length over time == total patient-minutes spent in the hospital queue."""
    res = run_simulation(SimConfig(seed=11), PRESET_POLICIES[2])
    h = res.cfg.horizon_min
    minutes = sum(((p.start if p.start is not None else h) - p.hospital_arrival)
                  for p in res.patients if p.hospital_arrival is not None)
    assert int(res.timeline["queue"].sum()) == minutes


def test_urgent_patients_wait_less_under_triage_than_fcfs():
    cfg = SimConfig(seed=2, arrivals_per_hour=11)
    fcfs, strict = PRESET_POLICIES[0], PRESET_POLICIES[1]
    w = lambda pol: patient_frame(run_simulation(cfg, pol)).query("triage <= 2")["wait"].mean()
    assert w(strict) < w(fcfs)
