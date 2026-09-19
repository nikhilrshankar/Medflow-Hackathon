"""
engine.py - the discrete-time hospital simulation (1-minute ticks).

PATIENT JOURNEY
---------------
  walk-in   : arrival -> [hospital queue] -> treatment -> discharge
  ambulance : call -> [dispatch queue] -> vehicle travels (pickup) -> [hospital queue]
              -> treatment -> discharge          (vehicle busy for 2*pickup + offload)

WHAT HAPPENS EVERY MINUTE (in this order)
-----------------------------------------
  1. Release   resources whose holding time has expired (doctor after the consult,
               bed + nurses after the stay, ambulance after it is back).
  2. Arrivals  new patients enter the hospital queue (walk-ins) or dispatch queue.
  3. Transit   ambulance patients that have reached the hospital join the queue.
  4. Dispatch  free ambulances are assigned to the highest-priority callers.
  5. Admit     scan the hospital queue in priority order and start treatment for
               every patient whose FULL requirement fits (see below).
  6. Record    snapshot of queues / resource usage for the dashboard.

CONFLICT-FREE ALLOCATION
------------------------
* Atomic, all-or-nothing: a patient gets bed + nurse(s) + doctor(s) in one step or
  nothing at all. Nobody ever holds half a set while waiting -> no deadlock.
* ResourcePool.allocate() raises CapacityViolation if in_use + request > capacity,
  or if the same patient is allocated twice -> over-booking is impossible.
* No pre-emption: when staff are reduced (shortage), people already treating
  patients finish; only NEW allocations see the smaller capacity.

BACKFILLING WITH RESERVATION (anti-starvation)
----------------------------------------------
If the top-priority patient cannot start (say a triage-1 patient needs 2 doctors
but only 1 is free), lower-priority patients may still use resources that patient
does NOT need (e.g. general beds), but every resource he is SHORT of is reserved
for him. So: no idle beds because of one blocked ICU patient, and no starvation of
patients who need scarce resources.
"""
from __future__ import annotations

import copy
import heapq
import itertools
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional

import pandas as pd

from .generator import generate_patients
from .models import ALL_RESOURCES, AMBULANCE_OFFLOAD_MIN, CARE_RESOURCES, Patient, SimConfig
from .policies import Policy, PriorityQueue


class CapacityViolation(RuntimeError):
    """Raised if the simulation ever tries to over-book a resource."""


class ResourcePool:
    """Tracks how many units of each resource are in use, and who holds them."""

    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.in_use: Dict[str, int] = {r: 0 for r in ALL_RESOURCES}
        self.held: Dict[Hashable, Dict[str, int]] = {}   # holder key -> units held

    def available(self, t: int) -> Dict[str, int]:
        cap = self.cfg.capacity_at(t)
        return {r: max(0, cap[r] - self.in_use[r]) for r in ALL_RESOURCES}

    def allocate(self, key: Hashable, req: Dict[str, int], t: int) -> None:
        if key in self.held:
            raise CapacityViolation(f"double allocation for {key!r}")
        cap = self.cfg.capacity_at(t)
        for r, n in req.items():
            if self.in_use[r] + n > cap[r]:
                raise CapacityViolation(f"{r}: {self.in_use[r]}+{n} > {cap[r]} at t={t}")
        for r, n in req.items():
            self.in_use[r] += n
        self.held[key] = dict(req)

    def release(self, key: Hashable, req: Dict[str, int]) -> None:
        held = self.held[key]
        for r, n in req.items():
            if held.get(r, 0) < n:
                raise CapacityViolation(f"{key!r} releasing more {r} than it holds")
            held[r] -= n
            self.in_use[r] -= n
        if not any(held.values()):
            del self.held[key]


@dataclass
class SimulationResult:
    cfg: SimConfig
    policy: Policy
    patients: List[Patient]
    timeline: pd.DataFrame      # one row per simulated minute


class HospitalSimulation:
    def __init__(self, cfg: SimConfig, policy: Policy, patients: List[Patient]) -> None:
        self.cfg = cfg
        self.policy = policy
        # Work on copies so the same patient list can be replayed under every policy.
        self.patients = [copy.copy(p) for p in patients]
        for p in self.patients:
            p.hospital_arrival = p.dispatch_time = p.start = p.end = None
        self.pool = ResourcePool(cfg)

    # ------------------------------------------------------------------
    def run(self) -> SimulationResult:
        cfg, pool = self.cfg, self.pool
        arrivals = sorted(self.patients, key=lambda p: (p.arrival, p.pid))
        i = 0
        hospital_q = PriorityQueue(self.policy)
        dispatch_q = PriorityQueue(self.policy)
        releases: list = []      # (time, seq, holder_key, req_dict)
        transit: list = []       # (hospital_arrival_time, seq, patient)
        seq = itertools.count()
        rows = []

        for t in range(cfg.horizon_min):
            # 1. release expired resources
            while releases and releases[0][0] <= t:
                _, _, key, req = heapq.heappop(releases)
                pool.release(key, req)

            # 2. arrivals
            while i < len(arrivals) and arrivals[i].arrival <= t:
                p = arrivals[i]
                i += 1
                if p.by_ambulance:
                    dispatch_q.push(p)
                else:
                    p.hospital_arrival = t
                    hospital_q.push(p)

            # 3. ambulances arriving at the hospital
            while transit and transit[0][0] <= t:
                _, _, p = heapq.heappop(transit)
                p.hospital_arrival = t
                hospital_q.push(p)

            # 4. dispatch ambulances to the highest-priority callers
            free_amb = pool.available(t)["ambulance"]
            while free_amb > 0 and len(dispatch_q):
                p = dispatch_q.pop()
                key = ("ambulance", p.pid)
                pool.allocate(key, {"ambulance": 1}, t)
                p.dispatch_time = t
                heapq.heappush(transit, (t + p.pickup, next(seq), p))
                back = t + 2 * p.pickup + AMBULANCE_OFFLOAD_MIN
                heapq.heappush(releases, (back, next(seq), key, {"ambulance": 1}))
                free_amb -= 1

            # 5. admit patients
            self._admit(t, hospital_q, releases, seq)

            # 6. record
            rows.append(self._snapshot(t, hospital_q, dispatch_q, len(transit)))

        return SimulationResult(cfg, self.policy, self.patients, pd.DataFrame(rows))

    # ------------------------------------------------------------------
    def _admit(self, t: int, hq: PriorityQueue, releases: list, seq) -> None:
        if not len(hq):
            return
        avail = self.pool.available(t)
        free = {r: avail[r] for r in CARE_RESOURCES}
        deferred: List[Patient] = []

        while len(hq) and any(v > 0 for v in free.values()):
            p = hq.pop()
            req = p.total_req
            short = [r for r, n in req.items() if free[r] < n]
            if not short:
                self.pool.allocate(p.pid, req, t)          # atomic, validated
                for r, n in req.items():
                    free[r] -= n
                p.start, p.end = t, t + p.stay
                # doctor(s) are freed after the consult, bed + nurse(s) after the stay
                heapq.heappush(releases, (t + p.consult, next(seq), p.pid, dict(p.consult_req)))
                heapq.heappush(releases, (t + p.stay, next(seq), p.pid, dict(p.stay_req)))
            else:
                for r in short:          # reserve whatever this patient is short of
                    free[r] = 0
                deferred.append(p)

        for p in deferred:               # blocked patients keep their place in line
            hq.push(p)

    def _snapshot(self, t: int, hq: PriorityQueue, dq: PriorityQueue, in_transit: int) -> dict:
        cap = self.cfg.capacity_at(t)
        row = {
            "t": t,
            "hour": t / 60.0,
            "queue": len(hq),
            "dispatch_queue": len(dq),
            "in_transit": in_transit,
            "oldest_wait": hq.oldest_wait(t),
        }
        for r in ALL_RESOURCES:
            row[f"use_{r}"] = self.pool.in_use[r]
            row[f"cap_{r}"] = cap[r]
        return row


def run_simulation(cfg: SimConfig, policy: Policy, patients: Optional[List[Patient]] = None) -> SimulationResult:
    """Convenience wrapper: generate patients from cfg.seed (if not given) and simulate."""
    if patients is None:
        patients = generate_patients(cfg)
    return HospitalSimulation(cfg, policy, patients).run()
