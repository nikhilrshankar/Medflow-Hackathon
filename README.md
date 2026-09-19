# MEDFLOW - Hospital Resource Management Simulator

*Prioritize patients. Optimize resources.* Built for BMSCE **Hack-a-Matics** (theme: VECTOR).

A minute-by-minute simulator of a hospital emergency flow: patients arrive with different
urgency, wait in a priority queue, and are matched to beds, ICU beds, doctors, nurses and
ambulances - without ever over-booking a resource. A Streamlit operations dashboard shows waits,
utilization and queues live, lets you inject emergency surges / staff shortages / equipment
failures, and compares scheduling strategies head to head.

## Run it

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # opens http://localhost:8501
```

Run the tests (about 2 seconds):

```bash
pytest -q
```

## Project layout

```
app.py                 Streamlit dashboard (presentation only)
medflow/
  models.py            parameters, Patient, scenario (surge / disruption) config
  generator.py         Poisson arrivals, log-normal stays
  policies.py          priority score + heapq priority queue   <-- scheduling logic
  engine.py            minute-by-minute simulation + conflict-free resource pool
  metrics.py           wait / utilization / SLA + independent safety audit
  experiments.py       fair strategy comparison (common random numbers)
tests/test_medflow.py  invariants and property tests
```

## The modelling in 60 seconds

**Arrivals** - non-homogeneous Poisson process: `lambda(t) = lambda0 * surge(t) * (1 + A sin(2*pi*(h-8)/24))`.
Each patient gets a triage level (1-5), ICU need, ambulance use and a log-normal stay.

**Priority score** - `score(t) = U[triage] + alpha * (t - arrival) + gamma * 60 / E[stay]`
(clinical urgency + waiting-time aging + optional short-stay bonus).
Because `alpha * t` is the same for everyone, ordering by the *static* part is identical to ordering by the
live score, so a plain `heapq` is exact: no re-sorting as patients age.

**Allocation** - a patient needs bed + nurse(s) + doctor(s) *together*; they are granted atomically or not at all
(no deadlock). The resource pool raises an error on over-booking or double allocation. If the top patient is
blocked, lower-priority patients may use resources he does not need, but what he is short of is reserved for
him (backfilling with reservation). Doctors are released after the consult, beds/nurses after the stay,
ambulances after the round trip.

**Stress tests** - surges multiply `lambda(t)`; staff shortages / equipment failures reduce capacity for a
window (no pre-emption: people already treating a patient finish).

**Metrics** - average / 90th-percentile wait (patients still waiting at the end are counted, not ignored),
utilization = mean in-use / normal capacity, share of patients seen within a per-triage target, queue length over time.

**Strategies compared** - FCFS, strict urgency, urgency + aging, urgency + aging + throughput bonus. All see the
same patients (common random numbers), averaged over several random days.

## Validation

* `verify_no_violations()` rebuilds resource usage from patient records alone and checks it never exceeds capacity.
* Tests check determinism, heap order = live-score order at any time, and that time-integrated queue length equals
  total patient-minutes waited.

## Disclosures (per hackathon rules)

* Libraries: Streamlit, Pandas, NumPy, Plotly (open source), pytest.
* Code was drafted with an AI coding assistant (Claude) and then reviewed, adapted and tested by the team.
* **AI component:** *(fill in honestly before submitting - see notes)*. The baseline simulator is
  classical stochastic simulation + priority scheduling and contains no machine-learning model.

## Suggested 2-3 minute demo flow

1. Patient arrivals + priority calculation (Live tab, scrub the clock, show the queue table and scores).
2. Resource allocation (utilization bars; mention atomic allocation + safety audit banner).
3. Scheduling simulation under stress (enable surge + staff shortage, press Play).
4. Performance statistics (KPIs, wait vs target) and the Strategy comparison tab.
