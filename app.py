"""
app.py - MEDFLOW operations dashboard (Streamlit).

Run:  streamlit run app.py

The page only *presents* results. All logic lives in the `medflow/` package:
    models.py      parameters, Patient, scenario config
    generator.py   Poisson arrivals + log-normal stays
    policies.py    priority score + heapq queue
    engine.py      minute-by-minute simulation + conflict-free resource pool
    metrics.py     wait / utilization / SLA + independent audit
    experiments.py fair policy comparison (common random numbers)
"""
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from medflow.engine import run_simulation
from medflow.experiments import compare_policies
from medflow.metrics import kpis, patient_frame, per_triage_table, verify_no_violations
from medflow.models import (
    ALL_RESOURCES, DEFAULT_CAPACITY, RESOURCE_LABELS, SLA_TARGET_MIN, TRIAGE_COLORS,
    TRIAGE_LABELS, Disruption, SimConfig, Surge,
)
from medflow.policies import POLICY_BY_NAME, PRESET_POLICIES, URGENCY_WEIGHTS, Policy

st.set_page_config(page_title="MEDFLOW | Hospital Resource Simulator", page_icon="🏥", layout="wide")

# ============================================================================
# Sidebar - scenario, resources, policy, stress tests
# ============================================================================
with st.sidebar:
    st.title("🏥 MEDFLOW")
    st.caption("Prioritize patients. Optimize resources.")

    st.header("Scenario")
    hours = st.select_slider("Simulated horizon (hours)", options=[12, 24, 48, 72], value=24)
    lam = st.slider("Baseline arrivals per hour", 2.0, 20.0, 8.0, 0.5)
    rhythm = st.slider("Daily rhythm strength", 0.0, 0.6, 0.3, 0.05,
                       help="Arrival rate swings +/- this fraction around the mean; busiest mid-afternoon.")
    seed = st.number_input("Random seed", 0, 100000, 42, help="Same seed = same patients, so runs are reproducible.")

    st.header("Resources")
    cap = {}
    for r in ALL_RESOURCES:
        cap[r] = int(st.number_input(RESOURCE_LABELS[r], 1, 200, DEFAULT_CAPACITY[r], key=f"cap_{r}"))

    st.header("Scheduling policy")
    policy_names = [p.name for p in PRESET_POLICIES] + ["Custom (tune below)"]
    choice = st.selectbox("Strategy", policy_names, index=2)
    aging = st.slider("Custom: aging rate (points/min waited)", 0.0, 1.0, 0.25, 0.05)
    gamma = st.slider("Custom: throughput bonus", 0.0, 30.0, 0.0, 1.0)
    if choice.startswith("Custom"):
        policy = Policy("Custom", dict(URGENCY_WEIGHTS), aging, gamma)
    else:
        policy = POLICY_BY_NAME[choice]

    st.header("Stress tests")
    surges, disruptions = [], []
    with st.expander("🚨 Emergency patient surge"):
        if st.checkbox("Enable surge", key="surge_on"):
            s0 = st.slider("Starts at hour", 0, hours - 1, min(6, hours - 1), key="s0")
            sd = st.slider("Lasts (hours)", 1, 12, 4, key="sd")
            sm = st.slider("Arrival-rate multiplier", 1.5, 6.0, 3.0, 0.5, key="sm")
            surges.append(Surge(s0 * 60, min(hours, s0 + sd) * 60, sm))
    with st.expander("🧑‍⚕️ Staff shortage"):
        if st.checkbox("Enable staff shortage", key="short_on"):
            f0 = st.slider("Starts at hour", 0, hours - 1, min(8, hours - 1), key="f0")
            fd = st.slider("Lasts (hours)", 1, 24, 6, key="fd")
            dl = st.slider("Doctors unavailable (%)", 0, 100, 40, 5, key="dl")
            nl = st.slider("Nurses unavailable (%)", 0, 100, 30, 5, key="nl")
            disruptions += [Disruption("doctor", f0 * 60, min(hours, f0 + fd) * 60, dl / 100),
                            Disruption("nurse", f0 * 60, min(hours, f0 + fd) * 60, nl / 100)]
    with st.expander("🛠️ Equipment / resource failure"):
        if st.checkbox("Enable resource failure", key="fail_on"):
            fr = st.selectbox("Resource", ["icu_bed", "general_bed", "ambulance"],
                              format_func=lambda r: RESOURCE_LABELS[r])
            e0 = st.slider("Starts at hour", 0, hours - 1, min(12, hours - 1), key="e0")
            ed = st.slider("Lasts (hours)", 1, 24, 4, key="ed")
            ep = st.slider("Capacity lost (%)", 10, 100, 40, 5, key="ep")
            disruptions.append(Disruption(fr, e0 * 60, min(hours, e0 + ed) * 60, ep / 100))

cfg = SimConfig(horizon_min=hours * 60, arrivals_per_hour=lam, diurnal_amplitude=rhythm,
                seed=int(seed), capacity=cap, surges=surges, disruptions=disruptions)


# ============================================================================
# Cached simulation runs (simulation is fast; caching keeps sliders snappy)
# ============================================================================
@st.cache_data(show_spinner=False)
def run_cached(cfg_json: str, policy_json: str):
    return run_simulation(SimConfig.from_json(cfg_json), Policy.from_json(policy_json))


@st.cache_data(show_spinner=False)
def compare_cached(cfg_json: str, n_reps: int, custom_json: str):
    pols = list(PRESET_POLICIES)
    if custom_json:
        pols.append(Policy.from_json(custom_json))
    return compare_policies(SimConfig.from_json(cfg_json), pols, n_reps)


res = run_cached(cfg.to_json(), policy.to_json())
pdf = patient_frame(res)
k = kpis(res, pdf)
tl = res.timeline
ok, peak_use = verify_no_violations(res)


# ============================================================================
# Helpers
# ============================================================================
def clock(t: int) -> str:
    return f"Day {t // 1440 + 1}, {t % 1440 // 60:02d}:{t % 60:02d}"


def shade_events(fig, **kw):
    """Shade surge / disruption windows (x axis in hours)."""
    for s in cfg.surges:
        fig.add_vrect(x0=s.start / 60, x1=s.end / 60, fillcolor="red", opacity=0.10, line_width=0,
                      annotation_text="surge", annotation_position="top left", **kw)
    for d in cfg.disruptions:
        fig.add_vrect(x0=d.start / 60, x1=d.end / 60, fillcolor="orange", opacity=0.10, line_width=0,
                      annotation_text=f"{RESOURCE_LABELS[d.resource].lower()} -{int(d.fraction * 100)}%",
                      annotation_position="bottom left", **kw)


def waiting_table(t: int, top: int = 12) -> pd.DataFrame:
    """Everyone in a queue at minute t, ordered by their LIVE priority score."""
    in_hosp = pdf["hospital_arrival"].le(t) & (pdf["start"].isna() | (pdf["start"] > t))
    in_disp = pdf["by_ambulance"] & pdf["arrival"].le(t) & (pdf["dispatch_time"].isna() | (pdf["dispatch_time"] > t))
    w = pdf[in_hosp | in_disp].copy()
    if w.empty:
        return w
    w["Stage"] = np.where(in_disp[w.index], "awaiting ambulance", "hospital queue")
    w["Waiting (min)"] = t - w["arrival"]
    w["Priority score"] = (w["static_score"] + res.policy.aging_rate * t).round(1)
    w["Needs"] = np.where(w["needs_icu"], "ICU bed", "ward bed")
    w["Triage"] = w["triage"]
    w["Patient"] = "P" + w["pid"].astype(str)
    return (w.sort_values("Priority score", ascending=False)
             [["Patient", "Triage", "Needs", "Stage", "Waiting (min)", "Priority score"]].head(top))


def queue_figure(t_now=None, height=300):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=tl["hour"], y=tl["queue"], name="Hospital queue", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=tl["hour"], y=tl["dispatch_queue"], name="Awaiting ambulance",
                             line=dict(color="#9467bd")))
    fig.add_trace(go.Scatter(x=tl["hour"], y=tl["oldest_wait"], name="Longest wait (min)",
                             line=dict(color="#d62728", dash="dot")), secondary_y=True)
    shade_events(fig)
    if t_now is not None:
        fig.add_vline(x=t_now / 60, line_color="black", line_width=2)
    fig.update_yaxes(title_text="Patients", secondary_y=False)
    fig.update_yaxes(title_text="Minutes", secondary_y=True)
    fig.update_xaxes(title_text="Hour of simulation")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.15))
    return fig


def render_live(t: int, key: str):
    row = tl.iloc[min(t, len(tl) - 1)]
    st.markdown(
        f"**{clock(t)}** &nbsp;·&nbsp; waiting in hospital: **{int(row['queue'])}** "
        f"&nbsp;·&nbsp; awaiting ambulance: **{int(row['dispatch_queue'])}** "
        f"&nbsp;·&nbsp; ambulances en route: **{int(row['in_transit'])}**"
    )
    cols = st.columns(len(ALL_RESOURCES))
    for col, r in zip(cols, ALL_RESOURCES):
        use, eff, nom = int(row[f"use_{r}"]), int(row[f"cap_{r}"]), cap[r]
        label = f"{RESOURCE_LABELS[r]}: {use}/{eff}" + (f" (normally {nom})" if eff != nom else "")
        col.progress(min(1.0, use / nom) if nom else 0.0, text=label)
    left, right = st.columns([3, 2])
    with left:
        st.caption("Queue, highest live priority first")
        tbl = waiting_table(t)
        if tbl.empty:
            st.success("Nobody is waiting right now.")
        else:
            st.dataframe(tbl, hide_index=True, width="stretch", key=f"tbl_{key}")
    with right:
        st.plotly_chart(queue_figure(t, height=280), width="stretch", key=f"qfig_{key}")


# ============================================================================
# Header + KPI row
# ============================================================================
st.title("🏥 MEDFLOW - Hospital Resource Operations Dashboard")
st.caption(f"Policy: **{policy.name}** · {len(res.patients)} patients simulated over {hours} h · seed {cfg.seed}")

c = st.columns(6)
c[0].metric("Average wait", f"{k['avg_wait']:.0f} min", help="All patients, incl. those still waiting at the end.")
c[1].metric("Critical wait (triage 1-2)", f"{k['avg_wait_critical']:.0f} min")
c[2].metric("Within wait target", f"{k['sla_pct']:.0f}%", help="Targets (min): " + ", ".join(
    f"T{t}: {m}" for t, m in SLA_TARGET_MIN.items()))
c[3].metric("Resource utilization", f"{100 * k['util_mean']:.0f}%", help="Mean of beds, ICU, doctors, nurses.")
c[4].metric("Peak queue", f"{k['peak_queue']:.0f} patients")
c[5].metric("Treated / arrived", f"{k['admitted']} / {k['arrivals']}", delta=f"-{k['still_waiting']} waiting", delta_color="inverse")

if ok:
    st.success("✅ Safety audit passed: zero capacity violations and zero double-allocations "
               "(re-derived independently from patient records).", icon=None)
else:
    st.error(f"❌ Capacity violation detected: {peak_use}")

tab_live, tab_compare, tab_log, tab_model = st.tabs(
    ["📈 Live operations", "🧭 Strategy comparison", "👥 Patient log", "🧮 Model & logic"])

# ============================================================================
# Tab 1 - live operations
# ============================================================================
with tab_live:
    a, b = st.columns([6, 1])
    t_hours = a.slider("Simulation clock (hours)", 0.0, float(hours), float(hours) / 2, 0.25)
    play = b.button("▶ Play", help="Animate the whole run", width="stretch")
    board = st.empty()
    t_view = int(min(t_hours * 60, cfg.horizon_min - 1))
    if play:
        step = max(10, cfg.horizon_min // 80)
        for tt in range(0, cfg.horizon_min, step):
            with board.container():
                render_live(tt, key=f"play{tt}")
            time.sleep(0.03)
    with board.container():
        render_live(t_view, key="static")

    st.divider()
    r1, r2 = st.columns(2)
    with r1:
        st.subheader("Queue length over time")
        st.plotly_chart(queue_figure(), width="stretch", key="queue_main")
    with r2:
        st.subheader("Resource utilization over time")
        fig = go.Figure()
        for r in ALL_RESOURCES:
            y = 100 * tl[f"use_{r}"].rolling(30, min_periods=1).mean() / cap[r]
            fig.add_trace(go.Scatter(x=tl["hour"], y=y, name=RESOURCE_LABELS[r]))
        shade_events(fig)
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="% of normal capacity (30-min avg)",
                          xaxis_title="Hour of simulation", legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch", key="util_main")

    r3, r4 = st.columns(2)
    with r3:
        st.subheader("Average wait by hour of treatment start")
        adm = pdf[pdf["admitted"]].copy()
        adm["hour"] = (adm["start"] // 60).astype(int)
        adm["Group"] = pd.cut(adm["triage"], [0, 2, 3, 5], labels=["Critical (1-2)", "Urgent (3)", "Low acuity (4-5)"])
        g = adm.groupby(["hour", "Group"], observed=True)["wait"].mean().reset_index()
        fig = px.line(g, x="hour", y="wait", color="Group", markers=True,
                      color_discrete_sequence=["#d62728", "#e0b000", "#1f77b4"])
        shade_events(fig)
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Wait (min)",
                          xaxis_title="Hour", legend=dict(orientation="h", y=1.15, title=None))
        st.plotly_chart(fig, width="stretch", key="wait_hour")
    with r4:
        st.subheader("Wait per triage level vs target")
        tri = per_triage_table(pdf)
        fig = go.Figure()
        fig.add_bar(x=[TRIAGE_LABELS[t] for t in tri["triage"]], y=tri["avg_wait"], name="Average",
                    marker_color=[TRIAGE_COLORS[t] for t in tri["triage"]])
        fig.add_scatter(x=[TRIAGE_LABELS[t] for t in tri["triage"]], y=tri["p90_wait"], name="90th percentile",
                        mode="markers", marker=dict(symbol="diamond", size=11, color="black"))
        fig.add_scatter(x=[TRIAGE_LABELS[t] for t in tri["triage"]], y=tri["sla_target"], name="Target",
                        mode="markers", marker=dict(symbol="line-ew", size=26, line=dict(width=3, color="green")))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Wait (min)",
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch", key="wait_triage")

# ============================================================================
# Tab 2 - strategy comparison
# ============================================================================
with tab_compare:
    st.subheader("Which scheduling strategy is best?")
    st.write("Every strategy sees **exactly the same patients** (same random arrivals), so differences come "
             "from the scheduling rule, not luck. Results are averaged over several random days.")
    n_reps = st.slider("Random days to average", 1, 15, 5)
    include_custom = st.checkbox("Also include the Custom strategy from the sidebar", value=choice.startswith("Custom"))
    if st.button("Run comparison", type="primary"):
        with st.spinner("Simulating..."):
            summ, tri = compare_cached(cfg.to_json(), n_reps, Policy("Custom", dict(URGENCY_WEIGHTS), aging, gamma).to_json()
                                       if include_custom else "")
        st.session_state["cmp"] = (summ, tri, n_reps)
    if "cmp" in st.session_state:
        summ, tri, reps = st.session_state["cmp"]
        st.caption(f"Averages over {reps} random days for the scenario currently set in the sidebar "
                   "(re-run after changing it).")
        show = summ[["policy", "avg_wait", "avg_wait_critical", "avg_wait_low", "p90_wait", "max_wait",
                     "sla_pct", "util_mean", "still_waiting"]].copy()
        show["util_mean"] = 100 * show["util_mean"]
        show.columns = ["Strategy", "Avg wait", "Critical wait (T1-2)", "Low-acuity wait (T4-5)", "90th pct wait",
                        "Longest wait", "Within target %", "Utilization %", "Still waiting at end"]
        st.dataframe(show.round(1), hide_index=True, width="stretch")
        fig = px.bar(tri, x="triage", y="avg_wait", color="policy", barmode="group",
                     labels={"triage": "Triage level", "avg_wait": "Average wait (min)", "policy": "Strategy"})
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.15, title=None))
        st.plotly_chart(fig, width="stretch", key="cmp_bar")
        st.info("**How to read this:** FCFS is fair but ignores clinical urgency, so critical patients wait longer. "
                "Strict triage protects critical patients but lets low-acuity patients wait the longest (starvation). "
                "Adding *aging* trades a little critical-patient speed for much better worst-case waits; the "
                "throughput-aware variant additionally favours short stays to free beds sooner.")

# ============================================================================
# Tab 3 - patient log
# ============================================================================
with tab_log:
    log = pdf.drop(columns=["static_score"]).copy()
    log["triage"] = log["triage"].astype(int)
    st.dataframe(log, hide_index=True, width="stretch")
    st.download_button("Download CSV", log.to_csv(index=False).encode(), "medflow_patients.csv", "text/csv")

# ============================================================================
# Tab 4 - the maths, for the judges
# ============================================================================
with tab_model:
    st.markdown("### 1. Arrivals")
    st.latex(r"N(t)\sim \text{Poisson}(\lambda(t)),\qquad \lambda(t)=\lambda_0\cdot \text{surge}(t)\cdot\Big(1+A\sin\tfrac{2\pi(h-8)}{24}\Big)")
    st.write("Non-homogeneous Poisson process: a daily rhythm plus optional emergency surges. Triage level, "
             "ICU need and ambulance use are drawn from per-level probabilities; stays are log-normal "
             "(positive and right-skewed, the standard model for length of stay).")
    st.markdown("### 2. Priority score")
    st.latex(r"\text{score}_i(t)=U_{\text{triage}_i}+\alpha\,(t-\text{arrival}_i)+\gamma\,\frac{60}{E[\text{stay}_i]}")
    st.write("Urgency weight **U**, waiting-time aging **α** (prevents starvation), optional short-stay bonus **γ** "
             "(better bed turnover; uses the class-average stay, never the patient's true stay).")
    st.markdown("**Why a plain `heapq` is exact:** the aging term α·t is the same for every patient, so it never "
                "changes the ordering. The heap key `-(U + γ·bonus − α·arrival)` is fixed at insertion and always "
                "orders patients exactly like the live score. No re-sorting every minute.")
    st.markdown("### 3. Allocation without conflicts")
    st.markdown(
        "- A patient needs **bed + nurse(s) + doctor(s)** simultaneously; they are granted **atomically** or not at all.\n"
        "- The resource pool raises an error on any over-booking or double allocation.\n"
        "- Doctors are released after the consult, beds and nurses after the stay, ambulances after the round trip.\n"
        "- **Backfilling with reservation:** if the top patient is blocked, lower-priority patients may use resources "
        "he does not need, but everything he is short of is reserved for him.\n"
        "- Staff shortages are **non-pre-empting**: only new allocations see the reduced capacity."
    )
    st.markdown("### 4. Metrics")
    st.latex(r"\text{utilization}_r=\frac{\overline{\text{in-use}_r(t)}}{\text{normal capacity}_r},\qquad "
             r"\text{wait}=\text{time in queue (censored at the end of the run)}")
    st.write("Waits of patients still queued at the end are counted (censored) so no policy can look good by "
             "ignoring people. *Within target %* uses per-triage limits: " +
             ", ".join(f"T{t} ≤ {m} min" for t, m in SLA_TARGET_MIN.items()) + ".")
    st.markdown("### 5. Validation")
    st.markdown("- Independent audit re-derives resource usage from patient records and checks it never exceeds capacity.\n"
                "- `pytest` suite also checks determinism, heap-order = live-score order, and that queue-length "
                "bookkeeping equals total patient-minutes waited.")
