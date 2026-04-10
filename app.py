import streamlit as st
import pandas as pd
import numpy as np
from itertools import chain, combinations

st.header("Design Healthcare System")

# ---------------- UI HELP ----------------
with st.expander("ℹ️ How it works"):
    st.write("""
    This tool simulates risk using Monte Carlo method.
    You define threats, mitigations and their relationships.
    The system evaluates all combinations of controls and finds optimal ones.
    """)

# ---------------- INPUT ----------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Threats")

    threats_df = st.data_editor(pd.DataFrame({
        "ID": [1, 2],
        "Likelihood": [0.3, 0.2],
        "Harm mean": [0.6, 0.7],
        "Harm variance": [0.02, 0.03]
    }), num_rows="dynamic")

with col2:
    st.subheader("Mitigations")

    mitigations_df = st.data_editor(pd.DataFrame({
        "ID": [1, 2],
        "Cost min": [100, 200],
        "Cost max": [300, 400],
        "Effectiveness mean": [0.5, 0.6],
        "Effectiveness variance": [0.02, 0.01]
    }), num_rows="dynamic")

st.subheader("Mapping (Threat → Mitigation)")
mapping_df = st.data_editor(pd.DataFrame({
    "Threat": [1, 2],
    "Mitigation": [1, 2]
}), num_rows="dynamic")

threshold = st.slider("Risk Threshold", 0.01, 1.0, 0.2)

simulations = st.slider("Monte Carlo Runs", 100, 2000, 500)

# ---------------- CORE ----------------

def all_subsets(ids):
    return list(chain.from_iterable(combinations(ids, r) for r in range(len(ids)+1)))


def sample_beta(mu, var):
    if mu <= 0 or mu >= 1 or var <= 0:
        return 0
    common = (mu*(1-mu)/var) - 1
    a = mu * common
    b = (1-mu) * common
    return np.random.beta(a, b)


def classify_risk(r, threshold):
    p = r / threshold
    if p < 0.1: return "low"
    elif p < 0.3: return "low-medium"
    elif p < 0.5: return "medium"
    elif p < 0.7: return "medium-high"
    else: return "high"


lambda_weights = {
    "low": 0.1,
    "low-medium": 0.3,
    "medium": 0.5,
    "medium-high": 0.7,
    "high": 0.9
}


def simulate_subset(threats, mitigations, mapping, subset):
    total_cost = 0
    residuals = []

    eff_map = {}

    for _, m in mitigations.iterrows():
        if m["ID"] in subset:
            eff = sample_beta(m["Effectiveness mean"], m["Effectiveness variance"])
            cost = np.random.uniform(m["Cost min"], m["Cost max"])
            eff_map[m["ID"]] = eff
            total_cost += cost

    for _, t in threats.iterrows():
        L = t["Likelihood"]
        H = sample_beta(t["Harm mean"], t["Harm variance"])

        mitigation_factor = 1.0
        for _, row in mapping.iterrows():
            if row["Threat"] == t["ID"] and row["Mitigation"] in subset:
                mitigation_factor *= (1 - eff_map.get(row["Mitigation"], 0))

        R = L * H * mitigation_factor

        #
        if R > threshold:
            return None, None, None, None
        if threshold < 0.05:
            st.warning("Very strict threshold may eliminate all solutions.")

        residuals.append(R)

    total_risk = sum(residuals)

    # breakdown
    breakdown = {"low":0, "low-medium":0, "medium":0, "medium-high":0, "high":0}
    for r in residuals:
        breakdown[classify_risk(r, threshold)] += 1

    # RSI
    rsi = sum(lambda_weights[k]*v for k,v in breakdown.items())

    return total_risk, total_cost, breakdown, rsi


# ---------------- RUN ----------------

if st.button("Run Full Analysis"):

    mitigation_ids = mitigations_df["ID"].tolist()
    subsets = all_subsets(mitigation_ids)

    results = []

    progress = st.progress(0)

    for i, subset in enumerate(subsets):

        risks = []
        costs = []
        breakdowns = []
        rsis = []

        for _ in range(simulations):
            r, c, b, rsi = simulate_subset(threats_df, mitigations_df, mapping_df, subset)

            if r is not None:
                risks.append(r)
                costs.append(c)
                breakdowns.append(b)
                rsis.append(rsi)

        if len(risks) == 0:
            continue

        avg_breakdown = {
            k: int(np.mean([b[k] for b in breakdowns])) for k in breakdowns[0]
        }

        results.append({
            "Subset": str(subset),
            "Risk": round(np.mean(risks), 3),
            "Cost": round(np.mean(costs), 2),
            "RSI": round(np.mean(rsis), 3),
            **avg_breakdown
        })

        progress.progress((i+1)/len(subsets))
    if len(results) == 0:
        st.error("No valid control sets found. Try increasing Risk Threshold.")
        st.stop()

    results_df = pd.DataFrame(results)

    # ---------------- OUTPUT ----------------

    st.success(f"Valid control sets: {len(results_df)}")

    colA, colB, colC = st.columns(3)
    colA.metric("Best Risk", round(results_df["Risk"].min(), 3))
    colB.metric("Lowest Cost", round(results_df["Cost"].min(), 2))
    colC.metric("Best RSI", round(results_df["RSI"].min(), 3))

    st.subheader("Top 5 Optimal Solutions")
    st.dataframe(results_df.sort_values("RSI").head(5))

    st.subheader("All Results")
    st.dataframe(results_df)

    st.subheader("Risk Distribution (best solution)")
    best = results_df.sort_values("RSI").iloc[0]
    st.write(f"Best subset: {best['Subset']}")
