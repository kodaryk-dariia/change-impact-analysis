import streamlit as st
import pandas as pd
import numpy as np
from itertools import chain, combinations
from model import simulate

st.header("Design Healthcare System")

# ---- INPUT TABLES ----

st.subheader("Threats")

threats_df = st.data_editor(pd.DataFrame({
    "ID": [1, 2],
    "Likelihood": [0.3, 0.2],
    "Harm mean": [0.6, 0.7],
    "Harm variance": [0.02, 0.03]
}), num_rows="dynamic")

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

threshold = st.number_input("Risk Threshold", value=0.1)

# ---- LOGIC ----

def all_subsets(ids):
    return list(chain.from_iterable(combinations(ids, r) for r in range(len(ids)+1)))


def sample_beta(mu, var):
    if mu <= 0 or mu >= 1 or var <= 0:
        return 0
    common = (mu*(1-mu)/var) - 1
    a = mu * common
    b = (1-mu) * common
    return np.random.beta(a, b)


def simulate_subset(threats, mitigations, mapping, subset):
    total_risk = 0
    total_cost = 0

    eff_map = {}
    cost_map = {}

    for _, m in mitigations.iterrows():
        if m["ID"] in subset:
            eff = sample_beta(m["Effectiveness mean"], m["Effectiveness variance"])
            cost = np.random.uniform(m["Cost min"], m["Cost max"])
            eff_map[m["ID"]] = eff
            cost_map[m["ID"]] = cost
            total_cost += cost

    for _, t in threats.iterrows():
        L = t["Likelihood"]
        H = sample_beta(t["Harm mean"], t["Harm variance"])

        mitigation_factor = 1.0
        for _, row in mapping.iterrows():
            if row["Threat"] == t["ID"] and row["Mitigation"] in subset:
                mitigation_factor *= (1 - eff_map.get(row["Mitigation"], 0))

        risk = L * H * mitigation_factor
        total_risk += risk

    return total_risk, total_cost


# ---- RUN ----

if st.button("Run Full Analysis"):

    mitigation_ids = mitigations_df["ID"].tolist()
    subsets = all_subsets(mitigation_ids)

    results = []

    for i, subset in enumerate(subsets):
        risks = []
        costs = []

        for _ in range(300):  # Monte Carlo
            r, c = simulate_subset(threats_df, mitigations_df, mapping_df, subset)
            risks.append(r)
            costs.append(c)

        avg_risk = np.mean(risks)
        avg_cost = np.mean(costs)

        # RSI (твоя логіка)
        rsi = avg_risk / threshold if threshold > 0 else 0

        results.append({
            "Subset": str(subset),
            "Risk": round(avg_risk, 3),
            "Cost": round(avg_cost, 2),
            "RSI": round(rsi, 3)
        })

    results_df = pd.DataFrame(results)

    # ---- OUTPUT ----

    st.subheader("All Control Sets")
    st.dataframe(results_df)

    st.subheader("Top 5 Optimal (by RSI)")
    st.dataframe(results_df.sort_values("RSI").head(5))

    st.subheader("Risk Distribution (last subset)")
    st.line_chart(risks)

    # save for next pages
    st.session_state["design_results"] = results_df
