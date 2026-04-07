import streamlit as st
from model import simulate, compare
from data import get_sample_data
from benchmark import generate_benchmark, percentile

st.set_page_config(page_title="Risk Tool", layout="wide")

st.title("Healthcare Risk Decision Tool")

threats, controls, mapping = get_sample_data()

# ---- sidebar ----
mode = st.sidebar.selectbox(
    "Mode",
    ["Design System", "Change Analysis", "Benchmark"]
)

# ---- DESIGN ----
if mode == "Design System":
    st.header("Create System")

    selected = []
    for c in controls:
        if st.checkbox(f"Control {c['id']}"):
            selected.append(c["id"])

    if st.button("Run Simulation"):
        result = simulate(threats, controls, mapping, selected)

        st.subheader("Results")
        st.metric("Risk", round(result["risk_mean"], 3))
        st.metric("Cost", round(result["cost_mean"], 2))

        st.line_chart(result["risk_dist"])

        st.session_state["last_result"] = result
        st.session_state["selected"] = selected


# ---- CHANGE ----
elif mode == "Change Analysis":
    st.header("Analyze Change")

    if "selected" not in st.session_state:
        st.warning("Run Design first")
    else:
        base = simulate(threats, controls, mapping, st.session_state["selected"])

        st.write("Add all controls as change?")
        if st.button("Apply Change"):
            changed = simulate(threats, controls, mapping, [c["id"] for c in controls])
            diff = compare(base, changed)

            st.metric("Δ Risk", round(diff["delta_risk"], 3))
            st.metric("Δ Cost", round(diff["delta_cost"], 2))


# ---- BENCHMARK ----
elif mode == "Benchmark":
    st.header("Benchmark")

    if "last_result" not in st.session_state:
        st.warning("Run simulation first")
    else:
        dataset = generate_benchmark(threats, controls, mapping, simulate)

        user_risk = st.session_state["last_result"]["risk_mean"]
        perc = percentile(user_risk, dataset)

        st.metric("Your Percentile", f"{round(perc, 1)}%")

        st.write("Lower is better (lower risk)")
        st.line_chart(dataset)
