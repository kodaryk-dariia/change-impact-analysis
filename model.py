import numpy as np
from scipy.stats import beta

# ---- distributions ----

def beta_params(mu, var):
    if mu <= 0 or mu >= 1 or var <= 0:
        return None, None
    common = (mu * (1 - mu)) / var - 1
    return mu * common, (1 - mu) * common


def sample_beta(mu, var):
    params = beta_params(mu, var)
    if params[0] is None:
        return 0
    return beta.rvs(params[0], params[1])


# ---- simulation ----

def simulate(threats, controls, mapping, selected_controls, n=2000):
    risks = []
    costs = []

    for _ in range(n):
        total_risk = 0
        total_cost = 0

        # sample controls
        control_samples = {}
        for c in controls:
            if c["id"] in selected_controls:
                eff = sample_beta(c["eff_mean"], c["eff_var"])
                cost = np.random.uniform(c["cost_min"], c["cost_max"])
                control_samples[c["id"]] = (eff, cost)
                total_cost += cost

        # threats
        for t in threats:
            L = t["likelihood"]
            H = sample_beta(t["harm_mean"], t["harm_var"])

            mitigation_factor = 1.0
            for c_id in mapping.get(t["id"], []):
                if c_id in control_samples:
                    eff, _ = control_samples[c_id]
                    mitigation_factor *= (1 - eff)

            risk = L * H * mitigation_factor
            total_risk += risk

        risks.append(total_risk)
        costs.append(total_cost)

    return {
        "risk_mean": float(np.mean(risks)),
        "cost_mean": float(np.mean(costs)),
        "risk_dist": risks
    }


def compare(base, changed):
    return {
        "delta_risk": changed["risk_mean"] - base["risk_mean"],
        "delta_cost": changed["cost_mean"] - base["cost_mean"]
    }
