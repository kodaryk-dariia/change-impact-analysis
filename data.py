def get_sample_data():
    threats = [
        {"id": 1, "likelihood": 0.3, "harm_mean": 0.6, "harm_var": 0.02},
        {"id": 2, "likelihood": 0.2, "harm_mean": 0.7, "harm_var": 0.03},
        {"id": 3, "likelihood": 0.4, "harm_mean": 0.5, "harm_var": 0.01},
    ]

    controls = [
        {"id": 1, "eff_mean": 0.5, "eff_var": 0.02, "cost_min": 100, "cost_max": 300},
        {"id": 2, "eff_mean": 0.6, "eff_var": 0.01, "cost_min": 200, "cost_max": 400},
        {"id": 3, "eff_mean": 0.4, "eff_var": 0.03, "cost_min": 50, "cost_max": 200},
    ]

    mapping = {
        1: [1, 2],
        2: [2],
        3: [3]
    }

    return threats, controls, mapping
