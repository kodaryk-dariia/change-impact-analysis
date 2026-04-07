import numpy as np

def generate_benchmark(threats, controls, mapping, simulate_fn):
    results = []

    for _ in range(50):
        selected = np.random.choice(
            [c["id"] for c in controls],
            size=np.random.randint(1, len(controls)+1),
            replace=False
        )
        res = simulate_fn(threats, controls, mapping, selected)
        results.append(res["risk_mean"])

    return results


def percentile(user_value, dataset):
    return sum(x <= user_value for x in dataset) / len(dataset) * 100
