"""Recover two outputs using the same product and linear features."""

import numpy as np
from pathlib import Path

from dimensr_mimo import MIMOSymbolicRegressor, SearchConfig


def main():
    rng = np.random.default_rng(42)
    X = rng.uniform(-2.0, 2.0, size=(160, 3))
    Y = np.column_stack((2.0 * X[:, 0] * X[:, 1] + 0.5 * X[:, 2],
                         -X[:, 0] * X[:, 1] + 3.0 * X[:, 2]))
    model = MIMOSymbolicRegressor(
        config=SearchConfig(random_state=42),
        feature_names=["a", "b", "c"],
        target_names=["response_1", "response_2"],
    ).fit(X[:120], Y[:120], X_val=X[120:140], Y_val=Y[120:140],
          input_bounds=[[-2.0, 2.0]] * 3)
    for name, expression in zip(model.target_names_, model.expressions_):
        print(f"{name} = {expression}")
    prediction = model.predict(X[140:])
    print("Held-out RMSE per output:", np.sqrt(np.mean((prediction - Y[140:]) ** 2, axis=0)))
    print("Declared-domain audit:", model.audit_domain())
    output = Path("outputs/shared_system.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    restored = MIMOSymbolicRegressor.load(output)
    np.testing.assert_allclose(restored.predict(X[140:]), prediction)


if __name__ == "__main__":
    main()
