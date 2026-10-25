"""Small development example; no claim of a blind performance benchmark."""
import numpy as np
import torch
from dimensr_mimo import DimenSRMIMORegressor, SearchConfig

torch.set_num_threads(1)
rng = np.random.default_rng(21)
X = rng.uniform(-1, 1, (160, 2))
Y = np.column_stack((X[:, 0] ** 2 + X[:, 1], X[:, 0] ** 2 - X[:, 1]))
cfg = SearchConfig(population_size=16, epochs=3, gp_generations=2,
                   selection_generations=3, max_evaluations=500, random_state=0,
                   constant_restarts=1, constant_max_nfev=20, polish_candidates=2)
model = DimenSRMIMORegressor(cfg).fit(X[:120], Y[:120], X_val=X[120:], Y_val=Y[120:],
                                    input_bounds=[[-1, 1], [-1, 1]])
print(model.formulas_)
print(model.diagnostics_)
model.save('example-model.json')
assert np.array_equal(model.predict(X), DimenSRMIMORegressor.load('example-model.json').predict(X))
