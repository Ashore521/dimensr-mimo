The first run stopped after saving shared_polynomial_s0.json. Writing the metrics
checkpoint raised `TypeError: Object of type int64 is not JSON serializable`:
the shared_features diagnostic had a NumPy scalar type. No results checkpoint
was produced. The diagnostic is now converted to a Python int. This partial run
is retained; the completed run uses the separate results_release directory.
