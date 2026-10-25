# Engineering validation

Local environment: `docs/environment.txt` records the algorithm/runtime environment used for calibration and the frozen first-round study. After the study, `wheel==0.48.0` was added solely for distribution building. This addition did not alter the frozen runtime versions or study files.

- 28 pytest checks passed, including installed CSV fit/predict entrypoints, shared-node evaluation, parameter identity, binding/unbinding changing fitted residuals, shared nonlinear parameter fitting, joint Rademacher coordinates, units and scale separation, strict invalid/unresolved rejection, actual neural/GP/NSGA-II execution, every requested switch, validation-label isolation, model corruption rejection and save/load equality.
- The original vendored files match all 16 recorded upstream SHA256 values. Original repository remains clean at its pinned revision.
- The first published revision passed four GitHub Actions jobs: Windows/Linux × Python 3.10/3.12, including tests and wheel/sdist builds. The final revision reruns this matrix.
- The first-round study completed 70/70 systems and all 70 method-specific execution gates. Every model/status freeze, original prediction hash, audit and recomputed metric passed independent verification. The maximum metric reconstruction difference is 8.67e-19.
- JSON is an inference/audit format; optimizer and GRU continuation checkpoints are not provided. Timing is cooperative within API fits and externally bounded in the study.

These checks establish the reported implementation behavior. They do not establish an accuracy advantage: the full joint method did not outperform the independent DimenSR reference in this first study. See FIRST_ROUND.zh-CN.md for the scientific result and missing coverage.
