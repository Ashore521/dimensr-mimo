# Independent post-run review: locked-candidate new output combinations

## Verdict and material findings

The sealed evidence supports `technical_pass=true` and `adoption_pass=false`. Independent reconstruction found no unresolved discrepancy within the checks described below (`post_review_checks.json: issues=[]`). This is a bounded consistency review, not comprehensive scientific acceptance or a release acceptance receipt. The parent retains final acceptance and release decisions.

All 32 systems completed and were natively verified. The candidate preserves 32/32 paired output protections and 16/16 foundation identities. Both methods accept only 5/16 systems, failing the unchanged candidate-all-16 acceptance requirement. Their clean-error and graph-size guards pass by equality. All 16 paired final graphs, float64 constant bytes and complete best-candidate JSON payloads are exactly identical. Fourteen pairs select foundation candidates in both arms; the two `reciprocal_quadratic_norm_n0` pairs select extension candidates in both arms. There is no observed final-selection gain and no observed paired performance degradation in this batch.

| Metric | base | completion |
|---|---:|---:|
| Complete systems | 16/16 | 16/16 |
| Fully accepted systems | 5/16 | 5/16 |
| Mean clean NRMSE | 0.021925807049642965 | 0.021925807049642965 |
| Mean shared graph size | 22.625 | 22.625 |
| log_affine_ratio accepted | 0/4 | 0/4 |
| reciprocal_quadratic_norm accepted | 4/4 | 4/4 |
| oscillatory_affine_saturation accepted | 0/4 | 0/4 |
| exponential_affine_saturation accepted | 1/4 | 1/4 |

The exponential acceptance is `n1`, seed 10241. No same-task archived control exists: `historical_control_comparison=not_applicable_no_same_task_archive` is appropriate; no archived-control identity success was substituted. These are authored, locked-candidate new output combinations, not external blind or real-data validation. Supplementation and new starts are compared jointly, so this evidence does not isolate component effects or establish why selections coincide.

## Original failures retained

The original shared-graph-size and independent-parameter limits are 24 and 6. Precision limits are 0.0001 for n0 and 0.03 observed / 0.02 clean for n1. Output indices below are zero-based; listed precision failures occur on train, validation, observed test and clean test. Structure violations follow directly from the displayed size/parameter values. All 11 completion failures and exact error vectors/reasons are retained in `post_review_checks.json: completion_failures` and the original `SUMMARY.json`.

| Case | Seed | Graph size / parameters | Precision-failing outputs |
|---|---:|---:|---|
| log_affine_ratio_n0 | 10241 | 28 / 8 | 1 |
| log_affine_ratio_n0 | 10242 | 29 / 9 | 1 |
| log_affine_ratio_n1 | 10241 | 29 / 9 | 1 |
| log_affine_ratio_n1 | 10242 | 27 / 8 | 1 |
| oscillatory_affine_saturation_n0 | 10241 | 28 / 8 | 0, 1 |
| oscillatory_affine_saturation_n0 | 10242 | 26 / 7 | 0, 1 |
| oscillatory_affine_saturation_n1 | 10241 | 24 / 7 | 0, 1 |
| oscillatory_affine_saturation_n1 | 10242 | 24 / 6 | 0, 1 |
| exponential_affine_saturation_n0 | 10241 | 24 / 7 | 0, 1 |
| exponential_affine_saturation_n0 | 10242 | 30 / 8 | 0 |
| exponential_affine_saturation_n1 | 10242 | 25 / 7 | None; structure only |

## Independent numerical and selection checks

- Reconstructed all 16 data streams from the specified 20261104, case-indexed seeds 10241/10242 and 96/48/192 train/validation/test sizes. Noise uses train-only clean standard deviations; NRMSE uses observed-training standard deviations. Matched training/validation array hashes against freeze/results. Checked frozen/test-access chronology, fixed matrix/configuration, source locks and pre-run review/test bindings. Test evaluation followed model freezing; no test-driven reselection was performed by this review.
- Evaluated the 32 selected raw graphs independently on all three splits; recomputed training, validation, observed-test and clean-test errors, graph structure/sharing, unit legality and acceptance membership. Reloaded all 32 native and all 32 PhysicalModel exports and compared predictions bitwise with raw evaluation. All 32 native domain rechecks were enclosed; this finite numerical-screen result is not a proof over all real-valued inputs.
- Checked every saved supplementation and closure parent/child training error and protection decision, graph units/bounds, counts/caps, training-pool hash and validation retention. Invalid `None` children remained invalid, never zero-error or protected successes.
- Recomputed all 384 returned-root raw-observed-training-Y scores and rankings (151 structured, 203 implicit, 30 compositional), with zero source-score discrepancy. Checked canonical/grammar forms and outer-source metadata using native utilities; reconstructed exact joint warm-route assembly with zero discrepancy. No initializer or linear enumeration was rerun.
- Reaggregated paired/family results, all 11 failure memberships/reasons, aggregate means and saved discovery counters. Supplementary checks compared all 864 callback rows with model histories after excluding callback-only output/monotonic fields; checked delivery train/validation vectors and score means; verified serialized best-payload identity and float64 constant-byte identity for all 16 pairs.

Maximum absolute discrepancies were training scale 5.56e-17, selected train 2.78e-17, selected validation 2.09e-17, selected observed/clean test 0, event parent 2.78e-17 and event child 1.12e-16. Source-score, joint-warm and aggregate-mean discrepancies were zero. Exact values are recorded in `post_review_checks.json: maximum_absolute_differences`.

## Mechanism and undefined-value accounting

There are 226 supplementation events and 757 closure events (983 total). Reconstructed new/valid/protected/validation-retained counts are:

| Mechanism / arm | New | Valid | Protected | Validation retained |
|---|---:|---:|---:|---:|
| Supplementation / base | 83 | 80 | 25 | 25 |
| Supplementation / completion | 143 | 139 | 68 | 68 |
| Closure / base | 391 | 391 | 220 | 220 |
| Closure / completion | 366 | 364 | 195 | 195 |

The invalid counts are three base supplementation children, four completion supplementation children and two completion closure children. The new compositional transform produced 30 systems, 29 valid, 13 protected and retained: exponential 16/16 valid with zero retained; reciprocal 14 new, 13 valid and retained; log-ratio and oscillatory returned no compositional roots. None produced an acceptance gain.

Saved linear-work counters sum to 1,231,284 total subproblems, including 609,048 implicit subproblems; the implicit count is a subset and must not be added again. The compositional count is 6,142. These are reconciled saved counters, not independently repeated linear work. Neural, GP, NSGA-II and saved Rademacher mechanism evidence were checked; Rademacher estimates were not independently recomputed.

## Frozen evidence, versions and budgets

Checked all 8,691 live frozen inputs and their 8,691 archived copies, exact archive membership, all 352 run artifacts and their cell/member hash chains, exact 32 job directories, and the complete 41-file source lock plus source copies. Prerequisite checks preserve the original historical failed verdict, the accepted sealed correction and the development acceptance with its exact eight-key receipt. All original thresholds, denominators, failures and frozen artifacts remain unchanged.

The frozen runtime receipt records NumPy 2.5.4, SciPy 1.18.1, Torch 2.14.1+cpu, SymPy 1.14.0, scikit-learn 1.9.1 and psutil 7.2.2. The source lock SHA-256 is `51103c4e7085da90695f7845df5a312ca6ff693dc686bbc651772bf63c74b452`; per-file bytes, rather than a mutable release label, bind this reviewed algorithm.

Execution exited 0 without termination. Supervisor wall time was 1621.8271069526672 s; verification elapsed was 1597.9054596424103 s. The unchanged budget was 2880 s with 600 s verification reserve, 52 s fit and 60 s outer cap, 1200 evaluations, serial one-thread execution. Maximum saved fit time was 39.765999999995984 s, cell wall 53.18800000000192 s and evaluations 465. No retry, extension, fitting, search, formal rescoring or formal verifier execution was performed by the reviewer.

Principal immutable SHA-256 bindings:

| File | SHA-256 |
|---|---|
| FREEZE.json | a98e8c3fda816efd4e2e4df0c73d6511bdfb514002179082d761461545c0a31e |
| ALL_MODELS_FROZEN.json | a837c1317fb1c5bc442f481b7c2cf5fea2c774b18e05700c2ef9675bb5b570ea |
| SCORES.json | e081622a3aa1fa16496a6b5324a6b1013586059523310c99d0503852d471945c |
| FROZEN_INPUT_ARCHIVE.json | 710c920cca8f37ee49c8c40278b5bcb1d80dae5e8ad18e83b85dd136a2a7bdfe |
| post_review_checks.json | 9db58b8d076435ae65204a74e9664a7aadc8f6d1d2444cd07458f4888da7a7ef |

## Reviewer checkpoint correction and limitations

The first numerical audit imported the predecessor study solely for matrix/gate lookup, producing the single reviewer finding `actual_fixed_configuration_matrix`. Its 32 numerical reconstructions used the correct new cases, independent data implementation and actual frozen specifications. The original executed source, numerical checkpoint and log are preserved as `post_review_validation/audit_01_wrong_stage_import.py`, `checkpoint_numeric.json` and `checkpoint_numeric.log`. Only the import was corrected in `audit.py`; it was not rerun. `resume_review.py` rechecked the affected new matrix/configuration, complete prerequisite gate, all immutable input/archive/run hashes and 16 best-payload identities, then reused the unchanged numerical checkpoint. The receipt separately binds the executed original script, corrected script, checkpoint and resume script. `extra_checks.json` binds the final main receipt. No original evidence was corrected or overwritten.

The reviewer did not independently recompute all metrics for the 5,836 saved candidates, Rademacher estimates, initializer/basis/linear enumeration, unpersisted failed-cache keys, RNG state or other unpersisted internal state. Full-pool and Rademacher verification therefore still rely on the preserved native verification evidence. Native canonical/grammar/outer-form/Builder and domain utilities were reused as stated. There is no manuscript/figure acceptance or package/release review in this assignment.

The final delivery integrity check and review-output hashes are in `post_review_validation/DELIVERY_QA.json`. No `POST_REVIEW_ACCEPTANCE.json` was written by this reviewer. This report supports returning the negative scientific result to the parent; any opt-in release must retain the failed adoption verdict and bounded claim scope.
