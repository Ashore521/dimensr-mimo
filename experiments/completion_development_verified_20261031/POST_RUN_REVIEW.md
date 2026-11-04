# Independent post-run review: observed development completion batch

**Disposition: supports the parent receiving this completed batch's technical and fixed development-adoption conclusions. No remaining material inconsistency was found in the assigned checks.** The original historical failed verdict and separately accepted historical correction remain unchanged. This report is not the eight-key predecessor acceptance receipt and does not start the new release-validation batch.

## Supported results

| Complete denominator | precision_repair control | compositional_completion |
|---|---:|---:|
| Completed / formally verified systems | 16/16 | 16/16 |
| Fully accepted under original precision and structure limits | 7/16 | 16/16 |
| Mean clean test NRMSE | 0.01602532135442316 | 0.0009509047060846458 |
| Mean shared graph size | 21.875 | 12.625 |
| Supplement new / valid / protected / validation-retained | 69 / 69 / 10 / 10 | 166 / 166 / 54 / 54 |
| Matching new / valid / protected / validation-retained | 239 / 239 / 131 / 131 | 376 / 376 / 326 / 326 |

Independent reconstruction confirms all 32 paired clean outputs satisfy control×1.1+1e-8, all seven accepted controls remain accepted, all 16 same-batch foundation identities match, and all 16 archived same-task repair controls match under the reviewed conditional timing rule. Candidate 16/16 acceptance, macro clean-error/mean-graph nonincrease and the original unit/domain/structure requirements all hold. No candidate failures were omitted: completion_failures is empty and both arms retain their complete 16-system denominators.

The nine acceptance gains are all four logarithmic_gain cases, all four algebraic_norm cases, and bilinear_saturation_n1/9132. Logistic_gain already had four accepted controls. Thresholds remained noiseless train/validation/observed/clean ≤1e-4, noisy observed≤0.03 and clean-test ≤0.02; graph ≤24, independent parameters ≤6, expanded nodes ≤30 and output depth ≤6.

## Immutable execution, pre-review and budget evidence

`post_review_validation/audit.py` and `extra_checks.py` are the review scripts; their receipts have issues=[]. The main check status is independent_reconstruction_consistent. All reviewer writes are confined to the assigned report/checks/post_review_validation paths.

- Independently hashed **4,084 live frozen inputs and all 4,084 archived original-byte copies**, exact archive membership, the **352** exact all-model run members and each CELL's actual members/cross-hashes. Checked the complete 41-source path set, live hashes and source copies, original failed/correction chains, pre-run independent review acceptance and test bindings. Initial and final immutable comparisons had zero mismatches. Added post-execution test adapters did not alter any frozen input byte.
- Verified PRE_RUN_ACCEPTANCE.status=accepted_for_single_development_execution, its binding to the actual independent harness review/checks and TEST_VALIDATION, source lock and pre-budget timestamp. Matrix and effective configurations match the frozen plan, seeds 9131/9132, old 96/48/192 streams and the two prescribed arm flags. The old prepared development directory remains unused; this was the authorized single continuation execution.
- Verified FREEZE→ordered CELL→ALL→SCORES→VERIFICATION chronology and cross-hashes, test-isolation flags and no worker test scoring. The supervisor records returncode 0, no termination, wall 898.8756775856018 s; verification records 32/32 passed, budget_met=true and 887.110121011734 s since the original budget start. All finish inside the unchanged 2880 s budget with 600 s verification reserve. Maximum recorded fit/cell time was 19.156000000002678/27.42199999999866 s; maximum unique-system evaluations was 434, below the original 52/60/1200 ceilings. No retry, extra job directory or budget extension is evidenced by the exact sealed matrix/member set and unchanged bounded implementation. This is source/receipt evidence, not an operating-system trace of unrecorded external activity.
- Rechecked the genuine old 32-row score table for complete, unique, finite rows and confirmed the original historical technical/adoption=false chain plus accepted isolated correction remain intact. No original historical POST_REVIEW_ACCEPTANCE was created.

## Numerical and source checks actually performed

1. Reused the previously independent raw JSON/unit/shape/data helper at its recorded SHA, without running its main entry. Independently rebuilt all **16 data streams**, including clean-training noise scales and train-observation error scales. Every training/validation hash matches FREEZE/result. Recomputed all **32 selected graphs'** train, validation, observed-test and clean-test errors, structures, parameter/sharing dimensions, units and fixed acceptance membership. Native estimator and PhysicalModel reload predictions matched the raw interpretation bitwise on all three splits. Rechecked 32 domain audits as enclosed and delivery/score mean/error fields. Maximum selected error discrepancy was 5.551115123125783e-17; observed/clean test discrepancies were zero. Domain enclosed keeps its finite floating-point screening scope.

2. Recomputed every parent and fitted child training error/protection decision in **235 supplementation events plus 615 matching events**. Checked parent provenance, fixed supplement parents, graph units/search bounds, unique event graphs, new/valid/protected/rejection counts, round/total evaluation caps, training-pool hashes and the validation-finite retained subset. Maximum event-child NRMSE discrepancy was 2.220446049250313e-16. Selected-system mechanism records show policy changes, valid neural candidates, GP new evaluations, at least two NSGA steps with new evaluations and positive recorded Rademacher counts. Compared all 864 callback rows with saved model histories after removing only callback output/monotonic fields.

3. For the 16 completion calls, independently evaluated **all 384 returned roots** against the original observed training Y: 119 compositional, 138 implicit and 127 structured. Recomputed MSE/max(var(Y),1e-12)+1e-8×nodes and ranking by score/size/repr; saved score discrepancies were zero. Checked returned/original positions, kind/slot consistency, canonical trees and finite constants, grammar/units/search bounds, declared log/sqrt outer forms, basis-index metadata and event root selection. Rebuilt exact joint event graphs/warm constants and checked their predictions against independently interpreted source trees. Canonicalization, grammar tracing, outer-form validation and graph Builder/clone utilities were reused; no initializer or coefficient search was rerun.

4. Independently compared complete model/result/delivery/pool and score evidence for 16 current controls versus their old same-task repair runs using the original frozen normalizer. Each pair has exactly 12 differences: four phase_budget epochs of epoch_allowance_seconds, neural_until_seconds and gp_until_seconds. All 192 classified paths were confirmed to be phase_budget in both files; no other normalized difference remains. Checked complete epochs, ordered distinct GP steps, 24 neural proposals and no global deadline stop in both runs. Minimum neural deadline margins were 3.384375000008731 s current / 1.958375000001979 s old; GP minima were 5.30128749999858 / 2.7442374999955064 s. The values have real scheduling semantics, but did not bind in these saved trajectories. This does not prove general time-insensitivity, equality of every failed-cache key or unpersisted RNG/internal state.

5. Independently aggregated all accepted members, protection pairs, foundation prefixes/counts/hashes/evaluations/effective-cache fingerprints, family tallies and both means against SUMMARY. Strict full denominators, all applicable technical/scientific gate values and the absence of completion failures agree with the recomputed evidence. Main REPORT's table and scope agree with those observations.

## Discovery evidence and claim applicability

Recorded linear-work counters reaggregate to 311,102 total, including 78,953 old implicit subproblems and 404 new-transform subproblems. The implicit count is a subset of total, not an additional amount to add again. These are checked saved counter sums; the underlying linear enumeration was not repeated.

| Family | New-transform returned roots | Valid complete systems containing new-transform roots | Protected/validation-retained such systems | Acceptance gains |
|---|---:|---:|---:|---:|
| logarithmic_gain | 32 | 17 | 17 | 4 |
| logistic_gain | 39 | 39 | 10 | 0 |
| algebraic_norm | 16 | 8 | 8 | 4 |
| bilinear_saturation | 32 | 20 | 0 | 1 |
| Total | 119 | 84 | 35 | 9 |

Overall supplementation produced 166 new/valid complete systems, 54 training-protected and validation-retained. New-root provenance demonstrates actual generated/retained structures where listed. Bilinear_saturation gained acceptance although its new-transform systems had zero protection retention; the gain cannot be attributed solely to that component. The comparison jointly changes always-supplement behavior and inverse-transform starts and does not estimate a single-component causal effect. These cases/seeds were previously observed and support development regression conclusions, not fresh blind/new-task or real-data generalization.

## Limits and next decision

The **5,597 saved candidates'** full train/validation metrics, Rademacher numerical estimates, all initializer/basis/linear enumerations and unpersisted failed-cache/RNG state were not independently recomputed. Formal native verification remains the evidence for its whole-pool/Rademacher checks. The independent selected/event/source calculations above are the actual added coverage; saved counters alone are not a new benchmark. No fitting, search, formal scorer/verifier entry, threshold/selection change or score overwrite was performed. The first reviewer script attempt stopped before numerical work on an extended-path/normal-root relative_to mismatch; only the reviewer script was corrected, and audit_01_path_representation.log preserves that incident.

The parent may accept this observed development conclusion and, if accepted, write the specified eight-key accepted_for_release_validation receipt. The new release-validation harness still must pass its full real prerequisite/archive gate using that receipt and complete its own freeze checks before execution. This reviewer wrote no acceptance receipt and launched no next batch. Final scientific and release acceptance remain the parent's decisions.

## Evidence hashes

- FREEZE.json: `3d35014a5ce368521e6adfc70d4db4e95cab36389576cb07f03f21f8982ebeea`
- ALL_MODELS_FROZEN.json: `0cf0deb18f4b941c93c49d43228c21d6254867e10d8d89b883e962add4798ef5`
- SCORES.json: `5a1a01da964001cd08df785dd62cd1a72931b5a801e6bd0f2b4abf4ccc93a707`
- VERIFICATION.json: `161b12fed35acf37b67527216a65f38ee05e6ae841b0fabdf93d219cb810d5e5`
- SUMMARY.json: `494f70ca34fa3c2272e33347795408924918c782b37d40660091910e9357e04c`
- FROZEN_INPUT_ARCHIVE.json: `49b6b9f0581567924f48d042d9b80c9cf75ea11fc43401c3df7beb28fbb5b474`
- post_review_checks.json: `306e06b0bbb45208f828ebd5b0fa91b7ff76406b3d7508a9614c3a22f1e49506`

The main check receipt records audit/helper/receipt hashes, tolerances, per-system results, source ranks, phase margins and immutable comparisons. The extra receipt records the additional full-denominator, history and work-accounting checks.
