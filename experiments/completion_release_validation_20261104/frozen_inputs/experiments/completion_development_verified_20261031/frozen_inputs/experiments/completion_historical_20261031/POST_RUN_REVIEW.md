# Historical completion batch: independent post-run review

## Disposition

The original batch remains **failed**. `VERIFICATION.json` records 17/34 verified systems; all 17 base/control rows failed with `ValueError('Archived control identity drift: model.json')`. `SUMMARY.json.technical_pass=false` and `.adoption_pass=false` remain authoritative. This review supplies additional evidence, does not rewrite formal verification, and does not permit development or creation of `POST_REVIEW_ACCEPTANCE.json`.

All 34 planned systems completed and were sealed. `EXECUTION.json` records worker returncode 0, no termination reason, wall time 865.0849244594574 s, and verification_passed=false. The bounded outer execution failed because verification failed, rather than because a cell was killed. Original verification elapsed time was 859.7208552360535 s. Maximum observed cell wall/fit times were 34.26500000001397/28.21899999998277 s; maximum evaluation count was 428, within the unchanged 60 s/52 s/1200 limits. The batch budget remained 2880 s, including the original 600 s verification reserve, one worker and one thread.

## Material findings

1. **The archived-control identity failure is real under the frozen contract, but its observed differences are narrow.** An independent implementation of the original normalizer and recursive comparison exactly reproduces `../completion_historical_identity_audit_20261031/DIAGNOSIS.json.rows`: each of 17 control pairs differs in precisely 12 `model.history` values, comprising four epochs of `epoch_allowance_seconds`, `neural_until_seconds`, and `gp_until_seconds`. After separately classifying only those history paths, no differences remain in normalized model/result/delivery/pool or the score excluding job/resources. The three fields were not in the original frozen timing whitelist. It would be a post hoc contract change to ignore them and call the original run passed.

2. **These fields have scheduling semantics; they are not intrinsically inconsequential metadata.** `src/dimensr_mimo/regressor.py:114` and `:210` show that the recorded epoch window and neural/GP deadlines derive from the remaining exploration deadline and feed stopping checks. In this observed pair of runs the saved phase evidence supports nonbinding deadlines: all 68 current and 68 prior neural events completed 24 proposals; neural deadline-minus-event-time minima were 2.4899999999906868 s current and 2.18399999999383 s prior. The 136 current/prior GP event margins had minima 4.964799999987008/4.643200000005891 s. Both runs recorded global deadline_reached=false for every control. Their other normalized saved numerical/event trajectories agree. This establishes agreement of the persisted evidence for these runs, not general independence from time allocation or equality of unpersisted RNG/runtime state.

3. **The base matching zeros in the original summary are a failure-reporting consequence, not evidence of absent matching.** In `../compositional_completion_20261031/verification.py:408`, cell verification and `closure_evidence` precede `control_identity`; the exception handler replaces the entire row and discards those already computed fields. Independent event/parent/child reconstruction plus a separately labelled read-only replay recovers base matching counts new/valid/protected/validation_retained = **394/394/115/115**, versus completion **390/390/127/127**. Original `SUMMARY.json.closure_by_method.base` remains zero, `closure_mechanism_pass=false`, and the official failed gates remain unchanged.

4. **Observed historical gains are supported, but acceptance is incomplete.** Independent selected-graph predictions and fixed-threshold calculations reproduce the following complete-denominator results. All original 11 accepted controls remain accepted; gains occur in `rational_n1/702` and `rational_variant/711`. All 34 paired clean outputs satisfy control×1.1+1e-8, and all 17 foundation payload/count/evaluation/effective-cache-fingerprint pairs agree.

| Metric | Base | Completion |
|---|---:|---:|
| Complete systems | 17/17 | 17/17 |
| Fully accepted | 11/17 | 13/17 |
| Mean clean NRMSE | 0.0022923970183390385 | 0.0017540199993836482 |
| Mean shared graph size | 21.176470588235293 | 19.235294117647058 |

The original graph≤24, independent parameters≤6, expanded nodes≤30, depth≤6 and case-specific error thresholds were retained. Four completion failures remain:

| Case/seed | Verified remaining failure |
|---|---|
| damped_n0/702 | graph 25>24; output1 exceeds 1e-4 on training, validation, observed and clean test (clean 0.003982324376281411) |
| damped_n1/701 | graph 30>24 and parameters 7>6 |
| damped_n1/702 | graph 26>24 |
| damped_variant/711 | graph 27>24, parameters 7>6 and clean output0 0.02193441936406219>0.02 |

Both specified historical targets pass the unchanged numerical conditions: damped_n0/701 graph20/parameters4, maximum train/validation/clean error 3.244997242801608e-16≤1e-12; rational_n0/702 graph15/parameters4, maximum corresponding error 3.250286231277542e-16≤1e-4. These target observations do not override failed technical/adoption gates.

5. **Discovery claims require family and comparison qualifications.** Recomputed compositional validation-retained counts are boundary1, damped0, polynomial0, products4, rational1. Acceptance gains are rational2 and zero in every other family. Damped produced 12 new valid compositional systems but none passed protection; polynomial returned no compositional roots. Counts of returned/event roots alone therefore do not establish retained discovery. The comparison jointly changes always-supplement mode and new starts and cannot identify a single-component causal effect. This is an already observed historical regression set, not blind or new-task evidence.

## Checks actually performed

The executable record is `post_review_validation/audit.py`; full results are in `post_review_checks.json` (status `review_consistent_with_original_failure`, issues=[]). It performed only read-only reconstruction of already sealed evidence:

- Matched all 966 live frozen inputs and their original-byte archive, all 374 sealed run files and exact cell/all membership, the 41-source lock and source copies. Initial and final hash comparisons all had zero mismatches. Verified freeze→training→ordered cells→all models→scores→verification chronology and cross-hashes.
- Checked all 34 exact effective configurations against frozen settings: only the prescribed precision_repair/compositional_completion arm choices differ. Reconstructed the 17 historical data streams independently (96 train, 48 validation, 192 test rows), their train-only noise/error scaling and train/validation hashes. The worker source (`../compositional_completion_20261031/harness.py:207`) supplies only train/validation to fitting and uses the frozen case specification; test generation is guarded by the all-model seal (`:46`). Receipt flags agree with this source path. This is source/receipt evidence, not an operating-system trace of every possible read.
- Independently interpreted all 34 selected raw graphs and their declared unit propagation, graph/sharing dimensions, constants and predictions; recomputed train/validation/observed/clean errors and acceptance. Reloaded 34 native estimators and 34 PhysicalModels; their predictions were bitwise equal to the raw interpretation on the three saved-data splits. Rechecked all 34 native domain audits as `enclosed`; this retains the existing finite floating-point screening meaning and is not a proof over every real input in the continuous box.
- Independently reconstructed 161 supplementation events and 784 matching events, including graph unit/shape legality, parent sources, parent/child training errors, protection decisions, new/valid/protected/validation-retained counts, evaluation caps and round limits, full training-pool hashes and the retained validation subset. Recomputed paired foundation identity, both means, accepted/failure membership, family tallies and both targets. Maximum absolute selected-error discrepancy was 6.938893903907228e-18; test/clean discrepancies were zero; maximum event-child discrepancy was 6.661338147750939e-16.
- Independently compared all five archived-control evidence classes and every diagnostic diff row. Read scheduling and fingerprint source. `sharing.py:70` and `regressor.py:345` hash successful/effective cache values only; identical fingerprints do not establish identity of every failed-cache key. No full runtime cache dictionary is persisted.
- Read-only `verifier.closure_evidence(job,frozen)` was additionally reused for 17 controls to recover the evidence lost by original failure-row replacement. It did not invoke official scoring/verification, control_identity, fitting, initialization or writing. The 17 candidate formal verification rows had originally passed; this review separately checked their saved event outcomes and selected predictions.
- Confirmed the absence of historical `POST_REVIEW_ACCEPTANCE.json` and development PREFLIGHT/BUDGET/FREEZE/TRAINING_STARTED/ALL_MODELS_FROZEN receipts.

## Limits and next decision

This review did not independently recompute all 5,508 saved candidates' train/validation metrics, Rademacher values, every source initializer's linear enumeration/returned-root ranking, or an unpersisted full failed-cache/RNG state. Original frozen verification and prior core/harness reviews remain the evidence for those unrerun checks; their existence is not represented as fresh independent recomputation. No search, fitting, benchmark, formal verifier entry point, development run, model reselection, threshold change or production-default change was performed. No manuscript/figure package was assigned or assessed.

If the parent proposes correcting the formal identity criterion, the review supports preparing a **new isolated read-only verification protocol/directory for unchanged sealed models**, explicitly listing the three history paths and preserving every original failed receipt and raw value. Such a protocol must distinguish observed trajectory equality from general scheduling equivalence and must not refit or reselect using the inspected tests. This report does not authorize or execute that correction and does not grant development continuation. Final scientific acceptance remains the parent's decision.

## Evidence identities

- FREEZE.json: `65b5993fe09e6590811d61ff2e93f6d49df11d7fe6b41b951b6a08ab224b51ec`
- ALL_MODELS_FROZEN.json: `1329953e409148f17c1f0febade04daefdee3d67aaf2ccf454bf769beab156f0`
- SCORES.json: `b052122f9393a0d318f5df07d9443a07062e5eb9bff585c478a59ea73b20f984`
- VERIFICATION.json: `928bc34f429ade20a83b97c611b0852eb4324368850e1437e19fa1565861434c`
- SUMMARY.json: `d6a63a6bc57a54ea02ee777666df15c8c82cd13261e77c7a9385d084cf5663b5`
- post_review_checks.json: `96cbf91bbbb617027f8091483b9650cbb640bf9d79b5ce867ca8401a688b556d`

The check receipt additionally records source/helper/script/identity-diagnosis hashes, numerical tolerances, all per-system results, phase margins and unchanged original gate values. Reviewer-only script incidents and their resolution are recorded in `post_review_validation/REVIEWER_LOG.md`.
