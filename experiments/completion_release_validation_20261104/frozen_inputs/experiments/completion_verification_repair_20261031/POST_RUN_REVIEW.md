# Independent terminal review: isolated verification of sealed results

**Disposition: the parent may receive the new isolated verification conclusion as supported.** No remaining material inconsistency was found in the assigned checks. This conclusion applies to unchanged historical models under the separately reviewed, conditional timing-classification contract. It does not change the original failed verdict, create a continuation receipt, or authorize development.

## Two distinct verdicts

| Evidence | Original historical contract | New isolated verification contract |
|---|---|---|
| Directory | completion_historical_20261031 | completion_verification_repair_20261031 |
| Formal verification | failed, 17/34 | passed, 34/34 |
| technical_pass / adoption_pass | false / false | true / true |
| Models, scores and selections | Original sealed evidence | Same evidence, unchanged |
| Control history comparison | Three scheduling fields were not excluded | Only three phase_budget history fields classified, subject to complete nonbinding-phase evidence |
| Development authorization | Absent | Explicitly false |

The original failure remains correct under its frozen identity rule. The new result is a separate verification of the sealed results, not a claim that the original run initially passed or that any numerical performance improved during this repair.

## Actual terminal checks

`post_review_validation/audit.py` performed only JSON/hash/receipt comparisons and reuse of bound independent evidence. `post_review_checks.json` reports `supported_isolated_verification_pass`, issues=[].

- **Immutable evidence:** independently checked all 245 new live inputs and all 245 original-byte archived copies, with exact archive membership; all original 966 live inputs and 966 archived copies; all 374 original sealed run artifacts; all original failure/post-review chain hashes; and the new receipt hashes. Initial and final checks had zero mismatches. The frozen final code and tests exactly match the independent pre-run review revision, including the restored whole-old-score-table denominator/uniqueness/completeness check. The original score table still has its complete 34-row denominator. The original FREEZE/ALL/SCORES/VERIFICATION/SUMMARY/EXECUTION and original false gate values remain unchanged.
- **Chronology and budget:** original model seal→score→failed verification precede the new freeze. New freeze was 1793427087.4418066; new verified time was 1793427143.4465795. EXECUTION records returncode=0, termination_reason=null, wall_seconds=56.37834596633911 and no_training_performed=true. Recorded freeze+outer-wall end is 1793427143.8201525, **388.217276096344 s before** the original absolute deadline 1793427532.0374286. The frozen deadline exactly equals min(new start+600, original BUDGET start+2880); no budget reset or development budget was used. The outer end is reconstructed from the recorded start and elapsed duration; no independent operating-system execution trace was collected.
- **Complete verification rows:** checked exactly 34 ordered, unique jobs, all verified=true, with full numerical/structure/mechanism/domain/pool/acceptance and matching fields, no error rows, and precisely 34 success log lines. All 17 candidate rows are byte-value-equivalent as parsed JSON to their original successful verification rows. Each newly restored control row's selected metrics and structure agree with the prior independent raw-graph reconstruction; its other original repair-verification fields agree with the archived repair row, apart from the changed job label and separately checked matching/identity fields. Every control has archived_control_identity=true. Candidate and structure-feasible counts match the unchanged pools; saved candidate counts total 5,508.
- **Scientific observations and gates:** compared every new selected clean/observed error, structure, accepted member and matching count against the previous independent 34-system reconstruction. The new methods, pairs, target outcomes and family discovery tallies match that evidence. All original summary values are identical except the exact eight expected fields: technical_pass, adoption_pass, archived_control_identity_pass, closure_mechanism_pass, closure_by_method, original_verdict, verification_of_sealed_results and development_authorized. The full denominator, 34 protected outputs, 17 foundation identities, preserved original 11 accepted controls, macro error/graph nonincrease and both fixed historical target conditions remain satisfied. Four completion failures remain explicitly listed; candidate_all_accepted remains false, as allowed by the unchanged historical-stage contract.
- **No scope expansion:** source/call-path review from the pre-run checkpoint remains bound by hashes. The new implementation calls the original read-only cell/provenance/summary checks, does not call fit/search/initializer or the score-writing entry, and writes its verification/summary/execution only here. No new training or score output exists in this directory. The original POST_REVIEW_ACCEPTANCE.json is absent; original development startup receipts remain absent. The new summary explicitly says original_verdict=failed, verification_of_sealed_results=true and development_authorized=false.

## Reproduced observations

| Metric | Base | Completion |
|---|---:|---:|
| Complete / verified systems under the new contract | 17/17 | 17/17 |
| Fully accepted by original numerical/structure thresholds | 11/17 | 13/17 |
| Mean clean NRMSE | 0.0022923970183390385 | 0.0017540199993836482 |
| Mean shared graph size | 21.176470588235293 | 19.235294117647058 |
| Matching new / valid / protected / validation-retained | 394 / 394 / 115 / 115 | 390 / 390 / 127 / 127 |

The two acceptance gains remain rational_n1/702 and rational_variant/711. The four remaining completion failures remain damped_n0/702 (graph and precision), damped_n1/701 (graph and parameters), damped_n1/702 (graph), and damped_variant/711 (graph, parameters and clean precision). Neither thresholds nor selections changed. Compositional validation-retained counts remain boundary1/damped0/polynomial0/products4/rational1; these do not support retained compositional discovery in every family or a single-component causal claim. The experiment remains observed historical regression evidence.

## Reused evidence and limits

The original independent post-run review and its 11 MB check receipt already reconstructed 34 selected raw graphs/predictions/unit/shape results, 161 supplementation events and 784 matching events. Their hashes are frozen here and rechecked, and all relevant model/data files remain identical; those numerical calculations were **not rerun**. The independent pre-run review and its bound check receipt applied the conditional phase classifier to both models of all 17 control pairs. Complete epochs, 24 neural proposals, finite ordered deadlines, strict positive neural/GP event margins and no global-deadline stop were confirmed there. Their data and reviewed implementation are unchanged. This supports the three-path classification in these observed runs, not general time-insensitivity or equality of unpersisted failed-cache/RNG/internal state.

This terminal review did not rerun formal verification, fitting, data/task generation, initializer enumeration, the test suite, all 5,508 candidate metrics or Rademacher calculations. The original verifier's rerun of its saved-candidate checks is formal execution evidence; it is not represented as fresh independent recomputation of the entire pool. Native domain `enclosed` retains its existing finite floating-point screening scope. No manuscript/figure package was assigned. The result can be received for this precise verification contract; final scientific acceptance remains the parent's responsibility. Any subsequent stage still needs a new isolated directory and independently reviewed continuation gate. No such gate, acceptance receipt or execution was created by this reviewer.

## Evidence identities

- New FREEZE.json: `9acaae3ca056c538ea5dd1e57c16195725805aad2435ba1ea96a789cbffb7868`
- New VERIFICATION.json: `414e7d4b262d2d01b38f2e2967a45dd78a0c6e9e073c75f9676e41dfb503d40d`
- New SUMMARY.json: `51b2fe7e3b627a2bb758ec010acd4f71a95b8bc1b5589df87f6915abf5713fb5`
- New EXECUTION.json: `c93c933726d713edacb14677762a57b663640f61d8dd703c58dd204e556c814c`
- New post_review_checks.json: `976c7462732e11d4d2b55619b0df0a0763c8de11707fb1626c54ce243c393829`
- Reused original numerical/event checks: `96cbf91bbbb617027f8091483b9650cbb640bf9d79b5ce867ca8401a688b556d`

The check receipt records the complete original/new evidence hashes, all per-row comparison checks, allowed summary differences, immutable-file results, provenance of reused reviews and deadline arithmetic.
