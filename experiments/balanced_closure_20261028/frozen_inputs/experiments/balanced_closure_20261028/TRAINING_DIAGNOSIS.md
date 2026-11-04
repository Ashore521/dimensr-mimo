# damped_n0 / seed 701: protected closure training-coverage diagnosis

Historical diagnostic snapshot: this report was completed before the 24-per-round scheduling change. References to "current" source and its equality to the protected-closure snapshot refer to that earlier diagnostic time, not the balanced-closure source frozen for this study.

Scope: only the completed protected_closure_20261028 unit's saved training candidates/closure events, current and frozen sharing.py, matching historical training graphs, and deterministic matched_proposals enumeration. Existing parameter vectors were evaluated only on training X; no fit, search, full closure, domain audit or benchmark was executed. Training X was obtained with noise_families_20261026.cases.inputs('damped_n0','train',701), which constructs X without calling latent/generate/training_data. No SCORES, clean/test/validation labels or validation-error fields were used. The sole intended write is this report; the rational diagnosis is retained.

## Narrow conclusion

The one-round stop was caused by spending all 48 new-evaluation allowance in round 0. There is direct evidence of a recursive compression opportunity on saved states, not only motivation to reserve a second round. In particular, allocating the first 24 evaluations to round 0 would retain an already saved, fitted and protected intermediate state whose next matched proposal has exactly the historical 20-node graph topology and zero change to its inherited training predictions. This does not establish that a revised run would pass ordinary fitting, interval acceptance, validation/test selection or the full scientific recommendation.

The suggestion that the old 20-node result requires topology absent from the random base path, or removal of near-zero additions, is refuted for this specific saved unit: the required starting topology is present in base candidate 147, and the 20-node proposal keeps its near-zero additive constant.

## Why only one round executed

`src/dimensr_mimo/sharing.py:81-86` checks a global new-evaluation limit of 48 at the start of each round. The inner evaluation loop (`sharing.py:113-118`) uses the same limit and has no per-round reserve. Therefore the first round can consume all 48, making the second round exit immediately.

Saved `model.json -> diagnostics.sharing_closure`:

| Field | Value |
|---|---:|
| base_candidate_count | 158 |
| base_evaluations | 368 |
| rounds | 1 |
| proposals | 94 |
| cache_skips / structure_rejections / prediction_rejections | 0 / 0 / 0 |
| new_evaluations / valid | 48 / 48 |
| protected / protection_rejections | 26 / 22 |

The sharing_closure events.jsonl record has evaluations=416, phase_seconds approximately 1.032 and elapsed_seconds approximately 19.829. Thus the 1200 evaluation cap and 52-second fitting deadline were not the stopping reason at this point. Protected retained states existed, so the `if not retained` exit was not the reason either.

## Exact existing two-step route to the old graph

All candidate indices are zero-based model.json candidates indices. Closure event numbers below are one-based indices into diagnostics.sharing_closure.events (not trace serial numbers).

| State | Origin | DAG size | Independent parameters | Saved training errors |
|---|---|---:|---:|---|
| Base candidate 147 | random base prefix | 27 | 6 | [2.3936137927403493e-16, 4.589119110633335e-16] |
| Extension candidate 163 | closure event 6, protected | 23 | 5 | [2.768805624605552e-16, 3.845980099106278e-16] |
| Enumerated next proposal | matched_proposals(candidate 163) | 20 | 4 | Not refit or evaluated against labels; inherited prediction unchanged |
| Historical matched candidate 121 | matched_sharing_20261028 | 20 | 4 | [1.700552365049716e-16, 3.2449972428016087e-16] |

Exact graph digests:

- Base 147: its provenance is checked by full system and constant-vector equality with event 6's saved parent.
- Extension 163: `9087e53090f4b63ef0d08b5555b30fa61ba4cec93747b4572628d79c1b7e54c2`.
- Next proposal and historical candidate 121 have the identical graph digest `af5ad975139e23cc80806e20e7d4ce9780eacccd984398b1fff2310d42f71fc1`.

The normalized RMS difference between that next proposal's warm predictions and extension 163's predictions is [0.0, 0.0] on all 96 training X rows. This is a comparison of existing program predictions, with no label access. If the stored parent training errors are accepted, unchanged predictions inherit those errors; this is not a fresh error recomputation.

Topology in the 20-node graph (parameter IDs p0-p3 are shared explicitly):

    y0 = p0 * cos(p1*x0) * x1 / (p2 + exp(p3*x0))
    y1 = p0 * sin(p1*x0) * x1 / exp(p3*x0)

The candidate-163 warm p2 is about -2.03341e-17 and remains in the graph. No near-zero-term deletion is performed. The step from candidate 163 shares the matching parameterized frequency argument; candidate 163 had already shared the parameterized exponential. The original base 147 already had shared amplitude and one additive/one bare exponential denominator. This suffices for the observed 27 -> 23 -> 20 route.

A second independent saved route exists through candidate 172, created by protected event 15 from base 147: its proposal has the same 20-node digest, and its prediction difference is [9.853429687460927e-17, 0.0]. Both intermediate events occur within the first 24 evaluations.

## What reserving 24 + 24 actually covers on saved states

The following is deterministic proposal enumeration using already saved fitted event results, not a second training run. For a hypothetical first-round cap 24, use the base prefix plus protected children among saved events 1-24, apply the unchanged parent ranking and inspect the next matched proposals. Do not fit these proposals.

- Events 1-24 contain 22 protected children; events 25-48 contain only 4 additional protected children.
- The original second-round top16 rule on base plus those first-24 retained children selects 14 extensions and 2 base candidates. Candidate 163 is rank 15 (one-based); candidate 172 is rank 1.
- Excluding the exact complete base evaluation cache and first-24 evaluated graph IDs, this top16 yields 56 proposal occurrences, 40 unique unseen graph IDs; 22 occurrences have maximum standardized prediction difference <1e-12.
- The exact historical size20/4-parameter topology is among them. Its best proposal distance is 0, making it first by the unchanged distance/size/error/digest ordering among these proposals. It is structurally in bounds under the original per-output node/depth budget.
- Using all 26 saved protected children after all 48 events still yields 38 unique cache-unseen recursive graph IDs, including the same size20 topology. The current code simply has no remaining new-evaluation allowance to try them.

This establishes a concrete training-side coverage opportunity under a 24+24 allocation. It does not prove that the final state would remain unchanged after least_squares, pass domain audit, survive selection or retain the full tight benefit across train/validation/clean-test requirements. No future algorithm is declared successful.

## Parent cap and warm sorting

`sharing.py:87` ranks by training mean error, then complexity and graph digest, and keeps 16 parents. The initial top16 base indices are:

    142, 90, 8, 147, 88, 7, 86, 9, 10, 11, 70, 12, 13, 14, 84, 15

All seven base states with both saved training errors <=1e-4 are included (indices 7,8,86,88,90,142,147). Thus the initial top16 cap did not exclude the known accurate starting states or base 147, and the second-round top16 would not exclude the identified useful intermediates. The cap can affect other paths; parents outside it were not exhaustively studied, so its general effect remains unknown.

`sharing.py:104-110,113` sorts by maximum per-output warm prediction change before size and parent error, with same-graph deduplication. Re-enumeration from the original top16 gives exactly 94 raw proposal occurrences and 65 unique graphs; the first 48 graph/warm pairs match all saved closure events exactly. All 65 first-round proposals have size >=23; no one-step proposal from these parents reaches size20.

| First-round rank | Maximum warm prediction change | DAG size | Parameters |
|---|---:|---:|---:|
| 1 | 0 | 25 | 7 |
| 21 | 1.416e-14 | 23 | 6 |
| 22 | 2.353e-14 | 23 | 6 |
| 24 | 0.034937120060404905 | 26 | 6 |
| 48 | 0.770504659255038 | 26 | 7 |

This ordering selects useful intermediates early but then spends the remaining global allowance on other one-step changes. It does not expand a retained child until the next round. Therefore the demonstrated limitation is the interaction of global budget exhaustion with batched breadth-before-recursion, rather than proof that warm sorting or the 16-parent cap alone is defective. Reserving a second round changes budget allocation; it does not authorize increasing total budget.

## Cache and source verification

The historical search_lineage_20261028/runs/damped_n0_s701_random/trace.jsonl reconstructs a full 368-graph evaluator cache. Replacing graph references in its state records with actual graphs and respecting evaluation event insertion order reproduces the current closure base_search_fingerprint exactly:

    c10e591bd3cd99834e21e5fe591e0166e1ccc7ac3d0c6bc6abf3cd6d92f896c6

This permits exact cache exclusion for the recursive enumeration above; exclusion was not approximated using only the 158 delivered base candidates. Current sharing.py equals the protected_closure frozen sharing.py byte-for-byte, SHA256:

    d6f62512722cc43cc47da86ffee5b3f75b786cadb0a884eeb166810f6349e19e

## Limits and remaining decision

The saved evidence supports testing a per-round reserve within the existing 48 total new evaluations. It does not support claiming success by increasing round count alone: without reserving budget the existing two-round loop still stops after one, and other tasks may lack the same topology or warm parameter agreement. This diagnosis did not evaluate new proposals through joint fitting/domain checks and used no validation or held-out outcomes. Acceptance and generalization remain unresolved.

Input SHA256 values:

- `experiments/protected_closure_20261028/runs/damped_n0_s701_closure/model.json`: `27814fa8acbfe94a2e117382dbf127b8bc27e1253ea3c1932775bdedd8e0d604`.
- `experiments/protected_closure_20261028/runs/damped_n0_s701_closure/events.jsonl`: `d69160c4012a296f3274ef7f9d3744e92aad8902ee765617359bf46b67dd3787`.
- `experiments/search_lineage_20261028/runs/damped_n0_s701_random/trace.jsonl`: `fa9d15eb00e671404633f60e2f0765546fbeb703b64d35a599676c5594de16c9`.
- `experiments/matched_sharing_20261028/runs/damped_n0_s701_matched/model.json`: `3a57792620718c8fb5801a7cff8d0a9383c6eb9bf87971c9427d857f58b1087e`.
