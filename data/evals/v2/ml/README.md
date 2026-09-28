# SentinelVoice V2-C offline intent/risk experiments

V2-C1 uses only controlled synthetic and hand-reviewed SentinelVoice
utterances. It did not download, ingest, or train on BANKING77 or any other
public or external dataset. Production transcripts are not repository data.
These sources remain separate in every artifact and report.

The classifier is offline and advisory. It is not imported by the agent or
voice runtime and has no authority over authentication, confirmation,
ownership, tool permission, or protected-action execution.

## Dataset construction

`intent_risk_seed.json` remains the unchanged 54-example V2-A source. The
builder adds hand-authored SentinelVoice requests, deterministic template/slot
expansions, and difficult contrast cases. Slots contain only synthetic V2
merchants, amounts, dates, account types, statuses, and card endings. No
external LLM or API generated paraphrases; deterministic augmentation keeps
provenance reviewable and regeneration byte-identical.

Version `2026-09-26.v2c1` contains 603 examples and 261 groups:

| Split | Examples | Groups | Examples per intent |
|---|---:|---:|---:|
| train | 405 | 72 | 45 |
| validation | 108 | 108 | 12 |
| locked_test | 90 | 81 | 10 |

The source categories are 54 unchanged V2-A seed examples, 378 deterministic
template expansions, 99 hand-authored validation examples, and 72 hand-authored
contrast/challenge examples. Every family has one `group_id`, and groups never
cross splits. Validation rejects duplicate IDs, blank text, invalid labels,
intent/risk incompatibility, cross-split groups, exact duplicate text, and
normalized duplicates after case, whitespace, and superficial punctuation are
removed.

The original 18 locked-test examples are unchanged. The expanded locked test
has 90 examples in 81 groups and is frozen by the dataset SHA-256 in the
manifest. Model, feature, hyperparameter, calibration, and abstention selection
use train plus validation only. `model_selection.json` is written with
`locked_test_evaluated: false` before the trusted local model artifact is used
for the one locked-test evaluation recorded in `classifier_report.json`.

## Models and selection

Both classifiers use a union of word TF-IDF 1–2 grams and `char_wb` TF-IDF 3–5
grams. Logistic Regression is the probability-capable candidate for log loss,
multiclass Brier score, confidence, and abstention experiments. Linear SVM is a
strong sparse-text comparison without probability calibration; group isolation
is not weakened to calibrate it.

The bounded `C` search is selected lexicographically on validation only:
intent macro-F1, intent-derived protected-write recall, available calibration,
then lower complexity. Linear SVM with `C=0.5` won validation and is the final
selected intent classifier. It reached locked-test intent macro-F1 `0.8195`,
above the rule baseline (`0.7317`) and majority baseline (`0.0222`). This is an
honest bounded synthetic result, not evidence of production readiness.

The best Logistic Regression probability candidate uses `C=2.0`. Its frozen
validation-only abstention rule requires a top-1/top-2 margin of at least `0.2`.
It covers 55.6% of validation examples with 100% selective accuracy and 100%
recall among evaluated protected-write examples. On the locked test it covers
56.7% with 100% selective accuracy. Abstention remains advisory metadata.

The independent risk model is deliberately experimental and redundant because
risk labels are deterministic from intent. Its disagreement with
intent-derived risk can indicate uncertainty, but it can never authorize an
action.

The grouped Logistic Regression learning curve rises from validation macro-F1
`0.5666` at 20% of training groups to `0.8247` at 80%, then `0.8160` at 100%.
The plateau and small decline suggest that future gains need more varied,
carefully reviewed examples rather than more variants of the same templates.
Synthetic phrasing also cannot establish generalization to public banking
corpora or real conversations.

## Frozen CFPB external evidence

The CFPB external evaluation loads the unchanged V2-C1 artifact only after its
model-selection metadata, internal report, artifact SHA-256, text-free labels,
and narrative-source linkage have been verified. The primary LinearSVC is
scored on 1,776 records with exact single-label targets; 24 genuine
multi-intent records remain in the 1,800-record holdout and receive only a
separate supported-intent membership score.

The supplied frozen local run reached 39.92% primary accuracy, 0.1136 macro-F1,
and 0.5033 weighted-F1, while the multi-intent membership score was 37.5%.
This exposes substantial domain shift and is not directly comparable to the
balanced controlled-synthetic locked test as a like-for-like benchmark. The
separate frozen Logistic path improved selective accuracy to 72.13% but
collapsed coverage to 10.30% (183 of 1,776 records). It does not replace or
validate the primary SVM.

These external results did not select, retrain, tune, or modify V2-C1.

## Frozen V2-C3 development contract

V2-C3 is a systematic next-generation model-development phase motivated by
the measured external domain shift. It does not tune against the already seen
BANKING77 test, CLINC test/processed evaluation, CFPB, or V2-C1 locked-test
results. Those remain historical evaluation evidence and may be rerun only
after a future V2-C3 configuration is frozen.

The configuration-only `v2c3_data_registry.json` and
`v2c3_experiment_contract.json` freeze the rules before execution. Later work
may compare bounded word/character TF-IDF, LSA, and frozen local
`BAAI/bge-small-en-v1.5` representations with LinearSVC,
LogisticRegression, SGDClassifier, and RidgeClassifier. It may compare direct
nine-way classification with a supported/current-then-intent hierarchy, using
group-aware `StratifiedGroupKFold`, bounded finalist optimization, predeclared
safety gates, and separately reported uncertainty/OOD behavior. No training,
embedding, cross-validation, model selection, or lockbox materialization occurs
in the contract step.

V2-C3 development may later use the existing internal train/validation splits
and conditionally eligible BANKING77 train and CLINC finance train/validation
examples. External eligibility is mapping-status-specific: exact and
unsupported categories are automatic candidates, while near and ambiguous
examples require utterance-level review. A taxonomy mapping can never invent
a protected-write request. A deterministic 80/20 development/fresh-lockbox
split must be made before training, with duplicate and lineage groups kept
together. Generic CLINC OOS examples are not automatically SentinelVoice
unsupported examples.

CFPB remains evaluation/research-only in full. Neither the frozen 1,800 nor
other available narratives, annotations, semantic labels, or structured
metadata-derived targets may be used for V2-C3 fitting or selection. Bitext
remains pending acquisition, mapping, and a later governance amendment;
unverified bank-support transcripts remain prohibited.

## V2-C3 development dataset and fresh lockbox

Step 3 implements the frozen data decision without training a model.
`scripts/build_v2c3_development_dataset.py` reads only the internal V2-C1
train/validation splits, BANKING77 `train.csv`, and CLINC finance `train`/`val`.
It includes external `EXACT_MATCH` and `UNSUPPORTED` mappings, while recording
`NEAR_MATCH` and `AMBIGUOUS` rows only as excluded pending semantic review.
CLINC OOS, all test splits, CFPB, Bitext, and unverified transcripts are never
eligible inputs.

The builder freezes Unicode NFKC/lowercase/whitespace normalization and groups
exact-normalized duplicates across every eligible source. Conflicting targets
fail the build. External duplicate groups that overlap internal development
cannot enter the lockbox. Remaining external groups are assigned by stable
seeded ordering within source/target strata to approximately 80% development
and 20% fresh lockbox. Model predictions and confidence never affect membership.

The development artifact contains text for future fitting and group-aware CV.
The fresh-lockbox artifact contains only reconstruction identities and hashes,
not text. Its members cannot be used until the V2-C3 representation, model,
hyperparameters, and thresholds are frozen. The dataset manifest records input
and output hashes, exclusions, distributions, duplicate controls, actual split
ratios, and leakage checks. Generate or verify the artifacts explicitly:

```bash
sentinelvoice_env/bin/python scripts/build_v2c3_development_dataset.py --write
sentinelvoice_env/bin/python scripts/build_v2c3_development_dataset.py --check
```

V2-C3 also freezes a separate synthetic final challenge set before any Step 4
experiment. It contains exactly 270 newly authored examples: 30 for each of the
nine frozen intents. Contrast families cover task boundaries, especially
protected-write positives versus lost/stolen-card statements, unfamiliar-charge
statements, historical disputes, prior card freezes, vague complaints, and
unsupported operations that do not contain a current protected-action request.
Lineage-related contrasts share a group ID.

The challenge set is not development data and is not the external fresh
lockbox. Its text may be used only for the one final challenge evaluation after
the representation, model, hyperparameters, and thresholds are frozen. It is
permanently ineligible for fitting, feature construction, cross-validation,
model selection, and threshold selection. The deterministic builder verifies
balance, derives risk from the frozen contract, and rejects normalized-text
overlap with V2-C3 development, the external fresh lockbox, the internal locked
test, BANKING77 test, or CLINC test/OOS test. It does not read CFPB data, train a
model, or run an evaluation. Development, the external lockbox, and this
challenge definition are all frozen before Step 4 experimentation:

```bash
sentinelvoice_env/bin/python scripts/build_v2c3_challenge_set.py --write
sentinelvoice_env/bin/python scripts/build_v2c3_challenge_set.py --check
```

## Reproduce V2-C1

```bash
sentinelvoice_env/bin/python scripts/build_v2_intent_dataset.py
sentinelvoice_env/bin/python scripts/run_v2_intent_experiment.py
```

Only load `artifacts/v2/classifier/classifier.joblib` when it was generated
locally from this trusted repository. Latency in the report measures the full
local vectorization and prediction path and is labeled `local_ml`; it is not
LLM, orchestration, STT, or voice latency.
