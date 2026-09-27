# SentinelVoice V2-C offline intent/risk experiment

V2-C uses only controlled synthetic and hand-reviewed SentinelVoice utterances.
It does not download, ingest, or train on BANKING77 or any other public or
external dataset. Production transcripts are not repository data. These three
possible sources remain separate in every artifact and report.

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

## Reproduce

```bash
sentinelvoice_env/bin/python scripts/build_v2_intent_dataset.py
sentinelvoice_env/bin/python scripts/run_v2_intent_experiment.py
```

Only load `artifacts/v2/classifier/classifier.joblib` when it was generated
locally from this trusted repository. Latency in the report measures the full
local vectorization and prediction path and is labeled `local_ml`; it is not
LLM, orchestration, STT, or voice latency.
