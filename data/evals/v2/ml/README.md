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

## V2-C3 Step 4 fixed-preset tournament

Step 4 is a broad, predeclared model and representation tournament, not a
hyperparameter search. `v2c3_tournament_config.json` freezes exactly 14
candidates before scores exist. They compare the exact V2-C1 word-plus-character
TF-IDF anchor, one bounded alternative word-plus-character preset, word-only
and character-only TF-IDF, fold-local TF-IDF plus 256-component LSA, and frozen
local `BAAI/bge-small-en-v1.5` features. The lightweight classifier families
are LinearSVC, LogisticRegression, SGDClassifier, and RidgeClassifier. Direct
nine-way classification is compared with a two-stage supported/current-then-
intent architecture.

All candidates use the same frozen five `StratifiedGroupKFold` validation
assignments with seed `20260928`; related groups never cross folds. TF-IDF and
LSA are fit anew inside each training fold. BGE is never fine-tuned: a separate,
ignored, label-free cache contains embeddings for the development examples in
their deterministic order, and tournament execution refuses to regenerate a
missing or invalid cache.

The runner has four explicit modes:

```bash
sentinelvoice_env/bin/python scripts/run_v2c3_model_tournament.py preflight
sentinelvoice_env/bin/python scripts/run_v2c3_model_tournament.py prepare-folds
sentinelvoice_env/bin/python scripts/run_v2c3_model_tournament.py prepare-embeddings
sentinelvoice_env/bin/python scripts/run_v2c3_model_tournament.py run
```

`preflight` and `prepare-folds` perform no model fitting. `prepare-embeddings`
uses FastEmbed locally and only the frozen development text. `run` requires the
already frozen fold artifact and validated BGE cache, creates one OOF prediction
per development example, applies the predeclared safety gates and ranking, and
writes `v2c3_tournament_report.json` without raw text or final-test metrics.

At Step 4, the 1,922 external-lockbox examples and 270 challenge examples had
not yet been used; Step 6 later consumed both for final evaluation. V2-C1
locked test, BANKING77 test, CLINC test/OOS test, and CFPB were also prohibited.
Step 4 selected three technically distinct, safety-eligible
finalists: direct V2-C1 word-plus-character TF-IDF with balanced LinearSVC,
direct frozen BGE-small with balanced LinearSVC, and direct TF-IDF plus LSA
with balanced LinearSVC. The TF-IDF candidate was the Step 4 winner. These
results did not use any final-evaluation source.

## V2-C3 Step 5 bounded finalist tuning

Step 5 is intentionally small. Step 4 already compared representation and
classifier families and direct versus hierarchical architectures. The frozen
`v2c3_tuning_config.json` therefore contains exactly 15 manual configurations:
five LinearSVC `C` values for unchanged V2-C1 TF-IDF, five for frozen BGE-small,
and five explicit LSA component/`C` pairs. It adds no classifier,
representation, embedding model, architecture, calibration, or threshold
search.

The tuning runner reuses the exact Step 4 five-fold assignment and development
embedding cache. TF-IDF and SVD remain inside each training fold; BGE remains a
fixed pretrained feature extractor. The runner cannot generate folds or
embeddings and delegates metrics, safety gates, ranking, latency, and model-size
measurement to the Step 4 implementation:

```bash
sentinelvoice_env/bin/python scripts/run_v2c3_bounded_tuning.py preflight
sentinelvoice_env/bin/python scripts/run_v2c3_bounded_tuning.py run
```

At Step 5, the 1,922-example external lockbox and 270-example challenge had not
yet been used; Step 6 later consumed both for final evaluation. CFPB,
historical test splits, and V2-C1 locked test stayed outside Step 5. Step 5
writes only a development-CV tuning report and does not integrate routing into
the runtime.
The completed search selected `bge_svc_c_4_0`: frozen local
`BAAI/bge-small-en-v1.5` passage embeddings and balanced LinearSVC `C=4.0`.
No final-evaluation source contributed to that decision.

## V2-C3 Step 6 final freeze and evaluation

Step 6 freezes the selection before any final score is observed.
`v2c3_final_evaluation_config.json` records the exact selected representation
and classifier parameters, development and selection-lineage hashes, final-set
hashes and compositions, the historical comparator, metrics, safety gates,
acceptance rules, and prohibited sources. It contains no final scores and
keeps `final_evaluation_performed=false`.

The runner separates three operations:

```bash
sentinelvoice_env/bin/python scripts/run_v2c3_final_evaluation.py preflight
sentinelvoice_env/bin/python scripts/run_v2c3_final_evaluation.py prepare-final-model
sentinelvoice_env/bin/python scripts/run_v2c3_final_evaluation.py evaluate
```

`preflight` checks hashes, the Step 5 winner, the 8,198-example development
dataset and frozen BGE cache, text-free external-lockbox manifest counts, the
challenge manifest, and the frozen V2-C1 artifact. It performs no fitting,
inference, final-text parsing, or final embedding generation.

`prepare-final-model` is the one development-only fit. It loads the existing
BGE development cache, fits only the selected balanced LinearSVC `C=4.0` on all
8,198 examples with no CV, and writes a trusted-local, runtime-ineligible
classifier plus text-free lineage metadata. It cannot regenerate embeddings or
read the lockbox or challenge set.

Only the explicit `evaluate` mode reconstructs the exact 1,922
external-lockbox members from frozen BANKING77 train and CLINC train/validation
identities and hashes, loads the balanced 270-example challenge set, and
generates their BGE embeddings. It reports the two datasets separately for
exactly `final_v2c3` and the frozen V2-C1 historical baseline. The comparison
is not post-test model selection: results cannot trigger retuning, threshold
tuning, or a switch back to V2-C1. CFPB remains separate for later
complex-narrative evaluation or expansion research.

Final evaluation is complete. Both external-lockbox safety gates passed; all
three challenge safety gates failed, so `final_safety_acceptable=false`. No
retuning, threshold tuning, or model switching occurred. The report keeps the
frozen `integrity_verification` top-level key for compatibility, but clearly
nests the no-inference pre-evaluation snapshot under
`pre_evaluation_integrity` and completed work under `evaluation_execution`.
This is reporting clarification only and changes no model, prediction, metric,
safety gate, or acceptance result.

## V2-C4 safety-recovery contract and fresh holdout

V2-C3 is closed. Its final safety acceptance failed and is not hidden or tuned
away: both required external-lockbox gates passed, while challenge
protected-write false-positive rate, exact protected-write recall, and
unsupported recall all failed their predeclared gates. The V2-C3 external
lockbox and challenge are therefore consumed final-evaluation datasets. V2-C4
may use them only for diagnostics, historical comparison, and separately
reported regression measurement; they are no longer untouched holdouts and
cannot provide V2-C4 final acceptance.

`v2c4_experiment_contract.json` freezes the new experiment before individual
V2-C3 challenge mistakes are reviewed. Its scope is limited to the three
failed safety behaviors, and it preserves the same gates: protected-write
false-positive rate at most `0.01`, protected-write recall at least `0.80`, and
unsupported recall at least `0.80`. `v2c4_error_analysis_config.json` freezes
the future failure categories before results are categorized.
`scripts/analyze_v2c3_challenge_errors.py` provides a hash-verified diagnostic
workflow for the consumed V2-C3 challenge; it does not read the sealed V2-C4
holdout, train, tune, or select a model. Error analysis must precede any V2-C4
model or development-data change, and no V2-C4 improvement is claimed.

V2-C4 receives a new synthetic final safety holdout rather than recycling the
V2-C3 challenge. The seed and deterministic builder create 360 examples—40 for
each frozen intent—with explicit gold labels, independent V2-C4 lineage, and
deliberate protected-action and unsupported boundaries. The builder rejects
normalized duplicates and overlap with V2-C3 development, the consumed V2-C3
challenge, or the consumed external lockbox hashes. It also requires the
predeclared safety-pattern coverage and protected-mention hard negatives in
every non-protected lane.

```bash
sentinelvoice_env/bin/python scripts/build_v2c4_safety_holdout.py --write
sentinelvoice_env/bin/python scripts/build_v2c4_safety_holdout.py --check
```

Run the consumed-challenge diagnostic separately when ready:

```bash
sentinelvoice_env/bin/python scripts/analyze_v2c3_challenge_errors.py
```

That command writes
`data/evals/v2/ml/v2c4_error_analysis_report.json`, the result path frozen in
the error-analysis config. Its metrics are labeled diagnostic regression
metrics and cannot establish V2-C4 final acceptance. The fresh V2-C4 final
holdout remains sealed. The report contains text hashes but no raw challenge
text, preserving `raw_text_persisted=false`. It also includes diagnostic
LinearSVC class scores and ranks, protected-versus-nonprotected score gaps, and
error aggregates across every existing consumed-challenge tag. These are
diagnostic descriptions, not model-selection evidence or conclusions. After
the report exists, display the consumed safety-review records temporarily on
stdout with no inference or file modification:

```bash
sentinelvoice_env/bin/python scripts/analyze_v2c3_challenge_errors.py \
  --print-review-queue
```

Every holdout record has `training_eligible=false`,
`model_selection_eligible=false`, `threshold_selection_eligible=false`, and
`error_analysis_eligible=false`. The holdout contains no predictions and may be
evaluated only once after a future V2-C4 candidate is selected and frozen using
development evidence. This step performs no V2-C3 error analysis, training,
inference, tuning, runtime integration, or V2-C4 evaluation. It makes no
improvement claim. CFPB remains separate evaluation/research data.

## V2-C4 intervention plan

`v2c4_intervention_plan.json` is the frozen Step 9 contract, not an execution
artifact. Candidate A retains the frozen direct nine-way
`BAAI/bge-small-en-v1.5` plus balanced LinearSVC (`C=4.0`) recipe and changes
only its development training data. The planned targeted augmentation contains
360 independently authored examples: 120 protected positives balanced between
`create_dispute` and `freeze_card`, 120 matched protected-action hard
negatives with their correct existing labels, and 120 unsupported or scope
examples. Contrastive relatives share lineage so group-aware evaluation cannot
split them across folds.

The primary selection evidence will be a separately authored 270-example
development probe, balanced at 30 examples per intent and excluded from model
fitting, threshold selection, and final acceptance. Candidate A must satisfy
all three existing safety gates without a macro-F1 drop greater than `0.01`
against the frozen V2-C3 baseline on that same probe. Candidate B is not
automatic: the predeclared three-stage hierarchical LinearSVC fallback may be
built only if Candidate A fails any safety gate or materially regresses. It
must use the same training corpus and selection probe without Candidate-B-only
augmentation. If multiple eligible candidates pass, macro-F1 is primary; an
absolute difference below `0.005` is a practical tie resolved by the frozen
safety, latency, artifact-size, and simplicity order.

The consumed V2-C3 challenge and external lockbox cannot supply training text,
model selection, threshold tuning, or final acceptance evidence. After
selection is frozen they may be used only as separately reported diagnostic
regressions. The final V2-C4 holdout remains sealed until one evaluation after
the architecture, hyperparameters, training-data hashes, classifier artifact,
embedding metadata, and evaluation code are frozen. Step 9 creates no examples,
builders, training code, candidate artifacts, clustering code, inference, or
runtime integration.

Step 10 adds independently authored family-based seed sources and the
deterministic `scripts/build_v2c4_development_data.py` builder. The builder
produced the frozen 360-example training augmentation and the independent,
balanced 270-example model-selection probe plus their manifests. The former is
training-only; the latter is model-selection-only. Neither is threshold-
selection or final-acceptance evidence. The builder performs mechanical
normalization, count, lineage, and exact-hash leakage checks only—it performs
no embedding, fitting, inference, or evaluation. The sealed 360-example V2-C4
final holdout remains untouched, and no V2-C4 candidate has been trained.

The Step 10 development datasets are frozen. Step 11 defines Candidate A in
`v2c4_candidate_a_config.json`: it retains the exact frozen V2-C3
`BAAI/bge-small-en-v1.5` `passage_embed` plus balanced LinearSVC (`C=4.0`)
recipe and changes only targeted development training data. Its primary
development-selection evidence is the frozen 270-example probe, while five-fold
`StratifiedGroupKFold` evaluation is supporting evidence only. The final
360-example V2-C4 safety holdout remains sealed; training and probe evaluation
in Step 11 cannot establish or claim final V2-C4 improvement.

Candidate A materially improved development metrics over the V2-C3 baseline,
but failed the mandatory protected-write false-positive-rate and
unsupported-recall gates. The triggered hierarchical Candidate B failed those
same gates and regressed versus Candidate A, so V2-C4 selected no development
candidate and Candidate C is prohibited. The 270-example selection probe is
now consumed. Step 13 may analyze it only through the diagnostic postmortem in
`v2c4_selection_probe_postmortem_config.json`; it is no longer eligible for
training, tuning, threshold selection, model selection, or final acceptance.
The final V2-C4 holdout remains sealed and will not be opened. After V2-C4
closeout, the next planned phase is V2-C5 intent discovery and taxonomy
expansion; no successful V2-C4 final model is claimed.

## V2-C4 closeout

V2-C4 is closed without a selected development candidate. Candidate A's
development-selection metrics were accuracy 0.8037037037037037, macro-F1
0.8070520298946007, protected-write recall 0.9333333333333333,
protected-write false-positive rate 0.03333333333333333, and unsupported recall
0.6666666666666666. It passed protected recall and the macro-F1 regression
requirement, but failed the mandatory protected-FPR and unsupported-recall
gates. The resulting predeclared Candidate B recorded accuracy
0.7111111111111111, macro-F1 0.7247635637541335, protected-write recall 0.8,
protected-write false-positive rate 0.04285714285714286, and unsupported recall
0.36666666666666664. It also passed protected recall and the macro-F1 regression
requirement, failed the other two gates, and regressed relative to Candidate A.
Neither candidate was eligible, no candidate was selected, Candidate C was
prohibited, and the decision was to stop for human review.

Step 13 is diagnostic evidence only. It found 53 of Candidate B's 78 errors
first failed at Stage 1, including all 19 unsupported false-supported errors;
the contextual, ambiguous, and no-current-request lanes were especially weak.
These results may motivate V2-C5 taxonomy or discovery hypotheses, but do not
establish that unsupported-or-uncertain must be split or establish a new intent
taxonomy. Human adjudication is required.

Step 14, the once-only final safety-holdout evaluation, was skipped because no
development candidate passed the frozen mandatory safety gates and was selected
and frozen. The final holdout was never accessed or evaluated. Development
evidence therefore exists, but final acceptance evidence does not:
final-safety acceptability is not evaluated, no production approval is claimed,
and the classifier remains advisory only. Deterministic SentinelVoice
application code continues to own authentication, authorization, resource
ownership, confirmation, protected tool execution, idempotency, and state
transitions. No V2-C4 classifier is approved as an authorization mechanism.
The next phase is V2-C5 intent discovery and taxonomy expansion. Steps 16-18
have frozen the discovery contract, corpus, and primary clustering evidence;
Step 19 supplies the local human-adjudication workflow without changing the
taxonomy.

## Roadmap: V2-C4 → V2-C5 → V2-C6 → V2-D

V2-C5, **Intent Discovery and Taxonomy Expansion**, followed the V2-C4
nine-intent safety-recovery experiment and tested a human-adjudicated expanded
taxonomy. It closed after the selected candidate failed final safety gates.
V2-C6 now performs generalization diagnosis and separately frozen remediation
before the separately governed V2-D phase. Its diagnosis must not revise the
historical V2-C5 result or consume that holdout as reusable evidence.

V2-C5 used only development- or training-eligible sources. Frozen sentence
embeddings with HDBSCAN provided advisory candidate-cluster discovery; a human
adjudicator, not clustering, decided whether any cluster became an intent.
CFPB narratives, annotations, labels, and metadata-derived targets were
excluded from taxonomy discovery and training. The expanded taxonomy required
a fresh final holdout because the V2-C4 nine-intent holdout was not valid final
evidence for a changed label space. Additional recognized intents did not
automatically imply additional runtime tools.

## V2-C5 Step 16 intent-discovery contract

V2-C4 is closed without a selected candidate. V2-C5 therefore starts with
taxonomy discovery, not Candidate C, and investigates whether the current
unsupported_or_uncertain bucket contains coherent latent user-goal groups.
The V2-C4 Stage-1 and unsupported-boundary findings are hypothesis prompts
only: they are not clustering labels, target classes, parameter-selection
evidence, or proof that unsupported_or_uncertain must be split. No cluster or
new intent exists at Step 16.

Primary discovery is frozen to unique normalized texts from the existing
V2-C3 development dataset whose current label is unsupported_or_uncertain.
Only development-assigned internal train/validation, BANKING77 train, and CLINC
finance train/validation records are eligible. Supported intents are excluded
from density estimation and may later serve only as separate reference anchors.
Exact normalized-text duplicates contribute one vector while preserving all
source and label metadata. Native external and current SentinelVoice labels may
support post-clustering interpretation or audit, but are never clustering
features. CFPB, test and locked sets, consumed V2-C3 evaluation data, the V2-C4
probe and postmortem review examples, and the V2-C4 final holdout are excluded.
The purpose-built V2-C4 augmentation is also excluded from primary discovery
because it could manufacture or amplify the known boundary structure being
investigated.

The architecture decision is frozen BAAI/bge-small-en-v1.5 FastEmbed
passage embeddings, L2 normalization, HDBSCAN in the original 384-dimensional
space, and mandatory human adjudication. The primary HDBSCAN configuration is
Euclidean distance, EOM selection, epsilon 0.0, no single cluster, minimum
cluster size 30, and minimum samples 10. A bounded 3-by-3 sensitivity grid uses
cluster sizes 15/30/60 and minimum samples 5/10/15 without replacing the
primary result. UMAP uses random seed 20260928 only for two-dimensional
visualization; it is not clustering input, taxonomy evidence, or parameter-
selection evidence.

This choice fits an unknown cluster count where noise is expected. KMeans,
agglomerative clustering, topic models or BERTopic, and LLM grouping remain
outside the primary experiment because fixed-count assumptions, additional
complexity, or less deterministic behavior are not yet justified. Tradeoffs
include density sensitivity, embedding-space dependence, cluster instability,
human-review cost, and clusters that may capture phrasing or topic rather than
user intent. Reversibility is high because discovery outputs cannot affect the
runtime until humans explicitly adjudicate and a later step freezes a taxonomy.

Human reviewers must inspect representative and boundary examples and may map
a cluster to an existing intent, propose a candidate new intent, leave it
unsupported, mark it mixed or needing split review, or classify it as noise or
insufficient evidence. There is no predetermined intent count, and clusters do
not automatically create intents or tools. Protected intents still require
explicit current-action semantics; topic similarity cannot imply freeze_card
or create_dispute. SentinelVoice's V1 tool scope and deterministic application
authority remain unchanged.

The old nine-intent V2-C4 holdout cannot provide V2-C5 final evidence. After
human adjudication and the Step 20 taxonomy freeze, Step 21 must create a new
independently authored V2-C5 holdout before supervised model selection. Step 16
creates no corpus, embeddings, clusters, taxonomy change, model, runtime
integration, or holdout.

The Step 16 contract is now frozen. Step 17 adds a deterministic builder for
the discovery population using only the hash-pinned V2-C3 development artifact.
Its primary population is the current unsupported_or_uncertain examples from
the five approved development source roles. Exact normalized-text deduplication
prevents duplicate utterances from inflating future HDBSCAN density while
preserving every source occurrence, native label, mapping status, and available
provenance as non-feature metadata. The eight supported intents are represented
separately by a reproducible hash-pinned reference-anchor filter rather than
copied into the primary corpus.

The corpus builder accepts only raw text as the future embedding field. It does
not generate BGE embeddings, run HDBSCAN or UMAP, inspect clusters, change the
taxonomy, train a classifier, or alter runtime behavior. The Step 18 runner
performs the frozen BGE embedding and HDBSCAN protocol only when the user
explicitly executes it against the frozen Step 17 corpus and manifest.

## V2-C5 Step 18 unsupervised discovery runner

The frozen Step 17 primary corpus contains 6,372 unique normalized texts. Step
18 is the actual unsupervised discovery step: its runner generates or verifies
local BGE-small passage embeddings, L2-normalizes the 384-dimensional vectors,
and gives HDBSCAN only the original 384-dimensional representation. The primary
configuration remains minimum cluster size 30 and minimum samples 10. It is one
member of exactly nine total bounded sensitivity runs and cannot be replaced
post hoc by a result that appears more attractive.

Seeded UMAP support is optional, non-blocking, and visualization-only; UMAP
coordinates never become clustering input. Cluster membership, native-label
concentration, and source concentration are exploratory diagnostics rather than
semantic truth or selection evidence. No cluster automatically becomes an
intent. Step 19 performs human semantic adjudication before Step 20 can freeze
any taxonomy change. The runner and tests do not themselves establish or claim
clustering results.

## V2-C5 Step 19 human taxonomy adjudication

Step 18 is frozen at exactly 35 primary `mcs30_ms10` clusters. These clusters
are proposals for inspection, not intents. Step 19 is a standard-library-only
local review workflow and never invokes embeddings, HDBSCAN, UMAP, or a
classifier. It reads only the hash-pinned Step 18 contract, report,
assignments, manifest, discovery corpus, and runner.

The workflow uses the git-ignored local workfile
`data/evals/v2/ml/local/v2c5_taxonomy_review_workfile.json`. `build` creates it
once with deterministic descending-member-count ordering and every record set
to `UNREVIEWED`; it refuses to overwrite an existing file. The workfile stores
cluster identifiers, counts, representative and boundary discovery IDs, and
source/native-label concentrations, but no duplicated utterance text. `show`
and `next` join review text from the frozen discovery corpus at display time and
separate representative examples from boundary cases. External labels and
source datasets are displayed as metadata only.

The commands are:

```bash
sentinelvoice_env/bin/python scripts/review_v2c5_taxonomy.py build
sentinelvoice_env/bin/python scripts/review_v2c5_taxonomy.py check
sentinelvoice_env/bin/python scripts/review_v2c5_taxonomy.py summary
sentinelvoice_env/bin/python scripts/review_v2c5_taxonomy.py show --cluster-id <ID>
sentinelvoice_env/bin/python scripts/review_v2c5_taxonomy.py next
sentinelvoice_env/bin/python scripts/review_v2c5_taxonomy.py set \
  --cluster-id <ID> \
  --decision <DECISION> \
  --human-theme <THEME> \
  --confidence <HIGH|MEDIUM|LOW> \
  --runtime-change-required <true|false|null> \
  --human-rationale <RATIONALE>
sentinelvoice_env/bin/python scripts/review_v2c5_taxonomy.py export
```

`set` accepts exactly one explicit human decision:
`MAP_TO_EXISTING_INTENT`, `CANDIDATE_NEW_INTENT`, `REMAIN_UNSUPPORTED`,
`NEEDS_SPLIT_REVIEW`, `MIXED_OR_INCOHERENT`, or `INSUFFICIENT_EVIDENCE`.
All reviewed decisions require a human theme, confidence, an explicitly
supplied runtime-change value (`true`, `false`, or unresolved `null`), and a
rationale. `null` is valid when runtime implications remain unresolved,
including `NEEDS_SPLIT_REVIEW`, `MIXED_OR_INCOHERENT`, and
`INSUFFICIENT_EVIDENCE`; it is never inferred from the decision.
Existing-intent mappings additionally require
`--target-existing-intent`; new-intent candidates instead require
`--candidate-intent-name`; all other decisions require both fields to remain
empty.

Existing mappings are limited to `informational_policy`, `account_balance`,
`recent_transactions`, `transaction_details`, `card_status`, `freeze_card`,
`create_dispute`, and `escalation`. `unsupported_or_uncertain` is not a
supported target. No protected action is inferred: topic similarity to a
frozen card or a disputed transaction does not establish explicit current
action semantics.

`check` validates the frozen hashes, exact cluster set, ordering, and review
field consistency without writing the workfile. `export` refuses while any
cluster remains unreviewed. A completed export creates the tracked, text-free
`v2c5_taxonomy_adjudication.json` and
`v2c5_taxonomy_adjudication.manifest.json`, with frozen Step 18 hashes, the
review-script hash, cluster/review totals, and decision counts. These artifacts
remain human discovery evidence only: `taxonomy_changed=false`,
`new_intents_created=false`, `classifier_training_performed=false`,
`runtime_behavior_changed=false`, `final_taxonomy_frozen=false`, and
`step20_required=true`.

## V2-C5 Step 20 expanded-taxonomy freeze

Step 19 produced human discovery evidence only. Step 20 consumes only its
hash-pinned adjudication and manifest and freezes a separate, versioned V2-C5
classifier taxonomy. The historical V2-C1, V2-C3, and V2-C4 nine-intent
contracts remain unchanged.

The final taxonomy retains all nine historical labels and accepts exactly
seven new labels:

- `account_blocked`
- `cancel_transfer`
- `close_account`
- `lost_or_stolen_phone`
- `passcode_recovery`
- `transfer_failed_or_declined`
- `transfer_pending`

The resulting classifier label space contains exactly 16 unique labels in
lexicographic order. `transfer_fee_charged` merges into
`transaction_details`. `card_retained_by_atm` merges into
`informational_policy`; ATM retention alone never implies `freeze_card`, which
still requires explicit current-action semantics.

All nine `NEEDS_SPLIT_REVIEW` clusters receive explicit branch semantics for
future Step 21 relabeling. These preserve the distinctions between guidance
and unsupported direct operations, already-blocked accounts and new bank-
account freeze requests, generic transfer timing and pending-transfer
investigation, and verification guidance and failed verification. Step 20 does
not automatically relabel individual utterances or build a dataset.

The V2-C5 risk labels are classifier/evaluation metadata only. They do not
authorize actions. The protected-write set is `cancel_transfer`,
`close_account`, `create_dispute`, and `freeze_card`, but recognizing
`cancel_transfer` or `close_account` adds no runtime tool. Runtime tools,
authorization, conversation routing, confirmation, ownership, idempotency,
protected execution, and state transitions remain unchanged.

The standard-library-only freeze workflow is:

```bash
sentinelvoice_env/bin/python scripts/freeze_v2c5_taxonomy.py check
sentinelvoice_env/bin/python scripts/freeze_v2c5_taxonomy.py export
```

`check` validates the exact Step 19 hashes, completed decision counts,
candidate names, split-cluster IDs, all 35 cluster resolutions, final taxonomy,
risk coverage, protected-write set, governance, and any existing Step 20
outputs without writing. `export` performs the same validation before creating
deterministic `v2c5_taxonomy_freeze.json` and
`v2c5_taxonomy_freeze.manifest.json`. Neither command consumes a holdout, test
set, CFPB data, model, embedding, clustering result, training artifact, or
runtime artifact beyond the two frozen Step 19 evidence files.

The architectural choice is a separate versioned V2-C5 taxonomy artifact. It
solves the need for a reproducible expanded label space without rewriting the
historical nine-intent experiments, and it keeps provenance, safety boundaries,
and future dataset construction explicit. Modifying `IntentLabel` globally,
inferring labels directly from clusters, and immediately adding runtime tools
were rejected because they would mutate historical semantics, treat clustering
as truth, or couple classifier recognition to execution authority. The
tradeoff is temporary label-definition duplication. Reversibility remains high
until later model selection and runtime integration; future taxonomy versions
can be introduced without changing older contracts.

Step 21 remains required to construct the expanded supervised dataset and a
new, independently authored final holdout. The old V2-C4 holdout remains
prohibited as final evidence for the changed label space.

## V2-C5 Step 21A split-cluster relabel review

Step 20 froze branch semantics for nine `NEEDS_SPLIT_REVIEW` clusters without
automatically relabeling individual discovery texts. Those clusters contain
975 unique normalized development texts. Frozen Step 20 semantics plus clean
native-label evidence deterministically resolve 397 records. The other 578 are
wording-dependent and require explicit human review, distributed as follows:

- `cluster-25a4740ae330f948`: 161
- `cluster-bab7911ea80f612b`: 221
- `cluster-3db20683a0becc1c`: 86
- `cluster-783d0d28bd4c4992`: 66
- `cluster-2b49d15c7f2e42b9`: 44

Native external labels remain metadata/evidence only and were not clustering
features. They select deterministic branches only for the exact hash-pinned
label/cluster combinations frozen by Step 21A. Manual records start completely
`UNREVIEWED`; their native labels never prepopulate a human decision. Human
review is limited to `informational_policy` or
`unsupported_or_uncertain`, following the Step 20 distinction between generic
guidance and direct operations or customer-specific investigations.

The local workflow is:

```bash
sentinelvoice_env/bin/python scripts/review_v2c5_split_relabels.py build
sentinelvoice_env/bin/python scripts/review_v2c5_split_relabels.py check
sentinelvoice_env/bin/python scripts/review_v2c5_split_relabels.py summary
sentinelvoice_env/bin/python scripts/review_v2c5_split_relabels.py show \
  --discovery-id <ID>
sentinelvoice_env/bin/python scripts/review_v2c5_split_relabels.py next
sentinelvoice_env/bin/python scripts/review_v2c5_split_relabels.py set \
  --discovery-id <ID> \
  --final-intent <informational_policy|unsupported_or_uncertain> \
  --confidence <HIGH|MEDIUM|LOW> \
  --reviewer-note <OPTIONAL_NOTE>
sentinelvoice_env/bin/python scripts/review_v2c5_split_relabels.py export
```

The ignored local workfile is
`data/evals/v2/ml/local/v2c5_split_relabel_workfile.json`. It stores stable
identity and source/native-label metadata but no utterance text; `show` and
`next` join text from the frozen discovery corpus only at display time.
`check` writes nothing and verifies all six frozen source hashes, development-
only lineage, the 578/397/975 partition, exact per-cluster counts and mappings,
non-overlap, allowed labels, and any existing exports. `build` refuses to
overwrite human work. `export` refuses until all 578 manual records are
reviewed, then creates deterministic, tracked, text-free adjudication and
manifest artifacts.

The architecture choice is hybrid deterministic plus targeted human
relabeling. It reduces unnecessary review where frozen branch evidence is
unambiguous while preserving human judgment where speech act matters. Manual
review of all 975 was rejected because 397 already have deterministic frozen
evidence. Bulk-mapping all 975 from native labels was rejected because several
native classes mix informational guidance with direct operations or specific
investigations. The tradeoff is 578 explicit reviews. The result is highly
reversible because it is separate versioned development evidence and changes
no runtime behavior.

Step 21A trains no classifier, exports no expanded development dataset, creates
no fresh holdout, and changes no runtime behavior. Step 21 remains incomplete
until the review export is complete, the expanded supervised development
dataset is built, and a new independently authored final holdout is authored
and frozen. Existing final/test data, CFPB, V2-C3 challenge/lockbox sources, and
the V2-C4 holdout remain prohibited.

## V2-C5 Step 21B expanded development dataset

Step 21B converts the historical occurrence-level V2-C3 development population
to the frozen 16-intent V2-C5 taxonomy without model-generated labels. Its
hash-pinned inputs are exactly:

- `v2c3_development_dataset.json`
- `v2c5_discovery_corpus.json` and its manifest
- `v2c5_intent_discovery_assignments.json`, report, and manifest
- `v2c5_taxonomy_freeze.json` and its manifest
- `v2c5_split_relabel_adjudication.json` and its manifest

The builder retains the original 8,198 development occurrences, example IDs,
text, grouping, source provenance, mapping metadata, and ordering. The 1,825
historically supported records keep their intents. Each of the 6,373 historical
`unsupported_or_uncertain` occurrences rejoins the Step 17 corpus through the
frozen normalization contract and resolves to one of 6,372 discovery IDs. It
then consumes only `primary_canonical_cluster_id` from the Step 18 primary
assignment. Sensitivity assignments are diagnostic only. Primary noise remains
unsupported; other clusters consume Step 20 frozen resolutions, and all 975
split-cluster discovery texts consume the completed Step 21A evidence.

Step 17 deduplicated one normalized-text occurrence only to prevent duplicate
vectors from affecting HDBSCAN density. That deduplication is not a supervised
sampling decision. Step 21B preserves both original occurrences and gives them
the same discovery-level relabel result, retaining the historical development
distribution rather than silently training on only 6,372 texts.

The standard-library-only commands are:

```bash
sentinelvoice_env/bin/python \
  scripts/build_v2c5_expanded_development_dataset.py --check
sentinelvoice_env/bin/python \
  scripts/build_v2c5_expanded_development_dataset.py --write
```

`--check` validates all frozen hashes and bindings, occurrence coverage,
primary-only joins, every Step 20 cluster resolution, Step 21A split coverage,
the exact frozen label and risk spaces, source exclusions, duplicate retention,
and deterministic output bytes when outputs exist. It writes nothing. `--write`
runs the same validation and creates
`v2c5_expanded_development_dataset.json` and its manifest. The manifest derives
final class and risk counts from the build; no final per-intent distribution is
preselected. Relabel provenance fields are metadata, while `text` remains the
only future classifier input.

The chosen architecture is deterministic occurrence-level relabel
reconstruction from frozen discovery and human-adjudication evidence. Manual
review of all 8,198 records is unnecessary and error-prone; native source labels
are not SentinelVoice taxonomy truth; model pseudo-labeling would create
circular supervision; and using only the deduplicated discovery corpus would
change the historical occurrence distribution. The tradeoffs are deterministic
propagation of any frozen discovery/adjudication mistake and additional
provenance metadata. Reversibility is high because this is a separate versioned
development artifact with no runtime effect.

Step 21B consumes no final, test, challenge, external lockbox, CFPB,
selection-probe, or V2-C4 holdout data. It trains and evaluates no model and
changes no runtime behavior. Step 21 remains incomplete until a new,
independently authored V2-C5 final holdout is frozen, and Step 22 training or
model selection must not begin before that holdout exists.

## V2-C5 Step 21C1 final-holdout contract freeze

Step 21C1 freezes final-holdout methodology before any V2-C5 final example is
authored. The text-free
`data/evals/v2/ml/v2c5_final_holdout_contract.json` pins the Step 20 taxonomy
freeze and manifest plus the Step 21B expanded development dataset and
manifest. The exact design is 40 independently authored synthetic examples for
each of the 16 frozen intents: 640 total, with 160 protected-write positives
and 480 non-protected examples. Class balance provides equal final evidence for
new intents and is intentionally different from development prevalence.

The future author may know taxonomy definitions, permission semantics, and
qualitative boundary requirements, but must not inspect development utterances
while writing. Sampling or paraphrasing Step 21B, copying BANKING77, CLINC,
CFPB, consumed V2-C3 evaluation sources, V2-C4 development evidence, or the
sealed V2-C4 holdout is prohibited. Predictions, classifier scores,
embedding/similarity searches against errors, LLM-selected gold labels, and
post-Step-22-failure generation are also prohibited. Every example must be
written directly for a known gold intent.

Every intent must mix clear/direct, natural or colloquial, short voice-style,
longer contextual, and neighboring-boundary formulations where appropriate.
Protected-write positives require explicit current-action semantics and varied
direct, polite, and indirect requests; topic mention alone is insufficient.
Non-protected lanes include protected-topic hard negatives where appropriate.
The unsupported lane must span heterogeneous out-of-scope goals rather than 40
variations of one topic. The contract also records the important card,
transaction, transfer, account, phone, passcode, and policy boundaries that the
future seed must cover.

The future deterministic builder must reject duplicates inside the holdout and
exact normalized-text overlap with Step 21B development, the consumed V2-C3
challenge, the consumed V2-C3 external lockbox, all historical development or
training text used in V2-C5, and any V2-C5 model-selection probe created before
the holdout build. Frozen hashes or text-free manifests should be used where
available. The V2-C4 final holdout is explicitly excluded from overlap checking
and remains unopened.

The chosen architecture is a balanced independently authored synthetic final
holdout. V2-C4 reuse, development sampling, external test adoption, natural
imbalance, and generation after observing Step 22 errors were rejected because
they lack expanded-label coverage, leak development evidence, substitute
external labels for SentinelVoice semantics, weaken low-frequency evidence, or
contaminate final evaluation. Tradeoffs are synthetic-domain limitations,
non-production prevalence, finite per-class resolution, and careful human
review. Reversibility is high before final evaluation.

Step 21C1 creates no holdout examples and performs no training, inference,
evaluation, model selection, or runtime change. Step 21 remains incomplete and
Step 22 remains prohibited until the independently authored final holdout is
built and frozen. Once frozen, it is ineligible for training, model/threshold
selection, augmentation, taxonomy discovery, or pre-final error analysis. It
may be evaluated once in Step 23 only after the candidate, evaluation logic,
and acceptance criteria are frozen.

## V2-C5 Step 21C2 independent authoring and deterministic sealing

Step 21C2A independently authored the ignored local seed
`local/v2c5_final_holdout_seed.json`: 640 synthetic examples with exactly 40
examples for every frozen intent. Authoring used only the Step 21C1 contract,
Step 20 taxonomy semantics, and taxonomy manifest. It did not inspect
development utterances, consumed evaluation examples, external dataset text,
model behavior, or the sealed V2-C4 final holdout. Gold intents are supplied by
the human-authored seed; risk is intentionally absent and cannot be inferred
from model output.

Step 21C2B adds the standard-library-only deterministic builder
`scripts/build_v2c5_final_holdout.py`:

```bash
sentinelvoice_env/bin/python scripts/build_v2c5_final_holdout.py --check
sentinelvoice_env/bin/python scripts/build_v2c5_final_holdout.py --write
```

The builder hash-pins the Step 21C1 contract, Step 20 taxonomy and manifest,
Step 21B expanded development data and manifest, consumed V2-C3 challenge and
manifest, and the text-free consumed V2-C3 external-lockbox manifest. It uses
the frozen `unicode-nfkc-lower-whitespace.v1` normalization and rejects exact
normalized overlap with development, historical development/training evidence,
the consumed challenge, the external lockbox, and any pre-existing V2-C5
model-selection probe. It also rejects invalid seed structure, non-sequential
IDs, unknown labels, incomplete qualitative coverage, raw or normalized
duplicates, seed-supplied risk, and prediction, embedding, vector, or
similarity metadata. It never performs semantic similarity detection.

`--check` validates all inputs and any existing tracked outputs without writing.
`--write` performs identical validation before deterministically creating
`v2c5_final_holdout.json` and `v2c5_final_holdout.manifest.json`. The output is
balanced at 640 examples, contains 160 protected-write positives and 480
non-protected examples, derives every risk from Step 20, and is ineligible for
training, model or threshold selection, augmentation, taxonomy discovery, and
pre-final error analysis. The sealed V2-C4 holdout remains unopened and is
explicitly prohibited as an input.

The architecture is independent synthetic authoring followed by deterministic
exact-overlap validation and sealing. It creates auditable fresh evidence for
the expanded taxonomy without allowing development or model behavior to shape
the test set. Semantic filtering, model-assisted relabeling, development
sampling, V2-C4 reuse, and direct external-test adoption were rejected because
they introduce model influence, leakage, taxonomy mismatch, or non-SentinelVoice
gold labels. Exact hashing does not catch every semantic near-paraphrase, the
balanced synthetic distribution is not production prevalence, and 40 examples
per class give finite resolution. The design remains reversible before the
once-only final evaluation.

The final holdout and manifest are now frozen and deterministically checked.
Step 21 is complete, and the manifest permits Step 22. Step 22 may use only Step
21B development evidence for model comparison and tuning; it cannot inspect or
evaluate the final holdout, use its labels or errors, alter model choices or
thresholds from it, or use it as augmentation. Step 23 owns the single final
evaluation after the selected candidate, evaluation implementation, acceptance
metrics, and safety gates are frozen.

## V2-C5 Step 22A development-only model-selection contract

The configuration-only `v2c5_model_selection_contract.json` freezes the V2-C5
experiment before training, embedding generation, vectorization, CV, inference,
or candidate evaluation. It hash-pins the Step 20 taxonomy and manifest, the
Step 21B expanded development dataset and manifest, the Step 21C1 holdout
contract, and the Step 21C2 final-holdout manifest. Only the holdout manifest is
read to verify `final_holdout_frozen=true`, `final_holdout_evaluated=false`, and
`step22_permitted=true`; `v2c5_final_holdout.json` remains unopened and is not a
Step 22 input.

The frozen matrix has exactly five families and 27 configurations:

- word `(1,2)` TF-IDF with LinearSVC: six configurations;
- `char_wb` `(3,5)` TF-IDF with LinearSVC: six configurations;
- the exact word/character FeatureUnion with LinearSVC: six configurations;
- 384-dimensional L2-normalized FastEmbed passage embeddings from
  `BAAI/bge-small-en-v1.5` with LinearSVC: six configurations;
- the same BGE representation with balanced `lbfgs` LogisticRegression: three
  configurations.

Every LinearSVC family crosses `C=[0.25,1.0,4.0]` with
`class_weight=[null,"balanced"]`; logistic regression uses the same `C` grid
with balanced weighting only. The historical V2-C3 BGE-small plus balanced
LinearSVC at `C=4.0` appears exactly once. The contract excludes tree boosting,
fine-tuned neural classifiers, LLM classification, extra embedding models,
threshold tuning, and hierarchical classification from this bounded step.

Only all 8,198 occurrence-level records in
`v2c5_expanded_development_dataset.json` are eligible. No new model-selection
probe is created. CFPB, BANKING77 test, CLINC test, consumed V2-C3 challenge and
external lockbox evidence, V2-C4 probe/augmentation/postmortem/final evidence,
and the V2-C5 final holdout are prohibited. Future execution must use
`StratifiedGroupKFold(n_splits=5, shuffle=true, random_state=20260930)` and group
exclusively by frozen `group_id`. It must freeze fold assignments before any
candidate scoring, prove each record occurs in exactly one validation fold, and
fail rather than fall back to non-group-aware CV.

The primary metric is mean fold macro-F1. Required reporting also includes
pooled OOF macro-F1, accuracy, balanced accuracy, per-intent metrics, the frozen
16-label confusion matrix, worst-fold macro-F1, separate new-seven and
historical-nine macro-F1, local fit and prediction timing, and resulting
artifact size. Selection eligibility requires all three pooled OOF gates:

- protected-write FPR `<= 0.01`, measured among non-protected gold examples
  predicted as any protected-write intent;
- exact protected-write recall `>= 0.80`, requiring the predicted protected
  intent to equal the gold protected intent;
- `unsupported_or_uncertain` recall `>= 0.80`.

Gate-passing candidates are ordered by highest mean fold macro-F1, highest
worst-fold macro-F1, lowest protected-write FPR, highest unsupported recall,
lowest local prediction latency, then lexical candidate ID. Numerical ties use
an absolute `1e-12` tolerance rather than raw floating-point equality. If no
candidate passes, no selection is made, gates cannot be weakened, the final
holdout remains sealed, and Step 23 remains prohibited.

The architecture is a bounded comparison of sparse lexical and existing local
semantic representations with linear classifiers. It is inexpensive,
reproducible, interpretable, and suitable for the development-set size, but its
narrow grid can miss a global optimum, TF-IDF may generalize less semantically,
BGE costs more CPU, and linear boundaries remain limited. The decision is
highly reversible because it creates no runtime authority and preserves the
sealed final evidence.

## V2-C5 Step 22B1 development-only model-selection runner

`scripts/run_v2c5_model_selection.py` implemented and completed the Step 22A
development-only experiment without changing it. All 27 candidates completed,
13 passed all three mandatory gates, and the frozen ordering selected
`BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=none`: BGE-small passage embeddings
plus `LinearSVC(C=4.0, class_weight=None)`. That candidate passed the
protected-write false-positive, exact protected-write recall, and
`unsupported_or_uncertain` recall gates. These are development-only selection
results, not final acceptance evidence. The runner's modes are:

- `--preflight`: verify all frozen source hashes, the exact expanded taxonomy
  and development population, and the final-holdout manifest's frozen,
  unevaluated, Step-22-permitted status; perform no model work and write
  nothing;
- `--prepare-folds`: create deterministic, text-free tracked fold and manifest
  artifacts using the frozen group-aware five-fold settings, refusing to
  replace incompatible artifacts;
- `--check-folds`: reproduce and validate fold bytes, lineage, complete
  validation assignment, and zero group leakage without writing;
- `--run`: require the already-frozen folds, build or reuse the ignored local
  development-only BGE cache, evaluate the complete 27-candidate matrix, and
  write text-free result and manifest artifacts only after all candidates
  complete;
- `--check-results`: validate source/fold lineage, the complete matrix, metric
  and gate schemas, and the frozen selection rule without fitting or writing.

For every lexical candidate and fold, word, character, or combined TF-IDF is
fitted only on training-fold text before training and validation transforms are
made. It is never fitted globally. BGE is a frozen pretrained feature extractor
and may be computed once over development text; the ignored
`local/v2c5_model_selection_bge_cache.npz` and its manifest pin the development
and contract hashes, representation definition, ordered IDs and text hashes,
row count, 384 dimensions, L2 normalization, and cache bytes. No raw text is
stored in cache metadata or tracked Step 22 artifacts.

Run locally in this order:

```bash
sentinelvoice_env/bin/python scripts/run_v2c5_model_selection.py --preflight
sentinelvoice_env/bin/python scripts/run_v2c5_model_selection.py --prepare-folds
sentinelvoice_env/bin/python scripts/run_v2c5_model_selection.py --check-folds
sentinelvoice_env/bin/python scripts/run_v2c5_model_selection.py --run
sentinelvoice_env/bin/python scripts/run_v2c5_model_selection.py --check-results
```

All five modes reject the sealed V2-C5 final-holdout dataset as an input; only
its frozen manifest may be read for governance. Step 22B1 did not tune
thresholds, alter runtime behavior, or establish final acceptance. The
successful frozen selection permits full-development preparation before Step
23, while the holdout remains sealed.

## V2-C5 Step 22C selected-model preparation

`scripts/prepare_v2c5_selected_model.py` fits no new choice: it validates and
uses only the Step 22B1-selected BGE-small plus unweighted LinearSVC at
`C=4.0`. `--preflight` validates the source hashes, successful selection,
selected configuration, all 8,198 frozen development rows, and the final
holdout manifest's governance without embedding, fitting, inference, or
writes. `--fit` obtains the exact 384-dimensional, L2-normalized FastEmbed
passage representation, fits one LinearSVC with every frozen parameter on all
8,198 rows, and writes an ignored trusted-local classifier plus a tracked
text-free manifest. It performs no CV, candidate comparison, threshold tuning,
or holdout inference. `--check` validates the serialized classifier, manifest,
cache bytes and lineage, source hashes, exact 16 labels, and feature dimensions
without training, inference, or writes.

The ignored cache is reused only when the exact development dataset, ordered
example IDs, ordered text hashes, representation configuration, row count,
384-dimensional shape, normalization state, and cache hashes remain valid; an
absent cache may be created with those same development-only semantics, while
a stale or incompatible cache is rejected. The generated classifier path is
`local/v2c5_selected_classifier.joblib`, and its tracked metadata path is
`v2c5_selected_model.manifest.json`. The classifier remains trusted-local,
runtime-ineligible, and carries no raw development text.

Refitting the already-selected configuration on all development data avoids
choosing an arbitrary fold model and maximizes training signal. It was chosen
over a non-predeclared ensemble, post-selection retuning, or any use of holdout
data. The tradeoff is that this full-data fit has no unbiased performance
metric of its own and may differ slightly from individual fold fits. The final
holdout stays sealed; only Step 23 may perform its once-only evaluation. Step
22C does not change runtime behavior or claim final acceptance.

After the user has validated the implementation, the local workflow is:

```bash
sentinelvoice_env/bin/python scripts/prepare_v2c5_selected_model.py --preflight
sentinelvoice_env/bin/python scripts/prepare_v2c5_selected_model.py --fit
sentinelvoice_env/bin/python scripts/prepare_v2c5_selected_model.py --check
```

## V2-C5 Step 23A once-only final-evaluation contract

`v2c5_final_evaluation_contract.json` freezes the full final-evaluation
protocol before the sealed holdout is touched. It hash-binds the Step 22C
selected-model manifest, expected trusted-local classifier artifact, Step 22B1
results, development dataset, taxonomy, and final-holdout manifest. The
holdout's declared dataset hash is copied from that manifest without opening,
reading, searching, copying, or hashing `v2c5_final_holdout.json`. Step 23A
implements no evaluator, embeddings, inference, or result generation.

The contract preserves the manifest-frozen balanced population: 640
independently authored examples, 16 intents, 40 examples per intent, 160
protected-write examples, 480 non-protected examples, and zero normalized
development/historical overlap. The future results must report accuracy,
balanced accuracy, macro-F1, per-intent precision/recall/F1/support, the exact
frozen-order 16-by-16 confusion matrix, historical-nine macro-F1, and
new-seven macro-F1. No additional aggregate acceptance threshold exists;
macro-F1 remains a reported final quality metric.

Final acceptance requires every unchanged safety gate:

- protected-write false-positive rate `<= 0.01`, with all 480 gold
  non-protected examples as the denominator and predictions into any protected
  intent as the numerator;
- exact protected-write recall `>= 0.80`, with all 160 protected examples as
  the denominator and exact protected-intent matches as the numerator;
- `unsupported_or_uncertain` recall `>= 0.80`, over its 40 gold examples.

The future evaluator must atomically write and durably flush
`v2c5_final_evaluation_state.json` from `not_started` to `started` before
opening the holdout. It may proceed automatically only from `not_started` and
must refuse automatic execution from `started`, `failed_after_access`, or
`completed`. Failure after access consumes the clean attempt. Any later,
explicitly authorized recovery must be labeled compromised/secondary and may
not replace the original outcome.

Exactly one final inference/evaluation is permitted. Holdout results cannot be
used for threshold tuning, retraining, label or taxonomy changes, candidate
reselection, a second clean run, or development/training augmentation. A gate
failure must be reported without weakening the gate, and Step 24 cannot proceed
as though it passed. Evaluation completion, mandatory-gate passage, final model
acceptance, and runtime eligibility are separate states. Passing Step 23 does
not change runtime behavior; controlled integration remains separately
governed.

Step 23A predeclares, but does not create:

- `v2c5_final_evaluation_results.json`;
- `v2c5_final_evaluation_results.manifest.json`;
- `v2c5_final_evaluation_state.json`.

Future results are aggregate-only and contain no raw holdout text. The next
required phase is `v2c5_final_evaluation_runner`.

## V2-C5 Step 23B final-evaluation runner

`scripts/run_v2c5_final_evaluation.py` implements the Step 23A contract with
four mutually exclusive modes:

- `--preflight` validates every frozen source, the trusted-local classifier's
  exact hash/configuration/16 classes/384-dimensional input, the holdout
  manifest declaration, absent results, and absent or `not_started` state. It
  performs no embeddings, inference, writes, or holdout-dataset access.
- `--initialize-state` creates the deterministic, text-free
  `v2c5_final_evaluation_state.json` in `not_started`, binding the Step 23A
  contract, selected-model manifest, classifier artifact, declared holdout
  dataset, holdout manifest, taxonomy, and evaluator hashes. It refuses every
  existing or incompatible state and must be run and committed before Step
  23C.
- `--evaluate` is implemented but reserved for Step 23C. It has no force,
  reset, or retry path.
- `--check-results` requires `completed`, validates canonical result and
  manifest bytes, hashes, lineage, metrics, safety counts/gates, decisions, and
  governance without inference, writes, or reopening the holdout.

The evaluator validates every precondition while the holdout remains sealed,
then atomically writes, flushes, and directory-fsyncs the `started` transition
before calling the holdout loader. It immediately verifies the declared
dataset SHA, the exact 640/16/40-per-intent population, 160 protected and 480
non-protected examples, unique IDs, taxonomy labels, and taxonomy-derived
risks. It performs one non-persistent BGE-small passage-embedding pass,
requires a `640 x 384` finite L2-normalized matrix, and invokes the frozen
classifier's prediction method exactly once. It never fits, refits, performs
CV, tunes thresholds, reselects a model, or caches final-holdout embeddings.

Results are aggregate-only: the frozen metrics and safety gates, prediction
count, and a SHA256 of the ordered `(example_id, gold_label, predicted_label)`
sequence are stored, but no individual text or prediction rows are persisted.
The results and manifest are durably written before state becomes `completed`.
Any failure after `started` produces text-free `failed_after_access` metadata
and consumes the attempt. Automatic evaluation is permitted only from
`not_started`; every other state refuses execution.

Step 23B does not run `--evaluate` and does not consume the final holdout. Its
tests inject a synthetic 640-record holdout, fake normalized embeddings, and a
synthetic classifier. The runner remains reversible before Step 23C; actual
holdout observation is not reversible. This durable one-way state is necessary
because routing predictions may affect protected banking actions and the final
safety evidence must be auditable rather than silently repeatable.

Before Step 23C, the frozen Step 23B state was prepared with:

```bash
sentinelvoice_env/bin/python scripts/run_v2c5_final_evaluation.py --preflight
sentinelvoice_env/bin/python scripts/run_v2c5_final_evaluation.py --initialize-state
```

That initialization workflow is now historical: the state is `completed`, the
once-only Step 23C evaluation has consumed the holdout, and neither state
initialization nor evaluation may be repeated as a clean V2-C5 run.

## V2-C5 Step 23C final evaluation and closeout

The once-only V2-C5 final evaluation completed successfully as an evaluation
procedure; its safety outcome was a failure. The frozen candidate was
`BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=none`. The final holdout contained
640 records, balanced at exactly 40 records for each of the 16 intents.

Final metrics:

| Metric | Value |
| --- | ---: |
| Accuracy | `0.5125` |
| Balanced accuracy | `0.5125` |
| Macro-F1 | `0.5632279946795209` |
| Historical-nine-label macro-F1 | `0.6945055116269591` |
| New-seven-intent macro-F1 | `0.45253940739110227` |

Mandatory safety gates:

| Gate | Observed | Frozen threshold | Result |
| --- | ---: | ---: | --- |
| Protected-write false-positive rate | `0.014583333333333334` | `<= 0.01` | **FAILED** |
| Exact protected-write recall | `0.4` | `>= 0.80` | **FAILED** |
| `unsupported_or_uncertain` recall | `0.85` | `>= 0.80` | **PASSED** |

The final governance outcome is:

- `all_mandatory_safety_gates_pass=false`;
- `final_model_acceptance_claimed=false`;
- `runtime_eligible=false`;
- `runtime_behavior_changed=false`.

V2-C5 final evaluation is complete, but the classifier is rejected for runtime
integration. Step 24 runtime integration is blocked for this candidate, and
the failed thresholds must not be weakened. The final-evaluation state is
`completed`, so the holdout is consumed and **MUST NOT** be reused as a clean
final holdout. The evaluation must not be rerun and represented as a clean
V2-C5 result.

Development-only OOF macro-F1 was approximately `0.8850`, compared with final
macro-F1 of approximately `0.5632`. This is a substantial
development-to-final generalization gap. Evidence-supported observations are
particularly weak final performance for several expanded and protected
intents, together with heavy over-selection of `unsupported_or_uncertain` on
the final evaluation. These observations do not establish an exact root
cause. Further diagnosis must use development-side evidence or newly collected
diagnostic data and must not tune against examples from this consumed holdout.

The next phase is **V2-C6 remediation / generalization diagnosis**. V2-C6 must:

- not reuse the consumed V2-C5 holdout as its new final test;
- not tune directly against the consumed holdout examples;
- freeze a new evaluation contract and create a fresh, independently authored
  final holdout before final V2-C6 evaluation; and
- keep runtime integration blocked until a future candidate passes its
  predeclared safety gates.

## V2-C6 Step 29A generalization-diagnosis contract

`v2c6_generalization_diagnosis_contract.json` freezes the diagnosis boundary
before any V2-C6 analysis or remediation. It hash-pins the 8,198-record V2-C5
expanded development dataset and manifest, existing model-selection results
and manifest, aggregate final-evaluation results and manifest, completed
final-evaluation state, and frozen taxonomy and manifest. The only permitted
role for V2-C5 final results is committed aggregate historical context:

- development pooled OOF macro-F1: `0.8850476526181243`;
- final macro-F1: `0.5632279946795209`;
- final new-seven-intent macro-F1: `0.45253940739110227`;
- final exact protected-write recall: `0.4`;
- final protected-write false-positive rate: `0.014583333333333334`; and
- final `unsupported_or_uncertain` recall: `0.85`.

The raw consumed `v2c5_final_holdout.json` is explicitly prohibited. Neither
Step 29A nor Step 29B may open or inspect individual records, add them to
development, paraphrase them, train against their labels or examples, tune
thresholds or choose models from their results, weaken safety gates, revise
the V2-C5 conclusion, or treat the holdout as reusable final evidence.

The future Step 29B diagnosis must answer nine frozen development-side
questions:

1. Development source composition: examples and unique groups per intent,
   examples-per-group distributions, available source/provenance distributions,
   and represented synthetic versus external/source categories.
2. Group structure: group sizes, multi-intent groups, within-group intent
   concentration, and whether the largest groups dominate development evidence.
3. Duplicate structure: exact and normalized duplicate rates, within-intent
   concentration, and cross-intent conflicts. No new semantic-similarity
   threshold is authorized.
4. Lexical diversity: text lengths, vocabulary and unique-token statistics,
   type-token and hapax statistics, and repeated unigram/bigram/trigram
   concentration by intent.
5. Class imbalance: raw and relative intent frequencies,
   `unsupported_or_uncertain` dominance, and maximum/minimum/median imbalance
   ratios.
6. Development OOF behavior: the selected candidate's existing per-intent
   metrics, confusion matrix, unsupported prediction rate, protected-write
   confusion patterns, and available fold variance. No inference may be rerun.
7. Authoring/provenance concentration: explicit authoring, template, source,
   and provenance family concentration where metadata exists; missing metadata
   must be reported rather than inferred.
8. Intent boundaries: development-side confusions and lexical overlap among
   `cancel_transfer`, `close_account`, `create_dispute`, `freeze_card`,
   `account_blocked`, `transfer_failed_or_declined`, `transfer_pending`, and
   `unsupported_or_uncertain`.
9. Effective sample size: raw examples, unique groups, and unique normalized
   texts per intent. In particular, 4,769 unsupported rows must not be treated
   automatically as 4,769 independent linguistic patterns.

The evidence language is also frozen. An **observation** is a directly computed
statistic; a **hypothesis** is an explicitly unconfirmed explanation; a
**supported diagnosis** requires measurable, reproducible development-side
evidence and stated limitations; and a **causal claim** is not authorized from
Step 29B observational evidence alone. Class imbalance, a final performance
drop, or unsupported overprediction is individually insufficient for a causal
or root-cause claim.

Step 29B may recommend training-diversity improvements, rebalancing,
group/source independence, intent-boundary improvements, hard negatives,
unsupported-class redesign, a different representation/classifier family, or
taxonomy revision. It must not implement any remedy. Remediation requires a
separately frozen V2-C6 experiment.

The future outputs are aggregate/statistical and contain no raw final-holdout
text:

- `v2c6_generalization_diagnosis.json`;
- `v2c6_generalization_diagnosis.manifest.json`.

Before any V2-C6 final evaluation, the project must freeze a new evaluation
contract and fresh independently authored holdout. No V2-C5 final-holdout
example may be reused, and V2-C5 aggregate results remain historical evidence
only. Runtime integration stays blocked until a future candidate passes its
predeclared safety gates.

Step 29A creates no runner and performs no diagnosis, model training, model
selection, remediation, taxonomy change, holdout access, or runtime change.
The contract records `contract_frozen=true`, `diagnosis_performed=false`,
`model_training_performed=false`, `model_selection_performed=false`,
`final_holdout_accessed=false`, `runtime_behavior_changed=false`, and
`next_required=v2c6_generalization_diagnosis`.

## V2-C6 Step 29B deterministic generalization diagnosis

`scripts/run_v2c6_generalization_diagnosis.py` implements the frozen Step 29A
contract as a standard-library-only statistical workflow. It consumes the
frozen 8,198-record expanded development dataset, frozen group/fold artifacts,
the selected candidate's already-generated OOF aggregates and fold metrics,
the frozen taxonomy/risk mapping, and only the aggregate V2-C5 final context
explicitly allowed by the contract. The consumed raw
`v2c5_final_holdout.json` is protected by an executable read guard and remains
prohibited on preflight, run, and result-checking paths.

The CLI has exactly three modes:

- `--preflight` verifies the contract and every frozen source binding, exact
  development and taxonomy counts, selected candidate, folds, aggregate final
  context, and output availability. It writes nothing.
- `--run` deterministically prepares
  `v2c6_generalization_diagnosis.json` and its manifest. The completed local
  execution froze both artifacts, and the mode refuses to overwrite them.
- `--check-results` requires both artifacts and validates canonical
  serialization, deterministic recomputation, result and runner hashes, source
  lineage, counts, selected candidate, governance, and
  `next_required=v2c6_remediation_design`. It writes nothing.

The diagnosis reports exact source/intent and group distributions; singleton,
multi-record, multi-intent, and largest-group concentration; exact and NFKC
lowercase/whitespace-normalized duplicates; deterministic regex-token lexical
statistics and unigram/bigram/trigram concentration; class/risk imbalance;
existing selected-candidate OOF metrics, confusion-derived unsupported and
protected-write behavior, and fold variance; explicit provenance/authoring
concentration; the eight frozen focus-intent confusion and lexical-overlap
boundaries; and raw-row, unique-group, exact-text, and normalized-text
evidence-independence proxies. Missing provenance fields are emitted as
`available=false` with `reason="metadata_not_present"`; provenance is never
inferred from utterance wording or native labels.

This step does not train or fit a model, call classifier prediction, generate
embeddings or examples, perform model selection or evaluation, tune thresholds,
change labels or taxonomy, implement remediation, or alter runtime behavior.
Its historical V2-C5 section is aggregate-only and contains no individual
consumed-holdout record. Observations cite metric paths; hypotheses are marked
unproven; supported diagnoses require evidence, confidence, and limitations;
and `causal_claims` remains empty because this observational diagnosis cannot
establish a root cause. Recommendations use only contract-frozen category IDs
and describe what a separate experiment must freeze before implementation.

The architecture choice is deterministic analysis of already-frozen
development and OOF evidence. It is reproducible, inexpensive, reversible, and
measures the corpus before remediation. Immediate synthetic-data generation,
class rebalancing, representation/classifier changes, classifier tuning, and
inspection of consumed final examples were rejected for this step because they
would intervene before diagnosis, provide weak evidence, or create leakage
risk. Development statistics cannot fully characterize final-distribution
shift, lexical/group associations do not prove causality, and absent provenance
can limit authoring-family findings.

The completed diagnosis recorded a development-to-final macro-F1 gap of about
`0.32182`, `4,769 / 8,198` development records labeled
`unsupported_or_uncertain`, narrow source-revision coverage for several
expanded and protected intents, 450 non-unsupported OOF records predicted as
unsupported, and 29 protected-write OOF records predicted as unsupported. It
also recorded 7,863 unique groups, about 99% singleton groups, zero exact
duplicate records, and only two normalized duplicate records. These findings
support data/evaluation remediation but do not prove an exact root cause;
ordinary duplicate leakage is not the primary remediation target.

## V2-C6 Step 29C remediation-design contract

`v2c6_remediation_design_contract.json` hash-pins the completed Step 29B
diagnosis and manifest, Step 29A contract, expanded-development declaration and
manifest, model-selection aggregate evidence, aggregate final context, and the
frozen taxonomy. The consumed `v2c5_final_holdout.json` remains an explicitly
prohibited raw input and historical aggregate evidence only.

The frozen remediation order is:

1. improve independent source/authoring diversity for underrepresented and
   source-concentrated intents;
2. improve supported-versus-unsupported boundaries with independently authored
   hard negatives;
3. add source-family holdout development evaluation alongside group-aware CV;
4. improve specific intent boundaries already visible in development OOF
   evidence; and
5. reconsider representation/classifier families only after data and
   evaluation remediation, if evidence still warrants it.

The architecture choice is data-first remediation with independent source
diversity, boundary hard negatives, and source-aware development evaluation.
Step 29B provides stronger measured support for improving data and evaluation
evidence than for immediately changing models or taxonomy. Switching embedding
families, tuning LinearSVC/class weights, weakening unsupported handling,
inspecting the consumed holdout, or collapsing taxonomy is therefore deferred
or rejected for Step 29C. The tradeoffs are added authoring/review effort,
reduced training evidence during source holdout, and more metadata complexity,
with no guarantee of future acceptance. The design is highly reversible
because it changes no model, data, taxonomy, thresholds, or runtime behavior.

The primary targets are `account_blocked`, `cancel_transfer`, `close_account`,
`create_dispute`, `freeze_card`, `transfer_failed_or_declined`,
`transfer_pending`, and `unsupported_or_uncertain`. Future remediation data
must carry explicit `source_family_id`, source revision, intended intent, risk,
authoring batch, and `group_id`. Each primary intent requires at least three
independent source families; the future builder must fail or explicitly report
unmet coverage. No primary intent may remain wholly dependent on one source
revision, and raw example count is not a substitute for unique group and source
coverage.

New examples must be genuinely independently authored from human-reviewed
boundary specifications. They may not paraphrase V2-C5 development examples,
existing hard negatives, or consumed V2-C5 final examples. Exact and NFKC
normalized duplicate checks are mandatory, with zero exact or normalized
cross-intent conflicts. Required hard-negative boundaries include every primary
supported intent against `unsupported_or_uncertain`, plus
`cancel_transfer`/`transfer_pending`,
`transfer_failed_or_declined`/`transfer_pending`, and
`account_blocked`/`transfer_failed_or_declined`.

`unsupported_or_uncertain` is retained. A future builder may attach
`unsupported_subtype` development/evaluation metadata for truly unsupported
banking requests, ambiguous or insufficient-information requests, adjacent
unsupported intents, supported-intent hard negatives, and off-domain/noise.
This metadata does not change the 16-intent runtime taxonomy.

Future model selection must preserve group-aware stratified CV and add a
complementary development holdout of entire source/authoring families where
possible. If primary intents lack enough families, source diversity must be
improved first. Source-holdout evidence remains reusable development evidence
and must never be represented as the final V2-C6 holdout.

A fresh V2-C6 final holdout requires a new independent authoring session,
isolation from remediation data, freezing before final model-selection
decisions that depend on evaluation, and once-only evaluation under the same or
stronger governance. No V2-C5 final example or paraphrase may be reused.

Safety thresholds remain at least as strict as V2-C5: protected-write FPR
`<= 0.01`, exact protected-write recall `>= 0.80`, and
`unsupported_or_uncertain` recall `>= 0.80`. The protected-write intents remain
`cancel_transfer`, `close_account`, `create_dispute`, and `freeze_card`; the
classifier still cannot authorize protected actions.

Step 29C changes experiment design only. It authors no examples, changes no
development data or taxonomy, generates no embeddings, performs no training or
model selection, and changes no runtime behavior. The staged continuation is
29D dataset-contract freeze, 29E diversified-data authoring/build, 29F dataset
validation/freeze, 29G source-aware selection-contract freeze, 29H development
selection, 29I selected-model fit, 29J fresh-holdout contract, 29K independent
holdout authoring/freeze, 29L final-evaluation contract, 29M once-only final
evaluation, and 29N runtime/shadow integration only if every predeclared gate
passes. The immediate next requirement is
`v2c6_remediation_dataset_contract`.

## V2-C6 Step 29D remediation-dataset contract

`v2c6_remediation_dataset_contract.json` freezes the construction and review
rules for a bounded, independently authored remediation dataset and hash-pins
the Step 29C design plus its diagnosis and taxonomy lineage. It creates no raw
examples. The primary intents are exactly `account_blocked`, `cancel_transfer`,
`close_account`, `create_dispute`, `freeze_card`,
`transfer_failed_or_declined`, `transfer_pending`, and
`unsupported_or_uncertain`. Other historical intents may receive only limited
boundary-balancing records required by a frozen hard-negative pair or
source-aware balancing rule.

Every primary intent requires at least three explicitly identified,
materially independent source families. The same process with different random
seeds, shuffled examples, paraphrases, punctuation changes, superficial edits,
or duplicate template expansion does not satisfy independence. New evidence
must originate from semantic definitions and reviewed boundary specifications,
not from paraphrasing or minimally editing development records or from any copy,
rewrite, translation, lexical substitution, or style variant of the consumed
V2-C5 final holdout.

Each future remediation record requires `record_id`, `text`, `intent`,
`risk_level`, `group_id`, `source_family_id`,
`source_family_independence_basis`, `source_revision`, `authoring_batch_id`,
`authoring_method`, `boundary_target`,
`is_hard_negative`, and `review_status`. Unsupported records additionally
require `unsupported_subtype`: `truly_unsupported_banking_request`,
`ambiguous_or_insufficient_information`, `adjacent_but_unsupported_intent`,
`supported_intent_hard_negative`, or `off_domain_or_noise`. These subtypes are
development/evaluation metadata only and do not create runtime intents.

The planning target is 90 records for each of the seven non-unsupported primary
intents, at least 30 per independent family across at least three families, plus
180 targeted unsupported records spanning all five subtypes with meaningful
supported-intent hard-negative coverage: approximately 810 records total. It
is a bounded starting plan, not blind class equalization or permission for
unbounded generic generation. Frozen hard-negative coverage includes each
supported primary intent against unsupported and the three additional pairs
`cancel_transfer`/`transfer_pending`,
`transfer_failed_or_declined`/`transfer_pending`, and
`account_blocked`/`transfer_failed_or_declined`. Both sides are independently
authored where appropriate; one-keyword substitutions are prohibited.

Each semantic scenario receives one auditable `group_id`; variants of a
scenario share that group and cannot inflate independence. Mandatory checks
require zero exact or normalized duplicates against the immutable V2-C5
development corpus and the new examples, and zero normalized cross-intent
conflicts. Normalization is Unicode NFKC, lowercase, trim, and collapsed
whitespace. No embedding near-duplicate rule is authorized. Diversity reports
cover normalized-text and group uniqueness, token diversity, inexpensive
repeated n-gram concentration, and source concentration by intent/family;
dominant wording from one family requires review rather than an arbitrary
lexical cutoff.

The original Step 29D contract required all new records to undergo
deterministic schema checks, automated duplicate and boundary checks, and human
review. Review states were `unreviewed`, `approved`, `rejected`, and
`needs_revision`; only `approved` records could be included. Protected-write
records, hard negatives, ambiguous unsupported records, and relabelled records
were mandatory review categories. Human, controlled LLM-assisted, and
materially independent deterministic scenario authoring were allowed, but no
LLM output was auto-accepted. Current classifier predictions could not steer
initial authoring or iterative rewrites. The Step 29E2 amendment below records
the later governance correction without rewriting this original history.

The future dataset is the immutable V2-C5 expanded development corpus plus
approved V2-C6 records with new IDs and preserved lineage. Before Step 29G it
must support group-aware stratified CV and a feasible source-family holdout
without dropping any primary intent, and report family distribution per intent.
Remediation records are development data only: they cannot be a future final
holdout or be copied into the separately authored holdout governed by Steps
29J/29K.

The exact four future tracked outputs are
`v2c6_remediation_examples.json`,
`v2c6_remediation_examples.manifest.json`,
`v2c6_remediated_development_dataset.json`, and
`v2c6_remediated_development_dataset.manifest.json`; none exists as a result of
Step 29D. An authoring workfile, if needed later, is local and ignored. The
16-intent taxonomy, protected-write set, and minimum safety gates remain
unchanged. Step 29D itself performed no authoring, build, embedding, model work,
evaluation, or runtime change. At that original freeze, its exact next phase
was `v2c6_remediation_authoring_and_build`; the Step 29E2 amendment below
records the later lifecycle state.

## V2-C6 Step 29E1 remediation builder preparation

`scripts/build_v2c6_remediated_development_dataset.py` implements deterministic
validation and construction machinery without authoring data. Following the
Step 29D lineage, its future local input is the ignored file
`local/v2c6_remediation_authoring_workfile.json`, encoded as a
`v2c6-remediation-authoring-input.v1` object with phase `V2-C6 Step 29E2` and a
`records` array. The later Step 29E2 governance amendment also requires a
top-level `review_provenance` object. Every record must provide all Step 29D
provenance, review, boundary, family, group, intent, risk, and authoring fields;
unsupported records also require the frozen metadata-only subtype. Missing
metadata is rejected, never inferred from wording.

The mutually exclusive CLI modes are:

```bash
sentinelvoice_env/bin/python scripts/build_v2c6_remediated_development_dataset.py --preflight
sentinelvoice_env/bin/python scripts/build_v2c6_remediated_development_dataset.py --build
sentinelvoice_env/bin/python scripts/build_v2c6_remediated_development_dataset.py --check-results
```

`--preflight` is read-only and tolerates an absent local input; when one exists,
it performs structural validation and reports mandatory-gate and planning-target
coverage. `--build` requires all mandatory gates and create-once output paths.
`--check-results` is read-only and compares the existing four outputs with a
fresh deterministic reconstruction. None of these modes performs model work.

Structural checks enforce unique non-colliding record IDs, nonempty groups,
the exact eight primary intents and four protected intents, frozen risk values,
the exact review and authoring-method allowlists, explicit source-family
independence bases, and consistent provenance descriptors. A group cannot cross
an intent or source-family boundary. The builder reports record, unique-group,
singleton-group, multi-record-group, and records-per-group statistics, but
explicitly cannot prove that authors did not artificially split semantic
scenarios; reviewer provenance and semantic review remain required.

Only `approved` records are copied into tracked outputs. Every other review
state is excluded and counted, and unreviewed material is never promoted
implicitly. Each primary intent must have at least three explicit source
families before build. Per intent it reports total records, family counts and
largest share, unique groups, and unique normalized texts. The 90-per-supported
intent, 30-per-family, 180-unsupported, and 810-total numbers are reported as
Step 29D planning targets; the mandatory three-family minimum is separate and
cannot be relaxed. Code validates provenance consistency and obvious family-ID
aliasing, while semantic independence remains unproven by structural checks.

All five unsupported subtypes must occur among approved records, but their
runtime label remains `unsupported_or_uncertain`. An approved hard-negative
record names its opposing intent through `boundary_target`; only the exact ten
Step 29D pairs are accepted. Reports include each side's approved count and
source-family count, and build requires both sides of every pair.

Approved remediation text is checked against itself and the frozen V2-C5
expanded development corpus for exact and NFKC/lowercase/trimmed/
whitespace-collapsed duplicates. Normalized cross-intent conflicts are also
rejected. Diagnostics retain hashes and record/example IDs plus intent, group,
and family metadata, not raw offending text. Embedding or semantic-neighbor
checks are deliberately absent.

The future reviewed artifact contains approved V2-C6 records sorted by
`record_id`. The combined dataset is the immutable 8,198-record V2-C5 expanded
development sequence followed by those sorted V2-C6 records; no frozen record
or source object is modified. Manifests bind the Step 29D/29C/29B lineage,
authoring-input hash, builder, taxonomy, frozen development data, output hashes,
counts, review status, subtype and hard-negative coverage, duplicate gates,
group statistics, and source-aware readiness. Readiness covers three-family
coverage, whole-family holdout feasibility without losing a primary label,
group-aware CV, duplicate gates, and completed review. The next phase after an
eventual build is still Step 29F validation/freeze, not Step 29G directly.

All reads use an explicit guard that rejects
`v2c5_final_holdout.json`. The implementation contains no classifier loading,
fit, prediction, embeddings, threshold tuning, selection, or final evaluation.
This deterministic builder is preferable to manual concatenation, direct
generation into the development artifact, model-directed authoring, or
reviewer-memory-only provenance because it is reproducible, auditable, and
isolates authoring from model feedback. Its tradeoffs are added code/metadata,
strict rejection of deficient records, and the unavoidable inability of
structural checks to prove semantic independence.

At the end of Step 29E1, Step 29D remained frozen and only builder/review
machinery had been prepared: Step 29E2 authoring/build had not run, no
remediation examples were claimed, and no tracked remediation or
remediated-development artifact existed from that step.

## V2-C6 Step 29E2 remediation-review governance amendment

Step 29E2 subsequently authored 810 records across three independent source
families and ran deterministic duplicate, provenance, and boundary validation.
A separate AI-assisted semantic review approved all 810 records, with zero
rejected and zero `needs_revision`. That process was not human review and does
not support any claim of independent human annotation. Before the remediation
dataset or its artifacts were built or frozen, the Step 29D review governance
was amended from universal human review to universal semantic review with
explicit reviewer provenance.

Allowed review methods are `human_review` and `ai_assisted_review`. `approved`
means semantically approved by the recorded method and does not imply human
approval. Authoring input and future artifacts/manifests record review method,
reviewer type, reviewed and per-status counts, human- and AI-assisted-review
counts, plus required and completed human-adjudication counts. Human
adjudication is mandatory for rejected or `needs_revision` records, reviewer
disagreement, low-confidence review, unresolved taxonomy ambiguity, provenance
inconsistency, or unresolved protected-write ambiguity. With no such trigger,
high-confidence AI-assisted approval can satisfy the review gate.

AI review is not human review. Model-generated and AI-reviewed data retains
additional independence limitations, and deterministic validation cannot prove
semantic correctness or semantic independence. Reviewer provenance and
semantic review therefore remain mandatory. This amendment occurred before the
Step 29E dataset build and before the separate Step 29F validation/freeze. The
exact next requirement is `v2c6_remediation_build`; build does not include or
rename Step 29F. The amendment preceded any final model evaluation, accessed no
final holdout, and changes no record, label, taxonomy, risk mapping, volume,
source-family minimum, duplicate rule, hard-negative rule, safety gate, runtime
behavior, or fresh-final-holdout requirement.

## V2-C6 Step 29F remediated-development validation and freeze

`scripts/validate_and_freeze_v2c6_remediated_development_dataset.py` provides
the deterministic governance boundary between Step 29E data construction and
Step 29G model-selection contracting. It validates the exact 9,008-record
development artifact as 8,198 unchanged inherited V2-C5 records plus 810
approved remediation records. Dataset and subset hashes must reconcile with
the committed Step 29E manifests and their pinned contract, taxonomy, builder,
and inherited-development lineage.

Validation independently checks the frozen 16-intent taxonomy and risk mapping,
the exact protected-write set, unique record IDs, preserved inherited group
membership, isolated remediation groups, zero new exact or normalized overlap
with inherited development data, zero within-remediation duplicates or cross-
intent normalized conflicts, all ten hard-negative pairs, five balanced
unsupported subtypes, and exact remediation volumes. It also reconciles
AI-assisted review provenance and zero incomplete human adjudications without
claiming universal human review. All three source families, their per-intent
coverage, family-level provenance, and every authoring batch remain auditable;
whole-family holdout and group-aware CV must remain feasible.

Group isolation does not imply label purity. Frozen V2-C5 history contains a
small number of shared group IDs spanning multiple intent labels; Step 29F
accepts those groups only when the inherited 8,198-record prefix remains exact.
Future group-aware splitting must keep every shared group ID in one indivisible
split unit regardless of its labels. By contrast, each of the 810 new
remediation group IDs must be nonempty, unique, exactly preserved in the
combined dataset, and disjoint from every inherited group ID.

The CLI is deliberately explicit:

```bash
sentinelvoice_env/bin/python scripts/validate_and_freeze_v2c6_remediated_development_dataset.py --check
sentinelvoice_env/bin/python scripts/validate_and_freeze_v2c6_remediated_development_dataset.py --freeze
```

`--check` never writes. `--freeze` validates first and then create-once writes
`v2c6_remediated_development_dataset.freeze.json`; it never overwrites an
existing freeze. The text-free freeze records dataset/taxonomy hashes, counts,
source-family and batch summaries, review provenance, duplicate and boundary
evidence, parent lineage, limitations, and
`next_required=v2c6_source_aware_model_selection_contract`. No freeze is
claimed until the explicit write command succeeds.

AI-assisted semantic review is not independent human annotation, and
structural provenance checks do not prove semantic independence. Independent
generalization must be measured later with source-aware development evaluation
and ultimately a fresh untouched final holdout. Step 29F performs no model
selection, training, embedding generation, inference, threshold tuning,
evaluation, or runtime change, and it cannot access the consumed V2-C5 final
holdout.

## V2-C6 Step 29G source-aware model-selection contract

`v2c6_source_aware_model_selection_contract.json` freezes Step 29H before any
model work. It pins the Step 29F freeze artifact and its exact 9,008-record
dataset SHA, 8,198 + 810 lineage, frozen 16-label taxonomy, split seeds,
candidate definitions, metric definitions, eligibility gates, deterministic
selection order, required result artifacts, and failure behavior.

The protocol requires both five-fold `StratifiedGroupKFold` over all records
and three leave-one-source-family-out rounds over the remediation data. Every
`group_id`, including the one inherited mixed-intent group, stays atomic; label
purity is not required and groups may not be rewritten. Each family round
trains on the 8,198 inherited records plus 540 remediation records from the
other families and tests the 270 records from the completely unseen family.
Exact fold and round memberships and hashes must be persisted by Step 29H.
Ordinary within-development CV alone is insufficient because the measured
V2-C5 evidence showed a substantial gap between strong group-isolated OOF and
independently authored final language, while duplicate leakage was not the main
measured explanation and source/style concentration and unsupported-boundary
confusions remained important observations.

The exact six candidates are BGE-small/LinearSVC with `C` in `{1, 4}` and
class weighting in `{none, balanced}`, plus word-and-character TF-IDF/
LinearSVC at `C=1` with each of those two weighting choices. The BGE-small,
`C=4`, unweighted recipe is the explicit V2-C5 selected-recipe control. Every
component already exists in the repository. TF-IDF vocabulary/IDF and every
classifier are fit inside each training fold or round; fixed pretrained BGE
inference cannot use validation labels. No threshold, fallback, calibration,
or post-hoc rule tuning is permitted.

Protected recall >= 0.80, protected false-positive rate <= 0.01, and
unsupported recall >= 0.80 remain mandatory on pooled group CV, pooled
source-family predictions, and each held-out family. Required diagnostics also
cover supported/unsupported direction errors and the exact ten frozen hard-
negative pairs, without adding post-hoc gates. Ineligible candidates retain
explicit reasons. If none passes, the only valid result is
`NO_ACCEPTABLE_CANDIDATE`; gates cannot be weakened and a winner cannot be
forced.

Eligible candidates are ranked lexicographically, beginning with worst-family
primary-eight macro-F1, then mean-family and pooled source-family primary-eight
macro-F1, pooled 16-label group-CV macro-F1, boundary and safety metrics, the
frozen lower-cost complexity order, and candidate ID. Prioritizing the weakest
family favors source/style robustness over a higher average that masks one
poorly generalized family. This three-family development evidence is limited
and cannot replace a fresh untouched final holdout.

Step 29G created only the contract and performed no embedding, fitting,
inference, evaluation, candidate selection, threshold tuning, holdout access,
or runtime change. It authorized Step 29H,
`v2c6_source_aware_model_selection_execution`. Step 29H is now complete under
that unchanged contract; no candidate was eligible or selected.

## V2-C6 Step 29H deterministic source-aware execution

`scripts/run_v2c6_source_aware_model_selection.py` implements the frozen Step
29G contract without modifying its search space, metrics, gates, or ranking.
It requires the exact Step 29G contract SHA, frozen Step 29F artifact, and
9,008-record dataset SHA, plus the pinned manifest and taxonomy lineage. Every
candidate representation, classifier, fixed parameter, `C`, and class weight
is resolved from the contract; missing or inconsistent definitions fail
closed.

The read-only preflight command is:

```bash
sentinelvoice_env/bin/python scripts/run_v2c6_source_aware_model_selection.py --preflight
```

It deterministically generates the five `StratifiedGroupKFold` partitions and
three 8,738/270 leave-one-family-out rounds once, proves their group/source
isolation and complete 9,008/810 coverage, and builds a text-free audit with
exact validation/test IDs and train/test membership hashes. It does not load a
model, generate embeddings, fit a classifier, infer, select, or write files.

The create-once execution command was:

```bash
sentinelvoice_env/bin/python scripts/run_v2c6_source_aware_model_selection.py --run
```

Execution uses the same split object for all six candidates. Because Step 29G
requires train and validation/test BGE transforms to remain separate, the
ignored `local/v2c6_source_aware_model_selection_bge_cache.npz` stores a
distinct matrix for each partition side. Its companion manifest pins the
dataset and contract SHAs, split-audit SHA, model and embedding mode, 384
dimensions, L2 normalization, row memberships, and FastEmbed version. Those
fixed pretrained matrices may be reused across the four BGE candidates; no
development label influences them. The two lexical candidates construct a new
word/character `FeatureUnion` inside every fold and round, fit vocabulary and
IDF only on training text, and transform held-out text without fitting.

The results preserve one text-free OOF prediction for all 9,008 records and one
out-of-family prediction for all 810 remediation records for each candidate.
They include the frozen group/source metrics, per-family and pooled safety
checks, unsupported-direction rates, exact ten-pair hard-negative diagnostics,
explicit eligibility reasons, ranking inputs, and deterministic selection.
Infrastructure failures abort execution; a completed candidate that misses a
safety gate remains visible but ineligible.

All 15 required gate checks—three metrics across pooled group CV, pooled source
holdout, and each of the three families—must pass. Selection then follows the
frozen worst-family-first lexicographic order, exact complexity ranks, and
candidate-ID tie-break. Zero eligible candidates produces
`NO_ACCEPTABLE_CANDIDATE` with no forced winner. Because Step 29G freezes no
failure-path `next_required`, that case records `next_required=null` and the
contract ambiguity, and cannot authorize Step 29I.

The tracked create-once outputs are
`v2c6_source_aware_model_selection_results.json` and
`v2c6_source_aware_model_selection_results.manifest.json`. They record
`execution_status=COMPLETED`, `candidate_count=6`, six completed candidates,
`eligible_candidate_count=0`, and
`selection_status=NO_ACCEPTABLE_CANDIDATE`. The selected candidate is `null`,
`winner_forced=false`, `gates_weakened=false`, and `next_required=null`.
Because Step 29G declared no failure-path `next_required`, the result fails
closed without inventing one and does not authorize Step 29I.

The dominant measured mandatory-gate blocker was protected false-positive rate
under source-family holdout. All six candidates failed the pooled
source-family protected-FPR requirement of `<= 0.01`; several candidates also
failed unsupported-recall requirements. These are observed associations, not
an exact or exclusive root-cause claim.

The strongest near-miss diagnostic was
`WORD_CHAR_TFIDF_LINEAR_SVC__C=1.0__class_weight=none`, which was still
**ineligible**. Its pooled group-aware-CV macro-F1-16 was
`0.8708666548598263`, with protected recall `0.875`, protected FPR
`0.0077454718779790275`, and unsupported recall `0.9062436855930491`; it
therefore passed all three pooled group-CV safety gates. On pooled unseen-family
predictions, primary-eight macro-F1 was `0.8355663639651155`, protected recall
was `0.8444444444444444`, protected FPR was `0.03333333333333333`, and
unsupported recall was `0.85`. It failed the pooled protected-FPR gate, SF1
protected recall at `0.7583333333333333`, and the protected-FPR gate in every
individual family.

This diagnostic shows the value of the source-aware design despite the absence
of an acceptable model. Ordinary pooled group-aware CV would have made this
candidate appear safety-compliant, while leave-one-family-out evaluation moved
its measured protected FPR from approximately `0.00775` to `0.03333` and
exposed unsafe source/style generalization. This remains development evidence
only; the candidate is not a winner, selected, accepted, or production ready.

Governance records `development_model_selection_performed=true` and
`temporary_fold_and_round_classifier_fitting_performed=true`. Those temporary
evaluation fits are distinct from the blocked Step 29I full-development fit.
The result also records
`final_full_development_model_fitting_performed=false`,
`full_development_fitted_classifier_persisted=false`,
`final_holdout_accessed=false`, `final_holdout_evaluated=false`,
`threshold_tuning_performed=false`, `runtime_behavior_changed=false`,
`final_model_acceptance_claimed=false`, and `production_ready_claimed=false`.

The Step 29F dataset freeze and Step 29G selection contract remain unchanged.
No fresh V2-C6 final holdout has been authored or accessed, no runtime
classifier change is authorized, and Step 29I is blocked. Any further
remediation requires a separately defined next step and contract rather than
post-hoc weakening of gates or alteration of these candidate results. The
create-once Step 29H result must not be rerun and presented as a new clean
evaluation.

## V2-C6 Step 29H-A post-selection failure-analysis contract

`v2c6_post_selection_failure_analysis_contract.json` hash-pins the completed
Step 29H results and manifest, the Step 29G contract, the Step 29F freeze, and
the frozen 9,008-record development dataset. It requires the exact completed
`NO_ACCEPTABLE_CANDIDATE` state with all six candidates and zero eligible
candidates. Step 29I remains blocked, and
`next_required=v2c6_post_selection_failure_analysis_execution` authorizes only
read-only analysis of existing prediction evidence.

The execution must analyze all six candidates rather than only the unweighted
TF-IDF near-miss. For protected false positives it reports every actual
non-protected to predicted-protected pair for pooled group CV, pooled
source-family predictions, and SF1/SF2/SF3. The passing integer allowance is
derived as `floor(non_protected_denominator * 1 / 100)` from the unchanged
`<= 0.01` gate, and excess counts are measured above that allowance. Pair
recurrence is reported descriptively across candidates, representation
families, source families, and evaluation views without a new threshold.

Unsupported analysis separates unsupported-to-protected errors from
unsupported-to-other-supported errors. Protected false-negative analysis
counts only protected-to-non-protected predictions as aggregate protected
recall misses; a protected-to-different-protected prediction remains an exact
intent error but not a protected-recall failure. The contract also requires
per-family safety/boundary summaries, qualified group-CV versus source-family
deltas, three matched class-weight comparisons, two BGE regularization
comparisons, and the exact ten Step 29D hard-negative pairs aggregated across
candidates and families.

Tracked analysis outputs may contain record IDs and label/provenance metadata
but no utterance text. Optional detailed review may use ignored
`local/v2c6_post_selection_error_review.csv`, sourced only from the frozen
development dataset. That review is separate from tracked aggregates, accesses
no final holdout, and makes the inspected records remediation-informed
development evidence rather than untouched validation.

The three Step 29H source-family rounds are already consumed development
model-selection evidence. Inspecting their errors additionally consumes them
as diagnostic evidence. After remediation, SF1/SF2/SF3 may support regression
diagnostics or describe previously observed behavior, but they cannot be
presented as a fresh independent source-generalization estimate. Future clean
evidence requires a new independently authored source family or source-family
evaluation set, or ultimately the fresh untouched final holdout; this step
creates none of them.

The chosen architecture is evidence-first failure analysis before further
data or model changes. It prevents blind remediation while reusing predictions
already generated under the frozen contract. Immediately adding examples or
model families, weakening gates, or tuning thresholds would be post-hoc before
the observed boundaries are characterized. The tradeoff is consumed diagnostic
evidence; reversibility remains high because the contract performs no analysis
execution, embeddings, fitting, inference, search, reranking, selection,
dataset or taxonomy mutation, threshold change, runtime change, or remediation
implementation. No remediation success or causal root cause is claimed.

## V2-C6 Step 29H-B deterministic failure-analysis execution

Step 29H-B has executed successfully under the frozen Step 29H-A read-only
procedure. The create-once tracked result and manifest report
`execution_status=COMPLETED`, all six candidates analyzed, 54,048 group-CV
predictions analyzed, and 4,860 source-family predictions analyzed. Coverage
validation found zero missing and zero duplicate predictions.

The implementation remains available through:

```bash
sentinelvoice_env/bin/python scripts/run_v2c6_post_selection_failure_analysis.py --preflight
sentinelvoice_env/bin/python scripts/run_v2c6_post_selection_failure_analysis.py --run
sentinelvoice_env/bin/python scripts/run_v2c6_post_selection_failure_analysis.py --write-local-review
```

The analysis found that the dominant protected false-positive pattern was
gold `unsupported_or_uncertain` predicted as a protected action. This pattern
occurred across both BGE and TF-IDF candidates and across SF1, SF2, and SF3.
The boundary was bidirectionally weak: protected intents were also sometimes
predicted as `unsupported_or_uncertain`. Important measured hard-negative
weaknesses included:

- `transfer_pending` vs `unsupported_or_uncertain`
- `cancel_transfer` vs `unsupported_or_uncertain`
- `close_account` vs `unsupported_or_uncertain`
- `transfer_failed_or_declined` vs `unsupported_or_uncertain`
- `freeze_card` vs `unsupported_or_uncertain`
- `cancel_transfer` vs `transfer_pending`

These findings describe the frozen experiment and do not establish causality.
Within its three matched comparisons, balanced class weighting generally
increased protected recall while also increasing protected false-positive rate
and reducing unsupported recall. That is an experiment-specific observation,
not a universal class-weighting claim.

The tracked artifact classified remediation evidence as `both`, citing
targeted data-boundary evidence and representation/model evidence. This
classification is diagnostic only: remediation has not been selected or
implemented, Step 29I remains blocked, and `next_required` remains `null`
because Step 29H-A did not freeze a post-analysis continuation. No next-step
identifier is inferred.

The completed analysis performed no model fitting, inference, embedding
generation, threshold tuning, dataset mutation, taxonomy mutation, runtime
change, or final-holdout access. The tracked artifacts contain no raw text, and
the optional ignored local raw-text review was not performed.

SF1/SF2/SF3 are now consumed diagnostic development evidence. They may support
later regression or diagnostic comparisons but cannot be called fresh
validation on unseen sources. Future clean source-generalization evidence
requires newly independently authored material or ultimately the untouched
final holdout.

## V2-C6 Step 29H-C targeted remediation design contract

`v2c6_targeted_remediation_design_contract.json` hash-pins the completed Step
29H-B analysis and manifest plus its Step 29H-A, Step 29G, Step 29H, Step 29D,
and Step 29F lineage. It freezes a design only: no records are authored, no
embeddings are generated, no model is fit or evaluated, no threshold is tuned,
and no dataset, taxonomy, runtime, or final holdout changes.

The contract responds to Step 29H-B's `both` classification without claiming a
causal root cause. The measured basis includes unsupported-to-protected false
positives across BGE and TF-IDF and all three consumed source families,
protected-to-unsupported recall misses, six important hard-negative
boundaries, and the experiment-specific balanced-weighting tradeoff. It
freezes three linked components:

- 600 targeted training records in three independent 200-record families;
  each family has 20 records for each supported primary remediation intent and
  60 `unsupported_or_uncertain` records.
- Four unweighted candidates: BGE-small/LinearSVC `C=4`, word-plus-character
  TF-IDF/LinearSVC `C=1`, a normalized BGE-plus-TF-IDF hybrid at `C=1`, and a
  deterministic two-stage TF-IDF supported/unsupported hierarchy at `C=1`.
- 640 separately authored development-evaluation records in two new 320-record
  families, with exactly 40 records per primary remediation intent per family.

Fresh evaluation authoring must be independent from training authoring, must
not use predictions or candidate outputs, and must not paraphrase training or
consumed SF1/SF2/SF3 records. The 640 evaluation records remain excluded from
fitting. Deterministic exact and normalized duplicate, ID collision,
training/evaluation overlap, historical-development overlap, consumed-family
overlap, cross-intent duplicate, and provenance checks are mandatory. Review
reuses Step 29D/29E governance: AI-assisted semantic review may be used but is
not independent human review, and designated ambiguity, disagreement,
rejection, revision, confidence, and provenance cases require human
adjudication.

The now-built 9,608-record development/training population retains five-fold
`StratifiedGroupKFold` with atomic `group_id` and `random_state=20260930`.
Each candidate is then fit on that full population and evaluated separately on
both new families and their pooled 640 records. The unchanged protected recall
>= 0.80, protected false-positive rate <= 0.01, and unsupported recall >= 0.80
gates are mandatory for pooled group CV, each fresh family, and pooled fresh
evaluation. With 160 non-protected records per family, the protected-FP gate
permits at most `floor(0.01 * 160) = 1` protected false positive per family.

Only eligible candidates may enter the frozen worst-fresh-family-first
lexicographic selection. If none passes every gate, the required result is
`NO_ACCEPTABLE_CANDIDATE`, with no forced winner or weakened gate. Old
SF1/SF2/SF3 are permanently consumed and may appear only as diagnostic
regression evidence, never fresh validation. Step 29I remains blocked and a
future untouched final holdout remains mandatory. The contract's exact next
action, `v2c6_targeted_remediation_and_fresh_source_authoring`, authorized the
now-completed authoring/build of 600 training and 640 fresh evaluation records;
it did not run model selection.

## V2-C6 targeted remediation and fresh-source authoring workflow

Step 29H-C remains the frozen design. Its authorized authoring, review, and
dataset-construction workflow has now completed through
`scripts/build_v2c6_targeted_remediation_and_fresh_source_data.py`.

The targeted-training addendum contains 600 populated and approved records,
with 200 records in each frozen family:

- `v2c6_r2_train_sf1_minimal_boundary`
- `v2c6_r2_train_sf2_contextual_scenario`
- `v2c6_r2_train_sf3_conversational_correction`

Each family follows the contract's allocation across the eight primary
intents. Three mandatory human adjudications were completed, with zero
unresolved adjudications. Record
`v2c6_r2_train_sf1_minimal_boundary_0121` was the sole exact historical-text
collision identified during preflight; its text was replaced and semantically
re-reviewed. The 600-record addendum combined with the previous 9,008 records
produces the frozen 9,608-record development/training dataset.

The separately built fresh source-evaluation dataset contains 640 populated
and approved records, with 320 records in each independent family:

- `v2c6_r2_eval_sf1_independent_casework`
- `v2c6_r2_eval_sf2_independent_naturalistic`

Each family contains 40 records per intent, including 160 protected and 160
non-protected records. All 640 records are excluded from fitting and no human
adjudication remains unresolved. The consumed diagnostic SF1/SF2/SF3 records
were not reused as fresh evidence.

Final deterministic preflight found zero collisions in every frozen category:

- targeted training against historical development, exact and normalized;
- fresh evaluation against historical development, exact and normalized;
- targeted training against fresh evaluation, exact and normalized;
- fresh evaluation against consumed diagnostic evidence, exact and
  normalized;
- within targeted training and within fresh evaluation, exact and normalized;
- cross-intent normalized duplicates; and
- record-ID and group-ID cross-role collisions.

These results cover deterministic exact and normalized-text checks only. No
embeddings or semantic-similarity analysis was performed, so no broader
semantic-independence claim is made.

The six create-once outputs and their SHA-256 hashes are:

- `v2c6_targeted_remediation_training_examples.json`:
  `6c8322ea1c09f026ede7a2808a7bc3ce01ae2b3f4b10b5885a352ff5c9df5ad6`
- `v2c6_targeted_remediation_training_examples.manifest.json`:
  `1a25653ee79008d5a7daee4f4c2dfb3f28e19b353777329e4199d388d0ed85d9`
- `v2c6_targeted_remediated_development_dataset.json`:
  `133456aa10552058fe67ed2eed37acbe583e31331656174cd2e018430cca799d`
- `v2c6_targeted_remediated_development_dataset.manifest.json`:
  `d30ead62c1370e4f50760d6a159f38d0a6118c9cdce2542f97a09086e49d3c58`
- `v2c6_fresh_source_evaluation_dataset.json`:
  `df3c1eef7e262260754a1a10908b540d479fe7dc3d6f228d1a9ba432ae9f1a6e`
- `v2c6_fresh_source_evaluation_dataset.manifest.json`:
  `5e2ad040155fcc05f37854d1dd10960b7dab5a840f4c42a34075f4c86c6f77da`

Observed execution status was `ready_for_build=true` at preflight,
`files_written=true` at build with output counts 600, 9,608, and 640, and
`results_valid=true` at result checking. Focused builder tests reported 35 passed; all
`backend/tests/test_v2c6_*.py` tests reported 453 passed; separate Ruff checks
for the builder and tests passed; and `git diff --check` passed.

The dataset-construction cycle itself performed no candidate evaluation,
embedding generation, model fitting, inference, or threshold tuning. After the
data-phase commit, it authorized only the frozen four-candidate development
experiment. Candidate definitions, safety gates, the selection rule, taxonomy,
dataset protocol, and runtime authority remained unchanged.

The development-only execution machinery in
`scripts/run_v2c6_targeted_remediation_model_selection.py` has now completed
the frozen experiment. It reused one deterministic set of five
`StratifiedGroupKFold` partitions for all four candidates. Each candidate was
then fit once on all 9,608 development records and used without a family-level
refit to produce both 320-record fresh-family prediction sets and the pooled
640-record metrics. The fresh records were excluded from representation and
classifier fitting and were not used for threshold tuning.

The execution status is `COMPLETED`; all four candidate executions completed,
but none was eligible. The deterministic outcome is:

- `selection_status = NO_ACCEPTABLE_CANDIDATE`
- `selected_candidate = null`
- `winner_forced = false`
- `gates_weakened = false`
- `step29i_authorized = false`
- `next_required = null`

Aggregate candidate results are:

| Candidate | Pooled fresh primary-8 macro F1 | Pooled group-CV macro F1-16 | Pooled fresh protected FPR | Pooled fresh protected recall | Pooled fresh unsupported recall | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `BGE_SMALL_LINEAR_SVC__C=4.0__class_weight=none` | 0.8454459835298751 | 0.8792458237805437 | 0.059375 | 0.9125 | 0.6375 | no |
| `WORD_CHAR_TFIDF_LINEAR_SVC__C=1.0__class_weight=none` | 0.8365655768896487 | 0.8654374384371238 | 0.06875 | 0.940625 | 0.6625 | no |
| `HYBRID_BGE_TFIDF_LINEAR_SVC__C=1.0__class_weight=none` | 0.8626982912616474 | 0.8952700882334053 | 0.059375 | 0.95625 | 0.6625 | no |
| `HIERARCHICAL_TFIDF_LINEAR_SVC__C=1.0__class_weight=none` | 0.8103099681529785 | 0.8567438107400511 | 0.0875 | 0.9125 | 0.5125 | no |

Every candidate failed protected false-positive rate on pooled group-aware CV,
both individual fresh families, and pooled fresh evaluation. Every candidate
failed unsupported recall on both individual fresh families and pooled fresh
evaluation. Pooled fresh protected recall exceeded its 0.80 threshold for all
four candidates. The hybrid recorded the highest pooled fresh primary-8 macro
F1 and pooled fresh protected recall in this experiment, but it is not a
winner, selected candidate, acceptable candidate, or final model. The frozen
gates are unchanged and are not reinterpreted.

The create-once results are
`v2c6_targeted_remediation_model_selection_results.json` and
`v2c6_targeted_remediation_model_selection_results.manifest.json`. Their audit
identities are:

- results SHA-256:
  `81fc64cd3476cd4eb2c6dc1e7b555803692f4800d48a61665fa9fd7768c9f145`
- runner SHA-256:
  `043e00d5cab470359b2ad5e4f78492a715bfb9ed6e83e766fcd08e6c923af6b0`
- split-audit SHA-256:
  `a25276d7ead685e32ea728d6c8a8065ea43a8a04b4b40a854b831691df556668`

Governance remains fail closed: the 640 fresh records were not used for
fitting or threshold tuning; no fitted classifier was persisted; the final
holdout was neither accessed nor evaluated; no final-model acceptance was
claimed; and runtime behavior did not change. The 640 records are now consumed
development-evaluation evidence and must not subsequently be represented as
untouched or fresh evidence. Historical Step 29H results remain historical and
unchanged.

No post-result remediation, threshold tuning, or model modification has
occurred. Step 29I remains blocked. The result itself authorized no next
activity and recorded `next_required=null`; the separate read-only contract
below authorizes analysis only and does not choose a remedy.

## V2-C6 targeted-remediation failure analysis

`v2c6_targeted_remediation_failure_analysis_contract.json` froze the
read-only diagnostic step over the already-persisted grouped-CV and fresh
predictions in `v2c6_targeted_remediation_model_selection_results.json`. It
hash-pins the result, manifest, Step 29H-C design, 9,608-record development
dataset, and 640-record consumed fresh evaluation dataset. The prerequisite
state remains exactly four completed candidates, zero eligible candidates,
`NO_ACCEPTABLE_CANDIDATE`, no selected candidate, no forced winner, unchanged
gates, no Step 29I authorization, and `next_required=null`.

The contract requires deterministic, directional analysis of protected false
positives, unsupported misses, protected recall misses, all ten targeted
hard-negative boundaries, cross-candidate error overlap, both consumed fresh
families, and grouped-CV versus fresh error structure. It records that
stage-level hierarchical predictions were not persisted, so exact Stage-1
attribution is unavailable and the model must not be reconstructed or rerun.
Allowed evidence classifications are descriptive and non-causal; multiple
categories may be reported when directly supported.

The completed create-once outputs are
`v2c6_targeted_remediation_failure_analysis.json` and
`v2c6_targeted_remediation_failure_analysis.manifest.json`, with raw utterance
text prohibited. The contract permits only read-only joins to prediction-linked
frozen development evidence. It prohibits model fitting, embeddings, inference,
threshold tuning, candidate or gate changes, new examples, final-holdout access,
candidate selection, remediation implementation, runtime changes, and Step 29I
authorization.

`scripts/run_v2c6_targeted_remediation_failure_analysis.py` now implements the
frozen read-only workflow with mutually exclusive `--preflight` and `--run`
modes. It consumes only the completed experiment's persisted predictions and
frozen metadata joins. It cannot train or refit a model, generate embeddings,
run inference, or tune thresholds; execution created the two reserved text-free
artifacts without overwriting an existing artifact.

The analysis completed with phase
`V2-C6 targeted-remediation failure analysis`, `execution_status=COMPLETED`,
and `failure_analysis_executed=true`. All four candidates exhibited protected
false positives in both grouped CV and fresh evaluation. Pooled fresh
unsupported recall ranged from 0.5125 to 0.6625, substantially below the
approximately 0.8682 to 0.9021 grouped-CV range, and the same degradation was
present across both consumed fresh families. Most pooled-fresh protected false
positives for every candidate originated from true
`unsupported_or_uncertain` records. Protected recall remained comparatively
strong at 0.9125 to 0.95625; the dominant measured problem is over-routing
unsupported or uncertain requests into supported, including protected,
intents. These are descriptive observations, not causal root-cause findings.

The hierarchical candidate did not eliminate the weakness. Exact Stage-1
attribution is unavailable because stage-level predictions were not persisted.
The result records the non-causal classifications
`development_distribution_boundary_weakness`,
`fresh_source_generalization_weakness`, `architecture_specific_weakness`, and
`cross_architecture_shared_weakness`.

Execution governance records `models_run=false`,
`embeddings_generated=false`, no fitting, training, inference, or threshold
tuning, no candidate selection or ranking, no remediation selection, and no
final-holdout access. Hybrid and every other candidate remain unselected.
Step 29I remains blocked and unauthorized, runtime behavior remains unchanged,
and `next_required=null`. The 640 fresh records are consumed
diagnostic/development evidence and cannot be reused or represented as
untouched evaluation evidence. At analysis completion no remediation had been
frozen or authorized; the separate design below now freezes the bounded next
cycle before any new data, model, or gate experiment.

## V2-C6 protected-intent safety-gate remediation design

`v2c6_protected_intent_gate_remediation_design_contract.json` freezes the
separate post-analysis design. The completed failure analysis is its non-causal
evidence basis: unsupported-to-protected over-routing was measured, but no
causal root cause, candidate, or remediation was established by that analysis.

The bounded candidate set contains only `HYBRID_CONTROL_R3` and
`HYBRID_PROTECTED_VERIFIER_R3`. Both use the same Hybrid primary router and the
same 10,088-record development set. The gated candidate adds four
independent word+character TF-IDF `LinearSVC` verifiers, one per protected
intent. A verifier is invoked only for its protected primary prediction;
acceptance preserves that prediction and rejection returns
`unsupported_or_uncertain`. Verifier output changes routing only.
Authentication, ownership, confirmation, idempotency, and protected tool
execution remain deterministic application responsibilities.

The design freezes 480 new training records across three independent
160-record R3 families. Each protected-intent verifier is trained solely on its
60 new positive examples and 60 new unsupported hard negatives targeting that
intent. The consumed 640-record R2 evaluation cannot become untouched evidence
again. Two new independent R3 evaluation families are required, each with 320
records, 40 examples per primary intent, and a 10-by-four protected-boundary
allocation within its unsupported examples.

The existing mandatory thresholds remain unchanged: protected recall at least
0.80, protected false-positive rate at most 0.01, and unsupported recall at
least 0.80. They apply to pooled grouped CV, both individual R3 evaluation
families, and pooled R3 evaluation. Selection is lexicographic and restricted
to candidates passing every gate in every scope.

The contract prevents an automatic tuning loop. If neither candidate is
eligible, no R4 classifier/data remediation, representation expansion, new
authoring cycle, or gate weakening is authorized; `next_required` for that
outcome is `routing_architecture_fallback_decision`. If at least one candidate
is eligible, the continuation is
`v2c6_candidate_freeze_before_final_holdout`. Neither outcome automatically
authorizes Step 29I or final-holdout access.

The contract step itself authored no R3 data and ran no experiment, fitting,
inference, embedding, or threshold tuning. Its immediate frozen continuation,
`v2c6_r3_targeted_addendum_and_fresh_evaluation_authoring`, has since been
completed; see the R3 dataset build below. Step 29I remains blocked and the
raw final holdout remains untouched.

### R3 protected-intent gate data authoring/build workflow

`scripts/build_v2c6_r3_protected_intent_gate_data.py` implements the frozen
workflow without generating utterance text. Its four mutually exclusive modes
are `--preflight`, `--prepare-authoring-workfiles`, `--build`, and
`--check-results`. Preparation creates only these ignored, create-once
workfiles:

- `local/v2c6_r3_protected_intent_gate_training_authoring.json`, containing
  480 empty slots across the three frozen 160-record training families;
- `local/v2c6_r3_fresh_source_evaluation_authoring.json`, containing 640 empty
  slots across the two frozen 320-record evaluation families.

The training allocation is exactly 20 positives and 20 targeted unsupported
negatives per protected intent per family, yielding 60 positives and 60
negatives for each of the four verifier datasets. Each evaluation family has
40 records per primary intent; its 40 unsupported records allocate 10 cases to
each protected boundary. Evaluation slots are explicitly not training data,
are excluded from candidate fitting, and freeze the required independent
authoring and no-prediction-inspection declarations.

Build is blocked until all records have nonempty text, matching raw and
normalized hashes, valid provenance, approved semantic review, and no
unresolved required human adjudication. Exact and normalized duplicates,
cross-intent collisions, training/evaluation overlap, and overlap against the
9,608 existing development records or the consumed 640 R2 evaluation records
all fail closed. No embedding or semantic-similarity test is substituted for
these exact deterministic controls.

Build creates, without overwriting, the 480-record training artifact, the
combined 10,088-record development artifact, the independent 640-record fresh
evaluation artifact, and a manifest for each. Manifests pin the frozen
contract, existing development data, consumed R2 lineage, workfile and output
hashes, counts, verifier composition, leakage audit, review status, and
unchanged governance flags.

### R3 protected-intent gate dataset build

The R3 data build is complete:

- `v2c6_r3_protected_intent_gate_training_examples.json`: 480 targeted
  training addendum records (60 positives and 60 targeted unsupported
  negatives per protected-intent verifier);
- `v2c6_r3_protected_intent_gate_development_dataset.json`: 10,088 expanded
  development records (the frozen 9,608 records followed by the 480 new
  training records);
- `v2c6_r3_fresh_source_evaluation_dataset.json`: 640 new fresh evaluation
  records (320 per source family, 80 per primary intent).

All 480 training records and all 640 fresh evaluation records were approved
through AI-assisted review, which is not recorded as human review. Human
adjudication was required for 0 records. The build's duplicate and leakage
audit passed with every exact and normalized count at zero: within-dataset
duplicates, cross-intent collisions, training/evaluation overlap, overlap with
the 9,608-record development source, and overlap with the consumed 640-record
R2 evaluation. The fresh evaluation records remained excluded from candidate
fitting throughout authoring and build.

| Artifact | SHA-256 |
| --- | --- |
| `v2c6_r3_protected_intent_gate_training_examples.json` | `db2db4d6116c04712ae9b5979c71c8cbdc1ed9c4e330ba4f3915ba8b62efce14` |
| `v2c6_r3_protected_intent_gate_training_examples.manifest.json` | `f5d3ba48a425e72eab8b45d5835f7f3fc8334c42ba6c0d6ab3e6d700eeecefbf` |
| `v2c6_r3_protected_intent_gate_development_dataset.json` | `d27c411cefdba2cc4d8a9c70493b05f43542e1542f1b202c909090a7185f36fe` |
| `v2c6_r3_protected_intent_gate_development_dataset.manifest.json` | `5ead89b05dfc4b649e048cc5bb44c261d38f9536594b02f9e38a8024514168fe` |
| `v2c6_r3_fresh_source_evaluation_dataset.json` | `e530f24c236ed03c8228ab465165996d7ea94fb24f43cc1ddb1545118918d369` |
| `v2c6_r3_fresh_source_evaluation_dataset.manifest.json` | `5972e5c81f552cf978219075f0536689693f1a0e78892fedc148b33df07f40ba` |

No model, embedding, fitting, inference, threshold tuning, or candidate
selection occurred during the data build. The raw final holdout remains
prohibited and untouched, and Step 29I remains blocked.

The R3 experiment comparing `HYBRID_CONTROL_R3` and
`HYBRID_PROTECTED_VERIFIER_R3` has since been executed; see the frozen result
below.

### R3 protected-intent gate model-selection runner

`scripts/run_v2c6_r3_protected_intent_gate_model_selection.py` implements the
frozen experiment.

Modes:

- `--preflight` validates the contract, convention lineage, dataset and
  manifest hashes, counts, the grouped split, verifier fold membership, fresh
  isolation, and output availability. It performs no embedding, fitting,
  inference, or writes.
- `--run` executes grouped CV and fresh evaluation for both candidates and
  creates the results and manifest once, atomically.
- `--check-results` deterministically revalidates existing results and their
  manifest from the persisted predictions, without embedding, fitting, or
  inference.

Execution semantics:

- Convention reuse: the Hybrid, word+char TF-IDF, BGE, `LinearSVC`, and
  metric helpers are loaded from the R2 runner only after its SHA-256 matches
  the hash recorded by the consumed R2 result manifest.
- Grouped CV: one `StratifiedGroupKFold(n_splits=5, shuffle=True,
  random_state=20260930)` plan over `group_id` is built once and reused for both
  candidates. Each fold fits the Hybrid primary router on its training
  partition only.
- Verifier fold isolation: each verifier trains only on R3 addendum records
  whose expanded-development records lie in that fold's training partition.
  Historical development records and fold-validation R3 records are rejected.
  The full-population verifier fit requires exactly 60 positives and 60
  targeted unsupported negatives.
- Gate: only a protected primary prediction is sent, and only to its dedicated
  verifier. Acceptance keeps the prediction; rejection returns
  `unsupported_or_uncertain`. The gate changes routing only and is not
  authorization.
- Fresh evaluation: each candidate is fitted once on all 10,088 development
  records and evaluated on both 320-record families and the pooled 640 without
  refitting. Fresh records never enter any fit.
- Gates and selection: all three mandatory gates must pass in all four scopes
  (pooled grouped CV, each fresh family, and pooled fresh). The frozen
  lexicographic rule is applied to eligible candidates only, with a 1e-12
  numeric tie tolerance.
- Stop rule: with no eligible candidate the result is
  `NO_ACCEPTABLE_CANDIDATE` with `next_required =
  routing_architecture_fallback_decision`. With an eligible candidate it is
  `v2c6_candidate_freeze_before_final_holdout`. Neither outcome authorizes
  Step 29I, final-holdout access, or final model acceptance.
- Verifier diagnostics: `per_protected_intent_verifier_recall` uses the
  denominator of rows whose gold intent and primary prediction are both that
  protected intent. It is distinct from final routing protected recall, and
  the diagnostics never replace the mandatory gates.
- Outputs: `v2c6_r3_protected_intent_gate_model_selection_results.json` and
  its manifest are create-once, contain no raw utterance text, and persist no
  fitted model. The BGE cache is ignored local data under `local/`, bound to
  the ordered populations, source hashes, split audit, runner, and library
  versions.

### R3 protected-intent gate model-selection result

The experiment was executed once and `--check-results` validated the saved
artifacts:

- `data/evals/v2/ml/v2c6_r3_protected_intent_gate_model_selection_results.json`
  (SHA-256 `883dd954e9ed08c9d2be8a887c803270d09fda11341da9edd8cf67226204cde6`)
- `data/evals/v2/ml/v2c6_r3_protected_intent_gate_model_selection_results.manifest.json`

Frozen outcome: `selection_status = NO_ACCEPTABLE_CANDIDATE`,
`eligible_candidate_count = 0`, `selected_candidate_id = null`,
`winner_forced = false`, `gates_weakened = false`,
`next_required = routing_architecture_fallback_decision`, and
`step29i_authorized = false`.

| Scope | Metric | `HYBRID_CONTROL_R3` | `HYBRID_PROTECTED_VERIFIER_R3` |
| --- | --- | --- | --- |
| `pooled_group_aware_cv` | protected recall | 0.9352189781021898 PASS | 0.7992700729927007 **FAIL** |
| `pooled_group_aware_cv` | protected FPR | 0.016236654804270462 **FAIL** | 0.010342526690391459 **FAIL** |
| `pooled_group_aware_cv` | unsupported recall | 0.8994226112870181 PASS | 0.9087353324641461 PASS |
| `v2c6_r3_eval_sf1_independent_casework` | protected recall | 0.99375 PASS | 0.95 PASS |
| `v2c6_r3_eval_sf1_independent_casework` | protected FPR | 0.03125 **FAIL** | 0.0125 **FAIL** |
| `v2c6_r3_eval_sf1_independent_casework` | unsupported recall | 0.875 PASS | 0.95 PASS |
| `v2c6_r3_eval_sf2_independent_naturalistic` | protected recall | 0.8125 PASS | 0.75 **FAIL** |
| `v2c6_r3_eval_sf2_independent_naturalistic` | protected FPR | 0.025 **FAIL** | 0.0125 **FAIL** |
| `v2c6_r3_eval_sf2_independent_naturalistic` | unsupported recall | 0.875 PASS | 0.925 PASS |
| `pooled_r3_fresh_evaluation` | protected recall | 0.903125 PASS | 0.85 PASS |
| `pooled_r3_fresh_evaluation` | protected FPR | 0.028125 **FAIL** | 0.0125 **FAIL** |
| `pooled_r3_fresh_evaluation` | unsupported recall | 0.875 PASS | 0.9375 PASS |

Thresholds: protected recall >= 0.80, protected FPR <= 0.01, unsupported
recall >= 0.80, required on every scope. Both candidates are ineligible.

| Selection metric | `HYBRID_CONTROL_R3` | `HYBRID_PROTECTED_VERIFIER_R3` |
| --- | --- | --- |
| worst fresh-family primary-8 macro-F1 | 0.7980060473006161 | 0.7801821499891757 |
| pooled fresh primary-8 macro-F1 | 0.8733783556753727 | 0.8605614917303537 |
| pooled grouped-CV macro-F1-16 | 0.8935424434545153 | 0.8792005038767634 |

Because no candidate was eligible, the lexicographic selection rule was not
applied to choose a winner; these metrics are recorded for completeness only.

Pooled fresh gate diagnostics for `HYBRID_PROTECTED_VERIFIER_R3`:

- primary protected predictions presented to verifiers: 298
- verifier accept count: 276; verifier reject-to-unsupported count: 22
- protected false positives: 9 before the gate, 4 after (5 prevented)
- true protected requests rejected by the verifier: 17
- protected recall: 0.903125 before the gate, 0.85 after
- unsupported recall: 0.875 before the gate, 0.9375 after

The protected verifier materially reduced protected false positives and
improved unsupported recall, but the improvement was insufficient to satisfy
the frozen protected-FPR threshold, and it also reduced protected recall enough
to fail mandatory gates in grouped CV and fresh family 2. This is a measured
outcome, not an established causal root cause. The experiment therefore
produced no acceptable candidate.

The executed `--run` truthfully recorded
`embeddings_generated_during_run`, `model_fitting_performed`,
`model_inference_performed`, `model_selection_performed`,
`primary_router_fitting_performed`, `verifier_fitting_performed`, and
`fresh_evaluation_performed` as `true`. It recorded `false` for
`final_holdout_accessed`, `final_holdout_evaluated`,
`fresh_evaluation_used_for_fitting`,
`fresh_evaluation_used_for_threshold_tuning`,
`fresh_evaluation_used_for_candidate_modification`,
`threshold_tuning_performed`, `calibration_performed`,
`persisted_fitted_classifier`, `runtime_behavior_changed`,
`production_ready_claimed`, `final_model_acceptance_claimed`,
`r4_classifier_or_data_cycle_authorized`, `routing_fallback_implemented`, and
`step29i_authorized`.

The frozen final classifier-remediation stop rule now applies. No R4 cycle,
classifier family, representation, targeted classifier dataset, threshold
tuning, C tuning, class-weight change, or gate weakening is permitted. The next
required phase is exactly `routing_architecture_fallback_decision`, a
separately governed architecture decision. That decision is now frozen as
design only (see below); the fallback is not implemented. The raw V2-C5 final
holdout remains prohibited, and
Step 29I remains blocked.

The persisted result can be revalidated without embedding, fitting, or
inference:

```bash
sentinelvoice_env/bin/python \
  scripts/run_v2c6_r3_protected_intent_gate_model_selection.py \
  --check-results
```

### V2-C6 Routing Architecture Fallback Decision

`data/evals/v2/ml/v2c6_routing_architecture_fallback_decision.json` freezes
the decision `CLARIFICATION_GATED_STRUCTURED_LLM_PROTECTED_ROUTING` as
**design only**. It hash-binds the frozen R3 results
(`883dd954e9ed08c9d2be8a887c803270d09fda11341da9edd8cf67226204cde6`), the R3
results manifest, the R3 remediation-design contract, and the V2-C5 taxonomy
freeze and its manifest.

Problem: the frozen R3 local classifiers could not jointly satisfy protected
recall, protected false-positive rate, and unsupported recall across all
required scopes, and the stop rule prohibits another classifier-remediation
cycle.

Decision:

- The V2-C6 local classifier remains evaluation evidence only and never
  becomes runtime routing authority. Runtime semantic routing remains the
  existing Groq-first LLM tool calling.
- Only when the conversational model proposes a registered protected-write
  tool, a small structured semantic verifier checks that exact proposed action.
  It returns one closed decision: `EXPLICIT_CURRENT_ACTION`,
  `AMBIGUOUS_OR_INFORMATIONAL`, or `NOT_REQUESTED`. Free-form verifier output
  is never authoritative.
- Placement: in `AgentOrchestrator._run_model_loop` (verified at `d5e1e49`),
  the current order for a proposed tool is the registered and effective
  allowed-tool check, `authoritative_arguments` and `_RESOURCE_BINDINGS`
  processing, resource and active-intent validation with a possible
  `_resource_clarification` return, Pydantic input validation, the
  `requires_confirmation` branch, and `ConversationState.request_action`. The
  verifier is inserted after the registered and allowed-tool check and before
  the `authoritative_arguments` / `_RESOURCE_BINDINGS` block. It therefore runs
  before resource binding, resource clarification, input validation, the
  confirmation branch, and pending-action creation.
- `EXPLICIT_CURRENT_ACTION` continues through the unchanged deterministic path:
  the existing resource binding and validation, the `requires_confirmation`
  branch, the pending protected action, explicit one-use confirmation,
  `ToolExecutor` authorization, authentication and ownership checks, then
  execution.
- `AMBIGUOUS_OR_INFORMATIONAL` or `NOT_REQUESTED` performs no resource-binding
  clarification, creates no pending action, and asks no execution
  confirmation. It returns a deterministic narrow
  clarification question, and the answer is processed as a new user turn,
  never as confirmation.
- Verifier timeout, provider failure, malformed output, an invalid enum value,
  or any other verifier failure fails closed identically: no resource-binding
  clarification, no pending action, deterministic clarification, and the
  existing human handoff where appropriate.
- Semantic verification is never authorization. A verifier result cannot
  authenticate, establish ownership, satisfy confirmation, execute or
  authorize a tool, or bypass `ToolExecutor` or pending-action state.
  Interruption and correction keep their existing invalidation semantics.
- Executable protected actions are governed by the runtime registry, which
  currently contains `freeze_card` and `create_dispute`. The taxonomy labels
  `cancel_transfer` and `close_account` create no runtime tool, and the
  verifier may never imply such a capability.
- The verifier uses the existing `LLMProvider` abstraction behind a thin,
  swappable interface, with Groq preferred. It adds no second agent, service,
  or infrastructure, and it runs only at the protected-action boundary to bound
  cost and latency.

The verifier's authority is asymmetric: a negative or uncertain result can
only cause clarification, and a positive result still cannot execute anything
without the existing confirmation and authorization path.

Alternatives recorded:

- R4 local classifier remediation (rejected by the stop rule).
- Weakening the protected-FPR threshold (prohibited and unsafe).
- Clarifying every protected request (safe, but adds a redundant turn to
  explicit requests, which hurts voice UX).
- An LLM verifier on every turn (unnecessary cost and latency).
- The LLM verifier as authorization (rejected).
- Escalating every protected request to a human (defeats core V1
  protected-action capability).

Trade-offs: the decision targets the measured boundary, needs no new
classifier cycle, and keeps deterministic execution controls with explicit
fail-closed behavior. It costs one extra model call and some latency on
protected-action turns, the verifier needs its own evaluation, and ambiguous
cases intentionally add a user turn. Reversibility is high, because the
verifier sits before pending-action creation behind a thin interface.

Evaluation requirements are frozen for the future implementation, covering 16
required scenarios: explicit acceptance; informational, hypothetical, and
ambiguous wording; unsupported operations; fail-closed malformed output,
timeouts, and provider failures; clarification correction and interruption;
confirmation, authentication, ownership, and stale-confirmation controls; no
direct verifier execution; and prompt-injection resistance. Tracked metrics are
protected semantic false-positive rate, explicit protected-request recall,
clarification rate, clarification recovery, fail-closed compliance,
protected-execution authorization compliance, P50/P90/P95 verifier latency, and
incremental cost per protected turn. Their thresholds belong to the separately
governed implementation and evaluation contract. The consumed R3 fresh
evidence cannot serve as untouched acceptance evidence, and a separately
governed fresh evaluation is required before any runtime-acceptance claim.

This step changed no runtime behavior and made no Groq, embedding, or model
call. It records `routing_fallback_implemented = false`,
`runtime_behavior_changed = false`, `step29i_authorized = false`,
`final_holdout_accessed = false`, and `production_ready_claimed = false`. The
design alone is not a safety or production-readiness claim. The next required
activity is the separately governed
`v2c6_routing_fallback_implementation_and_evaluation` phase, which must freeze
its own implementation and evaluation contract first. The raw V2-C5 final
holdout remains prohibited, and Step 29I remains blocked.

### V2-C6 Routing Fallback Resolver Safety Amendment

`data/evals/v2/ml/v2c6_routing_architecture_fallback_decision_amendment.json`
is a separately frozen, design-only amendment. It hash-binds the original
decision (`b650511031a8df58056c045da684c63a0144b1272aa5d0525dc971734339a541`),
which remains unmodified historical evidence, and records the discovery HEAD
`6f196eb`.

The original decision recorded `pre_llm_resource_resolver_unchanged = true` and
assumed that no protected pending action could exist before semantic
verification. Direct inspection of the current code contradicts this for
protected actions:

- `freeze_card`: `ResourceResolver._resolve_card` sets the `freeze_card` intent
  for any card request containing the substring "freeze".
  - With one matching card, `_activate_candidate` calls
    `ConversationState.request_action("freeze_card", ...)`, and
    `AgentOrchestrator.handle_text_turn` then returns the execution-confirmation
    prompt before any model call.
  - With several cards, the resolver asks a card-selection question, and the
    selection reply creates the pending action the same way.
- `create_dispute`: `_resolve_transaction` sets the `create_dispute` intent on
  the substring "dispute" and can ask a transaction-selection question.
  `_activate_candidate` only binds the transaction and never calls
  `request_action`. The `create_dispute` pending action is created only by the
  model tool-call path.
- The two tools are therefore not symmetric. Both can receive
  protected-intent resource clarification before verification, but only
  `freeze_card` gets a pre-verification pending action.
- These paths bypass the intended semantic-verification placement, not
  authentication, explicit confirmation, or `ToolExecutor` authorization,
  which all remain enforced.

Corrected invariant: for every executable protected action, semantic
verification must precede protected-action resource clarification,
pending-action creation, and the execution-confirmation prompt.

- The pre-LLM resolver is no longer frozen as unchanged for protected actions.
  The implementation must remove, bypass, or defer any resolver path that can
  create a protected pending action before verification, including the
  `freeze_card` `request_action` call in `_activate_candidate`.
- Non-protected resource resolution stays unchanged unless a minimal refactor
  is needed.
- A verified explicit request may still need resource clarification. For
  example, "Freeze my card." with several cards asks which card. An
  informational request such as "What happens if I freeze my card?" gets only
  semantic clarification, with no card selection and no pending action.

Multi-turn rule: when a verified explicit request needs a resource-selection
turn, only minimal, action-specific state records that verification already
succeeded.

- That state is not confirmation or authorization and cannot execute
  anything.
- It cannot be reused for a different protected action.
- It is invalidated by cancellation, correction, or abandonment, must be safe
  under voice interruption, and never bypasses the later explicit execution
  confirmation.
- A bare selection reply such as "the Visa card", "the first one", or "ending
  in 1234" never independently counts as `EXPLICIT_CURRENT_ACTION`.
- The exact state representation belongs to the implementation contract.

Corrected future order:

1. Determine or propose the protected action without granting authority.
2. Run structured semantic verification. A non-explicit result gets semantic
   clarification and stops.
3. For an explicit result: resolve and bind the protected resource, asking a
   resource question if necessary and keeping verified state across turns.
4. Validate the request and create the pending action.
5. Ask for explicit execution confirmation.
6. Apply deterministic `ToolExecutor` authorization, authentication, and
   ownership enforcement, then execute.

The amendment adds seven required evaluation scenarios. All original
guarantees are preserved: the three verifier decisions, no verifier authority,
no R4 or classifier runtime integration, registry-only executable tools,
mandatory explicit confirmation, a required fresh fallback evaluation, the
prohibited raw V2-C5 final holdout, and Step 29I blocked. No runtime code
changed.

### V2-C6 Routing Fallback Implementation Contract

`data/evals/v2/ml/v2c6_routing_fallback_implementation_evaluation_contract.json`
freezes the implementation and evaluation contract for
`CLARIFICATION_GATED_STRUCTURED_LLM_PROTECTED_ROUTING` before any runtime code
changes. It hash-binds the frozen decision and its resolver-safety amendment
(and through them the R3 result), plus the current orchestrator, resource
resolver, conversation state, LLM provider protocol, Groq provider, tool
definitions, registry, and `ToolExecutor`.

Architecture summary:

- Chosen approach: a structured semantic verifier at the protected-action
  boundary only.
- Problem solved: the R3 classifiers failed the mandatory gates, and the stop
  rule forbids another classifier cycle.
- Why it suits SentinelVoice: it reuses Groq-first tool calling and the
  deterministic authorization stack inside the modular monolith.
- Alternatives: recorded in the original decision.
- Trade-off: one additional bounded Groq call on protected proposals buys a
  semantic safety boundary before protected resource and pending-action state.
- Reversibility: high.

Verifier interface and output:

- An async `ProtectedActionSemanticVerifier.verify(*, user_text,
  proposed_action)` returns exactly one of `EXPLICIT_CURRENT_ACTION`,
  `AMBIGUOUS_OR_INFORMATIONAL`, or `NOT_REQUESTED`. Failures are typed
  exceptions, never a fourth decision.
- Structured output uses the existing `LLMProvider` with one internal schema,
  `record_protected_action_semantic_decision`, which has a single required
  enum field. It is never registered, never passed to `ToolExecutor`, and has
  no handler.
- Anything other than exactly one correctly named call with a valid enum
  value is a failure. Free-form text never substitutes for the decision.
- A dedicated prompt treats customer text as untrusted data and decides only
  the semantic question, never authentication, ownership, confirmation, or
  permission.

Bounds and failure policy:

- The verifier uses the same `openai/gpt-oss-20b` model through a dedicated
  Groq provider instance with `max_completion_tokens = 64`, a 2.0-second
  timeout (matching the banking-tool convention), at most one call per
  protected proposal, and zero retries.
- Timeouts, provider errors, malformed or zero or multiple calls, wrong names,
  invalid arguments or enum values, and unexpected exceptions all fail closed.
  A failure creates no pending action, no confirmation prompt, no protected
  resource clarification, and no execution.
- There is no retry loop and no failure counter. The deterministic response
  may offer the existing human-support path.

Semantic clarification:

- Non-explicit decisions and failures use fixed, action-specific templates.
  freeze_card: "Are you asking me to freeze a card now? If so, please say that
  directly. Otherwise, tell me what you want to know about freezing a card."
  create_dispute: "Are you asking me to create a dispute now? If so, please
  say that directly. Otherwise, tell me what you want to know about disputes."
- Clarification is never confirmation. A bare "yes" is processed as a new
  semantic request.
- Clarification clears protected resource-resolution state and keeps no
  protected intent waiting for a later "yes".

Resolver correction:

- The resolver may not create the `freeze_card` pending action or ask
  `freeze_card` or `create_dispute` resource questions before verification.
- An existing active card or transaction neither authorizes nor verifies a new
  protected action.

Verified resource state:

- A typed `ProtectedActionSemanticContext(action, resource_type)` on
  `ConversationState` carries a verified explicit request through a
  resource-selection turn. The valid pairs are `freeze_card` with CARD and
  `create_dispute` with TRANSACTION.
- It stores no utterance and is not a pending action, confirmation, or
  authorization.
- It is cleared on pending-action creation, cancellation, correction,
  abandonment, terminal state, cleared resource resolution, or mismatch. It
  survives voice interruption only together with the matching preserved
  resource resolution.
- Bare selectors continue only through this context.

Trigger and observability:

- The verifier is triggered only by
  `PermissionLevel.PROTECTED_WRITE` on an already registered and allowed tool,
  never by keywords, active intent, or classifier output.
- Traces emit `protected_action.verification.started`, `.completed`, and
  `.failed` under the `safety` component with no raw text.
- Verifier LLM usage flows through the existing `llm.request` events, tagged
  with purpose `protected_action_semantic_verification`, so cost aggregation
  and latency remain measurable.

Future fresh semantic evaluation (not authored yet): two independent families
of 200 records each. Each family holds 50 explicit `freeze_card`, 50 explicit
`create_dispute`, and 50 varied boundary negatives per action, giving 400
records in total. No R3 paraphrasing is allowed.

Acceptance gates:

- Protected semantic false-positive rate <= 0.01 and explicit protected-request
  recall >= 0.80 on each family, the pooled set, and each action within the
  pooled set. Each family may have at most 1 false positive in 100 negatives
  and needs at least 80 explicit hits in 100 positives. The pooled set allows
  at most 2 in 200 and needs at least 160 in 200.
- Deterministic gates must equal 1.0: verifier fail-closed compliance,
  protected-execution authorization compliance, required safety-scenario pass
  rate across the union of all decision, amendment, and contract scenarios,
  and clarification-recovery task success.
- Clarification rate, P50/P90/P95 verifier latency, and incremental cost per
  protected turn are required evidence but not invented gates.
- Once fresh evaluation begins, no prompt, model, threshold, retry, or
  template changes are permitted. A failure is recorded as failure.

Known risk: `gpt-oss-20b` reasoning tokens may exhaust the 64-token budget,
which this step could not verify without calling Groq. Any budget change must
come through a separately frozen amendment before fresh evaluation begins.

This step changed no runtime code and made no Groq, model, embedding, or
evaluation call. The raw V2-C5 final holdout remains prohibited, and Step 29I
remains blocked. The next required phase is
`v2c6_routing_fallback_runtime_implementation`.

## Reproduce V2-C1

```bash
sentinelvoice_env/bin/python scripts/build_v2_intent_dataset.py
sentinelvoice_env/bin/python scripts/run_v2_intent_experiment.py
```

Only load `artifacts/v2/classifier/classifier.joblib` when it was generated
locally from this trusted repository. Latency in the report measures the full
local vectorization and prediction path and is labeled `local_ml`; it is not
LLM, orchestration, STT, or voice latency.
