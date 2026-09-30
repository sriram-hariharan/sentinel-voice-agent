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

## Roadmap: V2-C4 → V2-C5 → V2-D

V2-C5, **Intent Discovery and Taxonomy Expansion**, sits between the current
V2-C4 nine-intent safety-recovery experiment and the separately governed V2-D
phase. Its purpose is to discover whether the taxonomy should expand, not to
assume that it must.

V2-C5 may use only development- or training-eligible sources. Frozen sentence
embeddings with HDBSCAN provide advisory candidate-cluster discovery; a human
adjudicator, not clustering, decides whether any cluster becomes an intent.
There is no fixed target intent count. CFPB narratives, annotations, labels,
and metadata-derived targets are excluded from taxonomy discovery and training.
An expanded taxonomy requires a new fresh final holdout, because the V2-C4
nine-intent holdout is not valid final evidence for a changed label space.
Additional recognized intents do not automatically imply additional runtime
tools. These constraints continue to govern the implemented discovery steps.

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

## Reproduce V2-C1

```bash
sentinelvoice_env/bin/python scripts/build_v2_intent_dataset.py
sentinelvoice_env/bin/python scripts/run_v2_intent_experiment.py
```

Only load `artifacts/v2/classifier/classifier.joblib` when it was generated
locally from this trusted repository. Latency in the report measures the full
local vectorization and prediction path and is labeled `local_ml`; it is not
LLM, orchestration, STT, or voice latency.
