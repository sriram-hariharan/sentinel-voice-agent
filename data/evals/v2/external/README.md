# SentinelVoice V2-C2 external dataset qualification

V2-C2A records provenance and intended use only. It does not download a
dataset, run an external benchmark, retrain a model, alter the frozen V2-C1
classifier artifact, or integrate classification into runtime behavior.

## Data-governance boundary

The SentinelVoice application and demo use synthetic identities, accounts,
cards, transactions, disputes, authentication records, and session data only.
No public external corpus may become application customer/account data or be
loaded into the synthetic banking backend.

Offline evaluation may separately use synthetic, hybrid synthetic,
crowdsourced human-written, or publicly released and de-identified real
consumer-authored datasets. A real-world source must be public and trustworthy,
have documented provenance and reviewed use or redistribution terms, and have
been de-identified or intentionally released for public use. SentinelVoice
must not use private, leaked, proprietary, or scraped customer records, and
must not intentionally retain credentials, account secrets, or PII. Raw source
material stays segregated from synthetic application data. This permission is
for offline evaluation only; training on real consumer-derived data requires a
separate explicit decision.

## Why provenance is separate

External text can differ in origin, consent, privacy risk, label meaning, and
license. Keeping a separate source register prevents a public benchmark,
government complaint archive, synthetic vendor corpus, or rumored bank log
from being described as equivalent evidence. A source is not usable merely
because a copy appears online: its publisher, official location, license,
collection method, and privacy boundary must be documented first.

External evaluation comes before any retraining. The frozen V2-C1 model should
first be measured without adapting to the new source. That gives an honest
generalization result. Training on the evaluation source first would erase that
baseline and make it impossible to distinguish transfer from memorization or
source-specific tuning. Every source currently has
`allowed_for_future_training: false`; training would require a new, explicit
review.

## Qualified evaluation candidates

- [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data)
  is the primary external banking intent benchmark: 13,083 English queries,
  77 intents, and a CC BY 4.0 license. Its labels require an explicit mapping
  to SentinelVoice, and it is not a collection of released bank calls.
- [CFPB Consumer Complaint Database Narratives Archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/)
  contains real consumer-written financial complaints previously published
  through August 14, 2026. CFPB states that its published complaint data is
  freely available to use and analyze. Public narratives required consumer
  publication consent and underwent personal-information scrubbing intended to
  minimize, not eliminate, re-identification risk. The narratives are
  one-sided, unverified consumer allegations and are not a representative
  sample. SentinelVoice permits them only for offline language robustness,
  intent, ambiguity, and out-of-distribution evaluation—not model training or
  automatic labels.
- [Bitext Retail Banking](https://huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset)
  has approximately 25,545 rows under CDLA-Sharing-1.0. Its official card calls
  it a hybrid synthetic dataset generated with NLP/NLG and automated labeling,
  so it is a secondary cross-source benchmark rather than real-world evidence.
- [CLINC150/OOS](https://github.com/clinc/oos-eval) is a crowdsourced,
  general-domain benchmark under CC BY 3.0. Its explicit OOS partitions are
  useful for `unsupported_or_uncertain` evaluation, but its 150 in-scope labels
  are not banking labels and must not be merged directly with SentinelVoice.

## Rejected unverified “bank logs”

No official publicly released de-identified customer-support conversation logs
were verified for Bank of America, JPMorgan Chase, or Wells Fargo. The source
register therefore gives these candidates no URL or license and marks them
`not_found_officially`. Search results about customer service, research using
aggregate or de-identified account activity, video transcripts, scraped chat
examples, or third-party mirrors do not establish an official support-log
release. Such material remains unusable unless the bank publishes an official
source with adequate provenance, disclosure authority, de-identification, and
license terms.

## Metric boundary

All future external-source results must be reported per source and must remain
separate from V2-C1 controlled-synthetic train, validation, and locked-test
metrics. CFPB robustness observations must also remain separate from BANKING77
intent metrics because the sources have different units, labels, collection
processes, and intended uses. V2-C1 remains frozen and retains no runtime
authority.

`dataset_sources.json` is the machine-readable qualification register. Its
`allowed_for_initial_evaluation` field means only that a later phase may design
and approve an evaluation protocol; it does not authorize downloading during
V2-C2A.

## CFPB offline-evaluation qualification

The [public database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
permits analysis of published complaint data, while the
[narrative publication policy](https://www.consumerfinance.gov/complaint/data-use/)
documents that directly identifying information is not published. CFPB's
[scrubbing standard](https://files.consumerfinance.gov/f/documents/cfpb_narrative-scrubbing-standard_2023-05.pdf)
and privacy assessment describe opt-in publication consent and steps intended
to remove personal information; these controls reduce but do not eliminate
privacy risk. The official
[archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/)
covers previously published narratives received through August 14, 2026.

CFPB cautions that complaint narratives are unverified, one-sided allegations
and not a representative sample of consumer experience. SentinelVoice will not
use them to rank or judge financial institutions. Any future approved use is
limited to offline language robustness, intent, ambiguity, or OOD evaluation,
with metrics kept separate from synthetic and other external sources. The data
is not assumed to be perfectly anonymized or risk-free, and future model
training remains prohibited pending a separate review. V2-C2K changes policy
only: it downloads no CFPB data and runs no model evaluation.

## BANKING77 taxonomy mapping

V2-C2B qualifies the canonical BANKING77 taxonomy before any utterance data is
downloaded. `banking77_intent_mapping.json` records all 77 labels from PolyAI's
official `categories.json` at repository revision
`57ec275d8078af65b7731c2a98be812d844a6d6b`. The taxonomy file was retrieved on
September 26, 2026, and has SHA-256
`53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63`.

The qualification finds 11 exact matches, 13 near matches, 43 unsupported
categories, and 10 ambiguous categories. Exact and near candidate coverage is
recorded by SentinelVoice intent in the mapping artifact. A near match is not
clean evaluation data, and ambiguous entries require utterance-level review.
In particular, a source label about a lost, compromised, duplicate, or
unrecognized payment/card never authorizes `freeze_card` or `create_dispute`.
Protected writes still require explicit utterance semantics and the existing
deterministic confirmation boundary.

The unsupported taxonomy areas cluster around card issuance and delivery,
top-ups, transfers and beneficiaries, ATM/cash issue resolution, refunds,
identity/profile/account administration, and currency-conversion operations.
These are capability-gap observations only, not proposed intents or tools.
V2-C1 artifacts remain frozen, and no classifier training, evaluation, or
runtime integration occurs in V2-C2B.

## BANKING77 frozen raw data

V2-C2C stores the official `categories.json`, `train.csv`, and `test.csv`
bytes from PolyAI repository revision
`57ec275d8078af65b7731c2a98be812d844a6d6b` under `raw/banking77/`. The raw
files are immutable source material: do not normalize, clean, rewrite, or
relabel them in place. Their byte sizes, SHA-256 hashes, row counts, taxonomy
hash, and validation results are recorded in `raw/banking77/manifest.json`.

V2-C2C did not run classifier evaluation or model training against BANKING77.
Any mapping, sampling, or other transformation for evaluation must write to a
separate processed-data directory so the pinned source bytes remain unchanged
and independently verifiable.

## BANKING77 processed evaluation data

V2-C2D deterministically builds the external evaluation artifacts from the
canonical BANKING77 `test.csv` split only. The `train.csv` split is not used;
it remains reserved for a separately reviewed future training or augmentation
experiment so the test split stays a clean zero-shot external benchmark.

The processed data keeps two scored lanes separate. `EXACT_MATCH` examples
form the clean intent benchmark, while `UNSUPPORTED` examples form a distinct
banking unsupported/OOS benchmark with expected intent
`unsupported_or_uncertain`. `NEAR_MATCH` and `AMBIGUOUS` examples remain in an
unscored review pool without forced expected labels. The generated manifest
records input and output hashes, partition counts, coverage, and deterministic
build metadata. V2-C2D did not run BANKING77 classifier predictions, model
evaluation, model training, or runtime integration.

## First frozen-model external evaluation

V2-C2E evaluates the unchanged V2-C1 classifier artifact on BANKING77 without
training, retraining, remapping, or threshold tuning. The official primary
metrics use the frozen Linear SVM intent model. The exact-match intent lane and
unsupported/OOS banking lane are reported separately, and the near-match and
ambiguous review pool remains unscored.

The existing Logistic Regression probability path and validation-selected
abstention rule are reported separately as a frozen advisory abstention
analysis; they are not blended with, and do not validate, the SVM predictions.
The two-class external exact-match macro-F1 is not directly interchangeable
with the V2-C1 nine-intent locked-test macro-F1. All classifier outputs remain
offline advisory evidence with no runtime or authorization authority.

## CLINC150/OOS frozen raw data

V2-C2F pins the official CLINC OOS evaluation repository at commit
`828f8093932c8fe6ca7936c3d2e52903b1c523de` and stores its canonical
`data/data_full.json` plus the repository's CC BY 3.0 license under
`raw/clinc_oos/`. These files are immutable; their byte sizes, hashes, split
structure, and verified counts are recorded in `raw/clinc_oos/manifest.json`.

The CLINC OOS benchmark is interested only in the `oos_train`, `oos_val`, and
`oos_test` examples as non-banking `unsupported_or_uncertain`
safety/generalization data; it does not map CLINC's 150 ordinary intents into
that OOS dataset. A separately qualified finance subset is documented below.
No model training, threshold tuning, or classifier evaluation was run in this
raw-data phase. Any processed OOS evaluation set must be created separately
under `processed/clinc_oos/`.

## CLINC OOS processed evaluation data

V2-C2G deterministically extracts only the official 1,000-example `oos_test`
split into `processed/clinc_oos/`. The `oos_train` and `oos_val` splits remain
reserved, and none of CLINC's 150 in-scope intent examples are included. Every
processed example has expected SentinelVoice intent
`unsupported_or_uncertain`, making this a focused external test of rejecting
unrelated, non-banking language. No classifier predictions, model training, or
threshold tuning have been run on the processed CLINC dataset yet.

## CLINC finance-intent qualification

V2-C2H keeps the CLINC OOS benchmark unchanged and separately qualifies the
30 intents in CLINC's official `banking` and `credit_cards` domains. The
finance domains come from `data/domains.json` at the same pinned CLINC revision,
`828f8093932c8fe6ca7936c3d2e52903b1c523de`; they were not selected by keyword
matching. `clinc_finance_intent_mapping.json` records the source taxonomy hash,
the deterministic example-review method, and conservative user-goal mappings
to the existing SentinelVoice intent taxonomy.

This qualification does not add finance examples to the existing OOS set. It
may support a separate external banking benchmark in a later phase, after a
processed-data protocol is reviewed. Mappings are frozen before any classifier
evaluation, and ambiguous protected-action categories such as reporting fraud
or a lost card are not treated as authorization to dispute a transaction or
freeze a card. No predictions, model changes, training, or runtime integration
occur in V2-C2H.

## CLINC finance processed evaluation data

V2-C2I deterministically selects only the 900 examples belonging to the 30
qualified finance intents from CLINC's official `test` split. The `train`,
`val`, and all OOS splits remain excluded, as do non-finance test examples.
Frozen `EXACT_MATCH` and `UNSUPPORTED` mappings become separate scored lanes;
their metrics must not be combined. `NEAR_MATCH` and `AMBIGUOUS` examples stay
in an unscored review pool without forced expected labels.

The CLINC OOS processed evaluation remains a separate benchmark for unrelated,
non-banking language. V2-C2I runs no classifier predictions, training,
retraining, threshold tuning, or runtime integration.

## Frozen-model CLINC finance evaluation

V2-C2J evaluates the unchanged V2-C1 classifier on only the two scored CLINC
finance lanes, without retraining, remapping, or threshold tuning. The frozen
Linear SVM is the primary classifier. Exact-match metrics and unsupported
recall are reported separately, and the 180-example near/ambiguous review pool
remains unscored.

The frozen Logistic Regression probability path and its validation-selected
0.2 margin abstention rule are analyzed separately from the primary SVM. CLINC
finance results are external-source metrics and are not directly
interchangeable with the nine-intent V2-C1 controlled-synthetic locked-test
metrics. The evaluation remains offline and advisory only, with no runtime or
authorization authority.

## CFPB complete narratives-archive acquisition and profile

V2-C2L acquires and profiles all 21 official Full Records ZIP partitions from
the CFPB Consumer Complaint Database Narratives Archive. The archive declares
coverage from December 1, 2011 through August 14, 2026. The Full Records files
contain complaint rows received through August 31, 2026; this later record date
does not change the archive's declared narrative-publication cutoff. The raw
collection contains real consumer-authored complaint data and remains strictly
separate from SentinelVoice's synthetic application data.

This is recorded as a source-metadata discrepancy and qualification note:
`official_archive_declared_narrative_coverage_end` is `2026-08-14`, while
`observed_full_records_date_received_max` is `2026-08-31`. August 31 is not
claimed as the archive page's narrative-coverage end, and records received
after August 14 are retained without alteration or filtering.

The large ZIP payloads stay local and Git-ignored. Reproducibility comes from
the official source page and per-partition URLs, byte sizes, SHA-256 hashes,
member inventories, row counts, and date ranges in `raw/cfpb_narratives/manifest.json`,
plus the acquisition and profiling scripts. `raw/cfpb_narratives/profile.json`
contains deterministic aggregate statistics over the complete archive,
including structured CFPB product and issue distributions, broad
SentinelVoice-relevance groups, missingness, duplicate counts, narrative-length
statistics, and aggregate privacy-pattern signals. It retains no strings
matched by the privacy checks.

This phase performs acquisition and profiling only. It assigns no
SentinelVoice intent labels, runs no classifier predictions or evaluation,
trains no model, changes no classifier artifact, and adds no CFPB data to
application or runtime code. Any later intent labeling or evaluation must use
separately generated processed data and preserve the external-source boundary.

CFPB complaints are unverified, one-sided allegations and are not a
representative sample of consumers or institutions. Complaint counts must not
be used to rank companies. CFPB's publication and scrubbing controls reduce,
but do not eliminate, residual privacy risk, so raw narratives remain local and
must be handled as real consumer-authored data.

## CFPB structured-taxonomy mapping

V2-C2M inventories every Product/Sub-product/Issue/Sub-issue combination
observed among the complete archive's narrative-bearing rows and freezes a
hierarchical mapping before any classifier inference or narrative sampling.
The inventory retains low-frequency combinations and reports product,
product/issue, product/sub-product/issue, and full-tuple marginal counts.

The mapping is deliberately conservative because CFPB taxonomy describes the
product and issue represented by a complaint, not the exact conversational
action a SentinelVoice user is currently requesting. Complaint topics such as
fraud, an unauthorized transaction, a lost or stolen card, a purchase problem,
or dispute history do not authorize `freeze_card` or `create_dispute` and are
not exact protected-write labels. No structured rule maps directly to either
protected-write intent as `EXACT_MATCH`; explicit action semantics must be
established later from narrative text and remain subject to deterministic
authorization and confirmation controls.

CFPB narratives are often long-form and retrospective, may describe multiple
events or actions a company already took, and may contain multiple intents or
requested remedies. A later phase must keep taxonomy-derived scored examples,
narrative-semantic review examples, multi-intent/ambiguity robustness examples,
and unsupported/OOD examples separate. V2-C2M performs no narrative-level
classification, creates no final evaluation sample, runs no model prediction,
and trains no model.

## CFPB narrative-candidate planning

V2-C2N adds a deterministic aggregate planner for the complete CFPB narrative
corpus. It uses the frozen V2-C2M full-taxonomy assignments without promoting
near or ambiguous complaints to exact intent labels. The planner writes no
narrative text, complaint IDs, or individual narrative hashes to its output and
does not materialize another narrative dataset.

The future candidate plan summarizes exact duplicates within and across mapping
lanes, narrative lengths, privacy-screening signals, structured taxonomy
distributions, overlapping frozen candidate-intent counts, and several
deduplicated eligibility views. These aggregates will guide a later sampling
and semantic-review design; V2-C2N itself selects no final evaluation sample,
assigns no narrative-level SentinelVoice intent, runs no model inference, and
uses no CFPB data for training.

## CFPB narrative review-pool materialization

V2-C2O defines a deterministic, diversity-aware review/evaluation-candidate
pool with default sizes of 600 `NEAR_MATCH`, 1,200 `AMBIGUOUS`, and 2,000
`UNSUPPORTED` narratives. Eligibility is privacy-screened with the existing
CFPB patterns and globally deduplicated by exact narrative SHA-256. When the
same narrative occurs in multiple mapping lanes, ownership follows the frozen
precedence `AMBIGUOUS` > `NEAR_MATCH` > `UNSUPPORTED` > `EXACT_MATCH`.

The two semantic-review lanes target a 35% / 30% / 25% / 10% mix across
`<=500`, `501-1000`, `1001-2000`, and `>2000` characters. Deterministic quota
reallocation handles unavailable strata, while an approximately 20% issue cap
limits dominance unless relaxing it is necessary to fill a lane. Selection
also rotates across frozen overlapping candidate-intent sets, products,
issues, and source archives. Unsupported selection instead uses deterministic
round-robin coverage across products and product/issue/sub-issue/length
strata.

The trackable `processed/cfpb/review_pool_manifest.json` contains complaint
IDs, narrative hashes, CFPB taxonomy, frozen candidate intents, provenance,
selection metadata, and aggregate summaries, but no narrative text. Real
consumer-authored text is written only to the ignored local file
`processed/cfpb/local/cfpb_review_pool.jsonl`. This pool is not the final CFPB
evaluation set: V2-C2O assigns no final intent labels, runs no model inference,
performs no training, and defers semantic review to the next phase.

## CFPB Codex-assisted dual-pass annotation and adjudication

V2-C2P originally defined purely manual independent review for the 600
`NEAR_MATCH` and 1,200 `AMBIGUOUS` records. The methodology now used is
"Codex-assisted independent dual-pass annotation with Codex adjudication."
Pass A independently annotates all 1,800 records. Pass B independently
re-annotates every record selected by the existing review-required logic,
including the deterministic 10% QC sample. Pass C adjudicates every completed
A/B pair that cannot be safely resolved by exact agreement. These labels are
not purely human ground truth.

The exact narrative hashes remain a protected external-evaluation holdout.
Neither narratives nor annotations may be used for training, feature
selection, hyperparameter or threshold tuning, or model selection. Any future
CFPB training data must use disjoint hashes. The separate 2,000-record
`UNSUPPORTED`/OOD lane is not annotated in this phase.

Reviewers assign one of `SINGLE_SUPPORTED_INTENT`,
`MULTI_SUPPORTED_INTENT`, `UNSUPPORTED`, `UNCLEAR_OR_INSUFFICIENT`, or
`NO_CURRENT_REQUEST`. Multi-intent annotations retain every independently
present supported intent rather than forcing a single primary label.
`freeze_card` and `create_dispute` require explicit present action requests;
fraud, loss, theft, unauthorized activity, or prior disputes alone are not
protected-write labels.

Pass A, Pass B, Pass C, and any residual human reviewer must not receive V2-C1
predictions, SVM scores, logistic predictions or probabilities, margins,
abstention results, or classifier evaluation artifacts before labels and
scoring rules are frozen. This is LLM-assisted annotation, not
classifier-assisted labeling. CFPB taxonomy and frozen candidate intents may
be shown only as nonbinding context.

The ignored local stores are:

- `processed/cfpb/local/cfpb_llm_first_pass.jsonl` for Pass A;
- `processed/cfpb/local/cfpb_codex_second_pass.jsonl` for Pass B; and
- `processed/cfpb/local/cfpb_codex_adjudication.jsonl` for Pass C.

Narrative-bearing batches also remain under the ignored `local/` tree. All
imports validate source linkage and hashes, reject duplicates, preserve source
order, requeue invalid or missing results, prevent implicit replacement of a
successful result, and use flush/fsync plus atomic replacement. None of these
commands writes the canonical review workfile or freezes final labels.

Pass B membership is derived from Pass A's `human_review_required` field; no
record count is hardcoded. Its prepared rows are rebuilt from a source-only
allowlist and expose no Pass A category, intents, confidence, note, or
secondary-review field. They also exclude classifier-derived fields. Pass B
uses `CODEX_SECOND_PASS` provenance and remains resumable by successful
`narrative_sha256`, including across sessions or accounts.

A completed A/B pair has strong agreement only when category and normalized
supported intents match exactly, neither confidence is `LOW`, and both results
are valid. It is a safe provisional final only when neither pass is unclear,
multi-intent, `freeze_card`, `create_dispute`, or explicitly marked for
secondary review. Disagreement or any of those risks routes the record to Pass
C. Invalid or missing Pass-B rows remain pending and are requeued rather than
prematurely adjudicated.

Pass C receives the original source context, concise Pass A and Pass B
annotations and provenance, and deterministic adjudication reasons. It receives
no chain-of-thought and no classifier information. `CODEX_ADJUDICATOR` may
select A, select B, produce a corrected third label, or return `UNRESOLVED` with
a concise reason. A valid `RESOLVED` result requires a complete category,
intent, confidence, and note contract. Successful Pass-C results are hash-
resumable and cannot be overwritten without an explicit replacement option.

The standard-library residual reviewer CLI remains at
`scripts/review_cfpb_semantic_annotations.py`. It is not Pass B and it is not
Pass C. If genuine `UNRESOLVED` rows remain after Pass C, they remain in the
1,800-record holdout and cannot be exported as finalized. The completed
workflow has zero unresolved rows, so no residual human adjudication was used
and no record was dropped.

The Codex workflow is coordinated by
`scripts/annotate_cfpb_semantic_with_codex.py`. Existing Pass-A commands remain
backward compatible:

```bash
python scripts/annotate_cfpb_semantic_with_codex.py status
python scripts/annotate_cfpb_semantic_with_codex.py prepare --limit 25
python scripts/annotate_cfpb_semantic_with_codex.py import \
  --batch <batch.jsonl> --annotations <results.jsonl>
```

Pass B, comparison, Pass C, and end-to-end status use:

```bash
python scripts/annotate_cfpb_semantic_with_codex.py second-pass status
python scripts/annotate_cfpb_semantic_with_codex.py second-pass prepare --limit 25
python scripts/annotate_cfpb_semantic_with_codex.py second-pass import \
  --batch <pass-b-batch.jsonl> --annotations <pass-b-results.jsonl>
python scripts/annotate_cfpb_semantic_with_codex.py compare
python scripts/annotate_cfpb_semantic_with_codex.py adjudication prepare --limit 25
python scripts/annotate_cfpb_semantic_with_codex.py adjudication import \
  --batch <pass-c-batch.jsonl> --annotations <pass-c-results.jsonl>
python scripts/annotate_cfpb_semantic_with_codex.py workflow-status
```

All prepare commands default to 25 rows and permit `--limit` up to 200. The
script invokes no model or external API. `compare` emits deterministic counts;
`workflow-status` reports Pass A completion, Pass B required/complete/pending,
strong and safe agreement, Pass C required/resolved/unresolved/pending,
currently available provisional labels, and genuinely unresolved human-review
remaining. An optional local preview is text-free and records whether each
available provisional label came from Pass A, safe A/B agreement, or Pass C.
It is not a final frozen export.

## CFPB final semantic-label freeze and scoring contract

The tracked final artifact is
`processed/cfpb/cfpb_semantic_final_labels.jsonl`. It is generated and checked
by `scripts/export_cfpb_semantic_final_labels.py`:

```bash
python3 scripts/export_cfpb_semantic_final_labels.py --write
python3 scripts/export_cfpb_semantic_final_labels.py --check
```

The exporter makes no new semantic decision. For a record that never required
Pass B it uses the valid Pass-A annotation. For a record with a safe exact A/B
agreement it uses the agreed semantics. For a record routed to Pass C it
requires a successful `RESOLVED` decision and uses the Pass-C final semantics.
Missing, invalid, or `UNRESOLVED` required decisions make export fail.

The JSONL contains only versioned identity, source stratum, final semantic
category/intents/confidence, resolution provenance, Pass-B/Pass-C routing
flags, and the frozen primary scoring target. It contains no narrative,
complaint text, annotation/adjudication note, unresolved reason, classifier
prediction, score, probability, margin, abstention result, or V2-C1 evaluation
field. Rows preserve deterministic frozen source order, keys are serialized
deterministically, and writes use flush/fsync plus atomic replacement.

The exporter validates exactly 1,800 unique complaint IDs and narrative
hashes, complete one-to-one Pass-A/Pass-B/Pass-C coverage where required,
source linkage, the five-category contract, sorted unique intents from the
frozen eight supported intents, category/intent cardinality, provenance, and
the allowlisted output schema. Repeated runs from identical inputs must produce
identical bytes. Protected-write subsets (`freeze_card` and `create_dispute`)
are derived only from the final frozen semantic intents, never from keywords.

The frozen nine-class single-label scoring mapping is:

- `SINGLE_SUPPORTED_INTENT` -> its sole `final_supported_intents` value;
- `UNSUPPORTED` -> `unsupported_or_uncertain`;
- `UNCLEAR_OR_INSUFFICIENT` -> `unsupported_or_uncertain`;
- `NO_CURRENT_REQUEST` -> `unsupported_or_uncertain`; and
- `MULTI_SUPPORTED_INTENT` -> no forced single-label target.

Multi-intent records remain part of the 1,800-record holdout. They are excluded
only from primary metrics that require one exact target, including nine-class
accuracy and macro-F1. They must be reported separately using
`prediction_is_supported_intent = classifier_prediction in
final_supported_intents`, with multi-intent count, membership-hit count, and
membership accuracy. `unsupported_or_uncertain` is not a hit for a true
multi-intent record because it cannot appear among the frozen eight supported
intents.

The frozen label distribution has 24 multi-intent records. Therefore the
primary single-label subset contains 1,776 records, the excluded count is 24,
and primary coverage is `1776 / 1800 = 0.986666...` (98.6667%). This is scoring
coverage, not holdout deletion. The full export provenance is 1,541 Pass-A-only
labels, 120 safe A/B agreements, and 139 resolved Pass-C labels. Pass C has
zero unresolved records.

Label construction and scoring-contract definition consumed only the source
holdout plus Pass A/B/C stores. They did not read V2-C1 classifier outputs or
evaluation artifacts. That freeze phase claimed no classifier performance;
the subsequent frozen-model evaluation is documented separately below.

## Frozen-model CFPB external evaluation

`scripts/run_cfpb_external_evaluation.py` reproduces the final CFPB
external-generalization baseline and writes deterministic aggregate results to
`results/cfpb/frozen_v2c1_report.json`. Before loading the trusted artifact it
pins the classifier, model-selection report, V2-C1 report, final labels, and
review-pool manifest by SHA-256. It then verifies the manifest-pinned ignored
source pool and joins the exact 1,800 final-label hashes against its 3,800
narratives. The extra 2,000 `UNSUPPORTED` review-pool narratives are not
scored. Complaint IDs, mapping strata, source hashes, label counts, and the
1,776/24 scoring split must all reconcile.

The primary path is the unchanged word/character TF-IDF LinearSVC with
`C=0.5`. On the 1,776 exact single-label records, the supplied frozen local run
produced:

- accuracy `0.399212`;
- nine-class macro-F1 `0.113558`;
- weighted-F1 `0.503277`; and
- primary coverage `1776 / 1800 = 0.986667`.

The 24 `MULTI_SUPPORTED_INTENT` records remain separate: 9 predictions were
members of the frozen supported-intent sets, for `0.375000` membership
accuracy. This remains a single-label classifier and is not presented as a
multi-label solution.

Protected-write gold membership comes only from final semantic intents across
all 1,800 records. `freeze_card` had 1 gold-supported record, 7 predicted
positives, 0 true positives, and 7 false positives. `create_dispute` had 28
gold-supported records (including multi-intent membership), 79 predicted
positives, 7 true positives, and 72 false positives. These are evaluation
observations only and have no authorization or runtime effect.

The separate frozen Logistic Regression probability path (`C=2.0`) uses the
validation-selected confidence threshold `0.0`, top-two margin `0.2`, and no
required intent/risk agreement. On the 1,776 primary records it accepted 183
and abstained on 1,593, giving `0.103041` coverage and `0.896959` abstention.
Unabstained accuracy was `0.503378`; selective accuracy was `0.721311`, and
selective nine-class macro-F1 was `0.111918`. Abstention removed 831 Logistic
errors and 762 correct predictions, leaving 51 errors. It accepted no
`freeze_card` or `create_dispute` predictions. This advisory path neither
replaces nor validates the primary SVM.

The report contains aggregate metrics and integrity metadata only. It emits no
consumer narrative text and no per-record predictions, uses stable sorted JSON
without wall-clock timestamps, and explicitly records that no training or
tuning occurred. CFPB performance shows substantial domain shift and must not
be described as strong generalization or directly compared with the balanced
internal nine-intent test as a like-for-like benchmark. It did not alter the
frozen model, features, taxonomy, hyperparameters, or abstention rule.

Future model-development work may investigate better classical models,
feature changes, embeddings, calibration and OOD methods,
`StratifiedGroupKFold` or other group-aware validation, training augmentation,
and separately governed external training sources. This is not approval to
train on CFPB or other real consumer-derived narratives. Such training requires
a separate explicit V2-C3 data-governance decision, and any approved future
training hashes must remain disjoint from this consumed CFPB holdout.

Reproduction is intentionally explicit and inference-only:

```bash
sentinelvoice_env/bin/python scripts/run_cfpb_external_evaluation.py
```

Before beginning this changed methodology, inspect the canonical workfile with
the reviewer CLI's `--status`. If prior manual annotations must be cleared, the
only supported reset is the explicit command below. It first writes a
timestamped ignored backup, resets only non-`UNREVIEWED` annotation fields, and
reports the count; it never runs as part of batch preparation or import:

```bash
python scripts/review_cfpb_semantic_annotations.py \
  --reset-reviewed \
  --confirm-reset-reviewed RESET_REVIEWED_ANNOTATIONS
```

Legacy Pass-A preparation/import refuses to start while the canonical workfile
still has reviewed rows, preventing pre-methodology labels from being silently
mixed into the provenance model. Status and comparison commands are
non-mutating. The final tracked export and scoring rules are now frozen; V2-C1
evaluation remains a later, separate step and was not run in this phase.
