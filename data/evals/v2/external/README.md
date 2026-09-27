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
