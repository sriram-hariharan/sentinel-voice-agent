# SentinelVoice V2-C2 external dataset qualification

V2-C2A records provenance and intended use only. It does not download a
dataset, run an external benchmark, retrain a model, alter the frozen V2-C1
classifier artifact, or integrate classification into runtime behavior.

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
  through August 14, 2026. It is useful for language robustness, ambiguity,
  and multi-intent stress testing, not direct intent accuracy or automatic
  training labels. Complaints are one-sided consumer allegations and are not
  independently verified. The archive is not representative of all consumer
  experiences.
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
