# SentinelVoice V2 evaluation contracts

V2-A defines versioned benchmark inputs and result boundaries. It does not
train a classifier, choose a treatment model, route runtime requests, or replay
audio.

## Authority boundary

Intent and risk predictions are advisory metadata only. They must never set or
replace authentication, confirmation, allowed tools, resource ownership, or
customer identity. Deterministic application code remains authoritative.

## Frozen V1 boundary

`data/fixtures/banking.json` is the immutable canonical V1 fixture. Future V2
generated banking records must use separate synthetic customers, accounts,
cards, and transactions. They must not mutate or extend the canonical Avery
records because that would silently change V1 resource-resolution behavior.

## Dataset split boundary

Every intent/risk example has a `group_id`. All examples in one group must use
the same split, preventing paraphrases from the same source family from being
divided between training and evaluation. `locked_test` remains closed during
model and threshold selection.

## Measurement boundary

Every benchmark result declares exactly one `measurement_source`:

- `deterministic_fake`: scripted orchestration evaluation
- `local_ml`: local classifier training or inference
- `live_llm`: an actual LLM provider request
- `live_stt`: an actual STT provider request
- `live_voice`: the configured realtime voice path

Offline fake-provider milliseconds must never be reported as live LLM or voice
latency. Live provider results identify their provider and model and report
sample and failure counts.

## Future experiment gates

Safety invariants are fixed:

- unauthorized protected actions executed must remain zero;
- confirmation compliance must remain 100%;
- all frozen V1 deterministic scenarios must remain passing; and
- shadow classifier integration must not change tool execution or application
  output.

A classifier experiment must beat both the majority baseline and the
deterministic-rule baseline on the locked grouped test set. It must report
macro-F1, per-class metrics, protected-write recall, calibration, and
abstention coverage and accuracy. V2-A intentionally declares no confidence or
quality threshold before a baseline has been measured.

A routing experiment must use paired control and treatment live runs, preserve
safety and required task/tool correctness, and report cost per successful task,
P50/P90/P95 live LLM latency, failures, and sample counts. First measure control
variance. Then predeclare the minimum efficiency improvement and allowed
variance before treatment results are inspected. A threshold chosen after
viewing treatment results is invalid.

The control concept is the frozen V1 `openai/gpt-oss-20b` route. No treatment
model is selected by V2-A.
