# CFPB semantic-review guidelines

## Scope and holdout boundary

V2-C2P covers only the 600 `NEAR_MATCH` and 1,200 `AMBIGUOUS` records in the
frozen CFPB review pool. Do not annotate the 2,000 `UNSUPPORTED` records in
this phase.

The 1,800 reviewed narratives are an external-evaluation holdout. Their exact
narrative hashes must never be used for training, hyperparameter selection, or
model-selection decisions. They may be used for final external evaluation only
after the annotations are complete and frozen. Any later CFPB training corpus
must use narratives whose exact hashes are disjoint from this holdout.

Reviewers must assign labels without seeing a V2-C1 classifier prediction, SVM
score, logistic-regression probability, or classifier abstention result. CFPB
taxonomy and frozen candidate-intent metadata may be visible as context, but
they are hints rather than labels. Read the narrative itself and make an
independent semantic judgment.

All examples below are synthetic and are not CFPB narratives.

## Review categories

### `SINGLE_SUPPORTED_INTENT`

Use when one clear current request maps to exactly one supported intent.

Synthetic example: “What is my available checking balance?”

```json
{
  "review_category": "SINGLE_SUPPORTED_INTENT",
  "supported_intents": ["account_balance"]
}
```

### `MULTI_SUPPORTED_INTENT`

Use when two or more independently meaningful supported requests are present.
Keep every supported intent and sort the list alphabetically. Do not force a
single primary label.

Synthetic example: “Explain this card charge and freeze my card now.”

```json
{
  "review_category": "MULTI_SUPPORTED_INTENT",
  "supported_intents": ["freeze_card", "transaction_details"]
}
```

### `UNSUPPORTED`

Use when the current request is clear but falls outside SentinelVoice’s eight
supported capabilities. `supported_intents` must be empty.

Synthetic example: “Help me refinance my mortgage.”

### `UNCLEAR_OR_INSUFFICIENT`

Use when there is not enough semantic evidence to decide whether the text has
a supported, unsupported, or actionable request. Do not guess from product or
issue metadata. `supported_intents` must be empty.

Synthetic example: “They changed it again and nobody explained why.”

### `NO_CURRENT_REQUEST`

Use when the text only recounts history, prior contacts, a resolved event, or
background and expresses no meaningful current request SentinelVoice could act
on. `supported_intents` must be empty.

Synthetic example: “Last year the bank replaced my card after I called.”

## Supported intent semantics

The only supported intents allowed in `supported_intents` are:

- `informational_policy`: general policy, product, rule, fee, eligibility, or
  process information that does not require private account-specific state.
- `account_balance`: a request for current balance or available funds.
- `recent_transactions`: a request to view or list recent account activity.
- `transaction_details`: a request for information about a particular charge,
  withdrawal, deposit, merchant transaction, or transaction status/history.
- `card_status`: a request about whether a card is active, blocked, delivered,
  expired, usable, or otherwise its current status.
- `freeze_card`: an explicit present request to freeze, block, or lock the
  caller’s card and stop it from being usable.
- `create_dispute`: an explicit present request to open, file, start, or create
  a dispute or challenge a transaction.
- `escalation`: an explicit request, or clear requested outcome, to speak with
  a human or support representative.

`unsupported_or_uncertain` is not a supported intent for this annotation
workflow. For `UNSUPPORTED`, `UNCLEAR_OR_INSUFFICIENT`, and
`NO_CURRENT_REQUEST`, always use an empty `supported_intents` list.

## Protected-write boundary

`freeze_card` and `create_dispute` require explicit present action semantics.
They must never be inferred only from topic words or CFPB metadata.

Do not assign `freeze_card` merely because a card was lost or stolen, fraud or
an unauthorized charge occurred, or the writer mentions security. Assign it
only when the writer currently and unambiguously asks for the card to be
frozen, blocked, or locked.

Do not assign `create_dispute` merely because the complaint concerns fraud, an
unauthorized transaction, billing disagreement, or a dispute previously filed.
Assign it only when the writer currently and unambiguously asks to initiate or
file a dispute or challenge the transaction.

Synthetic contrast:

- “My card was stolen yesterday.” does not by itself establish `freeze_card`.
- “My card was stolen; block it now.” establishes `freeze_card`.
- “I disputed this charge last month.” does not establish a current
  `create_dispute` request.
- “Open a dispute for this charge.” establishes `create_dispute`.

## Retrospective text and current requests

CFPB narratives often recount long histories. Separate background from what
the writer currently wants. A historical event may supply context without
creating a supported intent. If the narrative includes both history and a
current request, label only the current supported request or requests. If it
contains history alone, use `NO_CURRENT_REQUEST`.

## Escalation

Assign `escalation` when speaking to a human or support representative is an
explicit requested outcome. Dissatisfaction, repeated contacts, or strong
emotion alone does not establish escalation.

Synthetic contrast:

- “Support transferred me three times last week.” is retrospective alone.
- “Connect me to a human representative now.” establishes `escalation`.

## Confidence

- `HIGH`: direct, explicit semantic evidence supports the annotation.
- `MEDIUM`: the interpretation is reasonable but depends on some context.
- `LOW`: material ambiguity remains.

LOW-confidence records should normally set `secondary_review_required` to
`true` and use `NEEDS_ADJUDICATION`. Do not use confidence to replace the
review category.

## Annotation notes

Write a short explanation of why the category and intents apply. Paraphrase;
do not copy long narrative passages, names, account details, contact details,
or other potentially identifying content. Keep notes within 500 characters.

## Adjudication

- `UNREVIEWED`: untouched initialization state; all annotation fields remain
  empty.
- `REVIEWED`: one reviewer completed a valid annotation.
- `NEEDS_ADJUDICATION`: a second reviewer or adjudicator must resolve material
  ambiguity or disagreement; set `secondary_review_required` to `true`.
- `ADJUDICATED`: the final decision was resolved through adjudication.

The normal frozen export requires every record to be `REVIEWED` or
`ADJUDICATED`. Partial export is for workflow inspection only and must remain
explicitly marked partial; it is not a frozen evaluation label set.
