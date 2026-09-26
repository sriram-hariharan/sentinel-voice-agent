# SentinelVoice V2 deterministic banking data

This directory contains a larger synthetic add-on used to make evaluation more
realistic and difficult. It is fully generated, contains no real customer or
banking data, and is domain-realistic rather than statistically representative
of any bank population.

The canonical V1 fixture at `data/fixtures/banking.json` remains immutable. V2
customers, accounts, cards, transactions, and disputes use separate stable IDs;
the generator never adds resources or transactions to Avery or another V1
customer.

## Reproduce the artifacts

From the repository root:

```bash
sentinelvoice_env/bin/python scripts/generate_v2_banking_data.py
```

The versioned configuration is
`data/generation/v2/banking_generation_config.json`. Version 1 uses seed
`20260926` and reference time `2026-09-26T12:00:00-04:00`. A local
`random.Random` instance, UUIDv5 identifiers, `Decimal` money, deterministic
ordering, and stable JSON serialization make repeated generation byte-identical.

`banking.generated.manifest.json` records configuration and generator versions,
record counts, provenance, and SHA-256 hashes for the configuration, generated
fixture, annotation sidecar, and immutable V1 fixture.

`suspicious_patterns.json` is evaluation metadata for deliberately constructed
synthetic patterns. It is not authoritative banking state, is not consumed by
the current SentinelVoice agent, and must not be interpreted as fraud labels or
fraud ground truth.

## Seed profiles

The default remains the exact V1 behavior:

```bash
sentinelvoice_env/bin/python scripts/seed_database.py --reset
```

The larger profile is explicit and composes the unchanged V1 fixture with the
separate generated add-on:

```bash
sentinelvoice_env/bin/python scripts/seed_database.py --reset --profile v2
```
