# Provenance Contract

The provenance ledger is append-only. Every state mutation is represented by a new event rather than deletion or in-place replacement.

Each event contains:

- `event_id`
- `event_type`
- `parent_event_ids`
- `timestamp`
- `tenant_id`
- `actor`
- `payload_hash`
- `chain_hash`

The chain hash commits to event order and payload hashes. Quarantine is represented by `SourceQuarantined`; rollback is represented by `RollbackPerformed`.
