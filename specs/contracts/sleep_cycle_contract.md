# Sleep Cycle Contract

Sleep-cycle learning is a prepared replay flow:

1. Build a source-diverse replay batch.
2. Prepare a learning event token.
3. Compute sparse update inputs.
4. Run the sparse update monitor.
5. Commit only if monitor verification passes.
6. Abort without visible committed mutation if verification fails.

Replay excludes quarantined evidence, enforces tenant isolation, applies authority thresholds, and caps single-source flooding.
