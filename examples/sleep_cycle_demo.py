# ruff: noqa: E402,I001

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vecl.provenance.ledger import ProvenanceLedger
from vecl.runtime.metrics import tenant_crossing_metric
from vecl.runtime.monitor import SparseUpdateMonitor
from vecl.sleep.consolidation import SleepCycleConsolidator
from vecl.sleep.replay import ReplayCandidate, ReplayPolicy


def main() -> None:
    before = np.array([1.0, 1.0])
    candidates = [
        ReplayCandidate(
            "clean-evidence", "clean-source", "sleep-demo", 1.0, 1.0, 1.0, ["c1"], datetime.now(UTC)
        ),
        ReplayCandidate(
            "low-trust-evidence",
            "low-source",
            "sleep-demo",
            10.0,
            10.0,
            0.01,
            ["c2"],
            datetime.now(UTC),
        ),
    ]
    monitor = SparseUpdateMonitor(ProvenanceLedger())
    report = SleepCycleConsolidator(monitor).run_cycle(
        candidates,
        ReplayPolicy(2, 0.5, 0.1, tenant_id="sleep-demo"),
        before,
        learning_rate=0.1,
        min_score=0.1,
    )
    after = (
        monitor.committed_memory_values if monitor.committed_memory_values is not None else before
    )
    print("Sleep cycle report:")
    print(report)
    print("Metrics:")
    print(
        {
            "sparse_update_slots_changed": float(np.count_nonzero(~np.isclose(before, after))),
            "sparse_update_norm": float(np.linalg.norm(after - before)),
            **tenant_crossing_metric(0),
        }
    )


if __name__ == "__main__":
    main()
