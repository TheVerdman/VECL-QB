from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ScheduledSleepCycle:
    tenant_id: str
    run_at: datetime
    policy_version: str


class SleepScheduler:
    def __init__(self) -> None:
        self._scheduled: list[ScheduledSleepCycle] = []

    def schedule(self, cycle: ScheduledSleepCycle) -> None:
        self._scheduled.append(cycle)

    def due(self, now: datetime, tenant_id: str | None = None) -> list[ScheduledSleepCycle]:
        cycles = [cycle for cycle in self._scheduled if cycle.run_at <= now]
        if tenant_id is not None:
            cycles = [cycle for cycle in cycles if cycle.tenant_id == tenant_id]
        return sorted(
            cycles, key=lambda cycle: (cycle.run_at, cycle.tenant_id, cycle.policy_version)
        )
