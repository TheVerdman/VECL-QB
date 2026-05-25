from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vecl.qb.specialist import Specialist, SpecialistRequest


@dataclass(frozen=True)
class SpecialistCard:
    specialist_id: str
    supported_task_types: set[str]
    trust_requirements: dict[str, Any]
    cost_hint: float
    latency_hint: float
    version: str
    effective_trust: float = 0.0
    description: str = ""


class QBRouter:
    def __init__(self, max_specialists: int | None = None) -> None:
        self.max_specialists = max_specialists
        self._registry: dict[str, tuple[SpecialistCard, Specialist]] = {}

    def register_specialist(self, card: SpecialistCard, specialist: Specialist) -> None:
        self._registry[card.specialist_id] = (card, specialist)

    @property
    def specialists(self) -> dict[str, Specialist]:
        return {
            specialist_id: specialist
            for specialist_id, (_card, specialist) in self._registry.items()
        }

    @property
    def cards(self) -> dict[str, SpecialistCard]:
        return {
            specialist_id: card for specialist_id, (card, _specialist) in self._registry.items()
        }

    def route(
        self, request: SpecialistRequest, max_specialists: int | None = None
    ) -> list[Specialist]:
        candidates = [
            (card, specialist)
            for card, specialist in self._registry.values()
            if request.task_type in card.supported_task_types and specialist.can_handle(request)
        ]
        ranked = sorted(
            candidates,
            key=lambda item: (-item[0].effective_trust, item[0].cost_hint, item[0].specialist_id),
        )
        limit = max_specialists if max_specialists is not None else self.max_specialists
        specialists = [specialist for _, specialist in ranked]
        return specialists if limit is None else specialists[:limit]
