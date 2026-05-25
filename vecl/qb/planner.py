from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from vecl.qb.specialist import Specialist, SpecialistRequest


@dataclass(frozen=True)
class ChainStep:
    step_id: str
    specialist_id: str
    inputs_from: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    expected_artifact_type: str = ""

    def __post_init__(self) -> None:
        if not self.step_id or not self.specialist_id:
            raise ValueError("step_id and specialist_id must be non-empty")
        object.__setattr__(self, "inputs_from", tuple(self.inputs_from))
        object.__setattr__(self, "parameters", dict(self.parameters))


@dataclass(frozen=True)
class ChainPlan:
    plan_id: str
    task_type: str
    steps: tuple[ChainStep, ...]

    def __post_init__(self) -> None:
        if not self.plan_id or not self.task_type:
            raise ValueError("plan_id and task_type must be non-empty")
        if not self.steps:
            raise ValueError("chain plan must contain at least one step")
        object.__setattr__(self, "steps", tuple(self.steps))
        _validate_step_ids(self.steps)
        topological_steps(self)


class ChainPlanner:
    def __init__(self, templates: Mapping[str, ChainPlan] | None = None) -> None:
        self._templates = dict(templates or {})

    def register_template(self, task_type: str, plan: ChainPlan) -> None:
        if task_type != plan.task_type:
            raise ValueError("template task_type must match plan task_type")
        self._templates[task_type] = plan

    def plan(
        self,
        request: SpecialistRequest,
        specialists: Mapping[str, Specialist] | Iterable[str] | None = None,
    ) -> ChainPlan:
        template = self._templates.get(request.task_type)
        if template is None:
            raise ValueError(f"no chain template registered for task_type: {request.task_type}")
        if specialists is not None:
            available = (
                set(specialists.keys()) if isinstance(specialists, Mapping) else set(specialists)
            )
            missing = sorted(
                {
                    step.specialist_id
                    for step in template.steps
                    if step.specialist_id not in available
                }
            )
            if missing:
                raise ValueError(f"chain template references unregistered specialists: {missing}")
        return template


def topological_steps(plan: ChainPlan) -> tuple[ChainStep, ...]:
    order = {step.step_id: index for index, step in enumerate(plan.steps)}
    remaining = {step.step_id: step for step in plan.steps}
    completed: set[str] = set()
    result: list[ChainStep] = []
    missing = sorted(
        {
            dependency
            for step in plan.steps
            for dependency in step.inputs_from
            if dependency not in remaining
        }
    )
    if missing:
        raise ValueError(f"chain step depends on unknown step_id: {missing}")
    while remaining:
        ready = [
            step
            for step in remaining.values()
            if all(dependency in completed for dependency in step.inputs_from)
        ]
        if not ready:
            raise ValueError("chain plan contains a dependency cycle")
        ready.sort(key=lambda step: order[step.step_id])
        for step in ready:
            result.append(step)
            completed.add(step.step_id)
            del remaining[step.step_id]
    return tuple(result)


def _validate_step_ids(steps: tuple[ChainStep, ...]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for step in steps:
        if step.step_id in seen:
            duplicates.add(step.step_id)
        seen.add(step.step_id)
    if duplicates:
        raise ValueError(f"duplicate chain step_id: {sorted(duplicates)}")
