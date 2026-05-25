from __future__ import annotations

import importlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only needed by Vertex Python 3.10 package.
    import tomli as tomllib  # type: ignore[no-redef]

from vecl.qb.router import QBRouter, SpecialistCard
from vecl.qb.specialist import Specialist


@dataclass(frozen=True)
class SpecialistDefinition:
    specialist_id: str
    class_path: str
    task_types: set[str]
    trust_anchor_id: str
    cost_hint: float
    latency_hint: float
    version: str


class SpecialistRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, SpecialistDefinition] = {}
        self._specialists: dict[str, Specialist] = {}

    @classmethod
    def load(cls, paths: Iterable[str | Path]) -> SpecialistRegistry:
        registry = cls()
        for path in paths:
            registry.load_file(path)
        return registry

    def load_file(self, path: str | Path) -> None:
        path = Path(path)
        data = _load_config(path)
        for entry in data.get("specialists", []):
            definition = _definition_from_entry(entry)
            specialist = _instantiate_specialist(definition)
            self.register(definition, specialist)

    def register(self, definition: SpecialistDefinition, specialist: Specialist) -> None:
        if definition.specialist_id in self._specialists:
            raise ValueError(f"duplicate specialist id: {definition.specialist_id}")
        self._definitions[definition.specialist_id] = definition
        self._specialists[definition.specialist_id] = specialist

    @property
    def definitions(self) -> dict[str, SpecialistDefinition]:
        return dict(self._definitions)

    @property
    def specialists(self) -> dict[str, Specialist]:
        return dict(self._specialists)

    def specialist(self, specialist_id: str) -> Specialist:
        return self._specialists[specialist_id]

    def cards(self) -> list[SpecialistCard]:
        cards = [
            SpecialistCard(
                specialist_id=definition.specialist_id,
                supported_task_types=set(definition.task_types),
                trust_requirements={"trust_anchor_id": definition.trust_anchor_id},
                cost_hint=definition.cost_hint,
                latency_hint=definition.latency_hint,
                version=definition.version,
            )
            for definition in self._definitions.values()
        ]
        return sorted(cards, key=lambda card: card.specialist_id)

    def register_with_router(self, router: QBRouter) -> None:
        for card in self.cards():
            router.register_specialist(card, self._specialists[card.specialist_id])


def _load_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        loaded = yaml.safe_load(path.read_text()) or {}
    elif suffix == ".toml":
        loaded = tomllib.loads(path.read_text())
    else:
        raise ValueError(f"unsupported specialist registry config: {path}")
    if not isinstance(loaded, dict):
        raise ValueError("specialist registry config must be a mapping")
    return loaded


def _definition_from_entry(entry: dict[str, Any]) -> SpecialistDefinition:
    required = [
        "id",
        "class_path",
        "task_types",
        "trust_anchor_id",
        "cost_hint",
        "latency_hint",
        "version",
    ]
    missing = [key for key in required if key not in entry]
    if missing:
        raise ValueError(f"specialist definition missing fields: {missing}")
    task_types = set(entry["task_types"])
    if not task_types:
        raise ValueError("specialist task_types must be non-empty")
    return SpecialistDefinition(
        specialist_id=str(entry["id"]),
        class_path=str(entry["class_path"]),
        task_types=task_types,
        trust_anchor_id=str(entry["trust_anchor_id"]),
        cost_hint=float(entry["cost_hint"]),
        latency_hint=float(entry["latency_hint"]),
        version=str(entry["version"]),
    )


def _instantiate_specialist(definition: SpecialistDefinition) -> Specialist:
    cls = _load_class(definition.class_path)
    specialist = cls(
        specialist_id=definition.specialist_id,
        task_types=definition.task_types,
        version=definition.version,
    )
    if not isinstance(specialist, Specialist):
        raise TypeError(f"{definition.class_path} did not produce a Specialist")
    return specialist


def _load_class(class_path: str) -> type[Any]:
    module_name, separator, attr_name = class_path.partition(":")
    if not separator:
        module_name, separator, attr_name = class_path.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"invalid class_path: {class_path}")
    module = importlib.import_module(module_name)
    cls = getattr(module, attr_name)
    if not isinstance(cls, type):
        raise TypeError(f"class_path is not a class: {class_path}")
    return cls
