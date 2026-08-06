from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable

from kannaadi.domain import ArchitectureGraph, ModelSpec


class ModelAdapter(ABC):
    """Backend-neutral contract used by the API and experiment engine."""

    @abstractmethod
    def load(self, spec: ModelSpec, progress: Callable[[str, str], None] | None = None) -> None: ...

    @abstractmethod
    def architecture(self) -> ArchitectureGraph: ...

    @abstractmethod
    def tokenize(self, prompts: Iterable[str]) -> Any: ...

    @abstractmethod
    def run(self, tokens: Any, requested_activations: Iterable[str] | None = None) -> Any: ...

    @abstractmethod
    def install_intervention(self, spec: Any) -> Any: ...

    @abstractmethod
    def supported_activation_points(self) -> list[str]: ...

    @abstractmethod
    def estimate_memory(self, request: dict[str, Any]) -> int: ...
