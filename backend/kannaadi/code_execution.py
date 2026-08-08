from __future__ import annotations

import ast
import base64
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from io import BytesIO, StringIO
import math
import sys
import threading
import time
import traceback
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class CodeExecutionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    code: str
    cell_id: str | None = Field(alias="cellId", default=None)
    trusted: bool = False
    active_run_id: str | None = Field(alias="activeRunId", default=None)
    selection: list[str] = Field(default_factory=list)


class CodeArtifact(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    kind: Literal["none", "text", "json", "table", "tensor", "image"]
    title: str
    data: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CodeExecutionResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    cell_id: str | None = Field(alias="cellId", default=None)
    status: Literal["complete", "error", "interrupted"]
    stdout: str
    stderr: str
    artifact: CodeArtifact
    error: str | None = None
    traceback: str | None = None
    duration_ms: int = Field(alias="durationMs")
    started_at: str = Field(alias="startedAt")
    finished_at: str = Field(alias="finishedAt")
    namespace_keys: list[str] = Field(alias="namespaceKeys", default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class CodeSessionStatus(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    state: Literal["idle", "running", "interrupting"]
    execution_id: str | None = Field(alias="executionId", default=None)
    started_at: str | None = Field(alias="startedAt", default=None)
    namespace_keys: list[str] = Field(alias="namespaceKeys", default_factory=list)
    trust_model: str = Field(alias="trustModel")


class CodeInterrupted(Exception):
    pass


class KannaadiResearchContext:
    """High-level helpers available as ``kannaadi`` inside Code Lab.

    Raw ``model`` and ``engine`` objects are also exposed for analyses that have
    not yet earned a GUI method. Helpers return ordinary Pydantic objects, tensors,
    dicts, and lists so results can flow back through the artifact serializer.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    @property
    def architecture(self) -> Any:
        return self.engine.architecture

    @property
    def runs(self) -> list[Any]:
        return self.engine.list_runs()

    def run(self, prompt: str, *, label: str | None = None, top_k: int = 10, seed: int = 0) -> Any:
        return self.engine.run_prompt(prompt, label=label, top_k=top_k, seed=seed)

    def get_run(self, run_id: str) -> Any:
        return self.engine.get_run(run_id)

    def cache(self, run_id: str) -> Any:
        return self.engine._artifacts(run_id).cache

    def tokens(self, run_id: str) -> Any:
        return self.engine._artifacts(run_id).tokens

    def logits(self, run_id: str) -> Any:
        return self.engine._artifacts(run_id).logits

    def attention(self, run_id: str, layer: int, head: int) -> Any:
        return self.engine.attention(run_id, layer, head)

    def activation(self, run_id: str, component_id: str) -> Any:
        return self.engine.activation(run_id, component_id)

    def residual_stream(self, run_id: str, *, position: int = -1, target_token_id: int | None = None) -> Any:
        return self.engine.residual_stream(run_id, position=position, target_token_id=target_token_id)

    @staticmethod
    def table(rows: Any, *, title: str = "Code result") -> dict[str, Any]:
        return {"__kannaadi_artifact__": "table", "title": title, "data": rows}

    @staticmethod
    def heatmap(values: Any, *, title: str = "Heatmap", x: list[str] | None = None, y: list[str] | None = None) -> dict[str, Any]:
        return {
            "__kannaadi_artifact__": "tensor",
            "title": title,
            "data": values,
            "metadata": {"visualization": "heatmap", "xLabels": x or [], "yLabels": y or []},
        }

    @staticmethod
    def publish(value: Any, *, title: str = "Code result", kind: str | None = None) -> dict[str, Any]:
        return {"__kannaadi_artifact__": kind, "title": title, "data": value}


class CodeSession:
    """Persistent, model-scoped local Python namespace for the integrated Code Lab.

    This is intentionally not described as a security sandbox. Desktop API tokens
    and an explicit trust bit prevent accidental execution, while a separate session
    namespace, interrupt flag, and restart operation provide operational isolation.
    """

    trust_model = (
        "Trusted local Python: code can access this model process and your computer. "
        "Imported workspaces never execute code automatically."
    )

    def __init__(self, engine: Any, adapter: Any, architecture: Any, model_id: str) -> None:
        self.engine = engine
        self.adapter = adapter
        self.architecture = architecture
        self.model_id = model_id
        self._execution_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._interrupt = threading.Event()
        self._state: Literal["idle", "running", "interrupting"] = "idle"
        self._execution_id: str | None = None
        self._started_at: str | None = None
        self._namespace: dict[str, Any] = {}
        self._reset_namespace()

    def _reset_namespace(self) -> None:
        context = KannaadiResearchContext(self.engine)
        self._namespace = {
            "__name__": "__kannaadi_code_lab__",
            "model": self.adapter.model,
            "tl_model": self.adapter.model,
            "engine": self.engine,
            "architecture": self.architecture,
            "kannaadi": context,
            "runs": self.engine.list_runs(),
            "active_run": None,
            "tokens": None,
            "cache": None,
            "selection": [],
        }

    def status(self) -> CodeSessionStatus:
        with self._state_lock:
            return CodeSessionStatus(
                state=self._state,
                executionId=self._execution_id,
                startedAt=self._started_at,
                namespaceKeys=self._public_keys(),
                trustModel=self.trust_model,
            )

    def interrupt(self) -> CodeSessionStatus:
        with self._state_lock:
            if self._state == "running":
                self._state = "interrupting"
                self._interrupt.set()
            return self.status_unlocked()

    def status_unlocked(self) -> CodeSessionStatus:
        return CodeSessionStatus(
            state=self._state,
            executionId=self._execution_id,
            startedAt=self._started_at,
            namespaceKeys=self._public_keys(),
            trustModel=self.trust_model,
        )

    def restart(self) -> CodeSessionStatus:
        with self._state_lock:
            if self._state != "idle":
                raise RuntimeError("Interrupt the running cell before restarting the Code Lab session")
            self._reset_namespace()
            return self.status_unlocked()

    def execute(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        if not request.trusted:
            raise PermissionError("Code execution requires explicit trust for this local workspace")
        if not request.code.strip():
            raise ValueError("Enter Python code before running the cell")
        if not self._execution_lock.acquire(blocking=False):
            raise RuntimeError("Another Code Lab cell is already running")
        execution_id = str(uuid4())
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        stdout = StringIO()
        stderr = StringIO()
        result: Any = None
        status: Literal["complete", "error", "interrupted"] = "complete"
        error: str | None = None
        trace: str | None = None
        try:
            with self._state_lock:
                self._execution_id = execution_id
                self._started_at = started_at
                self._state = "running"
                self._interrupt.clear()
            self._bind_context(request)
            tree = ast.parse(request.code, filename=f"kannaadi-cell-{request.cell_id or execution_id}", mode="exec")
            previous_trace = sys.gettrace()
            sys.settrace(self._trace_interrupts)
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    result = self._execute_tree(tree)
            finally:
                sys.settrace(previous_trace)
        except CodeInterrupted:
            status = "interrupted"
            error = "Execution interrupted"
        except BaseException as exc:  # surface research-code failures in the cell
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            trace = traceback.format_exc(limit=20)
        finally:
            finished_at = datetime.now(timezone.utc).isoformat()
            duration_ms = int((time.perf_counter() - started) * 1000)
            with self._state_lock:
                self._state = "idle"
                self._execution_id = None
                self._started_at = None
                self._interrupt.clear()
            self._execution_lock.release()
        artifact = self._serialize_artifact(result) if status == "complete" else CodeArtifact(kind="none", title="No result")
        return CodeExecutionResult(
            id=execution_id,
            cellId=request.cell_id,
            status=status,
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
            artifact=artifact,
            error=error,
            traceback=trace,
            durationMs=duration_ms,
            startedAt=started_at,
            finishedAt=finished_at,
            namespaceKeys=self._public_keys(),
            provenance={
                "modelId": self.model_id,
                "activeRunId": request.active_run_id,
                "selection": request.selection,
                "runtime": "trusted-local-python",
            },
        )

    def _bind_context(self, request: CodeExecutionRequest) -> None:
        active_run = None
        tokens = None
        cache = None
        if request.active_run_id:
            active_run = self.engine.get_run(request.active_run_id)
            artifacts = self.engine._artifacts(request.active_run_id)
            tokens = artifacts.tokens
            cache = artifacts.cache
        self._namespace.update({
            "runs": self.engine.list_runs(),
            "active_run": active_run,
            "tokens": tokens,
            "cache": cache,
            "selection": list(request.selection),
        })

    def _trace_interrupts(self, frame: Any, event: str, arg: Any) -> Any:
        del frame, event, arg
        if self._interrupt.is_set():
            raise CodeInterrupted()
        return self._trace_interrupts

    def _execute_tree(self, tree: ast.Module) -> Any:
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            prefix = ast.Module(body=tree.body[:-1], type_ignores=[])
            if prefix.body:
                exec(compile(prefix, tree_filename(tree), "exec"), self._namespace, self._namespace)
            expression = ast.Expression(body=tree.body[-1].value)
            result = eval(compile(expression, tree_filename(tree), "eval"), self._namespace, self._namespace)
        else:
            exec(compile(tree, tree_filename(tree), "exec"), self._namespace, self._namespace)
            result = self._namespace.get("result")
        self._namespace["_"] = result
        return result

    def _public_keys(self) -> list[str]:
        return sorted(key for key in self._namespace if not key.startswith("__"))

    @classmethod
    def _serialize_artifact(cls, value: Any) -> CodeArtifact:
        if value is None:
            return CodeArtifact(kind="none", title="No return value")
        if isinstance(value, dict) and "__kannaadi_artifact__" in value:
            requested_kind = value.get("__kannaadi_artifact__")
            nested = cls._serialize_artifact(value.get("data"))
            kind = requested_kind if requested_kind in {"text", "json", "table", "tensor", "image"} else nested.kind
            return CodeArtifact(
                kind=kind,
                title=str(value.get("title") or nested.title),
                data=nested.data,
                metadata={**nested.metadata, **dict(value.get("metadata") or {})},
            )
        if hasattr(value, "model_dump"):
            return CodeArtifact(kind="json", title=type(value).__name__, data=cls._json_safe(value.model_dump(by_alias=True)))
        module_name = type(value).__module__
        if module_name.startswith("torch") and hasattr(value, "detach"):
            tensor = value.detach().cpu()
            shape = list(tensor.shape)
            flattened = tensor.reshape(-1)
            count = int(flattened.numel())
            if count <= 10_000:
                data = cls._json_safe(tensor.tolist())
                truncated = False
            else:
                data = cls._json_safe(flattened[:256].tolist())
                truncated = True
            numeric = flattened.float() if count else flattened
            metadata = {
                "shape": shape,
                "dtype": str(tensor.dtype),
                "device": str(value.device),
                "count": count,
                "truncated": truncated,
            }
            if count:
                metadata.update({"minimum": float(numeric.min()), "maximum": float(numeric.max()), "mean": float(numeric.mean())})
            return CodeArtifact(kind="tensor", title="Tensor result", data=data, metadata=metadata)
        if hasattr(value, "to_dict") and type(value).__module__.startswith("pandas"):
            return CodeArtifact(kind="table", title="DataFrame result", data=cls._json_safe(value.to_dict(orient="records")), metadata={"columns": [str(item) for item in value.columns]})
        if module_name.startswith("matplotlib") and hasattr(value, "savefig"):
            buffer = BytesIO()
            value.savefig(buffer, format="png", dpi=144, bbox_inches="tight")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return CodeArtifact(kind="image", title="Figure result", data=f"data:image/png;base64,{encoded}", metadata={"mimeType": "image/png"})
        if isinstance(value, str):
            return CodeArtifact(kind="text", title="Text result", data=value)
        if isinstance(value, (list, tuple)) and value and all(isinstance(row, dict) or hasattr(row, "model_dump") for row in value):
            rows = [row.model_dump(by_alias=True) if hasattr(row, "model_dump") else row for row in value]
            return CodeArtifact(kind="table", title="Table result", data=cls._json_safe(rows))
        if isinstance(value, (dict, list, tuple, int, float, bool)):
            return CodeArtifact(kind="json", title="Python result", data=cls._json_safe(value))
        return CodeArtifact(kind="text", title=type(value).__name__, data=repr(value)[:100_000])

    @classmethod
    def _json_safe(cls, value: Any, depth: int = 0) -> Any:
        if depth > 8:
            return "<maximum serialization depth>"
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if hasattr(value, "model_dump"):
            return cls._json_safe(value.model_dump(by_alias=True), depth + 1)
        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)
        if isinstance(value, dict):
            items = list(value.items())[:2_000]
            result = {str(key): cls._json_safe(item, depth + 1) for key, item in items}
            if len(value) > len(items):
                result["__truncated__"] = len(value) - len(items)
            return result
        if isinstance(value, (list, tuple)):
            result = [cls._json_safe(item, depth + 1) for item in value[:10_000]]
            if len(value) > len(result):
                result.append(f"<{len(value) - len(result)} items truncated>")
            return result
        if hasattr(value, "tolist"):
            return cls._json_safe(value.tolist(), depth + 1)
        return repr(value)


def tree_filename(_tree: ast.AST) -> str:
    return "<kannaadi-code-lab>"
