"""Progress reporting for in-flight ModelEval runs.

A tiny event-based writer that atomically updates ``<output_dir>/progress.json``
at known points during evaluation so external systems (e.g., a web dashboard)
can poll for live progress without coupling to logs or internal state.

File shape (UTC ISO 8601 with 'Z' suffix; consumer is expected to convert to
local time)::

    {
      "started_at":  "2026-05-29T15:41:48.996Z",
      "last_update": "2026-05-29T15:42:01.123Z",
      "models": [
        {"name": "<label>", "docs_done": 0, "docs_total": 50, "completed": false},
        ...
      ]
    }

Absent file = pre-evaluation phase (still in few-shot generation, validation,
prompt preparation, etc.).  Consumers should treat the missing file as
"initialising" rather than as an error.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _utcnow_iso() -> str:
    """Return the current time as a UTC ISO 8601 string with a 'Z' suffix.

    :return: ISO 8601 timestamp like ``"2026-05-29T15:41:48.996Z"``.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically via tmp + rename.

    Prevents external pollers from observing a half-written file.

    :param path: Destination file path.
    :param content: Full file content to write.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(str(tmp), str(path))


def write_status(output_dir: "str | Path", status_message: str) -> None:
    """Write a minimal pre-evaluation progress file with a status message.

    Used during the setup phase (few-shot generation, prompt preparation, etc.)
    when there are no per-model document counts to report yet — but we still
    want to tell the user *what* is happening so the panel doesn't sit on a
    generic "Initialising..." text for many minutes.

    Once :class:`ProgressTracker` is initialised, the file is replaced with the
    full per-model schema (no ``status_message`` field).

    :param output_dir: The run's output directory.
    :param status_message: Human-readable description of the current setup step.
    """
    path = Path(output_dir) / "progress.json"

    if path.exists():
        try:
            existing = json.loads(path.read_text())
            started_at = existing.get("started_at") or _utcnow_iso()
        except (json.JSONDecodeError, OSError):
            started_at = _utcnow_iso()
    else:
        started_at = _utcnow_iso()

    payload = {
        "started_at": started_at,
        "last_update": _utcnow_iso(),
        "status_message": status_message,
    }
    _atomic_write(path, json.dumps(payload))


class ProgressTracker:
    """Per-run progress state, written atomically to ``progress.json``.

    Thread-safe: ``on_doc_complete`` may be invoked concurrently from multiple
    per-model callbacks (one per parallel model evaluation) and from the
    semaphored per-document tasks within each model.
    """

    def __init__(
        self,
        output_dir: "str | Path",
        model_names: Iterable[str],
        docs_per_model: int,
    ) -> None:
        """Initialise the tracker and write the initial progress file.

        :param output_dir: The run's output directory (``progress.json`` is
            placed inside it).
        :param model_names: Display names / labels for each model being evaluated,
            one entry per row in the UI.
        :param docs_per_model: Number of documents each model will process.
        """
        self._path = Path(output_dir) / "progress.json"
        self._lock = threading.Lock()
        self._started_at = _utcnow_iso()
        self._models: dict[str, dict] = {
            name: {
                "name": name,
                "docs_done": 0,
                "docs_total": docs_per_model,
                "completed": False,
            }
            for name in model_names
        }

        self._flush()

    def on_doc_complete(self, model_name: str) -> None:
        """Increment ``docs_done`` for ``model_name`` and rewrite ``progress.json``.

        Silently ignores unknown model names so a callback wiring mistake
        never crashes a real evaluation.

        :param model_name: Label of the model whose document just completed.
        """
        with self._lock:
            m = self._models.get(model_name)
            if m is None:
                return

            m["docs_done"] += 1
            if m["docs_done"] >= m["docs_total"]:
                m["completed"] = True

            self._flush_locked()

    def mark_model_completed(self, model_name: str) -> None:
        """Mark ``model_name`` as fully done regardless of per-doc events received.

        Safety net for evaluation paths that don't emit per-document callbacks
        (e.g., the decomposed evaluator).  After the path returns, call this so
        the UI doesn't display the model as stuck at 0 / k.

        :param model_name: Label of the model that just finished evaluation.
        """
        with self._lock:
            m = self._models.get(model_name)
            if m is None:
                return

            m["docs_done"] = m["docs_total"]
            m["completed"] = True

            self._flush_locked()

    def _flush(self) -> None:
        """Acquire the lock and write the current state."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Write the current state.  Caller must hold ``self._lock``."""
        payload = {
            "started_at": self._started_at,
            "last_update": _utcnow_iso(),
            "models": list(self._models.values()),
        }
        _atomic_write(self._path, json.dumps(payload))
