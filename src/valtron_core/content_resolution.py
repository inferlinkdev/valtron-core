"""Resolve document content and local file references against a base directory.

Kept dependency-free (stdlib only) so both ``evaluation/base.py`` (which
``runner.py`` is imported by) and ``runner.py`` itself can use it without a
circular import.
"""

from pathlib import Path
from typing import Any


def resolve_content(item: dict[str, Any], base_dir: Path) -> "str | dict[str, str | None]":
    """Return a document record's content, given inline or by reference.

    A record provides either ``content`` (inline, as before) or ``content_path``
    (a path to a file holding the text, resolved relative to ``base_dir`` when not
    already absolute) -- never both, never neither. Read once, here, so
    ``Document.content`` stays a plain string either way and nothing downstream
    needs to know which form the record used. Only applies to the plain-string
    form; a record whose ``content`` is already the multi-placeholder dict form
    is returned unchanged (no per-placeholder file reference).
    """
    doc_id = item.get("id", "?")
    has_inline = "content" in item
    has_path = item.get("content_path") is not None
    if has_inline and has_path:
        raise ValueError(
            f"document {doc_id!r} has both 'content' and 'content_path'; provide exactly one."
        )
    if has_inline:
        return item["content"]  # type: ignore[no-any-return]
    if has_path:
        path = absolutize_local_path(item["content_path"], base_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"content_path {item['content_path']!r} for document {doc_id!r} "
                f"not found (resolved to {path})."
            )
        return path.read_text(encoding="utf-8")
    raise ValueError(f"document {doc_id!r} has neither 'content' nor 'content_path'.")


def absolutize_local_path(raw: str, base_dir: Path) -> Path:
    """Resolve a possibly-relative local path against base_dir; pass through as-is if already absolute."""
    path = Path(raw)
    return path if path.is_absolute() else (base_dir / path).resolve()


def is_local_path(value: str) -> bool:
    """True for a value that names a local file rather than a URL or a data: URI.

    Attachments and content_path both accept HTTP(S) URLs and data: URIs verbatim
    (never resolved against a base directory); everything else is treated as a
    local path.
    """
    return not (
        value.startswith("http://") or value.startswith("https://") or value.startswith("data:")
    )
