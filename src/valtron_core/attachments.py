"""Shared attachment MIME detection utilities."""

from pathlib import Path

_EXT_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_MAGIC: list[tuple[bytes, str]] = [
    (b"%PDF-", "application/pdf"),
    (b"\x89PNG", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF8", "image/gif"),
    (b"RIFF", "image/webp"),
]


def detect_mime_hint(s: str) -> str:
    """
    Detect MIME type from a data URI header or file/URL extension without any I/O.
    Returns an empty string if the type cannot be determined.
    """
    if s.startswith("data:"):
        header = s.split(",")[0]
        return header.split(":")[1].split(";")[0]
    suffix = Path(s.split("?")[0]).suffix.lower()
    return _EXT_MIME.get(suffix, "")
