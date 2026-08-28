"""Shared attachment MIME detection, preflight checks, and message building."""

import base64
import urllib.request
from pathlib import Path

import litellm
import structlog

from valtron_core.models import Document

logger = structlog.get_logger()

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


def check_attachment_support(documents: list[Document], model_name: str) -> None:
    """
    Verify the model supports every attachment type across all documents before
    any evaluation runs. Uses extension/data-URI detection only, no I/O.

    Raises:
        ValueError: If any document has an attachment type the model cannot handle,
                    or if an attachment's type cannot be determined from its extension.
    """
    supported_exts = ", ".join(_EXT_MIME.keys())

    for doc in documents:
        if not doc.attachments:
            continue
        for attachment in doc.attachments:
            mime_type = detect_mime_hint(attachment)

            if not mime_type:
                raise ValueError(
                    f"Cannot determine attachment type for document '{doc.id}' "
                    f"(attachment: '{attachment}'). "
                    f"Supported extensions: {supported_exts}."
                )

            if mime_type.startswith("image/") and not litellm.supports_vision(model_name):
                raise ValueError(
                    f"Model '{model_name}' does not support image inputs, "
                    f"but document '{doc.id}' has an image attachment."
                )
            if mime_type == "application/pdf" and not litellm.supports_pdf_input(model_name):
                raise ValueError(
                    f"Model '{model_name}' does not support PDF inputs, "
                    f"but document '{doc.id}' has a PDF attachment."
                )


def load_attachment(s: str) -> tuple[bytes, str, bool]:
    """
    Load attachment data and detect its MIME type.

    Args:
        s: HTTP/HTTPS URL or local file path.

    Returns:
        (data, mime_type, is_url) where is_url indicates the source was a URL.
        For URL sources where MIME was determined from the extension alone,
        data is empty bytes. Callers that support URL passthrough can skip
        fetching in that case.
    """
    is_url = s.startswith(("http://", "https://"))

    # Data URI: data:<mime>;base64,<data>
    if s.startswith("data:"):
        header, _, b64 = s.partition(",")
        mime_type = header.split(":")[1].split(";")[0]
        return base64.b64decode(b64), mime_type, False

    # Detect MIME from extension first (strips query strings for URLs)
    mime_type = detect_mime_hint(s)

    if is_url:
        if mime_type:
            # Extension was sufficient. Callers that support URL passthrough
            # do not need the bytes.
            return b"", mime_type, True
        with urllib.request.urlopen(s) as resp:
            raw = resp.read()
            ct = resp.headers.get("Content-Type", "").split(";")[0].strip()
            mime_type = ct if ct else ""
    else:
        raw = Path(s).read_bytes()

    # Magic-byte fallback if MIME still unknown
    if not mime_type:
        for magic, mime in _MAGIC:
            if raw[: len(magic)] == magic:
                mime_type = mime
                break

    if not mime_type:
        mime_type = "application/octet-stream"

    return raw, mime_type, is_url


def build_message_content(
    prompt: str, attachments: list[str], model: str
) -> str | list[dict[str, str]]:
    """
    Build the user message content, adding attachment parts when present.

    Returns a plain string when there are no attachments, or a list of
    provider-appropriate content parts when there are.

    Each entry in ``attachments`` is an HTTP/HTTPS URL or a local file path.
    The file type is auto-detected from the extension or magic bytes. LiteLLM
    translates the content blocks to the correct format per provider.

    Raises:
        ValueError: If the model does not support the required input type.
    """
    if not attachments:
        return prompt

    parts: list[dict[str, str]] = [{"type": "text", "text": prompt}]

    for s in attachments:
        try:
            raw, mime_type, is_url = load_attachment(s)
        except Exception as e:
            logger.warning("attachment_load_failed", attachment=s, error=str(e))
            continue

        is_image = mime_type.startswith("image/")
        is_pdf = mime_type == "application/pdf"

        if not is_image and not is_pdf:
            logger.warning("attachment_unsupported_mime", attachment=s, mime_type=mime_type)
            continue

        is_data_uri = s.startswith("data:")

        if is_image:
            if not litellm.supports_vision(model):
                raise ValueError(f"Model '{model}' does not support image inputs.")
            # image_url is LiteLLM's universal format for images across all providers.
            if is_url or is_data_uri:
                # URL and data URIs can be passed directly, no encode/decode needed.
                parts.append({"type": "image_url", "image_url": {"url": s}})
            else:
                b64 = base64.b64encode(raw).decode()
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                    }
                )

        elif is_pdf:
            if not litellm.supports_pdf_input(model):
                raise ValueError(f"Model '{model}' does not support PDF inputs.")
            if is_data_uri:
                # Already a data URI, so pass it directly as file_data.
                parts.append({"type": "file", "file": {"file_data": s}})
            elif is_url and not raw:
                # URL passthrough: LiteLLM fetches and routes per-provider.
                parts.append({"type": "file", "file": {"file_id": s, "format": "application/pdf"}})
            else:
                # Local file, or a URL whose bytes were already fetched during MIME detection.
                b64_data = f"data:application/pdf;base64,{base64.b64encode(raw).decode()}"
                parts.append({"type": "file", "file": {"file_data": b64_data}})

    return parts
