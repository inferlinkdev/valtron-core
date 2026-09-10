from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from valtron_core.attachments import check_attachment_support, detect_mime_hint
from valtron_core.models import Document

pytestmark = pytest.mark.unit


def _doc(attachment: str) -> Document:
    return Document(id="doc-1", content="hello", attachments=[attachment])


class TestDetectMimeHint:
    def test_data_uri(self) -> None:
        assert detect_mime_hint("data:image/png;base64,AAAA") == "image/png"

    def test_extension(self) -> None:
        assert detect_mime_hint("https://example.com/a.png") == "image/png"

    def test_extension_with_query_string(self) -> None:
        assert detect_mime_hint("https://example.com/a.png?w=100") == "image/png"

    def test_unresolvable_extension(self) -> None:
        assert detect_mime_hint("https://example.com/imagefly.cgi?cid=1") == ""


class TestCheckAttachmentSupport:
    def test_passes_for_recognized_extension(self) -> None:
        with patch("valtron_core.attachments.litellm.supports_vision", return_value=True):
            check_attachment_support([_doc("https://example.com/a.png")], "gpt-4o-mini")

    def test_falls_back_to_head_request_for_extensionless_url(self) -> None:
        url = "https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=887"
        response = MagicMock()
        response.headers.get.return_value = "image/png"
        response.__enter__.return_value = response
        response.__exit__.return_value = False

        with (
            patch("valtron_core.attachments.urllib.request.urlopen", return_value=response) as m,
            patch("valtron_core.attachments.litellm.supports_vision", return_value=True),
        ):
            check_attachment_support([_doc(url)], "gpt-4o-mini")

        # The fallback must be a lightweight HEAD, not a full GET.
        request = m.call_args[0][0]
        assert request.get_method() == "HEAD"

    def test_raises_when_head_request_cannot_determine_type(self) -> None:
        url = "https://example.com/imagefly.cgi?cid=887"
        response = MagicMock()
        response.headers.get.return_value = ""
        response.__enter__.return_value = response
        response.__exit__.return_value = False

        with patch("valtron_core.attachments.urllib.request.urlopen", return_value=response):
            with pytest.raises(ValueError, match="Cannot determine attachment type"):
                check_attachment_support([_doc(url)], "gpt-4o-mini")

    def test_raises_when_head_request_errors(self) -> None:
        url = "https://example.com/imagefly.cgi?cid=887"
        with patch("valtron_core.attachments.urllib.request.urlopen", side_effect=OSError("boom")):
            with pytest.raises(ValueError, match="Cannot determine attachment type"):
                check_attachment_support([_doc(url)], "gpt-4o-mini")

    def test_local_path_without_extension_still_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot determine attachment type"):
            check_attachment_support([_doc("/tmp/some-file-no-extension")], "gpt-4o-mini")

    def test_raises_when_model_lacks_vision_support(self) -> None:
        with patch("valtron_core.attachments.litellm.supports_vision", return_value=False):
            with pytest.raises(ValueError, match="does not support image inputs"):
                check_attachment_support([_doc("https://example.com/a.png")], "text-only-model")
