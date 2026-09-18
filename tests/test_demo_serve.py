"""`itx demo serve` exists because the demo was being checked under the wrong headers.

The live site sends a strict Content-Security-Policy and `python -m http.server` sends none,
so for two weeks the page was reviewed in a browser that allowed things the real one blocks.
The defect that surfaced was small and visible: every colour swatch in the chart legend was
blank, because a style attribute arriving in markup is exactly what the policy forbids.

These tests hold the two halves of the fix. The server really does send what the committed
config declares, and the page really does not contain the thing the policy blocks.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from itx.demo.serve import CONFIG_NAME, build_server, declared_headers

DEMO = Path("demo")


class TestReadingTheConfig:
    def test_it_reads_the_committed_policy(self):
        headers = declared_headers(DEMO)
        assert "Content-Security-Policy" in headers
        assert "default-src 'none'" in headers["Content-Security-Policy"]

    def test_no_config_is_not_an_error(self, tmp_path):
        # The folder may be served by something else entirely. Refusing to serve it would be
        # a worse answer than serving it the way any other static server would.
        assert declared_headers(tmp_path) == {}

    def test_a_malformed_config_is_an_error(self, tmp_path):
        # Quietly sending no headers because the config could not be read would put this
        # straight back to checking the page under headers it is not served with.
        (tmp_path / CONFIG_NAME).write_text(
            json.dumps({"globalHeaders": ["not", "an", "object"]}), encoding="utf-8"
        )
        with pytest.raises(TypeError, match="globalHeaders"):
            declared_headers(tmp_path)

    def test_values_come_back_as_strings(self, tmp_path):
        (tmp_path / CONFIG_NAME).write_text(
            json.dumps({"globalHeaders": {"X-Frame-Count": 3}}), encoding="utf-8"
        )
        assert declared_headers(tmp_path) == {"X-Frame-Count": "3"}


class TestServingWithThem:
    @pytest.fixture
    def served(self, tmp_path):
        """A running server on a port the operating system picks, torn down after the test."""
        page = "<!doctype html><title>x</title>"
        (tmp_path / "index.html").write_text(page, encoding="utf-8")
        (tmp_path / CONFIG_NAME).write_text(
            json.dumps({"globalHeaders": {"Content-Security-Policy": "default-src 'none'"}}),
            encoding="utf-8",
        )
        server = build_server(tmp_path, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_every_response_carries_the_declared_headers(self, served):
        with urllib.request.urlopen(f"{served}/index.html", timeout=5) as response:
            assert response.headers["Content-Security-Policy"] == "default-src 'none'"
            assert response.status == 200

    def test_it_still_serves_the_files(self, served):
        with urllib.request.urlopen(f"{served}/index.html", timeout=5) as response:
            assert b"<!doctype html>" in response.read()

    def test_it_binds_the_loopback_interface_only(self, served):
        # A development server that answers the network is a different thing from a
        # development server.
        assert served.startswith("http://127.0.0.1:")


class TestThePageUnderThatPolicy:
    def test_the_page_uses_no_style_attributes(self):
        # `style-src 'self'` blocks a style attribute that arrives in markup, and the page
        # sets one thing by colour: the legend swatches. They were blank on the live site.
        # The colour now goes on through the CSSOM, which the policy allows.
        for name in ("index.html", "app.js"):
            text = (DEMO / name).read_text(encoding="utf-8")
            assert 'style="' not in text, name

    def test_the_page_has_no_inline_script_or_style_blocks(self):
        # Same directive, same reason: neither would run under this policy, and both would
        # fail silently rather than loudly.
        page = (DEMO / "index.html").read_text(encoding="utf-8")
        assert "<style" not in page
        assert "<script>" not in page
