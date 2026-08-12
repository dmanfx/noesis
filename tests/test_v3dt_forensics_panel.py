#!/usr/bin/env python3
"""Unit tests for V3DT forensics panel HTML rendering."""
from __future__ import annotations

from noesis.diagnostics.v3dt_forensics import render_panel_html


def test_render_panel_html_contains_markers() -> None:
    html = render_panel_html({"generated_at": "now", "pipeline": {}}, {"generated_at": "now", "cameras": {}})
    assert "V3DT Forensics Panel" in html
    assert "Snapshot" in html
    assert "Report" in html
