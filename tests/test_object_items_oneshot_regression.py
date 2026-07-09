"""Regression: Service Maker frame_meta.object_items lifetime rules.

object_items is a one-shot iterator of *transient* ObjectMetadata views.
Storing wrappers via list() for a second pass leaves dangling native pointers;
accessing class_id on those wrappers segfaults in libnvds_service_maker.
"""

from __future__ import annotations

from pathlib import Path


class _OneShotItems:
    def __init__(self, items):
        self._items = list(items)
        self._iterated = False

    def __iter__(self):
        if self._iterated:
            return iter(())
        self._iterated = True
        return iter(self._items)


def test_object_items_must_be_materialized_before_double_pass():
    """First consume empties a second getattr iteration (one-shot)."""
    items = [{"id": 1}, {"id": 2}]
    one_shot = _OneShotItems(items)

    first = list(one_shot)
    second = list(one_shot)
    assert first == items
    assert second == []


def test_hooks_do_not_list_object_items_for_second_pass():
    """Crash fix: never list(object_items) then re-touch stored wrappers."""
    repo = Path(__file__).resolve().parents[1]
    for rel in (
        "noesis/pipelines/hooks.py",
        "noesis/pipelines/hooks_v3dt_reimpl.py",
    ):
        text = (repo / rel).read_text(encoding="utf-8")
        assert "object_items = list(" not in text, (
            f"{rel} still materializes object_items into a list for a second pass; "
            "that pattern segfaults on ObjectMetadata.class_id"
        )
        assert "object_items = getattr(frame_meta, \"object_items\"" in text or (
            "object_items = getattr(frame_meta, 'object_items'" in text
        )
