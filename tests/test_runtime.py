"""Resumability must actually survive a process death, not just a clean exit."""

import json

from caliper.runtime import Checkpoint, pick_device


def test_checkpoint_resumes_partial_work(tmp_path):
    path = tmp_path / "run.jsonl"
    first = Checkpoint(path)
    for i in range(3):
        first.record(i, {"value": i * 10})

    # A new process opening the same file must see the completed work.
    second = Checkpoint(path)
    assert all(second.done(i) for i in range(3))
    assert not second.done(3)
    assert sorted(r["value"] for r in second.rows()) == [0, 10, 20]


def test_checkpoint_survives_a_truncated_final_line(tmp_path):
    """A kill mid-write leaves a partial line; it must not poison the resume."""
    path = tmp_path / "run.jsonl"
    ck = Checkpoint(path)
    ck.record("a", {"value": 1})
    with path.open("a") as fh:
        fh.write('{"value": 2, "_key": "b"')  # truncated, no newline

    try:
        recovered = Checkpoint(path)
    except json.JSONDecodeError:
        raise AssertionError("a truncated trailing line must not break resume")
    assert recovered.done("a")


def test_pick_device_returns_something_usable():
    assert pick_device("auto") in ("cpu", "cuda")
    assert pick_device("cpu") == "cpu"
