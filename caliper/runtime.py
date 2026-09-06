"""Device selection and resumable runs.

Two operational lessons are encoded here. Sessions die - several have during this
project, losing hours of fitting - so any run long enough to matter must be able to
resume from partial results. And hosted GPU sessions cap at ~12 hours, so a run must be
able to stop cleanly and continue in a later session rather than restarting.
"""

import json
import os
from pathlib import Path

import torch


def pick_device(requested="auto"):
    """Resolve a device string, reporting what was actually chosen."""
    if requested not in ("auto", None):
        return requested
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  device: cuda ({name}, {total:.1f} GB)", flush=True)
        return "cuda"
    print(f"  device: cpu ({torch.get_num_threads()} threads)", flush=True)
    return "cpu"


class Checkpoint:
    """Append-only per-item results, so a killed run resumes instead of restarting.

    Usage:
        ck = Checkpoint("results/e01_n100.jsonl")
        for item in items:
            if ck.done(item):
                continue
            ck.record(item, compute(item))
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done = {}
        if self.path.exists():
            bad = 0
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A kill mid-write leaves a partial trailing line. Drop it and
                    # recompute that one item rather than losing the whole run.
                    bad += 1
                    continue
                self._done[str(row["_key"])] = row
            if bad:
                print(f"  discarded {bad} incomplete record(s) from an interrupted run",
                      flush=True)
            if self._done:
                print(f"  resuming: {len(self._done)} items already complete",
                      flush=True)

    def done(self, key):
        return str(key) in self._done

    def record(self, key, payload):
        row = dict(payload)
        row["_key"] = key
        with self.path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())  # survive an abrupt session kill
        self._done[str(key)] = row

    def rows(self):
        return list(self._done.values())
