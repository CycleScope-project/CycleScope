from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(os.environ["CYCLESCOPE_RUNNER_ROOT"])
ART = Path(os.environ.get("BW_CORPUS_ROOT", str(ROOT / "artifacts" / "corpus")))


def write_result(suite: str, case_id: str, payload: dict) -> Path:
    d = ART / suite / case_id
    d.mkdir(parents=True, exist_ok=True)
    out = dict(payload)
    out.setdefault("id", case_id)
    out.setdefault("suite", suite)
    out.setdefault("gpu", "B300S")
    out.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    (d / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    if out.get("log"):
        (d / "log.txt").write_text(str(out["log"]))
    return d


def write_text(suite: str, case_id: str, name: str, text: str) -> Path:
    d = ART / suite / case_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(text)
    return p
