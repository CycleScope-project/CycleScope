"""Engine stall reasons -> ncu warp-stall classes.

``configs/stall/class_mapping.json``: ordered rules, first match wins;
``excluded`` reasons are not part of a warp's residency on hardware.
A rule may carry an ``operation`` regular expression: it applies only to
a row whose operation class is known and matches (never when the caller
gives no operation).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .hardware import CLASSES

SCHEMA = "cyclescope.stall-class-mapping.v1"
DEFAULT_PATH = Path(__file__).resolve().parents[3] / "configs" / "stall" / \
    "class_mapping.json"


class ClassMapping:
    def __init__(self, document: dict, source: str = "") -> None:
        if document.get("schema_version") != SCHEMA:
            raise ValueError(f"unsupported:stall_class_mapping_schema:"
                             f"{document.get('schema_version')} ({source})")
        self.excluded = [re.compile(p) for p in document.get("excluded", [])]
        self.rules = []
        for rule in document["rules"]:
            if rule["class"] not in CLASSES:
                raise ValueError(f"unsupported:stall_class:{rule['class']} "
                                 f"({source})")
            if rule.get("evidence") not in ("hypothesis", "calibrated"):
                raise ValueError(f"missing_fact:stall_rule_evidence:"
                                 f"{rule['reason']} ({source})")
            operation = rule.get("operation")
            if "operation" in rule and (not isinstance(operation, str)
                                        or not operation):
                raise ValueError(f"unsupported:stall_rule_operation:"
                                 f"{rule['reason']} ({source})")
            self.rules.append((re.compile(rule["reason"]),
                               re.compile(operation) if operation else None,
                               rule["class"]))
        self.documents = list(document["rules"])
        # optional: {"instructions": n, "evidence": ..., "source": ...};
        # an instruction's issue cycle is booked n instructions behind
        # its own pc (compare.py); absent = 0
        shift = document.get("issue_pc_shift") or {}
        if shift and shift.get("evidence") not in ("hypothesis",
                                                   "calibrated"):
            raise ValueError(f"missing_fact:stall_rule_evidence:"
                             f"issue_pc_shift ({source})")
        self.issue_pc_shift = int(shift.get("instructions", 0))
        if self.issue_pc_shift < 0:
            raise ValueError(f"unsupported:stall_issue_pc_shift:"
                             f"{self.issue_pc_shift} ({source})")
        self._cache: dict[tuple[str, str | None], str | None] = {}

    def rule_index(self, reason: str,
                   operation: str | None = None) -> int | None:
        """Index of the rule that classifies `reason` (on an instruction
        of class `operation`, if known); None when the reason is
        excluded."""
        if any(p.search(reason) for p in self.excluded):
            return None
        for index, (pattern, op_pattern, _) in enumerate(self.rules):
            if pattern.search(reason) and (
                    op_pattern is None or (
                        operation is not None
                        and op_pattern.search(operation))):
                return index
        raise ValueError(f"unsupported:stall_reason_unmapped:{reason}")

    def classify(self, reason: str,
                 operation: str | None = None) -> str | None:
        """Class of a reason; None when the reason is excluded.  A
        reason no rule matches is an error: the mapping must end with an
        explicit catch-all if one is wanted."""
        key = (reason, operation)
        if key not in self._cache:
            index = self.rule_index(reason, operation)
            self._cache[key] = None if index is None else \
                self.rules[index][2]
        return self._cache[key]


def load_mapping(path: Path | None = None) -> ClassMapping:
    path = Path(path) if path else DEFAULT_PATH
    return ClassMapping(json.loads(path.read_text()), str(path))
