"""Result file readers: ``GSIMRES``, ``GSIMJNL``, ``GSIMWLG`` (spec §8)."""

from __future__ import annotations

from typing import Any

from ..io.file_header import FileReader
from ..io.fixed_tables import Record, TableReader

RSLT = Record("<QQQQI", (
    "cycles", "issued", "completed", "hard_constraint_failures", "flags"))
SMSP = Record("<IIQQ", ("sm", "smsp", "first_active", "last_active"))
WAIT = Record("<IIQ", ("scope", "cause_kind", "cycles"))
RWAIT = Record("<IIQ", ("resource", "cause_kind", "cycles"))
TELE = Record("<IIQ", ("group", "counter", "value"))

JRNL = Record("<IIIQIIIQQ", (
    "warp", "smsp", "instance", "issue_cycle", "operation_class",
    "template", "wait_bucket", "wait_cycles", "object"))
WLDG = Record("<IIQI", ("warp", "cause_kind", "cycles", "count"))

RESULT_RECORDS = {"RSLT": RSLT, "SMSP": SMSP, "WAIT": WAIT,
                  "RWAT": RWAIT, "TELE": TELE}
JOURNAL_RECORDS = {"JRNL": JRNL}
WAIT_LEDGER_RECORDS = {"WLDG": WLDG}


def _read_tables(reader: FileReader,
                 records: dict[str, Record]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for tag, record in records.items():
        payload = reader.table(tag)
        if payload is None:
            continue
        table = TableReader(record, payload)
        out[tag] = [table.unpack(i) for i in range(len(table))]
    return out


def read_result(data: bytes | memoryview) -> dict[str, Any]:
    reader = FileReader(data, expect="result")
    result = _read_tables(reader, RESULT_RECORDS)
    rslt = result.get("RSLT")
    return {
        "cycles": rslt[0]["cycles"] if rslt else 0,
        "issued": rslt[0]["issued"] if rslt else 0,
        "completed": rslt[0]["completed"] if rslt else 0,
        "hard_constraint_failures":
            rslt[0]["hard_constraint_failures"] if rslt else 0,
        "flags": rslt[0]["flags"] if rslt else 0,
        "smsp_spans": result.get("SMSP", []),
        "wait_buckets": result.get("WAIT", []),
        "resource_waits": result.get("RWAT", []),
        "telemetry": result.get("TELE", []),
    }


def read_journal(data: bytes | memoryview) -> dict[str, Any]:
    reader = FileReader(data, expect="journal")
    return _read_tables(reader, JOURNAL_RECORDS)


def read_wait_ledger(data: bytes | memoryview) -> dict[str, Any]:
    reader = FileReader(data, expect="wait_ledger")
    return _read_tables(reader, WAIT_LEDGER_RECORDS)
