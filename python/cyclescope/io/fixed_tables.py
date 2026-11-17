"""Fixed-width record tables driven by ``struct.Struct`` layouts.

Record layouts are declared once in ``cyclescope.job.segment_format`` /
``cyclescope.job.container_format`` as ``Record(format, fields)`` so Python and
generated accessors share a single source of truth.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence


@dataclass(frozen=True)
class Record:
    """One fixed-width table record layout.

    ``fmt`` is a little-endian ``struct`` format string; ``fields`` lists the
    decoded field names in order (``None`` marks an unnamed pad slot).
    """

    fmt: str
    fields: tuple[str | None, ...]

    @property
    def struct(self) -> struct.Struct:
        return struct.Struct(self.fmt)

    @property
    def size(self) -> int:
        return self.struct.size

    def pack(self, **values: Any) -> bytes:
        named = [f for f in self.fields if f is not None]
        row = []
        fi = 0
        for f in self.fields:
            row.append(0 if f is None else values[f])
        return self.struct.pack(*row)

    def pack_row(self, values: Sequence[Any]) -> bytes:
        _pad = self.fields.count(None)
        _check(len(values) == len(self.fields) - _pad,
               f"row has {len(values)} values, record needs "
               f"{len(self.fields) - _pad}")
        out, vi = [], 0
        for f in self.fields:
            if f is None:
                out.append(0)
            else:
                out.append(values[vi])
                vi += 1
        return self.struct.pack(*out)

    def unpack(self, data: bytes | memoryview, index: int = 0) -> dict:
        start = index * self.size
        row = self.struct.unpack(data[start:start + self.size])
        return {f: v for f, v in zip(self.fields, row) if f is not None}


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class TableWriter:
    """Append-only writer for one record table."""

    def __init__(self, record: Record):
        self.record = record
        self._buf = bytearray()

    @property
    def count(self) -> int:
        return len(self._buf) // self.record.size

    def append(self, values: Sequence[Any]) -> None:
        self._buf.extend(self.record.pack_row(values))

    def append_dict(self, **values: Any) -> None:
        self._buf.extend(self.record.pack(**values))

    def payload(self) -> bytes:
        return bytes(self._buf)


class TableReader:
    """Row view over a table payload without per-row objects.

    ``rows()`` yields memoryview slices of ``record.size`` bytes;
    ``unpack(i)`` materialises one dict when needed.
    """

    def __init__(self, record: Record, payload: bytes | memoryview):
        view = memoryview(payload)
        if len(view) % record.size:
            raise ValueError(
                f"table payload {len(view)} is not a multiple of "
                f"{record.size}")
        self.record = record
        self.payload = view

    def __len__(self) -> int:
        return len(self.payload) // self.record.size

    def rows(self) -> Iterator[memoryview]:
        size = self.record.size
        for i in range(len(self)):
            yield self.payload[i * size:(i + 1) * size]

    def unpack(self, index: int) -> dict:
        return self.record.unpack(self.payload, index)

    def column(self, field: str) -> list:
        return [self.unpack(i)[field] for i in range(len(self))]


class Strings:
    """NUL-terminated string table writer."""

    def __init__(self):
        self._buf = bytearray()
        self._offsets: dict[str, int] = {}

    def offset(self, text: str) -> int:
        if text not in self._offsets:
            self._offsets[text] = len(self._buf)
            self._buf.extend(text.encode("utf-8") + b"\0")
        return self._offsets[text]

    def payload(self) -> bytes:
        return bytes(self._buf)


def read_string(payload: bytes | memoryview, offset: int) -> str:
    end = bytes(payload[offset:]).index(0)
    return bytes(payload[offset:offset + end]).decode("utf-8")


def write_strings(texts: Iterable[str]) -> bytes:
    out = bytearray()
    for text in texts:
        out.extend(text.encode("utf-8") + b"\0")
    return bytes(out)
