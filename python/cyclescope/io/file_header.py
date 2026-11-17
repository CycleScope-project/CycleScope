"""Common 64-byte file header + table directory (job-format spec §1).

All binary files share one shell: a fixed ``FileHeader``, a directory of
``table_count`` 32-byte entries, and 8-byte aligned table payloads.  Tables
are looked up by tag only; unknown tags are ignored for forward
compatibility.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Mapping

HEADER_BYTES = 72
DIRECTORY_ENTRY_BYTES = 32
ALIGN = 8

# magic -> human name (all are 8 bytes including the NUL pad).
MAGICS = {
    "GSIMCAP01": b"GSIMCAP0",
    "GSIMCAP02": b"GSIMCP2\0",
    "GSIMCAPS1": b"GSIMCPS\0",
    "GSIMPRG01": b"GSIMPRG0",
    "segment": b"GSIMSEG\0",
    "container": b"GSIMJOB\0",
    "result": b"GSIMRES\0",
    "journal": b"GSIMJNL\0",
    "wait_ledger": b"GSIMWLG\0",
}
MAGIC_TO_KIND = {v: k for k, v in MAGICS.items()}

FLAG_PAYLOAD_SHA256 = 0x1
UINT32_NONE = 0xFFFFFFFF

_HEADER = struct.Struct("<8sIIQQII32s")
_ENTRY = struct.Struct("<IIQQQ")


class FileFormatError(ValueError):
    """Malformed binary file."""


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise FileFormatError(message)


def tag_value(tag: str) -> int:
    """4-letter ASCII tag as its little-endian u32."""
    raw = tag.encode("ascii")
    _fail(len(raw) == 4, f"tag {tag!r} must be exactly 4 ASCII characters")
    return int.from_bytes(raw, "little")


def tag_name(value: int) -> str:
    return value.to_bytes(4, "little").decode("ascii", errors="replace")


def align_up(offset: int, alignment: int = ALIGN) -> int:
    return (offset + alignment - 1) // alignment * alignment


def write_file(kind: str, tables: list[tuple[str, int, bytes]],
               *, version: int = 1,
               payload_sha256: bool = False) -> bytes:
    """Assemble one binary file from (tag, record_bytes, payload) triples.

    ``record_bytes`` is the fixed record width declared in the directory;
    pass 0 for a byte blob table (STRS): then ``record_count`` carries the
    byte length.  Table order in the directory follows ``tables``.
    """
    magic = MAGICS[kind]
    table_count = len(tables)
    directory_offset = HEADER_BYTES
    cursor = align_up(directory_offset + table_count * DIRECTORY_ENTRY_BYTES)
    body_start = cursor
    entries = []
    blob = bytearray()
    for tag, record_bytes, payload in tables:
        if record_bytes:
            _fail(len(payload) % record_bytes == 0,
                  f"table {tag!r} payload is not a multiple of "
                  f"{record_bytes}")
            record_count = len(payload) // record_bytes
        else:
            record_count = len(payload)
        entries.append((tag_value(tag), record_bytes, record_count,
                        cursor, 0))
        blob.extend(b"\0" * (cursor - body_start - len(blob)))
        blob.extend(payload)
        cursor = align_up(cursor + len(payload))
    header = _HEADER.pack(magic, version, HEADER_BYTES, 0,
                          directory_offset, table_count,
                          FLAG_PAYLOAD_SHA256 if payload_sha256 else 0,
                          b"\0" * 32)
    directory = b"".join(
        _ENTRY.pack(tag, rec_bytes, count, off, res)
        for tag, rec_bytes, count, off, res in entries)
    out = bytearray(header + directory + blob)
    if payload_sha256:
        digest = hashlib.sha256(bytes(out)).digest()
        # file_bytes (offset 24) must be filled before hashing; patch order:
        # zero digest -> hash -> write digest; file_bytes written first.
        struct.pack_into("<Q", out, 16, len(out))
        digest = hashlib.sha256(bytes(out)).digest()
        out[40:72] = digest
    else:
        struct.pack_into("<Q", out, 16, len(out))
    return bytes(out)


class FileReader:
    """Directory-driven access to one binary file's tables."""

    def __init__(self, data: bytes | memoryview, *, expect: str | None = None):
        view = memoryview(data)
        _fail(len(view) >= HEADER_BYTES, "file shorter than header")
        magic, version, header_bytes, file_bytes, directory_offset, \
            table_count, flags, digest = _HEADER.unpack(view[:HEADER_BYTES])
        _fail(magic in MAGIC_TO_KIND, f"bad magic {magic!r}")
        _fail(expect is None or MAGIC_TO_KIND[magic] == expect,
              f"expected {expect}, got {MAGIC_TO_KIND[magic]}")
        _fail(version in (1, 2), f"unsupported format_version {version}")
        _fail(file_bytes == len(view), "file_bytes mismatch")
        self.magic = magic
        self.version = version
        self.kind = MAGIC_TO_KIND[magic]
        self.flags = flags
        self.payload_sha256 = bytes(digest) if flags & FLAG_PAYLOAD_SHA256 \
            else None
        self.tables: dict[int, tuple[int, int, int]] = {}
        for index in range(table_count):
            start = directory_offset + index * DIRECTORY_ENTRY_BYTES
            _fail(start + DIRECTORY_ENTRY_BYTES <= len(view),
                  "directory overruns file")
            tag, record_bytes, record_count, offset, _ = _ENTRY.unpack(
                view[start:start + DIRECTORY_ENTRY_BYTES])
            _fail(tag not in self.tables,
                  f"duplicate table tag {tag_name(tag)!r}")
            length = (record_count * record_bytes if record_bytes
                      else record_count)
            _fail(offset + length <= len(view),
                  f"table {tag_name(tag)!r} overruns file")
            self.tables[tag] = (record_bytes, record_count, offset)
        self._view = view

    def has_table(self, tag: str) -> bool:
        return tag_value(tag) in self.tables

    def require_table(self, tag: str) -> memoryview:
        value = tag_value(tag)
        _fail(value in self.tables, f"required table {tag!r} missing")
        record_bytes, record_count, offset = self.tables[value]
        length = record_count * record_bytes if record_bytes \
            else record_count
        return self._view[offset:offset + length]

    def table(self, tag: str) -> memoryview | None:
        value = tag_value(tag)
        if value not in self.tables:
            return None
        record_bytes, record_count, offset = self.tables[value]
        length = record_count * record_bytes if record_bytes \
            else record_count
        return self._view[offset:offset + length]

    def table_layout(self, tag: str) -> tuple[int, int] | None:
        value = tag_value(tag)
        if value not in self.tables:
            return None
        record_bytes, record_count, _ = self.tables[value]
        return record_bytes, record_count
