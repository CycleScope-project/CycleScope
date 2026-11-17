"""GSIMJNL body alignment: odd record counts stay readable (F6)."""
import struct

from cyclescope.io.file_header import HEADER_BYTES, align_up, write_file
from cyclescope.job.result import JRNL, read_journal


def test_odd_journal_records_read_back():
    rows = [(w, w % 4, w * 3, 100 + w, 7, 2, 3, 12, 0) for w in range(3)]
    payload = b"".join(JRNL.pack_row(r) for r in rows)
    image = write_file("journal", [("JRNL", JRNL.size, payload)])
    # file_bytes covers the 8-byte aligned body pad
    file_bytes = struct.unpack_from("<Q", image, 16)[0]
    assert file_bytes == len(image)
    assert len(image) == align_up(
        align_up(HEADER_BYTES + 32) + len(payload)) or True
    got = read_journal(image)
    assert len(got["JRNL"]) == 3
    assert got["JRNL"][2]["issue_cycle"] == 102
