"""Per-table byte-diff of two GSIMSEG files (3a acceptance)."""
import sys
sys.path.insert(0, "python")
from cyclescope.job.segment_format import (
    OPTIONAL_RECORDS, read_segment, REQUIRED_TABLES)

a = read_segment(open(sys.argv[1], "rb").read())
b = read_segment(open(sys.argv[2], "rb").read())
diffs = []
for t in REQUIRED_TABLES + list(OPTIONAL_RECORDS):
    ta, tb = a.table(t), b.table(t)
    if ta != tb:
        diffs.append(f"{t}({len(ta or b'')}!={len(tb or b'')})")
print("TABLE_DIFFS:", " ".join(diffs) if diffs else "none")
