#!/usr/bin/env bash
# engine_fixture_digest.sh <engine> <fixtures_dir> <out.tsv>
#   sha256 of the result and journal binaries for every *.job fixture;
#   raw files land in <out.tsv>.wrk/
# engine_fixture_digest.sh --compare <a.wrk> <b.wrk>
#   table-level compare of the *.r result files in two .wrk dirs:
#   every table except WAIT must be byte-identical (incl. row count)
set -u
if [ "${1:-}" = "--compare" ]; then
    python3 - "$2" "$3" <<'EOF'
import glob, os, struct, sys
def tables(p):
    d = open(p, 'rb').read()
    do = struct.unpack_from('<Q', d, 24)[0]
    tc = struct.unpack_from('<I', d, 32)[0]
    out = {}
    for i in range(tc):
        t, rb, cnt, off, res = struct.unpack_from('<IIQQQ', d, do+i*32)
        out[t] = (cnt, d[off:off+cnt*rb])
    return out
WAIT = 0x54494157
rc = 0
for pa in sorted(glob.glob(os.path.join(sys.argv[1], '*.r'))):
    n = os.path.basename(pa)
    pb = os.path.join(sys.argv[2], n)
    ta, tb = tables(pa), tables(pb)
    if ta.keys() != tb.keys():
        print(f"{n}: table set differs"); rc = 1; continue
    bad = [hex(t) for t in ta if t != WAIT and ta[t] != tb[t]]
    if bad:
        print(f"{n}: non-WAIT tables differ {bad}"); rc = 1
print("COMPARE_RC", rc)
sys.exit(rc)
EOF
    exit $?
fi
ENGINE="$1" DIR="$2" OUT="$3"
mkdir -p "$OUT.wrk"
: > "$OUT"
for job in "$DIR"/*.job; do
    name=$(basename "$job" .job)
    "$ENGINE" simulate --job "$job" --out "$OUT.wrk/$name.r" \
        --journal "$OUT.wrk/$name.j" >/dev/null 2>&1
    rc=$?
    rsha=$(sha256sum "$OUT.wrk/$name.r" 2>/dev/null | cut -d' ' -f1)
    jsha=$(sha256sum "$OUT.wrk/$name.j" 2>/dev/null | cut -d' ' -f1)
    printf "%s\t%s\t%s\t%s\n" "$name" "$rc" "${rsha:--}" "${jsha:--}" \
        >> "$OUT"
done
