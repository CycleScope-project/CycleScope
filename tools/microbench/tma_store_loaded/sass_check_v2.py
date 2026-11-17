#!/usr/bin/env python3
"""store_round (protocol-v2) SASS gate: sass_check_v2.py <cuobjdump -sass output>

16 k_round<STAMPS, MODE, STAGES> instantiations.  k = round-body duplication
factor (STS.128 / 4 for modes 0, 1, 3; UTMASTG count for mode 2).  One
prologue FENCE.VIEW.ASYNC (mbarrier init) is subtracted everywhere.
  mode 0 full : 4k STS.128, k MEMBAR, k FENCE, 2k BAR.SYNC 0x1,0x80, k UTMASTG,
                k UTMACMDFLUSH, k DEPBAR.LE SB0,<STAGES-1>
  mode 1 fence: 4k STS.128, k MEMBAR, k FENCE, 2k BAR.SYNC, no UTMASTG / DEPBAR
  mode 2 store: 2k BAR.SYNC, k UTMASTG, k UTMACMDFLUSH, k DEPBAR; 1 MEMBAR +
                1 FENCE (source-box prologue)
  mode 3 sts  : 4k STS.128, 2k BAR.SYNC, no MEMBAR / FENCE / UTMASTG / DEPBAR
  all: 1 UTMALDG, no STG .128
"""
import re, sys

text = open(sys.argv[1], errors="replace").read()
blocks = re.split(r"\n\s*Function : ", text)[1:]
ok, n = True, 0
for b in blocks:
    name = b.split("\n", 1)[0].strip()
    m = re.search(r"k_roundILi(\d)ELi(\d)ELi(\d)E", name)
    if not m:
        continue
    n += 1
    stamps, mode, stages = (int(x) for x in m.groups())
    ins = re.findall(r"/\*[0-9a-f]{4}\*/\s+(.*?);", b)
    def count(pat):
        return sum(1 for i in ins if re.search(pat, i))
    c = {
        "sts128": count(r"\bSTS\S*\.128\b"), "stg128": count(r"\bSTG\S*\.128\b"),
        "membar": count(r"\bMEMBAR\b"), "fence": count(r"\bFENCE\.VIEW\.ASYNC") - 1,
        "bar1": count(r"\bBAR\.SYNC\S*\s+0x1,\s*0x80"), "utmastg": count(r"\bUTMASTG"),
        "flush": count(r"\bUTMACMDFLUSH"),
        "depbar": count(r"\bDEPBAR\.LE\s+SB0,\s*0x%x\b" % (stages - 1)),
        "depbar_any": count(r"\bDEPBAR"), "utmaldg": count(r"\bUTMALDG"), "clock": count(r"SR_CLOCKLO"),
    }
    if mode == 2:
        k = c["utmastg"]
        want = {"bar1": 2 * k, "flush": k, "depbar": k, "depbar_any": k, "membar": 1, "fence": 1}
    else:
        k = c["sts128"] // 4 if c["sts128"] and c["sts128"] % 4 == 0 else 0
        want = {"sts128": 4 * k, "bar1": 2 * k}
        if mode == 0:
            want.update({"membar": k, "fence": k, "utmastg": k, "flush": k, "depbar": k, "depbar_any": k})
        elif mode == 1:
            want.update({"membar": k, "fence": k, "utmastg": 0, "depbar_any": 0})
        else:
            want.update({"membar": 0, "fence": 0, "utmastg": 0, "depbar_any": 0})
    want.update({"utmaldg": 1, "stg128": 0})
    bad = {key: (c[key], v) for key, v in want.items() if c[key] != v}
    if k < 1:
        bad["k"] = (k, ">=1")
    print(f"{'OK ' if not bad else 'BAD'} stamps={stamps} mode={mode} stages={stages} k={k} {c}" +
          (f" mismatch(found, want)={bad}" if bad else ""))
    ok = ok and not bad
print(f"kernels={n}")
sys.exit(0 if ok and n == 16 else 1)
