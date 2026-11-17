#!/usr/bin/env python3
"""port_pairs (protocol-v3) SASS gate: sass_check_v3.py <cuobjdump -sass output>

18 k_round<STAMPS, MODE, STAGES> instantiations (the 16 of v2 + mode 4 stg + mode 5 bars).  k = round-body duplication
factor (STS.128 / 4 for modes 0, 1, 3; UTMASTG count for mode 2).  One
prologue FENCE.VIEW.ASYNC (mbarrier init) is subtracted everywhere.
  mode 0 full : 4k STS.128, k MEMBAR, k FENCE, 2k BAR.SYNC 0x1,0x80, k UTMASTG,
                k UTMACMDFLUSH, k DEPBAR.LE SB0,<STAGES-1>
  mode 1 fence: 4k STS.128, k MEMBAR, k FENCE, 2k BAR.SYNC, no UTMASTG / DEPBAR
  mode 2 store: 2k BAR.SYNC, k UTMASTG, k UTMACMDFLUSH, k DEPBAR; 1 MEMBAR +
                1 FENCE (source-box prologue)
  mode 3 sts  : 4k STS.128, 2k BAR.SYNC, no MEMBAR / FENCE / UTMASTG / DEPBAR
  mode 4 stg  : 4k STG.128 (k = STG.128 / 4), 2k BAR.SYNC, no STS.128 / MEMBAR /
                FENCE / UTMASTG / DEPBAR.LE SB0,3 in the round
  all: 1 UTMALDG, >= 4 LDGSTS and >= 8 LDG .128 (the three loader kinds); the
       ldgsts loader holds one DEPBAR.LE SB0,0x3 and one DEPBAR.LE SB0,0x0, which
       are subtracted from the round counts (depbar - 1, depbar_any - 2);
       modes 0-3: no STG .128
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
        "depbar": count(r"\bDEPBAR\.LE\s+SB0,\s*0x%x\b" % (stages - 1)) - 1,
        "depbar_any": count(r"\bDEPBAR\.LE") - 2, "utmaldg": count(r"\bUTMALDG"),
        "clock": count(r"SR_CLOCKLO"), "ldgsts": count(r"\bLDGSTS"), "ldg128": count(r"\bLDG\.[A-Z.]*128\b"),
    }
    if mode == 5:
        k = c["bar1"] // 2 if c["bar1"] and c["bar1"] % 2 == 0 else 0
        want = {"sts128": 0, "stg128": 0, "membar": 0, "fence": 0, "utmastg": 0}
    elif mode == 4:
        k = c["stg128"] // 4 if c["stg128"] and c["stg128"] % 4 == 0 else 0
        want = {"stg128": 4 * k, "bar1": 2 * k, "sts128": 0, "membar": 0, "fence": 0, "utmastg": 0}
    elif mode == 2:
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
    want.update({"utmaldg": 1})
    if mode not in (4, 5):
        want["stg128"] = 0
    if c["ldgsts"] < 4 or c["ldg128"] < 8:
        want["ldgsts"] = ">=4"; want["ldg128"] = ">=8"
    bad = {key: (c[key], v) for key, v in want.items() if c[key] != v}
    if k < 1:
        bad["k"] = (k, ">=1")
    print(f"{'OK ' if not bad else 'BAD'} stamps={stamps} mode={mode} stages={stages} k={k} {c}" +
          (f" mismatch(found, want)={bad}" if bad else ""))
    ok = ok and not bad
print(f"kernels={n}")
sys.exit(0 if ok and n == 18 else 1)
