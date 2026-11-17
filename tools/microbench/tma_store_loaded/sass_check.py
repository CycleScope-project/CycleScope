#!/usr/bin/env python3
"""tma_store_loaded SASS gate: sass_check.py <cuobjdump -sass output>

Per k_epilogue<STAMPS, KIND, STAGES> instantiation (KIND 0 = tma, 1 = stg):
  tma: exactly 4 STS .128, 1 MEMBAR + 1 FENCE.VIEW.ASYNC (the async-proxy
       fence), 2 BAR.SYNC on named barrier 1 with 128 threads, 1 UTMASTG,
       1 UTMACMDFLUSH, 1 DEPBAR.LE SB0 with immediate STAGES - 1, 0 STG .128
  stg: exactly 4 STG .128, 0 UTMASTG, 0 DEPBAR, 0 named-barrier BAR.SYNC
  both: 1 UTMALDG (loader), 1 prologue FENCE.VIEW.ASYNC (the lowered
        fence.mbarrier_init of the loader mbarriers, outside the rounds),
        SR_CLOCKLO reads counted and printed
ptxas may duplicate the round body (loop peeling around the warm-round
test); the counts then scale by one integer factor k = STS/4 (tma) or
STG/4 (stg) that every per-round class must share.  0 kernels found or
!= 8 kernels = failure.
"""
import re, sys

text = open(sys.argv[1], errors="replace").read()
blocks = re.split(r"\n\s*Function : ", text)[1:]
ok, n = True, 0
for b in blocks:
    name = b.split("\n", 1)[0].strip()
    m = re.search(r"k_epilogueILi(\d)ELi(\d)ELi(\d)E", name)
    if not m:
        continue
    n += 1
    stamps, kind, stages = (int(x) for x in m.groups())
    ins = re.findall(r"/\*[0-9a-f]{4}\*/\s+(.*?);", b)
    def count(pat):
        return sum(1 for i in ins if re.search(pat, i))
    c = {
        "sts128": count(r"\bSTS\S*\.128\b"),
        "stg128": count(r"\bSTG\S*\.128\b"),
        "membar": count(r"\bMEMBAR\b"),
        "fence": count(r"\bFENCE\.VIEW\.ASYNC") - 1,  # minus the prologue mbarrier-init fence
        "bar1": count(r"\bBAR\.SYNC\S*\s+0x1,\s*0x80"),
        "utmastg": count(r"\bUTMASTG"),
        "flush": count(r"\bUTMACMDFLUSH"),
        "depbar": count(r"\bDEPBAR\.LE\s+SB0,\s*0x%x\b" % (stages - 1)),
        "depbar_any": count(r"\bDEPBAR"),
        "utmaldg": count(r"\bUTMALDG"),
        "clock": count(r"SR_CLOCKLO"),
    }
    if kind == 0:
        want = {"sts128": 4, "stg128": 0, "membar": 1, "fence": 1, "bar1": 2, "utmastg": 1, "flush": 1,
                "depbar": 1, "depbar_any": 1, "utmaldg": 1}
    else:
        want = {"stg128": 4, "utmastg": 0, "depbar_any": 0, "bar1": 0, "utmaldg": 1, "fence": 0, "membar": 0}
    per_round = c["sts128"] if kind == 0 else c["stg128"]
    k = per_round // 4 if per_round and per_round % 4 == 0 else 1
    fixed = {"utmaldg"}
    bad = {key: (c[key], v * (1 if key in fixed else k)) for key, v in want.items()
           if c[key] != v * (1 if key in fixed else k)}
    print(f"{'OK ' if not bad else 'BAD'} stamps={stamps} kind={kind} stages={stages} k={k} {c}" +
          (f" mismatch(found, want)={bad}" if bad else ""))
    ok = ok and not bad
print(f"kernels={n}")
sys.exit(0 if ok and n == 8 else 1)
