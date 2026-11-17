#!/usr/bin/env python3
"""model_store_step.py: two-rate reading of the store_step cells (written after the data, see protocol-v3 addendum).
  per-SMSP acceptance u1(D) = P(tail 0, mask 1, K=4) / 4 ; per-SM acceptance u4(D) = P(tail 0, mask 15, K=4) / 16
  S(mask, K) = max(K * u1, popcount * K * u4)            (store-pipe period of a writer)
  P(tail, mask, K, n) = max(P(tail, 0, K, n), S)          (tails 1-5; nothing fitted)
  fence: w = P(t6, mask, K) - P(t6, 0, K) - (A - (L(K) - L(1))), A = P(tail 0, mask, K) (measured), L(K) = P(t2, 0, K)
"""
import sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from read_store_step import load, P, G16
import statistics
def pop(m): return bin(m).count('1')
for dev in ['b300', 'h100']:
    d = load(dev, 'a')
    u1 = P(d, G16, 4, 0, 0, 1)[0] / 4; u4 = P(d, G16, 4, 0, 0, 15)[0] / 16
    print('=====', dev, 'u1 %.2f u4 %.2f' % (u1, u4))
    S = lambda m, k: max(k * u1, pop(m) * k * u4) if m else 0.0
    rows = []
    for key, per in sorted(d.items()):
        g = key[:4]; k, t, n, m = key[4:]
        if g != G16 or t in (0, 6) or m == 0: continue
        p0 = P(d, G16, k, t, n, 0)[0]
        w = [i for i in range(4) if (m >> i) & 1][0]
        pred = max(p0, S(m, k)); obs = per[w]
        rows.append((t, k, n, m, p0, obs, pred, obs - pred))
    bad = [r for r in rows if abs(r[7]) > max(0.10 * r[5], 8)]
    print('tails 1-5 cells %d, |obs-pred| > max(10%%, 8): %d; mean abs %.1f max %.1f' % (
        len(rows), len(bad), statistics.mean(abs(r[7]) for r in rows), max(abs(r[7]) for r in rows)))
    for r in rows:
        if abs(r[7]) > 6: print('   tail %d K=%d n=%d mask %2d  P0 %.1f obs %.1f pred %.1f  diff %+.1f%s' % (r + (' <-- miss' if r in bad else '',)))
    # tail 0 masks against S
    print('tail 0 K=4 by popcount: ' + '  '.join('%d: obs %s pred %.0f' % (c, sorted(set(round(P(d, G16, 4, 0, 0, m)[[i for i in range(4) if (m >> i) & 1][0]], 1) for m in range(1, 16) if pop(m) == c)), S((1 << c) - 1, 4)) for c in (1, 2, 3, 4)))
    # bystanders
    print('bystander (tail 3, mask 1, non-writers) - c_n: ' + '  '.join('n%d K%d %+.1f' % (n, k, max(P(d, G16, k, 3, n, 1)[1:]) - P(d, G16, k, 3, n, 0)[0]) for n in (8, 24, 64) for k in (1, 2, 4, 8)))
    # fence
    L = {k: P(d, G16, k, 2, 0, 0)[0] for k in (1, 2, 4, 8)}
    ws = []
    for m in (1, 3, 15):
        for k in (1, 2, 4, 8):
            A = P(d, G16, k, 0, 0, m)[0]
            E = P(d, G16, k, 6, 0, m)[0] - P(d, G16, k, 6, 0, 0)[0]
            w = E - (A - (L[k] - L[1])); w_noov = E - A
            ws.append((m, k, pop(m) * k * 512, A, E, w, w_noov))
    print('fence: clean cost P(t6,0,K)-P(t2,0,K) = ' + ' '.join('%.1f' % (P(d, G16, k, 6, 0, 0)[0] - L[k]) for k in (1, 2, 4, 8)))
    for r in ws: print('   mask %2d K=%d bytes %5d  A %.1f  dirty-clean %.1f  w (loop overlapped) %+.1f  w (no overlap) %+.1f' % r)
    print('   w median %.1f  (K=1 cells: %s)' % (statistics.median(r[5] for r in ws), ' '.join('%.1f' % r[5] for r in ws if r[1] == 1)))
    old = {'b300': (75.26, 569.0), 'h100': (74.51, 751.8)}[dev]
    print('   installed leaf at these bytes: ' + ' '.join('%.0f' % (old[0] + r[2] / old[1]) for r in ws))
