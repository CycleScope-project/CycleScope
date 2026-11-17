#!/usr/bin/env python3
"""read_store_step.py: tables of the store_step cells (shared protocol-v3), both runs."""
import json, os, sys, collections
R=(sys.argv[1] if len(sys.argv)>1 else
   os.environ['CYCLESCOPE_ARTIFACT_ROOT']+'/fg-line-sync/store-step')
def load(dev, run):
    d={}
    for l in open(f'{R}/{dev}-r1/run-{run}.jsonl'):
        r=json.loads(l)
        key=(r['wide'],r['stride'],r['kstep'],r['wshift'],r['k'],r['tail'],r['nfill'],r['mask'])
        d[key]=r['period_median']
    return d
G16=(1,64,16,0)
def P(d,g,k,t,n,m): return d.get(g+(k,t,n,m))
def fmt(p): return 'None' if p is None else '/'.join('%.1f'%x for x in p)
if __name__=='__main__':
  for dev in ['b300','h100']:
    a=load(dev,'a'); b=load(dev,'b')
    worst=max(abs(x-y)/max(x,1e-9) for k in a for x,y in zip(a[k],b[k]))
    print('=====',dev,'cells',len(a),'max a/b rel diff %.4f'%worst)
    d=a
    print('-- A: tail 0, G16, K=4: per-warp periods by mask')
    for m in range(1,16): print('  mask %2d (%s) %s'%(m,format(m,'04b')[::-1],fmt(P(d,G16,4,0,0,m))))
    for k in (1,2,8):
        for m in (1,3,15): print('  K=%d mask %2d %s'%(k,m,fmt(P(d,G16,k,0,0,m))))
    print('-- B: tail 3 (stores + FILL), G16')
    for n in (8,24,64):
        for k in (1,2,4,8):
            print('  n=%2d K=%d '%(n,k)+'  '.join('m%d %s'%(m,fmt(P(d,G16,k,3,n,m))) for m in (0,1,3,15)))
    print('-- C: tail 2 (BAR x2), G16, K=4 by mask')
    for m in range(0,16): print('  mask %2d (%s) %s'%(m,format(m,'04b')[::-1],fmt(P(d,G16,4,2,0,m))))
    for k in (1,2,8):
        print('  K=%d '%k+'  '.join('m%d %.1f'%(m,P(d,G16,k,2,0,m)[0]) for m in (0,1,3,15)))
    print('-- C: tail 1 (BAR x1)')
    print('  K=4 '+'  '.join('m%d %.1f'%(m,P(d,G16,4,1,0,m)[0]) for m in (0,1,3,7,15)))
    for k in (1,2,8): print('  K=%d '%k+'  '.join('m%d %.1f'%(m,P(d,G16,k,1,0,m)[0]) for m in (0,1,15)))
    print('-- D: tail 4 (FILL, BARx2) / tail 5 (BARx2, FILL), K=4')
    for t in (4,5):
        for n in (24,64): print('  tail %d n=%d '%(t,n)+'  '.join('m%d %.1f'%(m,P(d,G16,4,t,n,m)[0]) for m in (0,1,3,15)))
    print('-- E: geometries: tail 0 masks 1/3/15 (per-warp), tail 2 masks 0/1/3/15 (warp 0)')
    for name,g,k in [('G16same',(1,64,512,0),4),('G16shift',(1,64,16,16),4),('G4',(1,16,512,0),4),('N1',(0,4,128,0),8),('N16',(0,64,4,0),4),('N32',(0,128,4,0),4)]:
        print('  %-8s t0: '%name+'  '.join('m%d %s'%(m,fmt(P(d,g,k,0,0,m))) for m in (1,3,15)))
        print('  %-8s t2: '%name+'  '.join('m%d %.1f'%(m,P(d,g,k,2,0,m)[0]) for m in (0,1,3,15)))
    print('-- F: tail 6 (fence), G16:   P(t6) and Df = P(t6)-P(t2)')
    for k in (1,2,4,8):
        print('  K=%d '%k+'  '.join('m%d %.1f (Df %.1f)'%(m,P(d,G16,k,6,0,m)[0],P(d,G16,k,6,0,m)[0]-P(d,G16,k,2,0,m)[0]) for m in (0,1,3,15)))
    G4=(1,16,512,0)
    print('-- G: G4: tail 6, tail 2, Df; tail 3 n=24')
    for k in (1,2,4,8):
        print('  K=%d '%k+'  '.join('m%d t6 %.1f t2 %.1f (Df %.1f)'%(m,P(d,G4,k,6,0,m)[0],P(d,G4,k,2,0,m)[0],P(d,G4,k,6,0,m)[0]-P(d,G4,k,2,0,m)[0]) for m in (0,1,3,15)))
    for k in (4,8): print('  t3 n=24 K=%d '%k+'  '.join('m%d %s'%(m,fmt(P(d,G4,k,3,24,m))) for m in (0,1,15)))
