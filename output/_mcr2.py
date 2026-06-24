"""Chunked MCR runner: stage-based so each invocation fits in <45s.
  python _mcr2.py ci <regime>            -> compute bootstrap CIs -> output/_ci_<regime>.json
  python _mcr2.py run <regime> <a> <b>   -> sim runs [a,b) appended to output/_runs_<regime>.csv
  python _mcr2.py report <regime>        -> assemble table + joint MCR from the CSV + CI json
"""
import os, sys, json, time
import numpy as np, pandas as pd
REPO="/sessions/exciting-gracious-rubin/mnt/mscthesis_abm"; sys.path.insert(0,REPO)
from model.globals import ModelParams, V0, CALIBRATED, FV_CSV, day_start_steps
from model.simulation import Simulation
from run_simulation import build_traders
DATA={"calm":f"{REPO}/data/processed/ES_front_calm_1m.csv","stressed":f"{REPO}/data/processed/ES_front_stressed_1m.csv"}
ACF1=(1,5,10,20); ACF2=(1,5,10,20); HF=(0.03,0.04,0.05,0.06,0.07,0.08)
def _acf(x,k):
    if len(x)<=k: return np.nan
    x=x-x.mean(); v=float((x*x).sum()); return 0.0 if v==0 else float((x[:-k]*x[k:]).sum()/v)
def _afwd(x,c):
    vv=[_acf(x,l) for l in (c,c+1,c+2)]; vv=[u for u in vv if np.isfinite(u)]; return float(np.mean(vv)) if vv else np.nan
def _h1(r,f):
    a=np.abs(np.asarray(r,float)); a=a[np.isfinite(a)&(a>0)]; n=len(a)
    if n<100: return np.nan
    k=max(int(f*n),20)
    if k>=n: return np.nan
    s=np.sort(a)[::-1]; thr=s[k]
    if thr<=0: return np.nan
    xi=float(np.mean(np.log(s[:k])-np.log(thr))); return float(1/xi) if xi>0 else np.nan
def _hb(r): 
    vv=[_h1(r,f) for f in HF]; vv=[u for u in vv if np.isfinite(u)]; return float(np.mean(vv)) if vv else np.nan
MOM=(["ret_std"]+[f"acf_r_{c}" for c in ACF1]+[f"acf_absr_{c}" for c in ACF2]+["hill_tail_index"])
def moments(r):
    r=np.asarray(r,float); r=r[np.isfinite(r)]
    if len(r)<100: return {m:np.nan for m in MOM}
    a=np.abs(r); o={"ret_std":float(r.std()),"hill_tail_index":_hb(r)}
    for c in ACF1: o[f"acf_r_{c}"]=_afwd(r,c)
    for c in ACF2: o[f"acf_absr_{c}"]=_afwd(a,c)
    return o
def emp(regime):
    df=pd.read_csv(DATA[regime],index_col=0,parse_dates=True); mid=df["mid"].to_numpy(float)
    lr=np.diff(np.log(mid)); d=pd.DatetimeIndex(df.index).date; return lr[d[1:]==d[:-1]]
def cmd_ci(regime,B=2000,block=390,seed=0):
    r=emp(regime); n=len(r); rng=np.random.default_rng(seed); nb=n//block
    me=moments(r); samp={m:[] for m in MOM}
    for _ in range(B):
        st=rng.integers(0,n-block+1,size=nb); rs=np.concatenate([r[s:s+block] for s in st])
        b=moments(rs)
        for m in MOM: samp[m].append(b[m])
    out={}
    for m in MOM:
        v=np.asarray(samp[m]); v=v[np.isfinite(v)]; se=float(v.std(ddof=1))
        out[m]=dict(emp=float(me[m]),se=se,lo=float(me[m]-1.96*se),hi=float(me[m]+1.96*se))
    json.dump({"regime":regime,"n_emp":int(n),"B":B,"block":block,"cis":out},
              open(f"{REPO}/output/_ci_{regime}.json","w"),indent=2)
    print(f"CI {regime}: n_emp={n}, B={B} -> _ci_{regime}.json")
def _isim(mid,regime):
    r=np.diff(np.log(mid)); drop=[d-1 for d in day_start_steps(regime) if 0<d<=len(r)]
    return np.delete(r,drop) if drop else r
def cmd_run(regime,a,b,seed0=1000):
    theta=CALIBRATED[regime]
    nsteps=len(pd.read_csv(FV_CSV["stressed"])) if regime=="stressed" else 390*20
    p=ModelParams(n_fundamental=40,n_momentum=20,n_zi=40,v0=V0[regime],tick_size=0.25,dt_minutes=1.0,
                  fv_csv=FV_CSV[regime],stressed=(regime=="stressed"),**theta)
    path=f"{REPO}/output/_runs_{regime}.csv"
    for i in range(a,b):
        t0=time.time(); s=seed0+i; tr=build_traders(p,seed=s)
        hist=Simulation(p,tr,seed=s,ccp=None).run(nsteps)
        mid=pd.Series(hist["mid_price"]).ffill().bfill().to_numpy(); mid=mid[mid>0]
        m=moments(_isim(mid,regime)); m["seed"]=s
        df=pd.DataFrame([m]); hdr=not os.path.exists(path)
        df.to_csv(path,mode="a",header=hdr,index=False)
        print(f"  {regime} run idx{i} seed{s} {time.time()-t0:.1f}s",flush=True)
def cmd_report(regime):
    ci=json.load(open(f"{REPO}/output/_ci_{regime}.json"))["cis"]
    runs=pd.read_csv(f"{REPO}/output/_runs_{regime}.csv")
    print(f"\n===== MCR {regime.upper()}  (M={len(runs)} runs) =====")
    hdr=f"{'moment':<16}{'empirical':>13}{'CI_lo':>13}{'CI_hi':>13}{'mdl_med':>13}{'in?':>5}{'cov%':>7}"
    print(hdr); print("-"*len(hdr)); covs=[]; ia=np.ones(len(runs),bool)
    for m in MOM:
        c=ci[m]; mv=runs[m].to_numpy(); mv=mv[np.isfinite(mv)]; med=float(np.median(mv))
        inb=c["lo"]<=med<=c["hi"]; cov=float(np.mean((mv>=c["lo"])&(mv<=c["hi"])))*100; covs.append(cov)
        ia&=(runs[m].to_numpy()>=c["lo"])&(runs[m].to_numpy()<=c["hi"])
        print(f"{m:<16}{c['emp']:>13.5f}{c['lo']:>13.5f}{c['hi']:>13.5f}{med:>13.5f}{('Y' if inb else 'n'):>5}{cov:>7.1f}")
    print("-"*len(hdr)); print(f" mean per-moment coverage = {np.mean(covs):.1f}%")
    print(f" JOINT MCR (inside ALL {len(MOM)}) = {np.mean(ia)*100:.1f}%")
if __name__=="__main__":
    c=sys.argv[1]; reg=sys.argv[2]
    if c=="ci": cmd_ci(reg, B=int(os.environ.get("B",2000)))
    elif c=="run": cmd_run(reg,int(sys.argv[3]),int(sys.argv[4]))
    elif c=="report": cmd_report(reg)
