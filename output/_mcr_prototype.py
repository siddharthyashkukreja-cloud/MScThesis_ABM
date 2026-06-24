import os, sys, time
import numpy as np
import pandas as pd
REPO = "/sessions/exciting-gracious-rubin/mnt/mscthesis_abm"
sys.path.insert(0, REPO)
from model.globals import ModelParams, V0, CALIBRATED, FV_CSV, day_start_steps
from model.simulation import Simulation
from run_simulation import build_traders
DATA = {"calm": f"{REPO}/data/processed/ES_front_calm_1m.csv",
        "stressed": f"{REPO}/data/processed/ES_front_stressed_1m.csv"}
ACF1_CENTERS=(1,5,10,20); ACF2_CENTERS=(1,5,10,20); HILL_FRACS=(0.03,0.04,0.05,0.06,0.07,0.08)
def _acf(x,k):
    if len(x)<=k: return np.nan
    x=x-x.mean(); var=float((x*x).sum())
    return 0.0 if var==0.0 else float((x[:-k]*x[k:]).sum()/var)
def _acf_fwd(x,c):
    vals=[_acf(x,l) for l in (c,c+1,c+2)]; vals=[v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan
def _hill_one(r,frac):
    a=np.abs(np.asarray(r,float)); a=a[np.isfinite(a)&(a>0)]; n=len(a)
    if n<100: return np.nan
    k=max(int(frac*n),20)
    if k>=n: return np.nan
    s=np.sort(a)[::-1]; tk=s[:k]; thr=s[k]
    if thr<=0: return np.nan
    xi=float(np.mean(np.log(tk)-np.log(thr)))
    return float(1.0/xi) if xi>0 else np.nan
def _hill_banded(r,fracs=HILL_FRACS):
    vals=[_hill_one(r,f) for f in fracs]; vals=[v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan
MOMENTS=(["ret_std"]+[f"acf_r_{c}" for c in ACF1_CENTERS]+[f"acf_absr_{c}" for c in ACF2_CENTERS]+["hill_tail_index"])
def compute_moments(r):
    r=np.asarray(r,float); r=r[np.isfinite(r)]
    if len(r)<100: return {m:np.nan for m in MOMENTS}
    a=np.abs(r); out={"ret_std":float(r.std()),"hill_tail_index":_hill_banded(r)}
    for c in ACF1_CENTERS: out[f"acf_r_{c}"]=_acf_fwd(r,c)
    for c in ACF2_CENTERS: out[f"acf_absr_{c}"]=_acf_fwd(a,c)
    return out
def empirical_intraday(regime):
    df=pd.read_csv(DATA[regime],index_col=0,parse_dates=True)
    mid=df["mid"].to_numpy(float); lr=np.diff(np.log(mid)); dates=pd.DatetimeIndex(df.index).date
    return lr[dates[1:]==dates[:-1]]
def bootstrap_cis(regime,B=2000,block=390,seed=0):
    r=empirical_intraday(regime); n=len(r); rng=np.random.default_rng(seed); nb=n//block
    m_emp=compute_moments(r); samples={m:[] for m in MOMENTS}
    for _ in range(B):
        st=rng.integers(0,n-block+1,size=nb); rs=np.concatenate([r[s:s+block] for s in st])
        b=compute_moments(rs)
        for m in MOMENTS: samples[m].append(b[m])
    out={}
    for m in MOMENTS:
        v=np.asarray(samples[m]); v=v[np.isfinite(v)]; se=float(v.std(ddof=1))
        out[m]=dict(emp=float(m_emp[m]),se=se,lo=float(m_emp[m]-1.96*se),hi=float(m_emp[m]+1.96*se))
    return out,m_emp,n
def _intraday_sim(mid,regime):
    r=np.diff(np.log(mid)); drop=[d-1 for d in day_start_steps(regime) if 0<d<=len(r)]
    return np.delete(r,drop) if drop else r
def model_moment_runs(regime,M,n_steps,seed0=1000):
    theta=CALIBRATED[regime]
    p=ModelParams(n_fundamental=40,n_momentum=20,n_zi=40,v0=V0[regime],tick_size=0.25,dt_minutes=1.0,
                  fv_csv=FV_CSV[regime],stressed=(regime=="stressed"),**theta)
    rows=[]
    for i in range(M):
        s=seed0+i; tr=build_traders(p,seed=s)
        hist=Simulation(p,tr,seed=s,ccp=None).run(n_steps)
        mid=pd.Series(hist["mid_price"]).ffill().bfill().to_numpy(); mid=mid[mid>0]
        if len(mid)>1: rows.append(compute_moments(_intraday_sim(mid,regime)))
        print(f"   [{regime}] run {i+1}/{M} done",flush=True)
    return pd.DataFrame(rows)
def run_regime(regime,M,n_steps,B=2000,block=390):
    print(f"\n{'='*78}\n MCR — {regime.upper()}  (M={M}, n_steps={n_steps}, B={B}, block={block})\n{'='*78}")
    cis,m_emp,n_emp=bootstrap_cis(regime,B=B,block=block)
    runs=model_moment_runs(regime,M,n_steps)
    print(f"\n empirical intraday n={n_emp}; model runs={len(runs)}")
    hdr=f"{'moment':<16}{'empirical':>13}{'CI_lo':>13}{'CI_hi':>13}{'mdl_med':>13}{'in?':>5}{'cov%':>7}"
    print("\n"+hdr); print("-"*len(hdr)); covs=[]
    for m in MOMENTS:
        c=cis[m]; mv=runs[m].to_numpy(); mv=mv[np.isfinite(mv)]; med=float(np.median(mv))
        inb=c["lo"]<=med<=c["hi"]; cov=float(np.mean((mv>=c["lo"])&(mv<=c["hi"])))*100.0; covs.append(cov)
        print(f"{m:<16}{c['emp']:>13.5f}{c['lo']:>13.5f}{c['hi']:>13.5f}{med:>13.5f}{('Y' if inb else 'n'):>5}{cov:>7.1f}")
    mean_cov=float(np.mean(covs)); ia=np.ones(len(runs),dtype=bool)
    for m in MOMENTS:
        c=cis[m]; ia&=(runs[m].to_numpy()>=c["lo"])&(runs[m].to_numpy()<=c["hi"])
    joint=float(np.mean(ia))*100.0
    print("-"*len(hdr)); print(f" mean per-moment coverage = {mean_cov:.1f}%")
    print(f" JOINT MCR (inside ALL {len(MOMENTS)} CIs) = {joint:.1f}%")
    return dict(regime=regime,mean_cov=mean_cov,joint=joint)
if __name__=="__main__":
    M=int(os.environ.get("M",30)); B=int(os.environ.get("B",2000))
    only=sys.argv[1] if len(sys.argv)>1 else None
    NSTEPS={"stressed":len(pd.read_csv(FV_CSV["stressed"])),"calm":390*20}
    regimes=(only,) if only in ("calm","stressed") else ("stressed","calm")
    t0=time.time()
    for reg in regimes: run_regime(reg,M,NSTEPS[reg],B=B)
    print(f"\n total wall time {time.time()-t0:.0f}s")
