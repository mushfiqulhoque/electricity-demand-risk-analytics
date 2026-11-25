import numpy as np
import pandas as pd

NAMES = ["Low","Normal","High"]

def markov_chain(train):
    q1,q2 = train.quantile([1/3,2/3])
    states = np.where(train <= q1,0,np.where(train <= q2,1,2))
    counts = np.zeros((3,3))
    for a,b in zip(states[:-1],states[1:]):
        counts[int(a),int(b)] += 1
    P = counts / np.maximum(counts.sum(axis=1,keepdims=True),1)
    return pd.DataFrame(P,index=NAMES,columns=NAMES), states, (float(q1),float(q2))

def poisson_diagnostic(train):
    threshold = float(train.quantile(.95))
    daily = (train > threshold).astype(int).resample("D").sum()
    mean, var = float(daily.mean()), float(daily.var())
    dispersion = var/mean if mean else float("nan")
    return daily, {"threshold_kw":threshold,"daily_mean":mean,"daily_variance":var,
        "dispersion_index":dispersion,
        "interpretation":"Possible over-dispersion; simple homogeneous Poisson counts may be inadequate." if dispersion > 1.5 else "No strong over-dispersion signal in this simple check; this does not establish a Poisson process."}

def simulate_regimes(P,last_state,horizon=24,n=5000,seed=42):
    rng=np.random.default_rng(seed); P=np.asarray(P); state=np.full(n,int(last_state)); paths=np.zeros((n,horizon),int)
    for t in range(horizon):
        nxt=np.empty(n,int)
        for s in range(3):
            mask=state==s
            if mask.any(): nxt[mask]=rng.choice(3,size=mask.sum(),p=P[s])
        paths[:,t]=nxt; state=nxt
    high=(paths==2)
    return {"probability_any_high":float(high.any(axis=1).mean()),
            "expected_high_hours":float(high.sum(axis=1).mean()),
            "median_high_hour_fraction":float(high.mean(axis=1).mean())}
