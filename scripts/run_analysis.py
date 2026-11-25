from pathlib import Path
import json
from src.data import load_hourly
from src.forecasting import fit_forecasts
from src.stochastic import markov_chain, poisson_diagnostic, simulate_regimes

raw=Path("data/raw/LD2011_2014.txt")
if not raw.exists(): raise FileNotFoundError("Run python scripts/download_data.py first.")
hourly=load_hourly(raw)
Path("data/processed").mkdir(parents=True,exist_ok=True)
Path("reports").mkdir(exist_ok=True)
hourly.to_csv("data/processed/hourly_aggregate.csv")
pred,metrics,train=fit_forecasts(hourly)
pred.to_csv("data/processed/test_predictions.csv")
metrics.to_csv("reports/forecast_metrics.csv")
P,states,thresholds=markov_chain(train.load_kw)
P.to_csv("reports/markov_transition_matrix.csv")
daily,diag=poisson_diagnostic(train.load_kw)
daily.to_csv("data/processed/daily_peak_event_counts.csv",header=True)
sim=simulate_regimes(P.values,states[-1])
summary={"start":str(hourly.index.min()),"end":str(hourly.index.max()),
         "n_hourly":len(hourly),"regime_terciles_kw":thresholds,
         "poisson_diagnostic":diag,"markov_24h_simulation":sim}
Path("reports/analysis_summary.json").write_text(json.dumps(summary,indent=2))
print(metrics); print("\\nTransition matrix:\\n",P); print("\\nPoisson diagnostic:",diag); print("\\n24h Markov simulation:",sim)
