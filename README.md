# Probabilistic Electricity Demand Forecasting & Grid Reliability Analytics

A reproducible portfolio project combining time-series forecasting, machine learning, and stochastic-process analysis.

## Research questions
1. Can ML models beat a seasonal-naive benchmark for next-hour load prediction?
2. How persistent are low, normal, and high electricity-demand regimes?
3. Do daily high-load event counts look compatible with a simple Poisson count model, or are they over-dispersed?
4. What fraction of simulated future hours fall into the high-demand Markov regime?

## Dataset
Official UCI Machine Learning Repository dataset: [ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014). It contains 370 client series at 15-minute frequency, with kW readings. The script downloads the official archive (about 249 MB compressed; extracted file is much larger). Original time labels use Portuguese local time. Dataset citation: Trindade, A. (2015), DOI: 10.24432/C58C86, CC BY 4.0.

## Run
Python 3.10 or 3.11 recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_analysis.py
streamlit run app.py
```

## Methods
- Aggregate all client readings and resample 15-minute load to hourly mean load.
- Chronological holdout: final 60 days.
- Forecasting: 24-hour seasonal-naive benchmark, Random Forest, and quantile Gradient Boosting (10th/50th/90th percentiles).
- Markov chain: training-set terciles define Low/Normal/High states; adjacent-state frequencies estimate the transition matrix.
- Counting-process analysis: count hourly observations above the training 95th percentile per day. Compare count mean and variance as a basic Poisson diagnostic. Equal mean and variance is a Poisson property, but this diagnostic alone does not prove a Poisson process.
- Monte Carlo: simulate 24-hour regime paths from the estimated transition matrix.

## Important interpretation limits
This is Portuguese client-level historical data, not Bangladesh national demand or a live grid dataset. It contains no weather, generation capacity, outage, tariff, or reserve-margin data. “High risk” means high relative to a historical quantile threshold, not blackout probability. The Poisson diagnostic is not proof of independent/stationary increments. Markov states are statistical regimes, not official operating states. The forecast model predicts one hour ahead from lagged features; do not describe its score as a 24-hour-ahead forecast score.

## Academic alignment
The uploaded Stochastic Process notes define a counting process N(t) as the number of events up to t; a Poisson process additionally requires independent and stationary increments and Poisson interval counts with mean λt. The Markov-chain notes define the Markov property and transition probabilities, with multi-step transitions obtained from powers of the transition matrix. The uploaded Brockwell & Davis book supports the time-series framework: stationarity, ARMA/ARIMA, forecasting, and diagnostics. This first version deliberately focuses on the concepts that can be operationalized credibly with this dataset.

## Suggested extensions
Add weather data; compare SARIMA using rolling-origin validation; add conformal interval calibration; test negative-binomial/non-homogeneous event models if counts are over-dispersed; validate against a reliable Bangladesh load dataset before making local claims.
