# Research plan

**Title:** Probabilistic Electricity Demand Forecasting and Peak-Regime Risk Analysis Using Time Series, Markov Chains, and Machine Learning.

## Testable hypotheses
- H1: Random Forest or Gradient Boosting improves on seasonal naive in chronological holdout.
- H2: High-demand regimes are persistent (High → High transition probability).
- H3: Daily high-load event counts may be over-dispersed relative to a homogeneous Poisson count model.

These are hypotheses, not promised findings. Report negative results too.

## Viva/interview questions
1. Why is seasonal naive a meaningful benchmark?
2. Why is random train/test splitting inappropriate for time series?
3. What is the Markov property here?
4. Why must each transition-matrix row sum to one?
5. Interpret P(High → High).
6. What assumptions define a Poisson process?
7. Why is a variance-to-mean check insufficient to prove a Poisson process?
8. Why estimate thresholds on training data only?
9. Distinguish point forecasts from prediction intervals.
10. What extra data would be required for actual grid reliability claims?

## Before publishing
Run the pipeline yourself, inspect data quality, save package versions, interpret actual results, cite UCI, and do not publish unexecuted performance figures.
