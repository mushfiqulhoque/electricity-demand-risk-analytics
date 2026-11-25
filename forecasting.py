import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURES = [
    "hour", "weekday", "month", "weekend",
    "lag1", "lag2", "lag3",
    "lag24", "lag48", "lag168",
    "mean24", "std24", "mean168", "std168"
]


def make_features(df):
    x = df.copy()
    ix = x.index

    x["hour"] = ix.hour
    x["weekday"] = ix.dayofweek
    x["month"] = ix.month
    x["weekend"] = (ix.dayofweek >= 5).astype(int)

    for lag in [1, 2, 3, 24, 48, 168]:
        x[f"lag{lag}"] = x["load_kw"].shift(lag)

    for w in [24, 168]:
        x[f"mean{w}"] = x["load_kw"].shift(1).rolling(w).mean()
        x[f"std{w}"] = x["load_kw"].shift(1).rolling(w).std()

    return x.dropna()


def metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p)

    m = np.abs(y) > 1e-8

    mape = (
        np.mean(np.abs((y[m] - p[m]) / y[m])) * 100
        if m.any()
        else np.nan
    )

    return {
        "MAE": mean_absolute_error(y, p),
        "RMSE": mean_squared_error(y, p) ** 0.5,
        "MAPE_percent": mape
    }


def fit_forecasts(hourly, test_days=60):

    cutoff = hourly.index.max() - pd.Timedelta(days=test_days)

    f = make_features(hourly)

    tr = f[f.index < cutoff]
    te = f[f.index >= cutoff]

    if tr.empty or te.empty:
        raise ValueError(
            "Insufficient observations for selected holdout."
        )

    X = tr[FEATURES]
    y = tr["load_kw"]

    Xt = te[FEATURES]
    yt = te["load_kw"]

    pred = pd.DataFrame(index=te.index)

    pred["actual"] = yt

    # Seasonal naive benchmark
    pred["Seasonal naive (24h)"] = te["lag24"]

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        max_features=0.9,
        n_jobs=-1,
        random_state=42
    )

    rf.fit(X, y)

    pred["Random Forest"] = rf.predict(Xt)

    # Quantile Gradient Boosting
    qpred = {}

    for q in [0.1, 0.5, 0.9]:

        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            n_estimators=180,
            learning_rate=0.04,
            max_depth=2,
            random_state=42
        )

        model.fit(X, y)

        qpred[q] = model.predict(Xt)

    # Stack quantile predictions
    predictions = np.column_stack([
        qpred[0.1],
        qpred[0.5],
        qpred[0.9]
    ])

    # Prevent quantile crossing
    predictions = np.sort(predictions, axis=1)

    pred["lower_10"] = predictions[:, 0]

    pred["Gradient Boosting (median)"] = predictions[:, 1]

    pred["upper_90"] = predictions[:, 2]

    # Forecast evaluation
    rows = {}

    for name in [
        "Seasonal naive (24h)",
        "Random Forest",
        "Gradient Boosting (median)"
    ]:

        rows[name] = metrics(yt, pred[name])

    # Prediction interval evaluation
    rows["Quantile interval 10-90%"] = {
        "coverage": np.mean(
            (yt >= pred["lower_10"]) &
            (yt <= pred["upper_90"])
        ),

        "mean_width": np.mean(
            pred["upper_90"] - pred["lower_10"]
        )
    }

    return (
        pred,
        pd.DataFrame(rows).T,
        hourly.loc[hourly.index < cutoff]
    )