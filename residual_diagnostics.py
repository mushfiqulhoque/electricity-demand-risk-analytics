import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import skew, kurtosis

# Load predictions
df = pd.read_csv(
    "data/processed/test_predictions.csv",
    parse_dates=["timestamp"]
)

# Calculate Random Forest residuals
df["residual"] = df["actual"] - df["Random Forest"]

residuals = df["residual"].dropna()

# Basic residual statistics
print("\n========== RESIDUAL SUMMARY ==========")

print("Observations:", len(residuals))
print("Mean residual:", residuals.mean())
print("Median residual:", residuals.median())
print("Standard deviation:", residuals.std())
print("Skewness:", skew(residuals))
print("Excess kurtosis:", kurtosis(residuals))

# Ljung-Box test
print("\n========== LJUNG-BOX TEST ==========")

lb = acorr_ljungbox(
    residuals,
    lags=[24, 48, 168],
    return_df=True
)

print(lb)

# Residual autocorrelation plot
fig, ax = plt.subplots(figsize=(12, 5))

plot_acf(
    residuals,
    lags=168,
    ax=ax,
    zero=False
)

ax.set_title("Random Forest Residual Autocorrelation")
ax.set_xlabel("Lag (hours)")
ax.set_ylabel("Autocorrelation")

plt.tight_layout()
plt.savefig(
    "reports/residual_acf.png",
    dpi=300
)

plt.show()

# Residual distribution
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    residuals,
    bins=50,
    edgecolor="black"
)

ax.set_title("Distribution of Random Forest Residuals")
ax.set_xlabel("Residual (kW)")
ax.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(
    "reports/residual_distribution.png",
    dpi=300
)

plt.show()

# Residuals over time
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(df["timestamp"], df["residual"])

ax.axhline(0, linestyle="--")

ax.set_title("Random Forest Residuals Over Time")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Residual (kW)")

plt.tight_layout()
plt.savefig(
    "reports/residuals_over_time.png",
    dpi=300
)

plt.show()

print("\nDiagnostic plots saved in reports/")