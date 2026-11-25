import pandas as pd
import matplotlib.pyplot as plt

# Load test predictions
df = pd.read_csv(
    "data/processed/test_predictions.csv",
    parse_dates=["timestamp"]
)

# Calculate residuals
df["residual"] = df["actual"] - df["Random Forest"]

# Extract time components
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.day_name()
df["month"] = df["timestamp"].dt.month

# 1. Average residual by hour
hourly_residual = df.groupby("hour")["residual"].mean()

plt.figure(figsize=(12, 5))
hourly_residual.plot(marker="o")
plt.axhline(0, linestyle="--")
plt.title("Average Random Forest Residual by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Mean Residual (kW)")
plt.xticks(range(24))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reports/residual_by_hour.png", dpi=300)
plt.show()

# 2. Average residual by weekday
weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekday_residual = (
    df.groupby("weekday")["residual"]
    .mean()
    .reindex(weekday_order)
)

plt.figure(figsize=(10, 5))
weekday_residual.plot(kind="bar")
plt.axhline(0, linestyle="--")
plt.title("Average Random Forest Residual by Weekday")
plt.xlabel("Weekday")
plt.ylabel("Mean Residual (kW)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("reports/residual_by_weekday.png", dpi=300)
plt.show()

# 3. Average residual by month
monthly_residual = df.groupby("month")["residual"].mean()

plt.figure(figsize=(10, 5))
monthly_residual.plot(marker="o")
plt.axhline(0, linestyle="--")
plt.title("Average Random Forest Residual by Month")
plt.xlabel("Month")
plt.ylabel("Mean Residual (kW)")
plt.xticks(range(1, 13))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reports/residual_by_month.png", dpi=300)
plt.show()

# Print numerical summaries
print("\n===== HOURLY RESIDUALS =====")
print(hourly_residual.round(2))

print("\n===== WEEKDAY RESIDUALS =====")
print(weekday_residual.round(2))

print("\n===== MONTHLY RESIDUALS =====")
print(monthly_residual.round(2))

print("\nAll plots saved in reports/")
