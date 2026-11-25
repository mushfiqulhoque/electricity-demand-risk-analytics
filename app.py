from pathlib import Path
import json

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.visualization import (
    historical_demand,
    forecast_comparison,
    forecast_errors,
    prediction_interval,
    markov_heatmap,
    peak_event_distribution,
    peak_event_trend,
)


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Electricity Demand Risk Analytics",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Probabilistic Electricity Demand Forecasting")
st.subheader("Grid Reliability Analytics")

st.markdown(
    """
    An integrated analytical framework combining
    **Time Series Analysis, Stochastic Processes and Machine Learning**
    for electricity demand forecasting and peak-risk assessment.
    """
)

st.divider()


# =====================================================
# LOAD DATA
# =====================================================

BASE = Path(__file__).resolve().parent

HOURLY_FILE = BASE / "data/processed/hourly_aggregate.csv"
PRED_FILE = BASE / "data/processed/test_predictions.csv"
DAILY_FILE = BASE / "data/processed/daily_peak_event_counts.csv"

SUMMARY_FILE = BASE / "reports/analysis_summary.json"
METRICS_FILE = BASE / "reports/forecast_metrics.csv"
MARKOV_FILE = BASE / "reports/markov_transition_matrix.csv"


required_files = [
    HOURLY_FILE,
    PRED_FILE,
    DAILY_FILE,
    SUMMARY_FILE,
    METRICS_FILE,
    MARKOV_FILE,
]

missing = [str(p.name) for p in required_files if not p.exists()]

if missing:
    st.error("Some project output files are missing:")
    st.write(missing)
    st.info("Run the analysis pipeline before opening the dashboard.")
    st.stop()


@st.cache_data
def load_data():

    hourly = pd.read_csv(
        HOURLY_FILE,
        parse_dates=["timestamp"]
    )

    predictions = pd.read_csv(
        PRED_FILE,
        parse_dates=["timestamp"]
    )

    daily = pd.read_csv(
        DAILY_FILE,
        parse_dates=["timestamp"]
    )

    with open(SUMMARY_FILE, encoding="utf-8") as f:
        summary = json.load(f)

    metrics = pd.read_csv(
        METRICS_FILE,
        index_col=0
    )

    markov = pd.read_csv(
        MARKOV_FILE,
        index_col=0
    )

    return hourly, predictions, daily, summary, metrics, markov


try:
    hourly, pred, daily, summary, metrics, markov = load_data()

except Exception as e:
    st.error(f"Unable to load project data: {e}")
    st.stop()


# =====================================================
# OVERVIEW METRICS
# =====================================================

simulation = summary["markov_24h_simulation"]
poisson = summary["poisson_diagnostic"]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Hourly observations",
    f"{summary['n_hourly']:,}"
)

col2.metric(
    "Random Forest MAPE",
    f"{metrics.loc['Random Forest', 'MAPE_percent']:.2f}%"
)

col3.metric(
    "High-demand probability",
    f"{simulation['probability_any_high']:.1%}"
)

col4.metric(
    "Peak-count dispersion",
    f"{poisson['dispersion_index']:.2f}"
)

st.caption(
    f"Study period: {summary['start']} to {summary['end']}"
)

st.divider()


# =====================================================
# DASHBOARD NAVIGATION
# =====================================================

tabs = st.tabs([
    "Overview",
    "Demand Analysis",
    "Forecasting",
    "Prediction Uncertainty",
    "Markov Chain",
    "Peak Risk"
])


# =====================================================
# TAB 1: OVERVIEW
# =====================================================

with tabs[0]:

    st.header("Project Overview")

    st.write(
        """
        This project investigates electricity demand using a hybrid
        statistical and machine-learning framework.

        The analytical framework contains three major components:
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Time Series")
        st.write(
            """
            - Historical demand analysis
            - Lag-based feature engineering
            - Seasonal naive benchmark
            - Forecast evaluation
            """
        )

    with c2:
        st.subheader("Machine Learning")
        st.write(
            """
            - Random Forest regression
            - Quantile Gradient Boosting
            - Forecast error analysis
            - Prediction intervals
            """
        )

    with c3:
        st.subheader("Stochastic Processes")
        st.write(
            """
            - Markov demand regimes
            - Transition probability matrix
            - Poisson dispersion diagnostic
            - Monte Carlo simulation
            """
        )

    st.divider()

    st.subheader("Historical Electricity Demand")

    st.plotly_chart(
        historical_demand(hourly),
        use_container_width=True
    )

    st.info(
        "The high-demand probability shown in this dashboard is a "
        "model-based simulation result. It is not a probability of "
        "electricity outage or grid failure."
    )


# =====================================================
# TAB 2: DEMAND ANALYSIS
# =====================================================

with tabs[1]:

    st.header("Historical Demand Analysis")

    recent_days = st.slider(
        "Select historical period (days)",
        min_value=7,
        max_value=365,
        value=30,
        key="demand_days"
    )

    recent = hourly.tail(recent_days * 24)

    st.plotly_chart(
        px.line(
            recent,
            x="timestamp",
            y="load_kw",
            title=f"Electricity Demand: Last {recent_days} Days",
            labels={
                "timestamp": "Date",
                "load_kw": "Demand (kW)"
            }
        ),
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Average demand",
        f"{hourly['load_kw'].mean():,.2f} kW"
    )

    c2.metric(
        "Maximum demand",
        f"{hourly['load_kw'].max():,.2f} kW"
    )

    c3.metric(
        "Minimum demand",
        f"{hourly['load_kw'].min():,.2f} kW"
    )

    st.subheader("Demand Distribution")

    fig = px.histogram(
        hourly,
        x="load_kw",
        nbins=60,
        title="Distribution of Hourly Electricity Demand",
        labels={"load_kw": "Demand (kW)"}
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =====================================================
# TAB 3: FORECASTING
# =====================================================

with tabs[2]:

    st.header("Electricity Demand Forecasting")

    st.caption(
        "One-step-ahead predictions using lagged features. "
        "The 24-hour seasonal naive model is used as a benchmark."
    )

    st.subheader("Model Performance")

    st.dataframe(
        metrics.style.format("{:.4f}"),
        use_container_width=True
    )

    st.subheader("Actual vs Predicted Demand")

    fig = forecast_comparison(pred.tail(14 * 24))

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Forecast Error Distribution")

    st.plotly_chart(
        forecast_errors(pred),
        use_container_width=True
    )

    st.subheader("Forecast Performance Comparison")

    model_metrics = metrics.loc[
        ["Seasonal naive (24h)", "Random Forest",
         "Gradient Boosting (median)"],
        ["MAE", "RMSE"]
    ].reset_index(names="Model")

    fig = px.bar(
        model_metrics.melt(
            id_vars="Model",
            var_name="Metric",
            value_name="Error"
        ),
        x="Model",
        y="Error",
        color="Metric",
        barmode="group",
        title="Forecast Error Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =====================================================
# TAB 4: PREDICTION UNCERTAINTY
# =====================================================

with tabs[3]:

    st.header("Probabilistic Forecasting")

    st.write(
        """
        Quantile Gradient Boosting estimates the conditional
        10th, 50th and 90th percentiles of electricity demand.
        """
    )

    interval = metrics.loc["Quantile interval 10-90%"]

    c1, c2 = st.columns(2)

    c1.metric(
        "Empirical interval coverage",
        f"{interval['coverage']:.2%}"
    )

    c2.metric(
        "Mean interval width",
        f"{interval['mean_width']:,.2f} kW"
    )

    st.plotly_chart(
        prediction_interval(pred.tail(14 * 24)),
        use_container_width=True
    )

    st.caption(
        "Nominal interval coverage is 80%. Empirical coverage "
        "is measured on the held-out observations. Overall coverage "
        "does not establish calibration across all conditions."
    )


# =====================================================
# TAB 5: MARKOV CHAIN
# =====================================================

with tabs[4]:

    st.header("Stochastic Demand Regime Analysis")

    st.write(
        """
        Electricity demand is classified into Low, Normal and High
        regimes. A first-order Markov chain is used to estimate
        transitions between these states.
        """
    )

    st.subheader("Transition Probability Matrix")

    st.plotly_chart(
        markov_heatmap(markov),
        use_container_width=True
    )

    st.dataframe(
        markov.style.format("{:.4f}"),
        use_container_width=True
    )

    st.subheader("24-Hour Monte Carlo Simulation")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Probability of at least one high hour",
        f"{simulation['probability_any_high']:.2%}"
    )

    c2.metric(
        "Expected high-demand hours",
        f"{simulation['expected_high_hours']:.2f}"
    )

    c3.metric(
        "Median high-hour fraction",
        f"{simulation['median_high_hour_fraction']:.2%}"
    )

    st.warning(
        "These are model-based simulation estimates, not "
        "real-world outage probabilities."
    )


# =====================================================
# TAB 6: PEAK RISK
# =====================================================

with tabs[5]:

    st.header("Peak-Demand Event Analysis")

    c1, c2 = st.columns(2)

    c1.metric(
        "Peak-event threshold",
        f"{poisson['threshold_kw']:,.2f} kW"
    )

    c2.metric(
        "Dispersion index",
        f"{poisson['dispersion_index']:.3f}"
    )

    st.subheader("Daily Peak-Event Trend")

    st.plotly_chart(
        peak_event_trend(daily),
        use_container_width=True
    )

    st.subheader("Peak-Event Distribution")

    st.plotly_chart(
        peak_event_distribution(daily),
        use_container_width=True
    )

    st.subheader("Poisson Process Diagnostic")

    col1, col2 = st.columns(2)

    col1.metric(
        "Daily mean",
        f"{poisson['daily_mean']:.3f}"
    )

    col2.metric(
        "Daily variance",
        f"{poisson['daily_variance']:.3f}"
    )

    st.info(poisson["interpretation"])

    st.caption(
        "The dispersion index is a diagnostic statistic. "
        "Overdispersion alone does not establish the exact "
        "underlying point-process distribution."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Probabilistic Electricity Demand Forecasting & Grid "
    "Reliability Analytics | Statistics • Time Series • "
    "Stochastic Processes • Machine Learning"
)

st.caption(
    "Data source: UCI Electricity Load Diagrams 2011–2014."
)