import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# --------------------------------------------------
# 1. HISTORICAL ELECTRICITY DEMAND
# --------------------------------------------------

def historical_demand(hourly):

    fig = px.line(
        hourly,
        x="timestamp",
        y="load_kw",
        title="Historical Electricity Demand (2011–2015)",
        labels={
            "timestamp": "Date",
            "load_kw": "Electricity Demand (kW)"
        }
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified"
    )

    return fig


# --------------------------------------------------
# 2. ACTUAL VS PREDICTED DEMAND
# --------------------------------------------------

def forecast_comparison(predictions):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=predictions["timestamp"],
        y=predictions["actual"],
        name="Actual Demand",
        line=dict(width=2)
    ))

    for column in [
        "Seasonal naive (24h)",
        "Random Forest",
        "Gradient Boosting (median)"
    ]:

        fig.add_trace(go.Scatter(
            x=predictions["timestamp"],
            y=predictions[column],
            name=column,
            line=dict(width=1.5)
        ))

    fig.update_layout(
        title="Actual vs Predicted Electricity Demand",
        xaxis_title="Date",
        yaxis_title="Demand (kW)",
        template="plotly_white",
        hovermode="x unified"
    )

    return fig


# --------------------------------------------------
# 3. FORECAST ERROR DISTRIBUTION
# --------------------------------------------------

def forecast_errors(predictions):

    fig = go.Figure()

    for column in [
        "Seasonal naive (24h)",
        "Random Forest",
        "Gradient Boosting (median)"
    ]:

        errors = predictions["actual"] - predictions[column]

        fig.add_trace(go.Histogram(
            x=errors,
            name=column,
            opacity=0.65,
            nbinsx=50
        ))

    fig.update_layout(
        title="Forecast Error Distribution",
        xaxis_title="Forecast Error (kW)",
        yaxis_title="Frequency",
        barmode="overlay",
        template="plotly_white"
    )

    return fig


# --------------------------------------------------
# 4. QUANTILE PREDICTION INTERVAL
# --------------------------------------------------

def prediction_interval(predictions):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=predictions["timestamp"],
        y=predictions["upper_90"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="Upper 90%"
    ))

    fig.add_trace(go.Scatter(
        x=predictions["timestamp"],
        y=predictions["lower_10"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        name="10–90% Prediction Interval"
    ))

    fig.add_trace(go.Scatter(
        x=predictions["timestamp"],
        y=predictions["actual"],
        mode="lines",
        name="Actual Demand",
        line=dict(width=1.5)
    ))

    fig.update_layout(
        title="Probabilistic Electricity Demand Forecast",
        xaxis_title="Date",
        yaxis_title="Demand (kW)",
        template="plotly_white",
        hovermode="x unified"
    )

    return fig


# --------------------------------------------------
# 5. MARKOV TRANSITION MATRIX
# --------------------------------------------------

def markov_heatmap(matrix):

    fig = px.imshow(
        matrix,
        text_auto=".3f",
        color_continuous_scale="Blues",
        aspect="auto",
        title="Markov Chain: Demand Regime Transition Matrix",
        labels={
            "x": "Next State",
            "y": "Current State",
            "color": "Probability"
        }
    )

    fig.update_layout(template="plotly_white")

    return fig


# --------------------------------------------------
# 6. DAILY PEAK EVENT DISTRIBUTION
# --------------------------------------------------

def peak_event_distribution(daily):

    date_column = "timestamp"

    if date_column not in daily.columns:
        date_column = daily.columns[0]

    count_candidates = [
        c for c in daily.columns
        if c != date_column
    ]

    if not count_candidates:
        raise ValueError("No peak-event count column found.")

    count_column = count_candidates[0]

    fig = px.histogram(
        daily,
        x=count_column,
        nbins=30,
        title="Distribution of Daily Peak Events",
        labels={
            count_column: "Peak Events per Day",
            "count": "Number of Days"
        }
    )

    fig.update_layout(template="plotly_white")

    return fig


# --------------------------------------------------
# 7. DAILY PEAK EVENT TREND
# --------------------------------------------------

def peak_event_trend(daily):

    date_column = "timestamp"

    if date_column not in daily.columns:
        date_column = daily.columns[0]

    count_column = [
        c for c in daily.columns
        if c != date_column
    ][0]

    fig = px.line(
        daily,
        x=date_column,
        y=count_column,
        title="Daily High-Demand Event Frequency",
        labels={
            date_column: "Date",
            count_column: "Peak Events"
        }
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified"
    )

    return fig
