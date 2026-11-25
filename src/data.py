from pathlib import Path
import zipfile
import requests
import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"


def download_dataset(path="data/raw/LD2011_2014.txt"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return path

    archive = path.with_suffix(".txt.zip")
    print("Downloading official UCI archive; this is a large file...")

    with requests.get(URL, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(archive, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

    with zipfile.ZipFile(archive) as z:
        member = next(
            n for n in z.namelist()
            if n.endswith("LD2011_2014.txt")
        )

        with z.open(member) as src, open(path, "wb") as dst:
            while chunk := src.read(1024 * 1024):
                dst.write(chunk)

    return path


def load_hourly(path):
    raw = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        low_memory=False
    )

    raw = raw.rename(columns={raw.columns[0]: "timestamp"})
    raw["timestamp"] = pd.to_datetime(
        raw["timestamp"],
        errors="coerce"
    )

    raw = (
        raw.dropna(subset=["timestamp"])
        .set_index("timestamp")
        .sort_index()
    )

    numeric = raw.apply(pd.to_numeric, errors="coerce")

    aggregate_15m = numeric.sum(axis=1, min_count=1)

    hourly = aggregate_15m.resample("h").mean().dropna()

    hourly = hourly.to_frame("load_kw")
    hourly.index.name = "timestamp"

    # Correct three identified spring DST aggregation anomalies.
    dst_anomalies = pd.to_datetime([
        "2012-03-25 01:00:00",
        "2013-03-31 01:00:00",
        "2014-03-30 01:00:00"
    ])

    for timestamp in dst_anomalies:
        previous_hour = timestamp - pd.Timedelta(hours=1)
        next_hour = timestamp + pd.Timedelta(hours=1)

        if (
            previous_hour in hourly.index
            and timestamp in hourly.index
            and next_hour in hourly.index
        ):
            original_value = hourly.loc[timestamp, "load_kw"]

            corrected_value = (
                hourly.loc[previous_hour, "load_kw"]
                + hourly.loc[next_hour, "load_kw"]
            ) / 2

            hourly.loc[timestamp, "load_kw"] = corrected_value

            print(
                f"DST correction: {timestamp} | "
                f"{original_value:.2f} -> {corrected_value:.2f} kW"
            )

    return hourly
