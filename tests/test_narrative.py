import numpy as np
import pandas as pd

from src.anomalies import Anomaly
from src.confidence import ConfidenceRating
from src.energy_signature import EnergySignatureResult
from src.narrative import monthly_narrative


def _clean_df(n=24, start="2023-01-01", values=None) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="MS")
    consumption = values if values is not None else np.full(n, 300.0)
    return pd.DataFrame({"month_start": idx, "consumption_kwh": consumption})


def _merged_and_result(clean_df: pd.DataFrame, fitted_values: np.ndarray) -> tuple[pd.DataFrame, EnergySignatureResult]:
    idx = clean_df["month_start"]
    merged_df = pd.DataFrame({"month_start": idx, "days_in_month": idx.dt.days_in_month})
    fitted_avg_daily = fitted_values / merged_df["days_in_month"].to_numpy()
    result = EnergySignatureResult(
        intercept=0.0,
        intercept_se=0.0,
        heating_slope=0.0,
        heating_se=0.0,
        heating_pvalue=float("nan"),
        cooling_slope=0.0,
        cooling_se=float("nan"),
        cooling_pvalue=float("nan"),
        r_squared=0.5,
        adj_r_squared=0.5,
        durbin_watson=2.0,
        n_obs=len(idx),
        fitted=pd.Series(fitted_avg_daily, index=pd.DatetimeIndex(idx)),
        resid=pd.Series(0.0, index=pd.DatetimeIndex(idx)),
    )
    return merged_df, result


def test_no_yoy_comparison_when_no_prior_year_data():
    clean_df = _clean_df(n=6, start="2024-01-01")
    text = monthly_narrative(pd.Timestamp("2024-06-01"), clean_df, None, None, [], None)
    assert "No year-on-year comparison" in text


def test_reports_percentage_increase():
    values = np.full(24, 300.0)
    values[23] = 333.0  # December of year 2 is 11% above December of year 1 (300)
    clean_df = _clean_df(n=24, values=values)

    text = monthly_narrative(pd.Timestamp("2024-12-01"), clean_df, None, None, [], None)

    assert "increased 11%" in text
    assert "Enable weather adjustment" in text  # no merged_df/energy_result supplied


def test_weather_explained_fraction_reported():
    values = np.full(24, 300.0)
    values[23] = 400.0  # +100 kWh vs the same month last year (index 11)
    clean_df = _clean_df(n=24, values=values)
    fitted = np.full(24, 300.0)
    fitted[23] = 380.0  # weather alone predicts +80 of that +100 -> 80%
    merged_df, result = _merged_and_result(clean_df, fitted)

    text = monthly_narrative(pd.Timestamp("2024-12-01"), clean_df, merged_df, result, [], None)

    assert "80%" in text
    assert "colder weather" in text


def test_weather_predicts_opposite_direction():
    values = np.full(24, 300.0)
    values[23] = 400.0  # actual increased
    clean_df = _clean_df(n=24, values=values)
    fitted = np.full(24, 300.0)
    fitted[23] = 280.0  # weather alone predicts a *decrease*
    merged_df, result = _merged_and_result(clean_df, fitted)

    text = monthly_narrative(pd.Timestamp("2024-12-01"), clean_df, merged_df, result, [], None)

    assert "opposite direction" in text


def test_weather_more_than_fully_explains_change():
    values = np.full(24, 300.0)
    values[23] = 320.0  # actual +20
    clean_df = _clean_df(n=24, values=values)
    fitted = np.full(24, 300.0)
    fitted[23] = 340.0  # weather alone predicts +40, more than the actual +20
    merged_df, result = _merged_and_result(clean_df, fitted)

    text = monthly_narrative(pd.Timestamp("2024-12-01"), clean_df, merged_df, result, [], None)

    assert "more than fully explains" in text


def test_weather_comparison_unavailable_when_month_missing_from_weather_data():
    values = np.full(24, 300.0)
    values[23] = 400.0
    clean_df = _clean_df(n=24, values=values)
    fitted = np.full(24, 300.0)
    merged_df, result = _merged_and_result(clean_df, fitted)
    merged_df = merged_df.iloc[:-1]  # drop the most recent month's weather data

    text = monthly_narrative(pd.Timestamp("2024-12-01"), clean_df, merged_df, result, [], None)

    assert "Weather-adjusted comparison is unavailable" in text


def test_no_yoy_comparison_when_month_itself_has_no_data():
    clean_df = _clean_df(n=24)
    text = monthly_narrative(pd.Timestamp("2099-01-01"), clean_df, None, None, [], None)
    assert "No year-on-year comparison" in text


def test_negligible_change_skips_weather_sentence():
    values = np.full(24, 300.0)
    values[23] = 300.2  # trivial change
    clean_df = _clean_df(n=24, values=values)
    fitted = np.full(24, 300.0)
    merged_df, result = _merged_and_result(clean_df, fitted)

    text = monthly_narrative(pd.Timestamp("2024-12-01"), clean_df, merged_df, result, [], None)

    assert "explained by" not in text
    assert "opposite direction" not in text


def test_multi_method_anomaly_triggers_recommendation():
    clean_df = _clean_df(n=24)
    anomaly = Anomaly(
        date=pd.Timestamp("2024-12-01"), methods=["rolling_zscore", "stl_esd"], direction="spike", rank_context="1st highest"
    )
    text = monthly_narrative(pd.Timestamp("2024-12-01"), clean_df, None, None, [anomaly], None)
    assert "2 independent anomaly detectors agree" in text
    assert "Recommendation: review heating" in text


def test_single_method_anomaly_not_independently_confirmed():
    clean_df = _clean_df(n=24)
    anomaly = Anomaly(date=pd.Timestamp("2024-12-01"), methods=["rolling_zscore"], direction="spike", rank_context="1st highest")
    text = monthly_narrative(pd.Timestamp("2024-12-01"), clean_df, None, None, [anomaly], None)
    assert "isn't independently confirmed" in text
    assert "No action recommended" in text


def test_no_anomaly_reports_none_detected_and_no_action():
    clean_df = _clean_df(n=24)
    text = monthly_narrative(pd.Timestamp("2024-12-01"), clean_df, None, None, [], None)
    assert "No statistically significant anomalies" in text
    assert "No action recommended" in text


def test_forecast_confidence_sentence_included():
    clean_df = _clean_df(n=24)
    rating = ConfidenceRating("High", "good")
    text = monthly_narrative(pd.Timestamp("2024-12-01"), clean_df, None, None, [], rating)
    assert "Forecast confidence remains high" in text
