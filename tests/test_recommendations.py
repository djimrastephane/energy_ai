import numpy as np
import pandas as pd

from src.anomalies import Anomaly
from src.confidence import ConfidenceRating
from src.energy_signature import EnergySignatureResult
from src.investigation import InvestigationChecklist, InvestigationItem
from src.recommendations import (
    generate_recommendations,
    recommend_heating_review,
    recommend_investigate_anomaly,
    recommend_more_data,
)


def _merged_df(n=26, start="2023-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="MS")
    return pd.DataFrame(
        {
            "month_start": idx,
            "avg_daily_hdd": np.full(n, 5.0),
            "avg_daily_cdd": np.zeros(n),
            "days_in_month": idx.days_in_month,
        }
    )


def _clean_df(n=26, start="2023-01-01", unit_rate=0.3) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="MS")
    consumption = np.full(n, 300.0)
    return pd.DataFrame(
        {
            "month_start": idx,
            "consumption_kwh": consumption,
            "cost_gbp": consumption * unit_rate,
            "unit_rate_gbp_per_kwh": np.full(n, unit_rate),
        }
    )


def _energy_result(resid_series, heating_p=0.001, heating_slope=1.6) -> EnergySignatureResult:
    return EnergySignatureResult(
        intercept=1.5,
        intercept_se=0.2,
        heating_slope=heating_slope,
        heating_se=0.1,
        heating_pvalue=heating_p,
        cooling_slope=0.0,
        cooling_se=float("nan"),
        cooling_pvalue=float("nan"),
        r_squared=0.7,
        adj_r_squared=0.69,
        durbin_watson=1.8,
        n_obs=len(resid_series),
        fitted=pd.Series(dtype=float),
        resid=resid_series,
    )


def test_recommend_heating_review_none_without_weather_data():
    clean_df = _clean_df()
    assert recommend_heating_review(clean_df, None, None) is None


def test_recommend_heating_review_none_when_heating_not_significant():
    merged_df = _merged_df()
    idx = merged_df["month_start"]
    resid = pd.Series(np.zeros(len(idx)), index=idx)
    energy_result = _energy_result(resid, heating_p=0.6)
    assert recommend_heating_review(_clean_df(), merged_df, energy_result) is None


def test_recommend_heating_review_none_without_complete_winter():
    # Only Jan-Nov present in every year -- no season ever has all of Dec/Jan/Feb.
    idx = pd.date_range("2023-01-01", periods=11, freq="MS")
    merged_df = pd.DataFrame(
        {"month_start": idx, "avg_daily_hdd": np.full(11, 5.0), "days_in_month": idx.days_in_month}
    )
    resid = pd.Series(np.zeros(11), index=idx)
    energy_result = _energy_result(resid, heating_p=0.001)
    assert recommend_heating_review(_clean_df(n=11), merged_df, energy_result) is None


def test_recommend_heating_review_none_when_winter_not_elevated():
    merged_df = _merged_df(n=26)
    idx = merged_df["month_start"]
    rng = np.random.default_rng(0)
    resid = pd.Series(rng.normal(0, 1.0, len(idx)), index=idx)  # small noise, no elevated winter
    energy_result = _energy_result(resid, heating_p=0.001)
    assert recommend_heating_review(_clean_df(n=26), merged_df, energy_result) is None


def test_recommend_heating_review_fires_with_plausible_savings():
    merged_df = _merged_df(n=26)
    idx = merged_df["month_start"]
    rng = np.random.default_rng(0)
    resid = pd.Series(rng.normal(0, 1.0, len(idx)), index=idx)
    # Latest complete winter is Dec 2024 / Jan 2025 / Feb 2025 -- elevate those residuals.
    winter_mask = idx.isin(pd.to_datetime(["2024-12-01", "2025-01-01", "2025-02-01"]))
    resid[winter_mask.tolist()] = 30.0
    energy_result = _energy_result(resid, heating_p=0.001, heating_slope=1.6)

    rec = recommend_heating_review(_clean_df(n=26), merged_df, energy_result, reduction_pct=0.10)

    assert rec is not None
    assert rec.estimated_saving_gbp is not None
    assert rec.estimated_saving_gbp > 0
    assert "winter 2024/2025" in rec.action
    assert rec.confidence == "Medium"


def test_recommend_more_data_none_unless_forecast_confidence_low():
    assert recommend_more_data(None, n_months=20) is None
    assert recommend_more_data(ConfidenceRating("High", "good"), n_months=20) is None


def test_recommend_more_data_fires_on_low_confidence():
    rec = recommend_more_data(ConfidenceRating("Low", "wide bands"), n_months=20)
    assert rec is not None
    assert rec.estimated_saving_gbp is None
    assert "20 months" in rec.action


def _anomaly() -> Anomaly:
    return Anomaly(
        date=pd.Timestamp("2024-12-01"),
        methods=["rolling_zscore", "stl_esd"],
        direction="spike",
        rank_context="1st highest of 35 months",
    )


def test_recommend_investigate_anomaly_none_on_low_confidence():
    checklist = InvestigationChecklist(
        date=pd.Timestamp("2024-12-01"), items=[InvestigationItem("Unknown", True, "nothing else supported")]
    )
    rating = ConfidenceRating("Low", "only one method")
    assert recommend_investigate_anomaly(_anomaly(), checklist, rating) is None


def test_recommend_investigate_anomaly_points_at_checked_cause():
    checklist = InvestigationChecklist(
        date=pd.Timestamp("2024-12-01"),
        items=[
            InvestigationItem("Colder weather", True, "HDD notably higher"),
            InvestigationItem("Unknown", False, "explained"),
        ],
    )
    rating = ConfidenceRating("Medium", "two methods agree")
    rec = recommend_investigate_anomaly(_anomaly(), checklist, rating)
    assert rec is not None
    assert "colder weather" in rec.action.lower()


def test_recommend_investigate_anomaly_generic_when_nothing_checked():
    checklist = InvestigationChecklist(
        date=pd.Timestamp("2024-12-01"), items=[InvestigationItem("Unknown", True, "nothing else supported")]
    )
    rating = ConfidenceRating("High", "all three methods agree")
    rec = recommend_investigate_anomaly(_anomaly(), checklist, rating)
    assert rec is not None
    assert "remains unexplained" in rec.action


def test_generate_recommendations_returns_empty_when_nothing_fires():
    recs = generate_recommendations(
        clean_df=_clean_df(),
        merged_df=None,
        energy_result=None,
        forecast_rating=ConfidenceRating("High", "good"),
        n_months=26,
    )
    assert recs == []


def test_generate_recommendations_collects_fired_rules():
    recs = generate_recommendations(
        clean_df=_clean_df(),
        merged_df=None,
        energy_result=None,
        forecast_rating=ConfidenceRating("Low", "wide bands"),
        n_months=26,
    )
    assert len(recs) == 1
    assert recs[0].title == "Collect more historical data"
