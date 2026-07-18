import numpy as np
import pandas as pd

from src.energy_signature import EnergySignatureResult
from src.investigation import build_investigation_checklist


def _clean_df(n=24, unit_rate=0.3, spike_month_idx=None, spike_rate=None) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="MS")
    consumption = np.full(n, 300.0)
    rates = np.full(n, unit_rate)
    if spike_month_idx is not None:
        rates[spike_month_idx] = spike_rate
    df = pd.DataFrame(
        {
            "month_start": idx,
            "consumption_kwh": consumption,
            "cost_gbp": consumption * rates,
            "unit_rate_gbp_per_kwh": rates,
        }
    )
    return df


def _merged_df(n=24, hdd_values=None) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="MS")
    hdd = hdd_values if hdd_values is not None else np.full(n, 5.0)
    return pd.DataFrame({"month_start": idx, "avg_daily_hdd": hdd, "avg_daily_cdd": np.zeros(n)})


def _energy_result(heating_p=0.001, heating_slope=1.6) -> EnergySignatureResult:
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
        n_obs=24,
        fitted=pd.Series(dtype=float),
        resid=pd.Series(dtype=float),
    )


def _item(checklist, label):
    return next(i for i in checklist.items if i.label == label)


def test_colder_weather_checked_when_hdd_up_and_heating_significant():
    clean_df = _clean_df(n=24)
    hdd = np.full(24, 5.0)
    hdd[23] = 8.0  # month 23 (Dec 2024) is notably colder than month 11 (Dec 2023)
    merged_df = _merged_df(n=24, hdd_values=hdd)
    energy_result = _energy_result(heating_p=0.001)

    checklist = build_investigation_checklist(pd.Timestamp("2024-12-01"), clean_df, merged_df, energy_result)

    assert _item(checklist, "Colder weather").checked is True


def test_colder_weather_unchecked_without_weather_data():
    clean_df = _clean_df(n=24)
    checklist = build_investigation_checklist(pd.Timestamp("2024-12-01"), clean_df, None, None)
    item = _item(checklist, "Colder weather")
    assert item.checked is False
    assert "off" in item.reason


def test_tariff_change_checked_on_large_rate_deviation():
    clean_df = _clean_df(n=24, unit_rate=0.30, spike_month_idx=23, spike_rate=0.45)
    checklist = build_investigation_checklist(pd.Timestamp("2024-12-01"), clean_df, None, None)
    assert _item(checklist, "Tariff change").checked is True


def test_tariff_change_unchecked_when_rate_stable():
    clean_df = _clean_df(n=24, unit_rate=0.30)
    checklist = build_investigation_checklist(pd.Timestamp("2024-12-01"), clean_df, None, None)
    assert _item(checklist, "Tariff change").checked is False


def test_heating_sensitivity_checked_when_significant():
    clean_df = _clean_df(n=24)
    checklist = build_investigation_checklist(
        pd.Timestamp("2024-12-01"), clean_df, None, _energy_result(heating_p=0.001)
    )
    assert _item(checklist, "Heating sensitivity").checked is True


def test_heating_sensitivity_unchecked_when_not_significant():
    clean_df = _clean_df(n=24)
    checklist = build_investigation_checklist(
        pd.Timestamp("2024-12-01"), clean_df, None, _energy_result(heating_p=0.8)
    )
    assert _item(checklist, "Heating sensitivity").checked is False


def test_heating_sensitivity_names_gas_when_fuel_is_gas():
    clean_df = _clean_df(n=24)
    checklist = build_investigation_checklist(
        pd.Timestamp("2024-12-01"), clean_df, None, _energy_result(heating_p=0.001), fuel="gas"
    )
    item = _item(checklist, "Heating sensitivity")
    assert item.checked is True
    assert "gas heating" in item.reason
    assert "electric" not in item.reason.lower()


def test_undeterminable_items_always_unchecked_with_honest_reason():
    clean_df = _clean_df(n=24)
    checklist = build_investigation_checklist(pd.Timestamp("2024-12-01"), clean_df, None, None)
    for label in ["New appliance", "Holiday occupancy", "Guests", "Home working"]:
        item = _item(checklist, label)
        assert item.checked is False
        assert "Not determinable" in item.reason


def test_unknown_checked_when_nothing_else_supported():
    clean_df = _clean_df(n=24, unit_rate=0.30)
    checklist = build_investigation_checklist(pd.Timestamp("2024-12-01"), clean_df, None, None)
    assert _item(checklist, "Unknown").checked is True
    assert checklist.checked_items == [_item(checklist, "Unknown")]


def test_unknown_unchecked_when_something_else_is_supported():
    clean_df = _clean_df(n=24, unit_rate=0.30, spike_month_idx=23, spike_rate=0.50)
    checklist = build_investigation_checklist(pd.Timestamp("2024-12-01"), clean_df, None, None)
    assert _item(checklist, "Unknown").checked is False
    assert _item(checklist, "Tariff change").checked is True


def test_colder_weather_unchecked_when_month_missing_from_weather_data():
    clean_df = _clean_df(n=24)
    merged_df = _merged_df(n=24).iloc[:-1]  # drop the most recent month
    item = _item(
        build_investigation_checklist(pd.Timestamp("2024-12-01"), clean_df, merged_df, _energy_result()),
        "Colder weather",
    )
    assert item.checked is False
    assert "No weather data available" in item.reason


def test_colder_weather_unchecked_without_prior_year_weather_data():
    clean_df = _clean_df(n=24)
    merged_df = _merged_df(n=24).iloc[12:]  # only the second year has weather data
    item = _item(
        build_investigation_checklist(pd.Timestamp("2024-12-01"), clean_df, merged_df, _energy_result()),
        "Colder weather",
    )
    assert item.checked is False
    assert "No prior-year weather data" in item.reason


def test_tariff_change_unchecked_when_month_missing_from_billing_data():
    clean_df = _clean_df(n=24)
    item = _item(build_investigation_checklist(pd.Timestamp("2030-01-01"), clean_df, None, None), "Tariff change")
    assert item.checked is False
    assert "No billing data available" in item.reason


def test_tariff_change_unchecked_with_too_little_history():
    clean_df = _clean_df(n=3)
    item = _item(build_investigation_checklist(pd.Timestamp("2023-03-01"), clean_df, None, None), "Tariff change")
    assert item.checked is False
    assert "Not enough prior months" in item.reason
