from src.benchmarking import compare_to_benchmark


def test_compare_to_benchmark_none_for_total_fuel():
    assert compare_to_benchmark(4000.0, "total", "uk", n_months=24, weather_adjusted=True) is None


def test_compare_to_benchmark_none_for_scotland_gas_gap():
    """No Scotland-specific gas consumption figure was found during research -- this must
    return None (an honest 'not available'), not silently fall back to the UK-wide figure
    under a Scotland label."""
    assert compare_to_benchmark(9500.0, "gas", "scotland", n_months=24, weather_adjusted=True) is None


def test_compare_to_benchmark_average_band_within_tolerance():
    # UK electricity reference is 2500 kWh/year, tolerance +/-15% -> [2125, 2875]
    result = compare_to_benchmark(2500.0, "electricity", "uk", n_months=24, weather_adjusted=True)

    assert result is not None
    assert result.band == "Average"
    assert result.reference_kwh == 2500.0


def test_compare_to_benchmark_below_average():
    result = compare_to_benchmark(1500.0, "electricity", "uk", n_months=24, weather_adjusted=True)
    assert result is not None
    assert result.band == "Below average"


def test_compare_to_benchmark_above_average():
    result = compare_to_benchmark(4000.0, "electricity", "uk", n_months=24, weather_adjusted=True)
    assert result is not None
    assert result.band == "Above average"


def test_compare_to_benchmark_scotland_electricity_uses_scotland_reference():
    result = compare_to_benchmark(3429.0, "electricity", "scotland", n_months=24, weather_adjusted=True)
    assert result is not None
    assert result.band == "Average"
    assert result.reference_kwh == 3429.0
    assert "Scotland" in result.source


def test_compare_to_benchmark_confidence_high_with_full_history_and_weather():
    result = compare_to_benchmark(2500.0, "electricity", "uk", n_months=24, weather_adjusted=True)
    assert result.confidence == "High"


def test_compare_to_benchmark_confidence_medium_without_weather_or_short_history():
    result = compare_to_benchmark(2500.0, "electricity", "uk", n_months=24, weather_adjusted=False)
    assert result.confidence == "Medium"

    result_short = compare_to_benchmark(2500.0, "electricity", "uk", n_months=6, weather_adjusted=True)
    assert result_short.confidence == "Medium"
