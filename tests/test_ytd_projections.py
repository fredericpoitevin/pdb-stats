"""Smoke tests for YTD and year-end estimate markers in pdb_dashboard.html."""

from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_HTML = ROOT / "pdb_dashboard.html"
ANNUAL_CSV = ROOT / "PDB Annual per method - Sheet1.csv"
TEMPERATURE_CSV = ROOT / "depositions_by_temperature.csv"
NEUTRON_CSV = ROOT / "depositions_by_neutron.csv"
NMR_CSV = ROOT / "depositions_by_nmr.csv"

# legendgroup, YTD trace name substring, year-end trace name substring
PROJECTION_PAIRS = [
    ("xray-annual", "to date", "year-end estimate"),
    ("cryoem-annual", "to date", "year-end estimate"),
    ("xray-temp", "Room temp X-ray to date", "projected"),
    ("neutron-annual", "to date", "year-end estimate"),
    ("nmr-annual", "to date", "year-end estimate"),
]


def elapsed_fraction(on_date: date | None = None) -> float:
    on_date = on_date or date.today()
    days_in_year = date(on_date.year, 12, 31).timetuple().tm_yday
    return on_date.timetuple().tm_yday / days_in_year


def parse_annual_chart_traces(html_path: Path = DASHBOARD_HTML) -> list[dict]:
    html = html_path.read_text(encoding="utf-8")
    match = re.search(
        r'Plotly\.newPlot\(\s*"annual-chart",\s*(\[.*?\]),\s*\{',
        html,
        re.DOTALL,
    )
    if not match:
        raise ValueError(f"annual-chart Plotly data not found in {html_path}")
    return json.loads(match.group(1))


def find_trace_value(traces: list[dict], legendgroup: str, name_substr: str) -> float | None:
    for trace in traces:
        if trace.get("legendgroup") != legendgroup:
            continue
        name = trace.get("name") or ""
        if name_substr not in name:
            continue
        y_vals = trace.get("y")
        if y_vals:
            return float(y_vals[0])
    return None


def current_year_from_traces(traces: list[dict]) -> int:
    for trace in traces:
        name = trace.get("name") or ""
        match = re.search(r"to date \((\d{4})\)", name)
        if match:
            return int(match.group(1))
    return date.today().year


@pytest.fixture(scope="module")
def annual_traces() -> list[dict]:
    if not DASHBOARD_HTML.exists():
        pytest.skip(f"{DASHBOARD_HTML} not found")
    return parse_annual_chart_traces()


@pytest.fixture(scope="module")
def projection_values(annual_traces: list[dict]) -> list[tuple[str, float, float]]:
    values: list[tuple[str, float, float]] = []
    for legendgroup, ytd_substr, ye_substr in PROJECTION_PAIRS:
        ytd = find_trace_value(annual_traces, legendgroup, ytd_substr)
        ye = find_trace_value(annual_traces, legendgroup, ye_substr)
        assert ytd is not None, f"Missing YTD trace for {legendgroup!r} ({ytd_substr!r})"
        assert ye is not None, f"Missing year-end trace for {legendgroup!r} ({ye_substr!r})"
        values.append((legendgroup, ytd, ye))
    return values


def test_ytd_matches_csv(annual_traces: list[dict]) -> None:
    year = current_year_from_traces(annual_traces)

    annual = pd.read_csv(ANNUAL_CSV)
    annual["Year"] = annual["Year"].astype(int)
    for col in (
        "Number of Structures Released Annually X-ray",
        "Number of Structures Released Annually cryoEM",
    ):
        annual[col] = annual[col].astype(str).str.replace(",", "").astype(int)
    annual_row = annual.loc[annual["Year"] == year].iloc[0]

    xray_ytd = find_trace_value(annual_traces, "xray-annual", "to date")
    cryo_ytd = find_trace_value(annual_traces, "cryoem-annual", "to date")
    assert xray_ytd == int(annual_row["Number of Structures Released Annually X-ray"])
    assert cryo_ytd == int(annual_row["Number of Structures Released Annually cryoEM"])

    temp = pd.read_csv(TEMPERATURE_CSV)
    temp_row = temp.loc[temp["Year"] == year].iloc[0]
    room_col = [c for c in temp.columns if "273" in c or "Room" in c.lower()][0]
    room_ytd = find_trace_value(annual_traces, "xray-temp", "Room temp X-ray to date")
    assert room_ytd == float(temp_row[room_col])

    neutron = pd.read_csv(NEUTRON_CSV)
    neutron_row = neutron.loc[neutron["Year"] == year].iloc[0]
    neutron_ytd = find_trace_value(annual_traces, "neutron-annual", "to date")
    assert neutron_ytd == int(neutron_row["Annual Neutron"])

    nmr = pd.read_csv(NMR_CSV)
    nmr_row = nmr.loc[nmr["Year"] == year].iloc[0]
    nmr_ytd = find_trace_value(annual_traces, "nmr-annual", "to date")
    assert nmr_ytd == int(nmr_row["Annual NMR"])


def test_year_end_estimate_formula(
    annual_traces: list[dict],
    projection_values: list[tuple[str, float, float]],
) -> None:
    year = current_year_from_traces(annual_traces)
    days_in_year = date(year, 12, 31).timetuple().tm_yday

    for legendgroup, ytd, ye in projection_values:
        if ytd <= 0:
            assert ye == 0.0, f"{legendgroup}: expected zero year-end estimate for zero YTD"
            continue

        implied_day = round(ytd * days_in_year / ye)
        assert 1 <= implied_day <= days_in_year, (
            f"{legendgroup}: implied day {implied_day} out of range for ytd={ytd}, ye={ye}"
        )

        expected_ye = ytd / (implied_day / days_in_year)
        assert abs(ye - expected_ye) < 0.01, (
            f"{legendgroup}: year-end estimate {ye} != ytd/elapsed ({expected_ye})"
        )

        if os.environ.get("CI") and year == date.today().year:
            assert implied_day == date.today().timetuple().tm_yday, (
                f"{legendgroup}: dashboard built for day {implied_day}, "
                f"expected {date.today().timetuple().tm_yday} on CI"
            )
            expected_today = ytd / elapsed_fraction()
            assert abs(ye - expected_today) < 0.01, (
                f"{legendgroup}: year-end estimate {ye} != today's projection ({expected_today})"
            )
