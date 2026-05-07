"""
Refresh PDB statistics from the RCSB Search API and rebuild the dashboard.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
from typing import Any

import requests


API_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
ROOT = Path(__file__).parent
ANNUAL_CSV = ROOT / "PDB Annual per method - Sheet1.csv"
TEMPERATURE_CSV = ROOT / "depositions_by_temperature.csv"
DASHBOARD_HTML = ROOT / "pdb_dashboard.html"

DATE_ATTRIBUTE = "rcsb_accession_info.initial_release_date"
METHOD_ATTRIBUTE = "rcsb_entry_info.experimental_method"
TEMPERATURE_ATTRIBUTE = "diffrn.ambient_temp"

METHODS = {
    "xray": {
        "label": "X-ray",
        "total_col": "Total Number of Entries Available X-ray",
        "annual_col": "Number of Structures Released Annually X-ray",
    },
    "cryoem": {
        "label": "EM",
        "total_col": "Total Number of Entries Available cryoEM",
        "annual_col": "Number of Structures Released Annually cryoEM",
    },
}


def _date_range_node(year: int) -> dict[str, Any]:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": DATE_ATTRIBUTE,
            "operator": "range",
            "value": {
                "from": f"{year}-01-01",
                "include_lower": True,
                "to": f"{year}-12-31",
                "include_upper": True,
            },
        },
    }


def _method_node(method_label: str) -> dict[str, Any]:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": METHOD_ATTRIBUTE,
            "operator": "exact_match",
            "value": method_label,
        },
    }


def _temperature_node(temp_min: float, temp_max: float) -> dict[str, Any]:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": TEMPERATURE_ATTRIBUTE,
            "operator": "range",
            "value": {
                "from": temp_min,
                "include_lower": True,
                "to": temp_max,
                "include_upper": True,
            },
        },
    }


def count_entries(nodes: list[dict[str, Any]], timeout: int = 120) -> int:
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": nodes,
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 1}},
    }
    response = requests.post(
        API_URL,
        json=query,
        timeout=timeout,
        headers={"Content-Type": "application/json"},
    )
    if response.status_code == 204 or not response.content:
        return 0
    response.raise_for_status()
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise ValueError(
            f"API returned non-JSON. Status: {response.status_code}. "
            f"Response preview: {response.content[:500]!r}"
        ) from exc
    return data.get("total_count", len(data.get("result_set", [])))


def annual_method_count(year: int, method_label: str) -> int:
    return count_entries([_date_range_node(year), _method_node(method_label)])


def annual_temperature_count(year: int, temp_min: float, temp_max: float) -> int:
    return count_entries(
        [
            _date_range_node(year),
            _method_node(METHODS["xray"]["label"]),
            _temperature_node(temp_min, temp_max),
        ]
    )


def build_annual_rows(start_year: int, end_year: int) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    totals = {key: 0 for key in METHODS}

    for year in range(start_year, end_year + 1):
        row: dict[str, int] = {"Year": year}
        for key, method in METHODS.items():
            annual = annual_method_count(year, method["label"])
            totals[key] += annual
            row[method["annual_col"]] = annual
            row[method["total_col"]] = totals[key]
        rows.append(row)
        print(
            f"{year}: "
            f"X-ray={row[METHODS['xray']['annual_col']]:,}, "
            f"cryoEM={row[METHODS['cryoem']['annual_col']]:,}"
        )

    return rows


def build_temperature_rows(start_year: int, end_year: int) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for year in range(start_year, end_year + 1):
        room_temp = annual_temperature_count(year, 273, 350)
        cryogenic = annual_temperature_count(year, 0, 150)
        rows.append(
            {
                "Year": year,
                "Room temp (273-350 K)": room_temp,
                "Cryogenic (<150 K)": cryogenic,
            }
        )
        print(f"{year}: room temp={room_temp:,}, cryogenic={cryogenic:,}")
    return rows


def _format_count(value: int) -> str:
    return f"{value:,}"


def write_annual_csv(rows: list[dict[str, int]], output_path: Path = ANNUAL_CSV) -> None:
    fieldnames = [
        "Year",
        METHODS["xray"]["total_col"],
        METHODS["xray"]["annual_col"],
        METHODS["cryoem"]["total_col"],
        METHODS["cryoem"]["annual_col"],
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["Year"], reverse=True):
            writer.writerow(
                {
                    "Year": row["Year"],
                    METHODS["xray"]["total_col"]: _format_count(row[METHODS["xray"]["total_col"]]),
                    METHODS["xray"]["annual_col"]: _format_count(row[METHODS["xray"]["annual_col"]]),
                    METHODS["cryoem"]["total_col"]: _format_count(row[METHODS["cryoem"]["total_col"]]),
                    METHODS["cryoem"]["annual_col"]: _format_count(row[METHODS["cryoem"]["annual_col"]]),
                }
            )
    print(f"Wrote {output_path}")


def write_temperature_csv(rows: list[dict[str, int]], output_path: Path = TEMPERATURE_CSV) -> None:
    fieldnames = ["Year", "Room temp (273-350 K)", "Cryogenic (<150 K)"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_path}")


def _clean_int(value: str) -> int:
    return int(str(value).replace(",", ""))


def summarize_annual_differences(new_rows: list[dict[str, int]], existing_path: Path = ANNUAL_CSV) -> None:
    if not existing_path.exists():
        return

    old_rows: dict[int, dict[str, int]] = {}
    with existing_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            old_rows[int(row["Year"])] = {
                column: _clean_int(row[column])
                for column in (
                    METHODS["xray"]["annual_col"],
                    METHODS["cryoem"]["annual_col"],
                )
            }

    differences: list[tuple[int, str, int, int]] = []
    for row in new_rows:
        old = old_rows.get(row["Year"])
        if old is None:
            continue
        for column in (METHODS["xray"]["annual_col"], METHODS["cryoem"]["annual_col"]):
            new_value = row[column]
            old_value = old[column]
            if new_value != old_value:
                differences.append((row["Year"], column, old_value, new_value))

    if not differences:
        print("Annual counts match the existing CSV.")
        return

    print(f"Annual count differences versus existing CSV: {len(differences)} cells")
    for year, column, old_value, new_value in differences[:12]:
        print(f"  {year} {column}: existing={old_value:,}, refreshed={new_value:,}")
    if len(differences) > 12:
        print(f"  ... {len(differences) - 12} more differences")


def refresh(
    start_year: int,
    end_year: int,
    annual_csv: Path = ANNUAL_CSV,
    temperature_csv: Path = TEMPERATURE_CSV,
    dashboard_html: Path = DASHBOARD_HTML,
    rebuild_dashboard: bool = True,
) -> None:
    print("Refreshing annual release-year method counts...")
    annual_rows = build_annual_rows(start_year, end_year)
    summarize_annual_differences(annual_rows, annual_csv)
    write_annual_csv(annual_rows, annual_csv)

    print("\nRefreshing release-year X-ray temperature counts...")
    temperature_rows = build_temperature_rows(start_year, end_year)
    write_temperature_csv(temperature_rows, temperature_csv)

    if rebuild_dashboard:
        from dashboard import build_dashboard

        build_dashboard(
            csv_path=annual_csv,
            output_path=dashboard_html,
            temperatures_path=temperature_csv,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh PDB CSV data from the RCSB Search API.")
    parser.add_argument("--start-year", type=int, default=1976)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--annual-csv", type=Path, default=ANNUAL_CSV)
    parser.add_argument("--temperature-csv", type=Path, default=TEMPERATURE_CSV)
    parser.add_argument("--dashboard-html", type=Path, default=DASHBOARD_HTML)
    parser.add_argument(
        "--no-rebuild-dashboard",
        action="store_true",
        help="Only refresh CSV files; do not rebuild pdb_dashboard.html.",
    )
    args = parser.parse_args()

    if args.start_year > args.end_year:
        parser.error("--start-year must be less than or equal to --end-year")

    refresh(
        start_year=args.start_year,
        end_year=args.end_year,
        annual_csv=args.annual_csv,
        temperature_csv=args.temperature_csv,
        dashboard_html=args.dashboard_html,
        rebuild_dashboard=not args.no_rebuild_dashboard,
    )


if __name__ == "__main__":
    main()
