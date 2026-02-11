"""
Query RCSB Search API for structure counts by deposition year and temperature range.
Outputs a CSV with room temperature (273-350 K) and cryogenic counts per year.
"""

import argparse
import csv
import sys
from pathlib import Path

import requests

API_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


def get_count(year: int, temp_min: float, temp_max: float) -> int:
    """Count structures deposited in year with ambient temp in [temp_min, temp_max] K."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "diffrn.ambient_temp",
                        "operator": "range",
                        "value": {
                            "from": temp_min,
                            "include_lower": True,
                            "to": temp_max,
                            "include_upper": True,
                        },
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.deposit_date",
                        "operator": "range",
                        "value": {
                            "from": f"{year}-01-01T00:00:00Z",
                            "include_lower": True,
                            "to": f"{year}-12-31T23:59:59Z",
                            "include_upper": True,
                        },
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 1}},
    }
    response = requests.post(
        API_URL,
        json=query,
        timeout=120,
        headers={"Content-Type": "application/json"},
    )
    # 204 No Content is a valid success response with no body
    if response.status_code == 204 or not response.content:
        return 0
    response.raise_for_status()
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError as e:
        raise ValueError(
            f"API returned non-JSON. Status: {response.status_code}. "
            f"Response preview: {response.content[:500]!r}"
        ) from e
    return data.get("total_count", len(data.get("result_set", [])))


def test_connectivity() -> bool:
    """Run a minimal query to verify API connectivity."""
    minimal = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_accession_info.deposit_date",
                "operator": "range",
                "value": {"from": "2020-01-01", "include_lower": True, "to": "2020-12-31", "include_upper": True},
            },
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 1}},
    }
    try:
        r = requests.post(API_URL, json=minimal, timeout=30, headers={"Content-Type": "application/json"})
        print(f"Connectivity test: status={r.status_code}, body_len={len(r.content)}")
        if r.content:
            d = r.json()
            print(f"  total_count={d.get('total_count', 'N/A')}")
        return bool(r.content)
    except Exception as e:
        print(f"Connectivity test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run connectivity test before querying")
    args = parser.parse_args()

    if args.debug:
        if not test_connectivity():
            print("API unreachable. Check network/proxy. Try: curl -X POST https://search.rcsb.org/rcsbsearch/v2/query -H 'Content-Type: application/json' -d '{}'")
            sys.exit(1)

    output_path = Path(__file__).parent / "depositions_by_temperature.csv"
    years = range(1976, 2026)  # PDB start through 2025

    rows = [["Year", "Room temp (273-350 K)", "Cryogenic (<150 K)"]]
    for year in years:
        rt_count = get_count(year, 273, 350)
        cryo_count = get_count(year, 0, 150)
        rows.append([year, rt_count, cryo_count])
        print(f"{year}: room temp={rt_count}, cryogenic={cryo_count}")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
