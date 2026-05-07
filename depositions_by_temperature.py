"""
Refresh X-ray temperature counts from the RCSB Search API by release year.
"""

import argparse
from datetime import date
from pathlib import Path

from fetch_pdb_stats import TEMPERATURE_CSV, build_temperature_rows, write_temperature_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1976)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--output", type=Path, default=TEMPERATURE_CSV)
    args = parser.parse_args()

    if args.start_year > args.end_year:
        parser.error("--start-year must be less than or equal to --end-year")

    rows = build_temperature_rows(args.start_year, args.end_year)
    write_temperature_csv(rows, args.output)


if __name__ == "__main__":
    main()
