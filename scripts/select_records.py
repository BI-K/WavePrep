"""
Entry point for WavePrep cohort/record selection.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on the path when called directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.select.record_selector import run_record_selection


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="select_records",
        description=(
            "Generate an input CSV for WavePrep by selecting MIMIC-III waveform "
            "records that match the clinical criteria defined in a config JSON."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help=(
            "Path to a select config JSON "
            "(e.g. configs/select/cardiac_surgery.json)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    df = run_record_selection(args.config)
    print(f"\nSelection complete. {len(df)} rows in output.")


if __name__ == "__main__":
    main()
