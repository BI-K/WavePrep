"""
Main orchestrator for record selection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .db_connection import get_connection
from .select_config import SelectConfig
from .waveform_filter import (
    get_waveform_records,
    load_duration_index,
    lookup_duration_from_wfdb,
)
from .filters.procedure_filter import apply_procedure_filter
from .filters.medication_filter import apply_medication_filter
from .filters.ventilation_filter import apply_ventilation_filter

logger = logging.getLogger(__name__)

RecordEntry = Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_record_selection(config_path: str) -> pd.DataFrame:
    """
    Execute the full selection pipeline defined by *config_path*.

    Returns the final DataFrame and writes it to
    {config.output.path}/{config.name}.csv.
    """
    config = SelectConfig.from_json(config_path)
    _configure_logging(config.name)

    logger.info("=" * 60)
    logger.info("WavePrep record selection: %s", config.name)
    logger.info("=" * 60)

    # ── Step 1: Enumerate candidate records ──────────────────────────────────
    logger.info("Step 1 — Enumerating candidate records …")
    records = get_waveform_records(config.database, config.record_requirements)
    logger.info("  Candidate records after waveform filter: %d", len(records))

    if not records:
        logger.warning("No candidate records found. Output CSV will be empty.")
        return _write_empty(config)

    # ── Step 2: Apply each cohort filter in sequence (AND logic) ─────────────
    with get_connection(config.database.login) as (_conn, cur):
        for i, filter_elem in enumerate(config.cohort_definition):
            before = len(records)
            logger.info(
                "Step 2.%d — Applying filter '%s' (%s) …",
                i + 1,
                filter_elem.filter_type,
                filter_elem.include_or_exclude,
            )

            if filter_elem.filter_type == "procedure":
                records = apply_procedure_filter(filter_elem, records, cur)

            elif filter_elem.filter_type == "medication":
                records = apply_medication_filter(filter_elem, records, cur)

            elif filter_elem.filter_type == "mechanical_ventilation":
                records = apply_ventilation_filter(filter_elem, records, cur)

            else:
                raise ValueError(
                    f"Unknown filter_type: '{filter_elem.filter_type}'. "
                    "Supported types: 'procedure', 'medication', "
                    "'mechanical_ventilation'."
                )

            logger.info("  %d → %d records", before, len(records))

            if not records:
                logger.warning(
                    "No records remain after filter '%s'. "
                    "Output CSV will be empty.",
                    filter_elem.filter_type,
                )
                return _write_empty(config)

    # ── Step 3: Resolve missing duration_seconds ──────────────────────────────
    logger.info("Step 3 — Resolving recording durations …")
    records = _resolve_durations(records, config)

    # ── Step 4: Build and write output CSV ────────────────────────────────────
    logger.info("Step 4 — Writing output CSV …")
    df = _build_output_dataframe(records)
    output_path = _write_csv(df, config)

    logger.info(
        "Done. Wrote %d rows to %s", len(df), output_path
    )
    logger.info("  Unique records: %d", df["record"].nunique())
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Duration resolution
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_durations(
    records: List[RecordEntry], config: SelectConfig
) -> List[RecordEntry]:
    """
    Fill in duration_seconds for records where it is None and offset_end is
    also None (procedure cohorts loaded from a precomputed JSON that doesn't
    carry duration).
    """
    needs_duration = [
        r for r in records
        if r.get("duration_seconds") is None and r.get("offset_end_seconds") is None
    ]

    if not needs_duration:
        return records

    logger.info(
        "  %d records need duration lookup …", len(needs_duration)
    )

    index_path = config.database.signal_duration_index

    duration_index: Dict[str, float] = (
        load_duration_index(index_path) if index_path else {}
    )

    for rec in records:
        if rec.get("duration_seconds") is not None or rec.get("offset_end_seconds") is not None:
            continue

        record_id = rec.get("record_id") or rec.get("record", "")
        duration: Optional[float] = None

        # Try CSV index first
        if duration_index:
            duration = duration_index.get(record_id)

        # Fall back to WFDB header read
        if duration is None:
            logger.debug("  WFDB fallback for duration of %s", record_id)
            duration = lookup_duration_from_wfdb(
                record_id,
                rec.get("subject", ""),
                config.database.matched_waveform_database,
            )

        if duration is not None:
            rec["duration_seconds"] = duration
            rec["offset_end_seconds"] = duration
        else:
            logger.warning(
                "  Could not determine duration for %s — row will be dropped.",
                record_id,
            )

    # Drop records where offset_end is still None
    before = len(records)
    records = [
        r for r in records
        if r.get("offset_end_seconds") is not None or r.get("duration_seconds") is not None
    ]
    # Final pass: fill offset_end from duration_seconds where still missing
    for r in records:
        if r.get("offset_end_seconds") is None and r.get("duration_seconds") is not None:
            r["offset_end_seconds"] = r["duration_seconds"]

    dropped = before - len(records)
    if dropped:
        logger.warning("  Dropped %d records with unresolvable duration.", dropped)

    return records


# ──────────────────────────────────────────────────────────────────────────────
# Output construction
# ──────────────────────────────────────────────────────────────────────────────

def _build_output_dataframe(records: List[RecordEntry]) -> pd.DataFrame:
    """
    Build the output DataFrame with columns:
        record, offset_start_seconds, offset_end_seconds

    Deduplication: if a record appears with multiple identical
    (record_id, offset_start, offset_end) tuples (e.g. matched by both ICD9
    and CPT), keep only one row.
    """
    rows = []
    for rec in records:
        record_id = rec.get("record_id") or rec.get("record", "")
        offset_start = rec.get("offset_start_seconds", 0.0)
        offset_end   = rec.get("offset_end_seconds")

        if offset_end is None:
            continue

        rows.append(
            {
                "record":                record_id,
                "offset_start_seconds": float(offset_start),
                "offset_end_seconds":   float(offset_end),
            }
        )

    df = pd.DataFrame(rows, columns=["record", "offset_start_seconds", "offset_end_seconds"])
    df = df.drop_duplicates().sort_values("record").reset_index(drop=True)
    return df


def _write_csv(df: pd.DataFrame, config: SelectConfig) -> Path:
    output_dir = Path(config.output.path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config.name}.csv"
    df.to_csv(output_path, index=False)
    return output_path


def _write_empty(config: SelectConfig) -> pd.DataFrame:
    df = pd.DataFrame(
        columns=["record", "offset_start_seconds", "offset_end_seconds"]
    )
    _write_csv(df, config)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────

def _configure_logging(name: str) -> None:
    """Ensure a console handler is attached if the root logger has none."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                              datefmt="%H:%M:%S")
        )
        root.addHandler(handler)
        root.setLevel(logging.INFO)
