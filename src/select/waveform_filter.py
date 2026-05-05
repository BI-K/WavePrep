"""
Waveform record enumeration and signal/duration filtering.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from .select_config import DatabaseConfig, RecordRequirements

logger = logging.getLogger(__name__)

RecordEntry = Dict[str, Any]  # keys: subject, record_id, duration_seconds (optional)

_WFDB_MAX_WORKERS = 16


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def get_waveform_records(
    db_config: DatabaseConfig,
    req: RecordRequirements,
) -> List[RecordEntry]:
    """Return candidate records that satisfy the signal/duration requirements."""

    if req.precomputed_records_path:
        logger.info(
            "Using precomputed records from %s", req.precomputed_records_path
        )
        return _load_precomputed(req.precomputed_records_path)

    if db_config.signal_duration_index:
        logger.info(
            "Using signal duration index CSV at %s", db_config.signal_duration_index
        )
        return _filter_from_csv_index(db_config.signal_duration_index, req)

    logger.warning(
        "No precomputed_records_path or signal_duration_index provided. "
        "Falling back to live WFDB queries against PhysioNet — "
        "this may take many hours for a full database scan."
    )
    return _filter_from_wfdb(db_config.matched_waveform_database, req)


# ──────────────────────────────────────────────────────────────────────────────
# Path 1 — precomputed JSON
# ──────────────────────────────────────────────────────────────────────────────

def _load_precomputed(path: str) -> List[RecordEntry]:
    """
    Load a pre-filtered record list from JSON.

    Accepted formats:
        [{"subject": "p00/p000020/", "record_id": "p000020-2183-04-28-17-47n"}, ...]
        [{"subject": "p00/p000020/", "record_id": "...", "duration_seconds": 12345.0}, ...]
    """
    with open(path, encoding="utf-8") as fh:
        raw: List[Dict] = json.load(fh)

    records: List[RecordEntry] = []
    for entry in raw:
        record_id = entry.get("record_id") or entry.get("record", "")
        subject = entry.get("subject", _subject_from_record_id(record_id))
        records.append(
            {
                "subject": subject,
                "record_id": record_id,
                "duration_seconds": entry.get("duration_seconds"),
            }
        )

    logger.info("Loaded %d records from precomputed JSON (%s)", len(records), path)
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Path 2 — local CSV signal/duration index
# ──────────────────────────────────────────────────────────────────────────────

def _filter_from_csv_index(index_path: str, req: RecordRequirements) -> List[RecordEntry]:
    """
    Filter records using a pre-computed signal/duration CSV.

    Expected CSV columns: record_id, signal, duration_seconds
    Optional columns:     subject, sampling_frequency
    """
    df = pd.read_csv(index_path)

    # ── recording type filter ─────────────────────────────────────────────────
    if req.recording_type == "numeric":
        df = df[df["record_id"].str.endswith("n")]
    elif req.recording_type == "non-numeric":
        # non-numeric: no trailing 'n', no '_' in name (waveform master records)
        df = df[~df["record_id"].str.endswith("n") & ~df["record_id"].str.contains("_")]

    # ── duration filter ───────────────────────────────────────────────────────
    df = df[df["duration_seconds"] > req.min_duration_seconds]

    if df.empty:
        logger.info("No records passed duration/type filter from CSV index.")
        return []

    # ── signal completeness filter ────────────────────────────────────────────
    # Each (record_id, signal) is one row.  Keep only records that have ALL
    # required signals with sufficient duration.
    required = set(req.required_signals)

    def _has_all_signals(grp: pd.DataFrame) -> bool:
        return required.issubset(set(grp["signal"].tolist()))

    valid_record_ids = (
        df.groupby("record_id")
        .filter(_has_all_signals)["record_id"]
        .unique()
        .tolist()
    )

    df_valid = df[df["record_id"].isin(valid_record_ids)].drop_duplicates(
        subset=["record_id"]
    )

    records: List[RecordEntry] = []
    for _, row in df_valid.iterrows():
        record_id = row["record_id"]
        subject = row.get("subject", _subject_from_record_id(record_id))
        records.append(
            {
                "subject": subject,
                "record_id": record_id,
                "duration_seconds": float(row["duration_seconds"]),
            }
        )

    logger.info(
        "Found %d records in CSV index after signal/duration filter.", len(records)
    )
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Path 3 — live WFDB remote query (PhysioNet)
# ──────────────────────────────────────────────────────────────────────────────

def _scan_subject(
    subject: str,
    database_name: str,
    recording_type: str,
    required: set,
    min_duration_seconds: float,
) -> List[RecordEntry]:
    """Fetch and filter all records for one subject. Runs in a thread."""
    import wfdb  # type: ignore  # cached after first import; safe across threads

    try:
        record_list = wfdb.get_record_list(f"{database_name}/{subject}")
    except Exception as exc:
        logger.debug("Skipping subject %s: %s", subject, exc)
        return []

    if recording_type == "numeric":
        record_list = [r for r in record_list if r.endswith("n")]
    elif recording_type == "non-numeric":
        record_list = [r for r in record_list if not r.endswith("n") and "_" not in r]

    results: List[RecordEntry] = []
    for record in record_list:
        try:
            header = wfdb.rdheader(
                record,
                pn_dir=f"{database_name}/{subject}",
                rd_segments=True,
            )
        except Exception as exc:
            logger.debug("Skipping record %s: %s", record, exc)
            continue

        if header.sig_name is None:
            continue
        if not required.issubset(set(header.sig_name)):
            continue
        duration_s = header.sig_len / header.fs
        if duration_s <= min_duration_seconds:
            continue

        results.append(
            {
                "subject": subject,
                "record_id": record,
                "duration_seconds": duration_s,
            }
        )

    return results


def _filter_from_wfdb(
    database_name: str, req: RecordRequirements
) -> List[RecordEntry]:
    """
    Enumerate records directly from PhysioNet via WFDB using a thread pool.
    Requires internet access.  Each subject is scanned concurrently.
    """
    try:
        import wfdb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "wfdb is required for live PhysioNet queries. "
            "Install it with: pip install wfdb"
        ) from exc

    subject_list = wfdb.get_record_list(database_name)
    logger.info(
        "WFDB: found %d subjects in %s. Scanning with %d workers…",
        len(subject_list),
        database_name,
        _WFDB_MAX_WORKERS,
    )

    required = set(req.required_signals)
    records: List[RecordEntry] = []

    with ThreadPoolExecutor(max_workers=_WFDB_MAX_WORKERS) as pool:
        futures = {
            pool.submit(
                _scan_subject,
                subject,
                database_name,
                req.recording_type,
                required,
                req.min_duration_seconds,
            ): subject
            for subject in subject_list
        }
        with tqdm(total=len(futures), desc="Scanning WFDB subjects", unit="subject") as pbar:
            for future in as_completed(futures):
                records.extend(future.result())
                pbar.update(1)

    logger.info("WFDB scan complete. Found %d records.", len(records))
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _subject_from_record_id(record_id: str) -> str:
    """Derive subject directory path from a record ID.

    p000020-2183-04-28-17-47n  →  "p00/p000020/"
    """
    subject_part = record_id.split("-")[0]  # "p000020"
    dir_prefix = f"p{subject_part[1:3]}"   # "p00"
    return f"{dir_prefix}/{subject_part}/"


def load_duration_index(index_path: str) -> Dict[str, float]:
    """Load the signal-duration CSV once into a {record_id: duration_seconds} dict."""
    df = pd.read_csv(index_path, usecols=["record_id", "duration_seconds"])
    return dict(zip(df["record_id"], df["duration_seconds"].astype(float)))


def lookup_duration_from_index(
    record_id: str, index_path: str
) -> Optional[float]:
    """Look up a single record's duration_seconds from the signal duration CSV."""
    df = pd.read_csv(index_path)
    match = df[df["record_id"] == record_id]
    if match.empty:
        return None
    return float(match.iloc[0]["duration_seconds"])


def lookup_duration_from_wfdb(
    record_id: str, subject: str, database_name: str
) -> Optional[float]:
    """Read the recording duration directly from a WFDB header."""
    try:
        import wfdb  # type: ignore

        header = wfdb.rdheader(
            record_id,
            pn_dir=f"{database_name}/{subject}",
            rd_segments=True,
        )
        return header.sig_len / header.fs
    except Exception as exc:
        logger.debug(
            "Could not read WFDB header for %s/%s: %s", subject, record_id, exc
        )
        return None
