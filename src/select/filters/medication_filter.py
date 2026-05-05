"""
Medication-based cohort filter (time_span_level).
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..select_config import FilterElement
from .procedure_filter import _get_hadm_id, _parse_record_id

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def apply_medication_filter(
    filter_elem: FilterElement,
    records: List[Dict[str, Any]],
    cur: Any,
) -> List[Dict[str, Any]]:
    """Filter *records* by medication administration."""
    output: List[Dict[str, Any]] = []

    item_ids_cv: List[int] = filter_elem.item_ids_cv or []
    item_ids_mv: List[int] = filter_elem.item_ids_mv or []

    for rec in records:
        record_id: str = rec.get("record_id") or rec.get("record", "")
        if not record_id:
            continue

        try:
            subject_id, rec_start = _parse_record_id(record_id)
        except Exception as exc:
            logger.debug("Cannot parse record_id '%s': %s", record_id, exc)
            continue

        hadm_id = _get_hadm_id(subject_id, rec_start, cur)
        if hadm_id is None:
            logger.debug("No ICU stay for %s — skipping.", record_id)
            continue

        duration_s: Optional[float] = rec.get("duration_seconds")
        rec_end: Optional[datetime.datetime] = (
            rec_start + datetime.timedelta(seconds=duration_s)
            if duration_s is not None
            else None
        )

        # ── exclude mode: simple boolean check ───────────────────────────────
        if filter_elem.include_or_exclude == "exclude":
            has_med = _has_any_medication(subject_id, hadm_id, item_ids_cv, item_ids_mv, cur)
            if not has_med:
                out = dict(rec)
                out["offset_start_seconds"] = 0.0
                out["offset_end_seconds"] = float(duration_s) if duration_s is not None else None
                output.append(out)
            continue

        # ── include mode ──────────────────────────────────────────────────────
        if not filter_elem.constrain_to_condition_period:
            # Just check presence; return full recording if found
            has_med = _has_any_medication(subject_id, hadm_id, item_ids_cv, item_ids_mv, cur)
            if has_med:
                out = dict(rec)
                out["offset_start_seconds"] = 0.0
                out["offset_end_seconds"] = float(duration_s) if duration_s is not None else None
                output.append(out)
            continue

        # ── include + constrain_to_condition_period = True ────────────────────
        if rec_end is None:
            logger.warning(
                "Record %s has no duration_seconds; cannot compute medication "
                "time window. Skipping. Provide a signal_duration_index or "
                "include duration_seconds in precomputed_records_path JSON.",
                record_id,
            )
            continue

        intervals = _get_medication_intervals(
            subject_id, hadm_id, record_id, item_ids_cv, item_ids_mv, cur
        )
        min_dur = filter_elem.min_condition_duration_seconds or 0

        for interval_start, interval_end in intervals:
            windowed = _compute_window(
                interval_start, interval_end,
                rec_start, rec_end,
                duration_s,
                min_dur,
            )
            if windowed is not None:
                out = dict(rec)
                out["offset_start_seconds"] = windowed[0]
                out["offset_end_seconds"]   = windowed[1]
                output.append(out)

    return output


# ──────────────────────────────────────────────────────────────────────────────
# Boolean medication check (used for include-without-window and exclude)
# ──────────────────────────────────────────────────────────────────────────────

def _has_any_medication(
    subject_id: int,
    hadm_id: int,
    item_ids_cv: List[int],
    item_ids_mv: List[int],
    cur: Any,
) -> bool:
    """True if any of the given medications were given during this admission."""
    if item_ids_mv:
        cur.execute(
            "SELECT 1 FROM mimiciii.inputevents_mv "
            "WHERE subject_id = %s AND hadm_id = %s AND itemid = ANY(%s) LIMIT 1",
            (subject_id, hadm_id, item_ids_mv),
        )
        if cur.fetchone():
            return True

    if item_ids_cv:
        cur.execute(
            "SELECT 1 FROM mimiciii.inputevents_cv "
            "WHERE subject_id = %s AND hadm_id = %s AND itemid = ANY(%s) LIMIT 1",
            (subject_id, hadm_id, item_ids_cv),
        )
        if cur.fetchone():
            return True

    return False


# ──────────────────────────────────────────────────────────────────────────────
# Interval reconstruction
# ──────────────────────────────────────────────────────────────────────────────

def _get_medication_intervals(
    subject_id: int,
    hadm_id: int,
    record_id: str,
    item_ids_cv: List[int],
    item_ids_mv: List[int],
    cur: Any,
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """Return all continuous medication administration intervals for this record."""
    intervals: List[Tuple[datetime.datetime, datetime.datetime]] = []

    # ── MV: explicit starttime / endtime ─────────────────────────────────────
    if item_ids_mv:
        cur.execute(
            "SELECT itemid, starttime, endtime FROM mimiciii.inputevents_mv "
            "WHERE subject_id = %s AND hadm_id = %s AND itemid = ANY(%s)",
            (subject_id, hadm_id, item_ids_mv),
        )
        for _item_id, starttime, endtime in cur.fetchall():
            if starttime is not None and endtime is not None and starttime != endtime:
                intervals.append((starttime, endtime))

    # ── CV: reconstruct intervals from charttime + stopped flag ──────────────
    if item_ids_cv:
        cv_intervals = _reconstruct_cv_intervals(
            subject_id, hadm_id, record_id, item_ids_cv, cur
        )
        intervals.extend(cv_intervals)

    return intervals


def _reconstruct_cv_intervals(
    subject_id: int,
    hadm_id: int,
    record_id: str,
    item_ids_cv: List[int],
    cur: Any,
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """
    Reconstruct continuous administration intervals from inputevents_cv.

    Mirrors the pre-filter and calculate_start_end_time_medication_administrations
    logic in cohort_creation_vasopressors.py exactly:

        Keep rows where:
            (rate IS NOT NULL AND amount IS NULL AND rate != 0)
            OR stopped = 'Stopped'

        Then iterate sorted by (item_id, charttime):
            — Start a new interval at the first row of each (item_id) group.
            — Close and record the interval when stopped = 'Stopped'.
    """
    cur.execute(
        """
        SELECT itemid, charttime, amount, rate, stopped
        FROM   mimiciii.inputevents_cv
        WHERE  subject_id = %s
          AND  hadm_id    = %s
          AND  itemid     = ANY(%s)
        ORDER BY itemid, charttime
        """,
        (subject_id, hadm_id, item_ids_cv),
    )
    rows = cur.fetchall()

    if not rows:
        return []

    # Build DataFrame and apply the same pre-filter as the distribution code
    df = pd.DataFrame(rows, columns=["itemid", "charttime", "amount", "rate", "stopped"])

    active_mask = (
        df["rate"].notna() & df["amount"].isna() & (df["rate"] != 0.0)
    )
    stopped_mask = df["stopped"] == "Stopped"
    df = df[active_mask | stopped_mask].copy()
    df.sort_values(by=["itemid", "charttime"], inplace=True)

    intervals: List[Tuple[datetime.datetime, datetime.datetime]] = []
    current_item: Any = None
    starttime: Optional[datetime.datetime] = None

    for _, row in df.iterrows():
        item = row["itemid"]

        # New (item_id) group → reset and start fresh interval
        if item != current_item:
            current_item = item
            starttime = row["charttime"]

        if row["stopped"] == "Stopped":
            endtime = row["charttime"]
            if starttime is not None and starttime != endtime:
                intervals.append((starttime, endtime))
            # Next row for the same item starts a new interval
            starttime = None

    return intervals


# ──────────────────────────────────────────────────────────────────────────────
# Overlap / window computation
# ──────────────────────────────────────────────────────────────────────────────

def _compute_window(
    cond_start: datetime.datetime,
    cond_end: datetime.datetime,
    rec_start: datetime.datetime,
    rec_end: datetime.datetime,
    rec_duration_s: float,
    min_duration_s: int,
) -> Optional[Tuple[float, float]]:
    """
    Compute (offset_start_seconds, offset_end_seconds) for the overlap of
    a condition interval with a recording window.

    Mirrors transform_csv_to_csv_with_start_and_end in
    cohort_creation_vasopressors.py exactly.

    Returns None if:
        — the condition and recording do not overlap, or
        — the overlapping duration is shorter than min_duration_s.
    """
    # Three-way overlap check
    cond_starts_in_rec = rec_start <= cond_start <= rec_end
    cond_ends_in_rec   = rec_start <= cond_end   <= rec_end
    rec_inside_cond    = cond_start <= rec_start and rec_end <= cond_end

    if not (cond_starts_in_rec or cond_ends_in_rec or rec_inside_cond):
        return None

    # Overlap duration = min(ends) – max(starts)
    overlap_start = max(cond_start, rec_start)
    overlap_end   = min(cond_end,   rec_end)
    overlap_s     = (overlap_end - overlap_start).total_seconds()

    if overlap_s < min_duration_s:
        return None

    # Offsets relative to recording start
    offset_start = max((cond_start - rec_start).total_seconds(), 0.0)
    offset_end   = min(offset_start + overlap_s, rec_duration_s)

    if offset_start >= rec_duration_s:
        return None
    if (offset_end - offset_start) < min_duration_s:
        return None

    return offset_start, offset_end
