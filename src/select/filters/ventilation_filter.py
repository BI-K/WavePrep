"""
Mechanical-ventilation cohort filter (time_span_level).
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..select_config import FilterElement
from .procedure_filter import _get_hadm_id, _parse_record_id
from .medication_filter import _compute_window

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def apply_ventilation_filter(
    filter_elem: FilterElement,
    records: List[Dict[str, Any]],
    cur: Any,
) -> List[Dict[str, Any]]:
    """Filter *records* by mechanical ventilation (CPT 94002/94003)."""
    output: List[Dict[str, Any]] = []
    cpt_codes: List[str] = filter_elem.cpt_codes or ["94002", "94003"]

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

        # ── exclude mode ──────────────────────────────────────────────────────
        if filter_elem.include_or_exclude == "exclude":
            has_vent = _has_ventilation(subject_id, hadm_id, cpt_codes, cur)
            if not has_vent:
                out = dict(rec)
                out["offset_start_seconds"] = 0.0
                out["offset_end_seconds"] = float(duration_s) if duration_s is not None else None
                output.append(out)
            continue

        # ── include + no time windowing ───────────────────────────────────────
        if not filter_elem.constrain_to_condition_period:
            has_vent = _has_ventilation(subject_id, hadm_id, cpt_codes, cur)
            if has_vent:
                out = dict(rec)
                out["offset_start_seconds"] = 0.0
                out["offset_end_seconds"] = float(duration_s) if duration_s is not None else None
                output.append(out)
            continue

        # ── include + constrain_to_condition_period = True ────────────────────
        if rec_end is None:
            logger.warning(
                "Record %s has no duration_seconds; cannot compute ventilation "
                "time window. Skipping. Provide signal_duration_index or include "
                "duration_seconds in precomputed_records_path JSON.",
                record_id,
            )
            continue

        vent_window = _get_ventilation_window(subject_id, hadm_id, record_id, cur)
        if vent_window is None:
            continue

        vent_start, vent_end = vent_window
        min_dur = filter_elem.min_condition_duration_seconds or 0

        windowed = _compute_window(
            vent_start, vent_end,
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
# Database helpers
# ──────────────────────────────────────────────────────────────────────────────

def _has_ventilation(
    subject_id: int,
    hadm_id: int,
    cpt_codes: List[str],
    cur: Any,
) -> bool:
    """True if any of the given CPT codes appear for this admission."""
    cur.execute(
        "SELECT 1 FROM mimiciii.cptevents "
        "WHERE subject_id = %s AND hadm_id = %s AND cpt_cd = ANY(%s) LIMIT 1",
        (subject_id, hadm_id, cpt_codes),
    )
    return cur.fetchone() is not None


def _get_ventilation_window(
    subject_id: int,
    hadm_id: int,
    record_id: str,
    cur: Any,
) -> Optional[Tuple[datetime.datetime, datetime.datetime]]:
    """
    Return (ventilation_start, ventilation_end) for this record.

    Mirrors cohort_creation_ventilation.py:
        — Query cptevents for CPT code 94003 (continuation) only.
        — Take FIRST chartdate as window start, LAST chartdate as window end.
        — chartdate is day-resolution; interpret as start of that calendar day
          (HH:MM:SS = 00:00:00).  The end boundary is the START of the NEXT
          day so the full last ventilation day is included.
    """
    cur.execute(
        "SELECT chartdate FROM mimiciii.cptevents "
        "WHERE subject_id = %s AND hadm_id = %s AND cpt_cd = '94003' "
        "ORDER BY chartdate",
        (subject_id, hadm_id),
    )
    rows = cur.fetchall()
    if not rows:
        return None

    chartdates = [row[0] for row in rows if row[0] is not None]
    if not chartdates:
        return None

    # chartdate is a timestamp in MIMIC-III (stored as date, read as datetime)
    first_date = _to_date_midnight(min(chartdates))
    last_date  = _to_date_midnight(max(chartdates))

    # Discard degenerate windows (same start and end day)
    if first_date == last_date:
        return None

    return first_date, last_date


def _to_date_midnight(dt: Any) -> datetime.datetime:
    """Convert a date or datetime to midnight of that calendar day."""
    if isinstance(dt, datetime.datetime):
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if isinstance(dt, datetime.date):
        return datetime.datetime(dt.year, dt.month, dt.day, 0, 0, 0)
    # fallback: try string parse
    try:
        parsed = datetime.datetime.strptime(str(dt)[:10], "%Y-%m-%d")
        return parsed
    except ValueError:
        raise ValueError(f"Cannot convert {dt!r} to midnight datetime.")
