"""
Procedure-based cohort filter.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..select_config import FilterElement

logger = logging.getLogger(__name__)

# ±2-hour buffer applied to ICU stay in/out-times when matching a recording
_ADMISSION_BUFFER = datetime.timedelta(hours=2)


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def apply_procedure_filter(
    filter_elem: FilterElement,
    records: List[Dict[str, Any]],
    cur: Any,
) -> List[Dict[str, Any]]:
    """
    Filter *records* by surgical procedure (ICD-9 and/or CPT).

    Each element in filter_elem.criteria is tested; a record is considered
    matching if ANY criterion is satisfied (OR/union logic).

    Returns enriched record dicts with offset_start_seconds and
    offset_end_seconds set where duration is known.
    """
    output: List[Dict[str, Any]] = []

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
            logger.debug("No matching ICU stay for record %s — skipping.", record_id)
            continue

        # ── ICU-stay-count constraint ─────────────────────────────────────────
        # Procedures are recorded per admission. With more than one ICU stay
        # we cannot tell which stay followed the procedure.
        if (
            filter_elem.include_or_exclude == "include"
            and _get_icu_stay_count(hadm_id, cur) > 1
        ):
            continue

        # ── criterion matching (OR logic) ─────────────────────────────────────
        record_matches = _record_matches_any_criterion(
            subject_id, hadm_id, filter_elem.criteria or [], cur
        )

        keep = (
            (filter_elem.include_or_exclude == "include" and record_matches)
            or (filter_elem.include_or_exclude == "exclude" and not record_matches)
        )
        if keep:
            output.append(_enrich(rec))

    return output


# ──────────────────────────────────────────────────────────────────────────────
# Record ID parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_record_id(record_id: str) -> Tuple[int, datetime.datetime]:
    """Parse MIMIC record ID into (subject_id, recording_start).

    Format: p000020-2183-04-28-17-47n
    """
    parts = record_id.split("-")
    subject_id = int(parts[0].lstrip("p"))
    year   = int(parts[1])
    month  = int(parts[2])
    day    = int(parts[3])
    hour   = int(parts[4])
    minute = int(parts[5].rstrip("n"))
    return subject_id, datetime.datetime(year, month, day, hour, minute)


# ──────────────────────────────────────────────────────────────────────────────
# Database helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_hadm_id(
    subject_id: int,
    rec_start: datetime.datetime,
    cur: Any,
) -> Optional[int]:
    """Return the hadm_id whose ICU stay contains *rec_start* (±2-hour buffer)."""
    cur.execute(
        "SELECT hadm_id, intime, outtime FROM mimiciii.icustays WHERE subject_id = %s",
        (subject_id,),
    )
    for hadm_id, intime, outtime in cur.fetchall():
        if outtime is None:
            continue
        if (intime - _ADMISSION_BUFFER) <= rec_start <= (outtime + _ADMISSION_BUFFER):
            return int(hadm_id)
    return None


def _get_icu_stay_count(hadm_id: int, cur: Any) -> int:
    cur.execute(
        "SELECT COUNT(*) FROM mimiciii.icustays WHERE hadm_id = %s", (hadm_id,)
    )
    return int(cur.fetchone()[0])


def _record_matches_any_criterion(
    subject_id: int,
    hadm_id: int,
    criteria: list,
    cur: Any,
) -> bool:
    """Return True if the record satisfies at least one criterion (OR logic)."""
    for criterion in criteria:
        if criterion.type == "icd9":
            prefixes = _code_ranges_to_prefixes(criterion.code_ranges)
            if _matches_icd9(subject_id, hadm_id, prefixes, cur):
                return True

        elif criterion.type == "cpt":
            if _matches_cpt(subject_id, hadm_id, criterion.code_ranges, cur):
                return True

    return False


def _code_ranges_to_prefixes(code_ranges: List[Dict[str, int]]) -> List[str]:
    """
    Convert inclusive numeric ranges into ICD-9 code prefixes.

    Each range is read as leading digits of a code, with start zero-padded to
    the width of end, so {"start": 3500, "end": 3999} becomes
    ["35", "36", "37", "38", "39"].
    """
    prefixes: List[str] = []
    for rng in code_ranges:
        width = len(str(rng["end"]))
        prefixes.extend(_prefix_cover(rng["start"], rng["end"], width))
    return prefixes


def _prefix_cover(start: int, end: int, width: int) -> List[str]:
    """Prefixes whose width-digit expansions are exactly the numbers start..end."""
    if start == 0 and end == 10 ** width - 1:
        return [""]
    block = 10 ** (width - 1)
    first, last = start // block, end // block
    if first == last:
        return [str(first) + p for p in _prefix_cover(start % block, end % block, width - 1)]
    return (
        [str(first) + p for p in _prefix_cover(start % block, block - 1, width - 1)]
        + [str(d) for d in range(first + 1, last)]
        + [str(last) + p for p in _prefix_cover(0, end % block, width - 1)]
    )


def _matches_icd9(
    subject_id: int,
    hadm_id: int,
    prefixes: List[str],
    cur: Any,
) -> bool:
    """True if any procedure ICD-9 code for the stay starts with any prefix."""
    cur.execute(
        "SELECT icd9_code FROM mimiciii.procedures_icd "
        "WHERE subject_id = %s AND hadm_id = %s",
        (subject_id, hadm_id),
    )
    codes = [row[0] for row in cur.fetchall()]
    return any(
        code.startswith(prefix) for code in codes for prefix in prefixes
    )


def _matches_cpt(
    subject_id: int,
    hadm_id: int,
    code_ranges: List[Dict[str, int]],
    cur: Any,
) -> bool:
    """True if any CPT event for the stay falls in a range."""
    cur.execute(
        "SELECT cpt_number FROM mimiciii.cptevents "
        "WHERE subject_id = %s AND hadm_id = %s",
        (subject_id, hadm_id),
    )
    for (cpt_number,) in cur.fetchall():
        if cpt_number is not None:
            for rng in code_ranges:
                if rng["start"] <= int(cpt_number) <= rng["end"]:
                    return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Output enrichment
# ──────────────────────────────────────────────────────────────────────────────

def _enrich(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add offset fields to a procedure-cohort record.

    For procedure cohorts the full recording is included, so offset_start = 0
    and offset_end = duration_seconds (resolved later if None).
    """
    out = dict(rec)
    out["offset_start_seconds"] = 0.0
    # duration_seconds may be None here; record_selector.resolve_duration fills it
    dur = rec.get("duration_seconds")
    out["offset_end_seconds"] = float(dur) if dur is not None else None
    return out
