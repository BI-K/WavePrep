from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DatabaseLogin:
    host: str = "localhost"
    port: int = 5432
    dbname: str = "mimic_iii"
    user: str = "postgres"
    password: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatabaseLogin":
        return cls(
            host=d.get("host", "localhost"),
            port=d.get("port", 5432),
            dbname=d.get("dbname", "mimic_iii"),
            user=d.get("user", "postgres"),
            password=d.get("password"),
        )


@dataclass
class DatabaseConfig:
    matched_waveform_database: str
    login: DatabaseLogin
    signal_duration_index: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatabaseConfig":
        return cls(
            matched_waveform_database=d["matched_waveform_database"],
            login=DatabaseLogin.from_dict(d["login"]),
            signal_duration_index=d.get("signal_duration_index"),
        )


@dataclass
class RecordRequirements:
    required_signals: List[str]
    min_duration_seconds: int = 3600
    recording_type: str = "numeric"  # "numeric", "non-numeric", or "both"
    precomputed_records_path: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RecordRequirements":
        return cls(
            required_signals=d["required_signals"],
            min_duration_seconds=d.get("min_duration_seconds", 3600),
            recording_type=d.get("recording_type", "numeric"),
            precomputed_records_path=d.get("precomputed_records_path"),
        )


@dataclass
class ProcedureCriterion:
    """A single procedure criterion — either ICD-9 or CPT."""

    type: str  # "icd9" or "cpt"
    icd9_code_prefixes: Optional[List[str]] = None
    cpt_code_ranges: Optional[List[Dict[str, int]]] = None  # [{"start": int, "end": int}]
    cpt_codes: Optional[List[str]] = None  # exact CPT code strings

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProcedureCriterion":
        return cls(
            type=d["type"],
            icd9_code_prefixes=d.get("icd9_code_prefixes"),
            cpt_code_ranges=d.get("cpt_code_ranges"),
            cpt_codes=d.get("cpt_codes"),
        )


@dataclass
class FilterElement:
    """
    One element in cohort_definition[].  Elements are AND-combined; criteria
    within a "procedure" element are OR-combined (union).

    filter_type values:
        "procedure"              — stay_level:      ICD-9 / CPT code checks
        "medication"             — time_span_level: inputevents_cv + inputevents_mv
        "mechanical_ventilation" — time_span_level: CPT 94002 / 94003
    """

    filter_type: str
    level: str
    include_or_exclude: str  # "include" or "exclude"

    # ── procedure fields ─────────────────────────────────────────────────────
    criteria: Optional[List[ProcedureCriterion]] = None
    max_icu_stays_per_admission: Optional[int] = None

    # ── medication fields ─────────────────────────────────────────────────────
    item_ids_cv: Optional[List[int]] = None
    item_ids_mv: Optional[List[int]] = None

    # ── ventilation fields ────────────────────────────────────────────────────
    cpt_codes: Optional[List[str]] = None

    # ── time-windowing (medication + ventilation) ─────────────────────────────
    constrain_to_condition_period: bool = False
    min_condition_duration_seconds: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FilterElement":
        criteria: Optional[List[ProcedureCriterion]] = None
        if "criteria" in d:
            criteria = [ProcedureCriterion.from_dict(c) for c in d["criteria"]]
        return cls(
            filter_type=d["filter_type"],
            level=d["level"],
            include_or_exclude=d["include_or_exclude"],
            criteria=criteria,
            max_icu_stays_per_admission=d.get("max_icu_stays_per_admission"),
            item_ids_cv=d.get("item_ids_cv"),
            item_ids_mv=d.get("item_ids_mv"),
            cpt_codes=d.get("cpt_codes"),
            constrain_to_condition_period=d.get("constrain_to_condition_period", False),
            min_condition_duration_seconds=d.get("min_condition_duration_seconds"),
        )


@dataclass
class OutputConfig:
    path: str = "inputs/"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OutputConfig":
        return cls(path=d.get("path", "inputs/"))


@dataclass
class SelectConfig:
    name: str
    description: str
    database: DatabaseConfig
    record_requirements: RecordRequirements
    cohort_definition: List[FilterElement]
    output: OutputConfig

    @classmethod
    def from_json(cls, path: str) -> "SelectConfig":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            database=DatabaseConfig.from_dict(d["database"]),
            record_requirements=RecordRequirements.from_dict(d["record_requirements"]),
            cohort_definition=[FilterElement.from_dict(fe) for fe in d["cohort_definition"]],
            output=OutputConfig.from_dict(d["output"]),
        )
