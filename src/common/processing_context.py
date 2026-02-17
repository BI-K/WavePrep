"""Processing context passed through the signal processing pipeline."""

from dataclasses import dataclass, field
from typing import List, Any, Optional


@dataclass
class ProcessingContext:
    """Metadata and configuration passed through the processing pipeline.

    Replaces the ad-hoc ``metadata`` dict that was previously built up and
    passed between loading, processing, and visualization functions.
    """

    record_id: str = ""
    sampling_rate: float = 1.0
    min_record_duration: int = 0
    output_path_process_images: str = "outputs/reports/process_images"
    imputer_path: str = ""
    windowing_config: Any = None
    n_channels: int = 0
    available_channels: List[str] = field(default_factory=list)
