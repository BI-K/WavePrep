"""Data classes for signal data representation throughout the pipeline."""

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple

import numpy as np


class Window(NamedTuple):
    """A single observation/prediction window pair."""
    observation: np.ndarray
    prediction: np.ndarray


@dataclass
class SignalData(MutableMapping):
    """Multi-channel signal data for a single continuous snippet.

    Inherits from MutableMapping so it is recognized as a dict-like object
    by pandas, copy.deepcopy, and all existing code that uses dict access.
    """
    channels: Dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_numpy(cls, data: np.ndarray, channel_names: List[str]) -> 'SignalData':
        """Create from a 2D numpy array (samples x channels) and column names."""
        channels = {name: np.array(data[:, i]) for i, name in enumerate(channel_names)}
        return cls(channels=channels)

    def to_numpy(self) -> np.ndarray:
        """Convert to a 2D numpy array (samples x channels)."""
        if not self.channels:
            return np.array([])
        return np.column_stack(list(self.channels.values()))

    @property
    def channel_names(self) -> List[str]:
        return list(self.channels.keys())

    @property
    def n_samples(self) -> int:
        if not self.channels:
            return 0
        return len(next(iter(self.channels.values())))

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    # MutableMapping required methods

    def __getitem__(self, key: str) -> np.ndarray:
        return self.channels[key]

    def __setitem__(self, key: str, value: np.ndarray):
        self.channels[key] = value

    def __delitem__(self, key: str):
        del self.channels[key]

    def __iter__(self):
        return iter(self.channels)

    def __len__(self) -> int:
        return len(self.channels)


@dataclass
class WindowedData(MutableMapping):
    """Windowed signal data with observation/prediction pairs per channel.

    Inherits from MutableMapping for backward compatibility with dict access.
    """
    channels: Dict[str, List[Window]] = field(default_factory=dict)

    @property
    def channel_names(self) -> List[str]:
        return list(self.channels.keys())

    @property
    def n_windows(self) -> int:
        if not self.channels:
            return 0
        return len(next(iter(self.channels.values())))

    def __getitem__(self, key: str) -> List[Window]:
        return self.channels[key]

    def __setitem__(self, key: str, value: List[Window]):
        self.channels[key] = value

    def __delitem__(self, key: str):
        del self.channels[key]

    def __iter__(self):
        return iter(self.channels)

    def __len__(self) -> int:
        return len(self.channels)

    def __bool__(self) -> bool:
        if not self.channels:
            return False
        return any(len(windows) > 0 for windows in self.channels.values())
