# ==============================================================================
# audio_matcher/data_types.py
# ==============================================================================
from dataclasses import dataclass
from typing import TypeAlias
import numpy as np

AudioArray: TypeAlias = np.ndarray
Spectrogram: TypeAlias = np.ndarray
FeatureMatrix: TypeAlias = np.ndarray
TimeRange: TypeAlias = tuple[float, float]

@dataclass
class MatchResult:
    score: float
    start_f: float
    end_f: float
    method: str
    y_long: AudioArray | None = None
    y_query: AudioArray | None = None
    sr: int = 0
    vis_spec_query: Spectrogram | None = None
    vis_spec_long: Spectrogram | None = None
    whisper_text: str = ""
    error: str | None = None

    @property
    def is_success(self) -> bool:
        return self.error is None
