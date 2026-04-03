from dataclasses import dataclass, asdict
from typing import List


@dataclass
class AnalogItem:
    id: str
    title: str
    sold_price: float


@dataclass
class VisionAssessment:
    coefficient: float
    reason: str


@dataclass
class PriceEstimate:
    fast: float
    balanced: float
    max: float


@dataclass
class AnalysisResult:
    prices: PriceEstimate
    analogs: List[AnalogItem]
    vision: VisionAssessment

    def to_dict(self) -> dict:
        return asdict(self)