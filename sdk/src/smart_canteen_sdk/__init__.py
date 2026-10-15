"""Smart Canteen SDK: pure recommendation API with optional adapters and server."""
from .engine import CanteenEngine
from .errors import CanteenError
from .schemas import DataSnapshot, FinalResult, Proposal, RecommendationRequest, Selection

__version__ = "0.1.0"
__all__ = ["CanteenEngine", "CanteenError", "DataSnapshot", "FinalResult", "Proposal", "RecommendationRequest", "Selection"]
