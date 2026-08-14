"""CLIP-based feature extraction, caching, classification, and evaluation."""

from seedlearn.clip.metrics import (
    EvaluationResult,
    compare_results,
    compute_metrics,
    print_results_summary,
    save_evaluation_results,
)
from seedlearn.clip.simpleshot import (
    FewShotClassifier,
    NearestSupportMatch,
    PredictionDetail,
    SimpleShot,
    l2_normalize,
)

__all__ = [
    # metrics
    "EvaluationResult",
    "compare_results",
    "compute_metrics",
    "print_results_summary",
    "save_evaluation_results",
    # simpleshot
    "FewShotClassifier",
    "NearestSupportMatch",
    "PredictionDetail",
    "SimpleShot",
    "l2_normalize",
]

# Lazy imports for encoder/cache (require pybioclip)


def __getattr__(name: str):
    if name == "FeatureExtractor":
        from seedlearn.clip.encoder import FeatureExtractor
        return FeatureExtractor
    if name == "CachedFeatureExtractor":
        from seedlearn.clip.cache import CachedFeatureExtractor
        return CachedFeatureExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
