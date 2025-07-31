"""UMA model implementation."""

try:
    from .calculator import (
        AVAILABLE_UMA_MODELS,
        AVAILABLE_UMA_TASKS,
        UMACalculator,
        create_uma_calculator,
    )
    from .embeddings import UMAASECalculator, UMAEmbeddingExtractor

    UMA_AVAILABLE = True
except ImportError:
    UMA_AVAILABLE = False
    UMACalculator = None
    UMAEmbeddingExtractor = None
    UMAASECalculator = None
    create_uma_calculator = None
    AVAILABLE_UMA_MODELS = []
    AVAILABLE_UMA_TASKS = []

INFO = {
    "name": "UMA",
    "description": "Universal Models for Atoms by Meta",
    "default_model": "uma-s-1",
    "default_task": "omat",
    "supports_embeddings": True,
    "supports_relaxation": True,
}

__all__ = [
    "UMACalculator",
    "UMAEmbeddingExtractor",
    "UMAASECalculator",
    "create_uma_calculator",
    "AVAILABLE_UMA_MODELS",
    "AVAILABLE_UMA_TASKS",
    "UMA_AVAILABLE",
    "INFO",
]
