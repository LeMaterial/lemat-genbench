"""Equiformer v2 model implementation."""

try:
    from .calculator import EquiformerCalculator, create_equiformer_calculator
    from .embeddings import EquiformerASECalculator, EquiformerEmbeddingExtractor

    EQUIFORMER_AVAILABLE = True
except ImportError:
    EQUIFORMER_AVAILABLE = False
    EquiformerCalculator = None
    EquiformerEmbeddingExtractor = None
    EquiformerASECalculator = None
    create_equiformer_calculator = None


INFO = {
    "name": "Equiformer",
    "description": "Equiformer v2: Transformer for molecules and materials",
    "default_model": "custom",
    "supports_embeddings": True,
    "supports_relaxation": True,
}

__all__ = [
    "EquiformerCalculator",
    "EquiformerEmbeddingExtractor",
    "EquiformerASECalculator",
    "create_equiformer_calculator",
    "EQUIFORMER_AVAILABLE",
    "INFO",
]
