"""ORB model implementation."""

try:
    from .calculator import AVAILABLE_ORB_MODELS, ORBCalculator, create_orb_calculator
    from .embeddings import ORBEmbeddingExtractor

    ORB_AVAILABLE = True
except ImportError:
    ORB_AVAILABLE = False
    ORBCalculator = None
    ORBEmbeddingExtractor = None
    create_orb_calculator = None
    AVAILABLE_ORB_MODELS = []

INFO = {
    "name": "ORB",
    "description": "Orbital Materials' ORB force fields",
    "default_model": "orb_v3_conservative_inf_omat",
    "supports_embeddings": True,
    "supports_relaxation": True,
}

__all__ = [
    "ORBCalculator",
    "ORBEmbeddingExtractor",
    "create_orb_calculator",
    "AVAILABLE_ORB_MODELS",
    "ORB_AVAILABLE",
    "INFO",
]
