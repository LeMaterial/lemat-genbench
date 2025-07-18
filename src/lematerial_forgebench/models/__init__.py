"""Models package for ML interatomic potentials.

This package provides a unified interface for different ML interatomic potentials
including ORB, MACE, Equiformer v2, and UMA.
"""

from lematerial_forgebench.models.base import (
    BaseEmbeddingExtractor,
    BaseMLIPCalculator,
    CalculationResult,
    EmbeddingResult,
    get_energy_above_hull_from_total_energy,
    get_formation_energy_from_total_energy,
)
from lematerial_forgebench.models.registry import (
    get_calculator,
    get_equiformer_calculator,
    get_mace_calculator,
    get_model_info,
    get_orb_calculator,
    get_uma_calculator,
    list_available_models,
    print_model_info,
)

# Try to import specific calculators if available
try:
    from lematerial_forgebench.models.orb.calculator import ORBCalculator
except ImportError:
    ORBCalculator = None

try:
    from lematerial_forgebench.models.mace.calculator import MACECalculator
except ImportError:
    MACECalculator = None

try:
    from lematerial_forgebench.models.equiformer.calculator import EquiformerCalculator
except ImportError:
    EquiformerCalculator = None

try:
    from lematerial_forgebench.models.uma.calculator import UMACalculator
except ImportError:
    UMACalculator = None

__all__ = [
    # Base classes
    "BaseMLIPCalculator",
    "BaseEmbeddingExtractor",
    "CalculationResult",
    "EmbeddingResult",
    # Registry functions
    "get_calculator",
    "list_available_models",
    "get_model_info",
    "print_model_info",
    "get_orb_calculator",
    "get_mace_calculator",
    "get_equiformer_calculator",
    "get_uma_calculator",
    # Utility functions
    "get_formation_energy_from_total_energy",
    "get_energy_above_hull_from_total_energy",
    # Specific calculators (if available)
    "ORBCalculator",
    "MACECalculator",
    "EquiformerCalculator",
    "UMACalculator",
]
