"""Benchmarks package for material generation evaluation.

This package contains benchmark implementations that combine multiple
metrics to provide comprehensive evaluation of generated materials.
Each benchmark represents a different aspect of quality assessment.
"""

from .base import BaseBenchmark, BenchmarkConfig, BenchmarkResult
from .distribution_benchmark import DistributionBenchmark
from .diversity_benchmark import DiversityBenchmark
from .hhi_benchmark import HHIBenchmark
from .multi_mlip_stability_benchmark import StabilityBenchmark
from .novelty_benchmark import NoveltyBenchmark
from .novelty_new_benchmark import (
    AugmentedNoveltyBenchmark,
    create_augmented_novelty_benchmark,
    create_computation_based_novelty_benchmark,
    create_high_precision_novelty_benchmark,
    create_property_based_novelty_benchmark,
    create_robust_novelty_benchmark,
)
from .sun_benchmark import SUNBenchmark
from .uniqueness_benchmark import UniquenessBenchmark
from .uniqueness_new_benchmark import UniquenessNewBenchmark
from .validity_benchmark import ValidityBenchmark

__all__ = [
    # Base classes
    "BaseBenchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    # Benchmark implementations
    "DistributionBenchmark",
    "DiversityBenchmark",
    "HHIBenchmark",
    "NoveltyBenchmark",
    "AugmentedNoveltyBenchmark",  # New enhanced novelty benchmark
    "SUNBenchmark",
    "StabilityBenchmark",
    "UniquenessBenchmark",
    "UniquenessNewBenchmark",
    "ValidityBenchmark",
    # Factory functions for enhanced novelty benchmark
    "create_augmented_novelty_benchmark",
    "create_computation_based_novelty_benchmark",
    "create_high_precision_novelty_benchmark", 
    "create_property_based_novelty_benchmark",
    "create_robust_novelty_benchmark",
]