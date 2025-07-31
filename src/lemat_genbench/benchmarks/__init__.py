"""Benchmarks package for material generation evaluation.

This package contains benchmark implementations that combine multiple
metrics to provide comprehensive evaluation of generated materials.
Each benchmark represents a different aspect of quality assessment.
"""

from .base import BaseBenchmark, BenchmarkConfig, BenchmarkResult
from .distribution_benchmark import DistributionBenchmark
from .hhi_benchmark import HHIBenchmark
from .multi_mlip_stability_benchmark import StabilityBenchmark
from .novelty_benchmark import NoveltyBenchmark
from .sun_benchmark import SUNBenchmark
from .uniqueness_benchmark import UniquenessBenchmark
from .validity_benchmark import ValidityBenchmark

__all__ = [
    # Base classes
    "BaseBenchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    # Benchmark implementations
    "DistributionBenchmark",
    "HHIBenchmark",
    "NoveltyBenchmark", 
    "SUNBenchmark",
    "StabilityBenchmark",
    "UniquenessBenchmark",
    "ValidityBenchmark",
]