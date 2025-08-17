"""LeMat-GenBench: Benchmark suite for generative models for materials.

This package provides a comprehensive benchmarking framework for evaluating 
material generation models across multiple metrics including validity, 
distribution, diversity, novelty, uniqueness, and stability.

This version includes enhanced benchmarks using new implementations:
- novelty_new: Enhanced novelty evaluation using augmented fingerprints
- uniqueness_new: Enhanced uniqueness evaluation using augmented fingerprints  
- sun_new: Enhanced SUN benchmark using augmented fingerprinting

The package includes both legacy and enhanced CLI interfaces:
- cli: Enhanced CLI with new benchmark implementations
- cli_legacy: Legacy CLI implementation
"""

from .cli import main
from .cli_legacy import main as main_legacy

__version__ = "0.2.0"

__all__ = [
    "main",         # Enhanced CLI function with new benchmarks (current)
    "main_legacy",  # Legacy CLI function  
]