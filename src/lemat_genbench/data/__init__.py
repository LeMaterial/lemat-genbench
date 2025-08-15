"""Data utilities for lemat-genbench.

This package provides utilities for loading and managing reference datasets,
including preprocessed fingerprints and other cached data.
"""

from .reference_fingerprint_loader import (
    ReferenceFingerprintDatabase,
    check_structure_novelty,
    load_reference_database,
)

__all__ = [
    "ReferenceFingerprintDatabase",
    "load_reference_database", 
    "check_structure_novelty",
]