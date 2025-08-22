"""Fingerprinting utilities for structure comparison.

This package provides various fingerprinting methods for crystal structures,
including the augmented fingerprinting approach that combines crystallographic
metadata with chemical composition.
"""

from .augmented_fingerprint import (
    AugmentedFingerprinter,
    get_augmented_fingerprint,
    record_to_anonymous_fingerprint,
    record_to_augmented_fingerprint,
    record_to_relaxed_AFLOW_fingerprint,
    record_to_strict_AFLOW_fingerprint,
)
from .crystallographic_analyzer import (
    CrystallographicAnalyzer,
    analyze_lematbulk_item,
    lematbulk_item_to_structure,
    structure_to_crystallographic_dict,
)
from .encode_compositions import (
    filter_df,
    get_all_compositions,
    lematbulk_item_to_structure,
    one_hot_encode_composition,
    process_chunk,
)
from .utils import get_fingerprint, get_fingerprinter

__all__ = [
    # Crystallographic analysis
    "CrystallographicAnalyzer",
    "structure_to_crystallographic_dict",
    "lematbulk_item_to_structure",
    "analyze_lematbulk_item",
    # Augmented fingerprinting
    "AugmentedFingerprinter",
    "record_to_augmented_fingerprint",
    "record_to_anonymous_fingerprint", 
    "record_to_relaxed_AFLOW_fingerprint",
    "record_to_strict_AFLOW_fingerprint",
    "get_augmented_fingerprint",
    # Composition encoding
    "get_all_compositions",
    "filter_df",
    "one_hot_encode_composition",
    "process_chunk",
    # Fingerprinting utilities
    "get_fingerprint",
    "get_fingerprinter",
]