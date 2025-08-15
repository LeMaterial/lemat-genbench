"""Fingerprinting utilities for structure comparison.

This package provides various fingerprinting methods for crystal structures,
including the augmented fingerprinting approach that combines crystallographic
metadata with chemical composition.
"""

from .crystallographic_analyzer import (
    CrystallographicAnalyzer,
    analyze_lematbulk_item,
    lematbulk_item_to_structure,
    structure_to_crystallographic_dict,
)

__all__ = [
    "CrystallographicAnalyzer",
    "structure_to_crystallographic_dict", 
    "lematbulk_item_to_structure",
    "analyze_lematbulk_item",
]