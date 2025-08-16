"""Reference fingerprint loader for novelty detection.

This module provides utilities to load and query the preprocessed LeMat-Bulk
fingerprints for novelty detection with StructureMatcher fallback support.
"""

import glob
import os
import pickle
from typing import Any, Dict, List, Optional, Set

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure

from lemat_genbench.utils.logging import logger


class ReferenceFingerprintDatabase:
    """Database for querying reference fingerprints with StructureMatcher fallback."""
    
    def __init__(self, reference_dir: str = "data/reference_fingerprints"):
        """Initialize the reference database.
        
        Parameters
        ----------
        reference_dir : str
            Directory containing preprocessed fingerprint files.
        """
        self.reference_dir = reference_dir
        self.fingerprint_set: Set[str] = set()
        self.structure_matcher = StructureMatcher()
        self._fingerprint_to_structures: Dict[str, List[Dict]] = {}
        self._loaded = False
        
    def load_fingerprints(self, load_structures: bool = False) -> None:
        """Load reference fingerprints from disk.
        
        Parameters
        ----------
        load_structures : bool, default=False
            Whether to load full structure data for StructureMatcher fallback.
            If False, only loads fingerprint strings for fast lookup.
        """
        if self._loaded:
            logger.info("Reference fingerprints already loaded")
            return
        
        logger.info(f"Loading reference fingerprints from {self.reference_dir}")
        
        # Load unique fingerprints set (always fast)
        fingerprint_set_file = os.path.join(self.reference_dir, "unique_fingerprints.pkl")
        if os.path.exists(fingerprint_set_file):
            with open(fingerprint_set_file, 'rb') as f:
                self.fingerprint_set = pickle.load(f)
            logger.info(f"Loaded {len(self.fingerprint_set):,} unique fingerprints")
        else:
            logger.warning(f"Unique fingerprints file not found: {fingerprint_set_file}")
            logger.info("Will load fingerprints from chunk files...")
            self._load_from_chunks(load_structures=False)
        
        # Load structure data if requested
        if load_structures:
            self._load_structure_data()
        
        self._loaded = True
        logger.info("Reference fingerprint database loaded successfully")
    
    def _load_from_chunks(self, load_structures: bool = False) -> None:
        """Load fingerprints from individual chunk files.
        
        Parameters
        ----------
        load_structures : bool
            Whether to load structure data.
        """
        chunk_files = glob.glob(os.path.join(self.reference_dir, "lemat_bulk_fingerprints_chunk_*.pkl"))
        chunk_files.sort()
        
        logger.info(f"Loading from {len(chunk_files)} chunk files...")
        
        for chunk_file in chunk_files:
            with open(chunk_file, 'rb') as f:
                chunk_data = pickle.load(f)
            
            for item in chunk_data:
                if item['success']:
                    fingerprint = item['fingerprint_string']
                    self.fingerprint_set.add(fingerprint)
                    
                    if load_structures:
                        if fingerprint not in self._fingerprint_to_structures:
                            self._fingerprint_to_structures[fingerprint] = []
                        self._fingerprint_to_structures[fingerprint].append(item)
        
        logger.info(f"Loaded {len(self.fingerprint_set):,} unique fingerprints from chunks")
    
    def _load_structure_data(self) -> None:
        """Load structure data for StructureMatcher fallback."""
        logger.info("Loading structure data for StructureMatcher fallback...")
        
        chunk_files = glob.glob(os.path.join(self.reference_dir, "lemat_bulk_fingerprints_chunk_*.pkl"))
        
        for chunk_file in chunk_files:
            with open(chunk_file, 'rb') as f:
                chunk_data = pickle.load(f)
            
            for item in chunk_data:
                if item['success']:
                    fingerprint = item['fingerprint_string']
                    if fingerprint not in self._fingerprint_to_structures:
                        self._fingerprint_to_structures[fingerprint] = []
                    self._fingerprint_to_structures[fingerprint].append(item)
        
        total_structures = sum(len(structures) for structures in self._fingerprint_to_structures.values())
        logger.info(f"Loaded structure data for {total_structures:,} reference structures")
    
    def is_fingerprint_novel(self, fingerprint: str) -> bool:
        """Check if a fingerprint is novel (fast lookup).
        
        Parameters
        ----------
        fingerprint : str
            Fingerprint string to check.
            
        Returns
        -------
        bool
            True if fingerprint is novel (not in reference set).
        """
        if not self._loaded:
            self.load_fingerprints(load_structures=False)
        
        return fingerprint not in self.fingerprint_set
    
    def is_structure_novel(self, structure: Structure, fingerprint: Optional[str] = None) -> bool:
        """Check if a structure is novel with StructureMatcher fallback.
        
        Parameters
        ----------
        structure : Structure
            Structure to check for novelty.
        fingerprint : str, optional
            Pre-computed fingerprint. If None, will compute from structure.
            
        Returns
        -------
        bool
            True if structure is novel.
        """
        from lemat_genbench.fingerprinting.augmented_fingerprint import (
            get_augmented_fingerprint,
        )
        
        if not self._loaded:
            self.load_fingerprints(load_structures=True)
        
        # Get fingerprint if not provided
        if fingerprint is None:
            fingerprint = get_augmented_fingerprint(structure)
            if fingerprint is None:
                logger.warning("Failed to compute fingerprint for structure")
                return True  # Assume novel if fingerprint fails
        
        # Fast check: if fingerprint is novel, structure is novel
        if fingerprint not in self.fingerprint_set:
            return True
        
        # Fingerprint collision: use StructureMatcher
        if fingerprint not in self._fingerprint_to_structures:
            logger.warning(f"Fingerprint {fingerprint} in set but no structure data loaded")
            return False  # Conservative: assume not novel
        
        # Compare with all structures having the same fingerprint
        reference_structures = self._fingerprint_to_structures[fingerprint]
        
        for ref_item in reference_structures:
            try:
                # Reconstruct pymatgen Structure from stored data
                ref_structure = self._reconstruct_structure(ref_item['structure_data'])
                
                # Compare with StructureMatcher
                if self.structure_matcher.fit(structure, ref_structure):
                    return False  # Found match - not novel
                    
            except Exception as e:
                logger.warning(f"Failed to reconstruct/compare structure: {e}")
                continue
        
        # No structural matches found despite fingerprint collision
        return True
    
    def _reconstruct_structure(self, structure_data: Dict) -> Structure:
        """Reconstruct pymatgen Structure from stored data.
        
        Parameters
        ----------
        structure_data : Dict
            Stored structure data.
            
        Returns
        -------
        Structure
            Reconstructed pymatgen Structure.
        """
        return Structure(
            lattice=structure_data['lattice_matrix'],
            species=structure_data['species'],
            coords=structure_data['coords'],
            coords_are_cartesian=False
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns
        -------
        Dict[str, Any]
            Statistics about the reference database.
        """
        if not self._loaded:
            self.load_fingerprints(load_structures=False)
        
        stats = {
            'total_unique_fingerprints': len(self.fingerprint_set),
            'structures_loaded': len(self._fingerprint_to_structures) > 0,
            'total_reference_structures': sum(len(structures) for structures in self._fingerprint_to_structures.values()) if self._fingerprint_to_structures else 0,
            'fingerprints_with_collisions': sum(1 for structures in self._fingerprint_to_structures.values() if len(structures) > 1) if self._fingerprint_to_structures else 0
        }
        
        return stats
    
    def get_summary(self) -> None:
        """Print database summary."""
        stats = self.get_statistics()
        
        print("📊 Reference Fingerprint Database Summary:")
        print(f"   Unique fingerprints: {stats['total_unique_fingerprints']:,}")
        print(f"   Structure data loaded: {stats['structures_loaded']}")
        print(f"   Total reference structures: {stats['total_reference_structures']:,}")
        
        if stats['structures_loaded']:
            print(f"   Fingerprints with collisions: {stats['fingerprints_with_collisions']:,}")
            collision_rate = stats['fingerprints_with_collisions'] / stats['total_unique_fingerprints'] * 100
            print(f"   Collision rate: {collision_rate:.2f}%")


# Convenience functions
def load_reference_database(reference_dir: str = "data/reference_fingerprints", 
                          load_structures: bool = False) -> ReferenceFingerprintDatabase:
    """Load reference fingerprint database.
    
    Parameters
    ----------
    reference_dir : str
        Directory containing reference fingerprints.
    load_structures : bool
        Whether to load structure data for StructureMatcher fallback.
        
    Returns
    -------
    ReferenceFingerprintDatabase
        Loaded database.
    """
    db = ReferenceFingerprintDatabase(reference_dir)
    db.load_fingerprints(load_structures=load_structures)
    return db


def check_structure_novelty(structure: Structure, 
                          reference_dir: str = "data/reference_fingerprints") -> bool:
    """Check if a structure is novel against reference database.
    
    Parameters
    ----------
    structure : Structure
        Structure to check.
    reference_dir : str
        Directory containing reference fingerprints.
        
    Returns
    -------
    bool
        True if structure is novel.
    """
    db = load_reference_database(reference_dir, load_structures=True)
    return db.is_structure_novel(structure)