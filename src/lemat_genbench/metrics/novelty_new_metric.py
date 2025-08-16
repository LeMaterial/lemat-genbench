"""New novelty metric using augmented fingerprinting for evaluating material structures.

This module implements an enhanced novelty metric that uses the new augmented
fingerprinting approach to compare generated structures against reference datasets.
The metric leverages the preprocessed augmented fingerprints attached to structures.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from pymatgen.core.structure import Structure

from lemat_genbench.fingerprinting.augmented_fingerprint import (
    get_augmented_fingerprint,
)
from lemat_genbench.metrics.base import BaseMetric, MetricConfig
from lemat_genbench.utils.logging import logger

warnings.filterwarnings("ignore", message="No oxidation states specified on sites!")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=r".*__array__.*copy.*"
)


@dataclass
class AugmentedNoveltyConfig(MetricConfig):
    """Configuration for the augmented novelty metric.

    Parameters
    ----------
    reference_fingerprints_path : str or None, default=None
        Path to precomputed reference fingerprints file. If None, uses default.
    reference_dataset_name : str, default="LeMat-Bulk"
        Name of the reference dataset for logging purposes.
    fingerprint_source : str, default="property"
        Source of fingerprints: "property" (from structure.properties), 
        "compute" (compute on-the-fly), or "auto" (try property first, compute if needed).
    symprec : float, default=0.01
        Symmetry precision for on-demand fingerprint computation.
    angle_tolerance : float, default=5.0
        Angle tolerance for on-demand fingerprint computation.
    fallback_to_computation : bool, default=True
        Whether to compute fingerprints if not found in structure properties.
    """

    reference_fingerprints_path: Optional[str] = None
    reference_dataset_name: str = "LeMat-Bulk"
    fingerprint_source: str = "auto"
    symprec: float = 0.01
    angle_tolerance: float = 5.0
    fallback_to_computation: bool = True


class AugmentedNoveltyMetric(BaseMetric):
    """Enhanced novelty metric using augmented fingerprinting.

    This metric evaluates the novelty of structures by comparing their augmented
    fingerprints against a reference dataset. It can use preprocessed fingerprints
    from structure properties or compute them on-demand using the enhanced
    augmented fingerprinting approach.

    The novelty score is defined as:
    N = |{x ∈ G | x ∉ T}| / |G|

    where G is the set of generated structures and T is the set of known materials.

    Parameters
    ----------
    reference_fingerprints_path : str or None, default=None
        Path to file containing reference fingerprints. If None, uses default path.
    reference_dataset_name : str, default="LeMat-Bulk"
        Name of the reference dataset for logging purposes.
    fingerprint_source : str, default="auto"
        Source of fingerprints: "property", "compute", or "auto".
    symprec : float, default=0.01
        Symmetry precision for fingerprint computation.
    angle_tolerance : float, default=5.0
        Angle tolerance for fingerprint computation.
    fallback_to_computation : bool, default=True
        Whether to compute fingerprints if not found in structure properties.
    name : str, optional
        Custom name for the metric.
    description : str, optional
        Description of what the metric measures.
    lower_is_better : bool, default=False
        Higher novelty values indicate more novel structures.
    n_jobs : int, default=1
        Number of parallel jobs to run.
    """

    def __init__(
        self,
        reference_fingerprints_path: Optional[str] = None,
        reference_dataset_name: str = "LeMat-Bulk",
        fingerprint_source: str = "auto",
        symprec: float = 0.01,
        angle_tolerance: float = 5.0,
        fallback_to_computation: bool = True,
        name: str | None = None,
        description: str | None = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "AugmentedNovelty",
            description=description
            or "Enhanced novelty metric using augmented fingerprinting",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )

        self.config = AugmentedNoveltyConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            reference_fingerprints_path=reference_fingerprints_path,
            reference_dataset_name=reference_dataset_name,
            fingerprint_source=fingerprint_source,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            fallback_to_computation=fallback_to_computation,
        )

        # Cache for reference fingerprints
        self._reference_fingerprints: Optional[Set[str]] = None
        self._reference_loaded = False

    def _load_reference_fingerprints(self) -> Set[str]:
        """Load reference fingerprints from file or default location.

        Returns
        -------
        Set[str]
            Set of reference fingerprints.
        """
        if self._reference_fingerprints is not None and self._reference_loaded:
            return self._reference_fingerprints

        try:
            # Determine data directory - default to the known location
            if self.config.reference_fingerprints_path:
                data_dir = Path(self.config.reference_fingerprints_path)
            else:
                data_dir = Path("data/augmented_fingerprints")
            
            logger.info(f"Loading reference fingerprints from {data_dir}")
            
            # Load directly from parquet file (simpler approach)
            unique_fp_parquet = data_dir / "unique_fingerprints.parquet"
            unique_fp_pkl = data_dir / "unique_fingerprints.pkl"
            
            reference_fingerprints = set()
            
            if unique_fp_parquet.exists():
                import pandas as pd
                logger.info(f"Loading unique fingerprints from {unique_fp_parquet}")
                df = pd.read_parquet(unique_fp_parquet)
                
                # Extract fingerprints from 'values' column
                if 'values' in df.columns:
                    reference_fingerprints = set(df['values'].tolist())
                else:
                    # Fallback: use first column if 'values' column not found
                    reference_fingerprints = set(df.iloc[:, 0].tolist())
                    
                logger.info(f"Loaded {len(reference_fingerprints):,} unique fingerprints from parquet")
                
            elif unique_fp_pkl.exists():
                import pickle
                logger.info(f"Loading unique fingerprints from {unique_fp_pkl}")
                with open(unique_fp_pkl, 'rb') as f:
                    reference_fingerprints = pickle.load(f)
                logger.info(f"Loaded {len(reference_fingerprints):,} unique fingerprints from pickle")
                
            else:
                logger.warning(f"No reference fingerprints found at {data_dir}")
                logger.info(f"Looked for: {unique_fp_parquet} or {unique_fp_pkl}")
                logger.info("Ensure fingerprint preprocessing has been run or provide custom path")
                reference_fingerprints = set()
            
            # Filter out any None or empty fingerprints
            reference_fingerprints = {fp for fp in reference_fingerprints if fp and isinstance(fp, str) and fp.strip()}
            
            if not reference_fingerprints:
                logger.warning("No valid reference fingerprints found - all structures will be considered novel")
            else:
                logger.info(f"Loaded {len(reference_fingerprints):,} valid reference fingerprints")
            
            self._reference_fingerprints = reference_fingerprints
            self._reference_loaded = True
            
            return reference_fingerprints
            
        except Exception as e:
            logger.error(f"Failed to load reference fingerprints: {e}")
            logger.warning("Using empty reference set - all structures will be considered novel")
            return set()

    def _get_structure_fingerprint(self, structure: Structure) -> Optional[str]:
        """Get augmented fingerprint for a structure.

        Parameters
        ----------
        structure : Structure
            Structure to fingerprint.

        Returns
        -------
        str or None
            Augmented fingerprint string, or None if computation failed.
        """
        fingerprint = None
        
        # Try to get from properties first if requested
        if self.config.fingerprint_source in ["property", "auto"]:
            fingerprint = structure.properties.get("augmented_fingerprint")
            
            if fingerprint:
                logger.debug("Using fingerprint from structure properties")
                return str(fingerprint)
        
        # Compute fingerprint if needed and allowed
        if (fingerprint is None and 
            self.config.fingerprint_source in ["compute", "auto"] and
            self.config.fallback_to_computation):
            
            try:
                logger.debug("Computing augmented fingerprint")
                fingerprint = get_augmented_fingerprint(
                    structure,
                    symprec=self.config.symprec,
                    angle_tolerance=self.config.angle_tolerance
                )
                
                if fingerprint:
                    return str(fingerprint)
                    
            except Exception as e:
                logger.warning(f"Failed to compute augmented fingerprint: {e}")
        
        return None

    def _get_compute_attributes(self) -> Dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        # Load reference fingerprints once
        reference_fingerprints = self._load_reference_fingerprints()

        return {
            "reference_fingerprints": reference_fingerprints,
            "fingerprint_source": self.config.fingerprint_source,
            "symprec": self.config.symprec,
            "angle_tolerance": self.config.angle_tolerance,
            "fallback_to_computation": self.config.fallback_to_computation,
        }

    @staticmethod
    def compute_structure(
        structure: Structure,
        reference_fingerprints: Set[str],
        fingerprint_source: str,
        symprec: float,
        angle_tolerance: float,
        fallback_to_computation: bool,
    ) -> float:
        """Check if a structure is novel using augmented fingerprinting.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to evaluate.
        reference_fingerprints : Set[str]
            Set of reference fingerprints.
        fingerprint_source : str
            Source of fingerprints.
        symprec : float
            Symmetry precision for computation.
        angle_tolerance : float
            Angle tolerance for computation.
        fallback_to_computation : bool
            Whether to compute if not found in properties.

        Returns
        -------
        float
            1.0 if the structure is novel (not in reference), 0.0 otherwise.
        """
        try:
            fingerprint = None
            
            # Try to get from properties first if requested
            if fingerprint_source in ["property", "auto"]:
                fingerprint = structure.properties.get("augmented_fingerprint")
            
            # Compute fingerprint if needed and allowed
            if (fingerprint is None and 
                fingerprint_source in ["compute", "auto"] and
                fallback_to_computation):
                
                try:
                    fingerprint = get_augmented_fingerprint(
                        structure,
                        symprec=symprec,
                        angle_tolerance=angle_tolerance
                    )
                except Exception as e:
                    logger.warning(f"Failed to compute augmented fingerprint: {e}")
                    return float("nan")
            
            if fingerprint is None:
                logger.warning("Could not obtain augmented fingerprint for structure")
                return float("nan")
            
            # Convert to string if needed
            fingerprint_str = str(fingerprint)
            
            # Check if fingerprint is in reference set
            is_novel = fingerprint_str not in reference_fingerprints
            
            return 1.0 if is_novel else 0.0

        except Exception as e:
            logger.warning(f"Error computing novelty: {e}")
            return float("nan")

    def aggregate_results(self, values: List[float]) -> Dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            Novelty values for each structure (1.0 for novel, 0.0 for known).

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        # Filter out NaN values
        valid_values = [v for v in values if not np.isnan(v)]

        if not valid_values:
            total_attempted = len(values)
            return {
                "metrics": {
                    "novelty_score": float("nan"),
                    "novel_structures_count": 0,
                    "total_structures_evaluated": 0,
                    "total_structures_attempted": total_attempted,
                    "fingerprinting_success_rate": 0.0,
                },
                "primary_metric": "novelty_score",
                "uncertainties": {},
            }

        # Calculate novelty metrics
        novel_count = sum(valid_values)
        total_count = len(valid_values)
        total_attempted = len(values)
        novelty_score = novel_count / total_count if total_count > 0 else 0.0
        success_rate = total_count / total_attempted if total_attempted > 0 else 0.0

        return {
            "metrics": {
                "novelty_score": novelty_score,
                "novel_structures_count": int(novel_count),
                "total_structures_evaluated": total_count,
                "total_structures_attempted": total_attempted,
                "fingerprinting_success_rate": success_rate,
            },
            "primary_metric": "novelty_score",
            "uncertainties": {
                "novelty_score": {
                    "std": np.std(valid_values) if len(valid_values) > 1 else 0.0
                }
            },
        }


# Factory functions for common configurations
def create_augmented_novelty_metric(
    reference_fingerprints_path: Optional[str] = None,
    fingerprint_source: str = "auto",
    **kwargs
) -> AugmentedNoveltyMetric:
    """Factory function to create augmented novelty metric with common configurations.

    Parameters
    ----------
    reference_fingerprints_path : str or None, default=None
        Path to reference fingerprints file.
    fingerprint_source : str, default="auto"
        Source of fingerprints.
    **kwargs
        Additional arguments for the metric.

    Returns
    -------
    AugmentedNoveltyMetric
        Configured augmented novelty metric.
    """
    return AugmentedNoveltyMetric(
        reference_fingerprints_path=reference_fingerprints_path,
        fingerprint_source=fingerprint_source,
        **kwargs,
    )


def create_property_based_novelty_metric(**kwargs) -> AugmentedNoveltyMetric:
    """Create novelty metric that only uses preprocessed fingerprints from properties."""
    return create_augmented_novelty_metric(
        fingerprint_source="property",
        fallback_to_computation=False,
        **kwargs
    )


def create_computation_based_novelty_metric(**kwargs) -> AugmentedNoveltyMetric:
    """Create novelty metric that computes fingerprints on-demand."""
    return create_augmented_novelty_metric(
        fingerprint_source="compute",
        **kwargs
    )


def create_robust_novelty_metric(**kwargs) -> AugmentedNoveltyMetric:
    """Create novelty metric with robust settings for most use cases."""
    return create_augmented_novelty_metric(
        fingerprint_source="auto",
        symprec=0.1,
        angle_tolerance=10.0,
        fallback_to_computation=True,
        **kwargs
    )