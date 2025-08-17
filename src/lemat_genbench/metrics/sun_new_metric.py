"""New SUN (Stable, Unique, Novel) metrics using augmented fingerprinting for material structures.

This module implements SUN and MetaSUN metrics that measure the proportion
of generated structures that are simultaneously stable (or metastable),
unique, and novel using the new augmented fingerprinting approach.
"""

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from pymatgen.core.structure import Structure

from lemat_genbench.metrics.base import BaseMetric, MetricConfig, MetricResult
from lemat_genbench.metrics.novelty_new_metric import AugmentedNoveltyMetric
from lemat_genbench.metrics.uniqueness_new_metric import UniquenessNewMetric
from lemat_genbench.utils.logging import logger

warnings.filterwarnings("ignore", message="No oxidation states specified on sites!")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=r".*__array__.*copy.*"
)


@dataclass
class SUNNewConfig(MetricConfig):
    """Configuration for the new SUN metric using augmented fingerprinting.

    Parameters
    ----------
    stability_threshold : float, default=0.0
        Energy above hull threshold for stability (eV/atom).
        0.0 for SUN (stable), higher values for MetaSUN (metastable).
    metastability_threshold : float, default=0.1
        Energy above hull threshold for metastability (eV/atom).
        Used for MetaSUN calculations.
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
    """

    stability_threshold: float = 0.0
    metastability_threshold: float = 0.1
    reference_fingerprints_path: Optional[str] = None
    reference_dataset_name: str = "LeMat-Bulk"
    fingerprint_source: str = "auto"
    symprec: float = 0.01
    angle_tolerance: float = 5.0
    fallback_to_computation: bool = True


class SUNNewMetric(BaseMetric):
    """Evaluate SUN (Stable, Unique, Novel) rate using augmented fingerprinting.

    This metric computes the proportion of structures that are simultaneously:
    1. Stable (e_above_hull <= stability_threshold)
    2. Unique (not duplicated within the generated set using augmented fingerprints)
    3. Novel (not present in reference dataset using augmented fingerprints)

    The SUN rate is defined as:
    SUN = |{x ∈ G | E_hull(x) ≤ 0, x ∉ T, x is unique}| / |G|

    Parameters
    ----------
    stability_threshold : float, default=0.0
        Energy above hull threshold for stability.
    metastability_threshold : float, default=0.1
        Energy above hull threshold for metastability (for MetaSUN).
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
        Higher SUN rates indicate better performance.
    n_jobs : int, default=1
        Number of parallel jobs to run.
    """

    def __init__(
        self,
        stability_threshold: float = 0.0,
        metastability_threshold: float = 0.1,
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
            name=name or "SUNNew",
            description=description
            or "Measures proportion of structures that are Stable, Unique, and Novel using augmented fingerprinting",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )

        self.config = SUNNewConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            stability_threshold=stability_threshold,
            metastability_threshold=metastability_threshold,
            reference_fingerprints_path=reference_fingerprints_path,
            reference_dataset_name=reference_dataset_name,
            fingerprint_source=fingerprint_source,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            fallback_to_computation=fallback_to_computation,
        )

        # Initialize sub-metrics using new augmented fingerprinting
        self.uniqueness_metric = UniquenessNewMetric(
            fingerprint_source=fingerprint_source,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            n_jobs=1,  # We'll handle parallelization at this level
        )

        self.novelty_metric = AugmentedNoveltyMetric(
            reference_fingerprints_path=reference_fingerprints_path,
            reference_dataset_name=reference_dataset_name,
            fingerprint_source=fingerprint_source,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            fallback_to_computation=fallback_to_computation,
            n_jobs=1,  # We'll handle parallelization at this level
        )

    def compute(self, structures: list[Structure]) -> MetricResult:
        """Compute the SUN metric on a batch of structures.

        This method efficiently computes SUN by:
        1. First identifying unique structures within the generated set
        2. Then checking novelty for only the unique structures
        3. Finally checking stability for structures that are both unique and novel

        Parameters
        ----------
        structures : list[Structure]
            List of pymatgen Structure objects to evaluate.

        Returns
        -------
        MetricResult
            Object containing the SUN metrics and computation metadata.
        """
        start_time = time.time()
        n_structures = len(structures)

        if n_structures == 0:
            return self._empty_result(start_time)

        try:
            # Step 1: Compute uniqueness for all structures
            logger.info("Computing uniqueness using augmented fingerprints...")
            uniqueness_result = self.uniqueness_metric.compute(structures)

            # Get the unique fingerprints to calculate proper unique count
            if hasattr(uniqueness_result, 'fingerprints') and uniqueness_result.fingerprints:
                unique_fingerprints = set(uniqueness_result.fingerprints)
                unique_count = len(unique_fingerprints)
            else:
                # Fallback: calculate from metrics if fingerprints not available
                unique_count = uniqueness_result.metrics.get("unique_structures_count", 0)

            # Select unique structures for further processing:
            # One representative from each unique fingerprint group
            unique_indices = self._get_unique_structure_indices(
                uniqueness_result, uniqueness_result.failed_indices
            )

            if not unique_indices:
                logger.info("No unique structures available for processing")
                return self._create_result(
                    start_time,
                    n_structures,
                    [],
                    [],
                    unique_count,
                    uniqueness_result.failed_indices,
                )

            logger.info(f"Found {len(unique_indices)} unique structures for processing")

            # Step 2: Check novelty for unique structures only
            logger.info("Computing novelty for unique structures using augmented fingerprints...")
            unique_structures = [structures[i] for i in unique_indices]
            novelty_result = self.novelty_metric.compute(unique_structures)

            # Identify which unique structures are novel 
            # (individual_values = 1.0 means novel)
            # FIXED: Properly handle index mapping between novelty results and 
            # original structure indices
            novel_among_unique_indices = []
            
            # The novelty_result.individual_values corresponds 1:1 with the 
            # unique_structures we passed
            # So novelty_result.individual_values[i] corresponds to 
            # unique_indices[i]
            for i, val in enumerate(novelty_result.individual_values):
                # Check if this novelty computation was successful and structure 
                # is novel
                if (i not in novelty_result.failed_indices and 
                    not np.isnan(val) and 
                    val == 1.0):
                    # Map back to original structure index
                    original_structure_idx = unique_indices[i]
                    novel_among_unique_indices.append(original_structure_idx)

            if not novel_among_unique_indices:
                logger.info("No novel structures among unique structures")
                return self._create_result(
                    start_time,
                    n_structures,
                    [],
                    [],
                    unique_count,
                    uniqueness_result.failed_indices + [
                        unique_indices[i] for i in novelty_result.failed_indices
                    ],
                )
            logger.info(
                f"Found {len(novel_among_unique_indices)} novel structures among unique"
            )

            # Step 3: Check stability for structures that are both unique and novel
            logger.info("Computing stability for unique and novel structures...")
            sun_indices, msun_indices = self._compute_stability(
                structures, novel_among_unique_indices
            )

            logger.info(
                f"Found {len(sun_indices)} SUN and {len(msun_indices)} MetaSUN structures"
            )

            # Combine all failed indices
            all_failed_indices = list(
                set(uniqueness_result.failed_indices + [
                    unique_indices[i] for i in novelty_result.failed_indices
                ])
            )

            return self._create_result(
                start_time,
                n_structures,
                sun_indices,
                msun_indices,
                unique_count,  # Correct unique count from fingerprints
                all_failed_indices,
            )

        except Exception as e:
            logger.error("Failed to compute SUN metric", exc_info=True)
            return MetricResult(
                metrics={self.name: float("nan")},
                primary_metric=self.name,
                uncertainties={},
                config=self.config,
                computation_time=time.time() - start_time,
                n_structures=n_structures,
                individual_values=[float("nan")] * n_structures,
                failed_indices=list(range(n_structures)),
                warnings=[f"Global computation failure: {str(e)}"] * n_structures,
            )

    def _get_unique_structure_indices(
        self, 
        uniqueness_result,
        failed_indices: List[int]
    ) -> List[int]:
        """Get indices of representative structures from each unique fingerprint group.

        Selects one representative from each unique fingerprint group by working
        directly with fingerprints to avoid collisions from identical individual values.

        Parameters
        ----------
        uniqueness_result : MetricResult
            Result from uniqueness metric containing fingerprints and individual values.
        failed_indices : List[int]
            Indices of structures that failed uniqueness computation.

        Returns
        -------
        List[int]
            Indices of representative structures from each unique fingerprint group.
        """
        selected_indices = []
        
        # Check if fingerprints are available
        if hasattr(uniqueness_result, 'fingerprints') and uniqueness_result.fingerprints:
            seen_fingerprints = set()
            fingerprint_idx = 0
            
            for struct_idx in range(len(uniqueness_result.individual_values)):
                if struct_idx not in failed_indices:
                    fingerprint = uniqueness_result.fingerprints[fingerprint_idx]
                    if fingerprint not in seen_fingerprints:
                        # First occurrence of this fingerprint - select as representative
                        selected_indices.append(struct_idx)
                        seen_fingerprints.add(fingerprint)
                    fingerprint_idx += 1
        else:
            # Fallback: use individual values (less accurate but better than nothing)
            logger.warning("Fingerprints not available, using individual values fallback")
            seen_values = set()
            
            for i, val in enumerate(uniqueness_result.individual_values):
                if i not in failed_indices and not np.isnan(val):
                    if val == 1.0:
                        # Unique structure - always select
                        selected_indices.append(i)
                    elif val not in seen_values:
                        # First occurrence of this duplicate group - select as representative
                        selected_indices.append(i)
                        seen_values.add(val)
        
        return sorted(selected_indices)

    def _compute_stability(
        self, structures: list[Structure], candidate_indices: List[int]
    ) -> tuple[List[int], List[int]]:
        """Compute stability for candidate structures.

        Parameters
        ----------
        structures : list[Structure]
            All structures.
        candidate_indices : List[int]
            Indices of structures to check for stability.

        Returns
        -------
        tuple[List[int], List[int]]
            Tuple of (SUN indices, MetaSUN indices).
        """
        sun_indices = []
        msun_indices = []

        for idx in candidate_indices:
            structure = structures[idx]

            # Extract e_above_hull from structure properties
            e_above_hull = structure.properties.get("e_above_hull_mean", None)
            if e_above_hull is None:
                e_above_hull = structure.properties.get("e_above_hull", None)

            if e_above_hull is None:
                logger.warning(
                    f"Structure {idx} missing e_above_hull in properties, "
                    "please compute it first using StabilityPreprocessor"
                )
                continue

            try:
                e_above_hull_val = float(e_above_hull)

                # Check if stable (SUN)
                if e_above_hull_val <= self.config.stability_threshold:
                    sun_indices.append(idx)
                # Check if metastable (MetaSUN) - note: stable structures are NOT included in MetaSUN
                elif e_above_hull_val <= self.config.metastability_threshold:
                    msun_indices.append(idx)

            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid e_above_hull value for structure {idx}: {e}")
                continue

        return sun_indices, msun_indices

    def _create_result(
        self,
        start_time: float,
        n_structures: int,
        sun_indices: List[int],
        msun_indices: List[int],
        unique_count: int,  # Now properly calculated from fingerprints
        failed_indices: List[int],
    ) -> MetricResult:
        """Create MetricResult from computed indices."""
        # Calculate rates
        sun_rate = len(sun_indices) / n_structures if n_structures > 0 else 0.0
        msun_rate = len(msun_indices) / n_structures if n_structures > 0 else 0.0

        # Create individual values
        individual_values = np.zeros(n_structures)

        # Assign values: 1.0 for SUN structures, 0.5 for MetaSUN structures, 0.0 for others
        for idx in sun_indices:
            individual_values[idx] = 1.0
        for idx in msun_indices:
            individual_values[idx] = 0.5

        # Set failed structures to NaN
        for idx in failed_indices:
            if idx < n_structures:
                individual_values[idx] = float("nan")

        # Calculate additional metrics
        unique_rate = unique_count / n_structures if n_structures > 0 else 0.0
        total_sun_msun = len(sun_indices) + len(msun_indices)
        combined_rate = total_sun_msun / n_structures if n_structures > 0 else 0.0

        metrics = {
            "sun_rate": sun_rate,
            "msun_rate": msun_rate,
            "combined_sun_msun_rate": combined_rate,
            "sun_count": len(sun_indices),
            "msun_count": len(msun_indices),
            "unique_count": unique_count,  # Now correctly calculated from unique fingerprints
            "unique_rate": unique_rate,
            "total_structures_evaluated": n_structures,
            "failed_count": len(failed_indices),
        }

        return MetricResult(
            metrics=metrics,
            primary_metric="sun_rate",
            uncertainties={
                "sun_rate": {"std": 0.0},  # Deterministic given inputs
                "msun_rate": {"std": 0.0},
            },
            config=self.config,
            computation_time=time.time() - start_time,
            n_structures=n_structures,
            individual_values=individual_values.tolist(),
            failed_indices=failed_indices,
            warnings=[],
        )

    def _empty_result(self, start_time: float) -> MetricResult:
        """Create result for empty structure list."""
        return MetricResult(
            metrics={
                "sun_rate": float("nan"),
                "msun_rate": float("nan"),
                "combined_sun_msun_rate": float("nan"),
                "sun_count": 0,
                "msun_count": 0,
                "unique_count": 0,
                "unique_rate": float("nan"),
                "total_structures_evaluated": 0,
                "failed_count": 0,
            },
            primary_metric="sun_rate",
            uncertainties={},
            config=self.config,
            computation_time=time.time() - start_time,
            n_structures=0,
            individual_values=[],
            failed_indices=[],
            warnings=[],
        )

    @staticmethod
    def compute_structure(structure: Structure, **compute_args: Any) -> float:
        """Compute metric for a single structure.

        This method is required by the base class but not used directly
        for SUN calculation since we need to evaluate all structures together.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to evaluate.
        **compute_args : Any
            Additional keyword arguments.

        Returns
        -------
        float
            Always returns 0.0 as this method is not used directly.
        """
        return 0.0

    def aggregate_results(self, values: List[float]) -> Dict[str, Any]:
        """Aggregate results into final metric values.

        This method is required by the base class but not used directly
        for SUN calculation since we override the compute method.

        Parameters
        ----------
        values : list[float]
            Individual values (not used directly).

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        return {
            "metrics": {"sun_rate": 0.0},
            "primary_metric": "sun_rate",
            "uncertainties": {},
        }


class MetaSUNNewMetric(SUNNewMetric):
    """Evaluate MetaSUN (Metastable, Unique, Novel) rate using augmented fingerprinting.

    This is a convenience class that extends SUNNewMetric with different default
    thresholds optimized for metastability evaluation.
    """

    def __init__(self, metastability_threshold: float = 0.1, **kwargs):
        # Set stability_threshold to metastability_threshold for primary metric
        kwargs.setdefault("stability_threshold", metastability_threshold)
        kwargs.setdefault("name", "MetaSUNNew")
        kwargs.setdefault(
            "description",
            "Measures proportion of structures that are Metastable, Unique, and Novel using augmented fingerprinting",
        )

        super().__init__(metastability_threshold=metastability_threshold, **kwargs)

    @property
    def primary_threshold(self) -> float:
        """Get the primary threshold for this metric."""
        return self.config.metastability_threshold


# Factory functions for common configurations
def create_sun_new_metric(
    reference_fingerprints_path: Optional[str] = None,
    fingerprint_source: str = "auto",
    **kwargs
) -> SUNNewMetric:
    """Factory function to create SUN metric with new augmented fingerprinting.

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
    SUNNewMetric
        Configured SUN metric with augmented fingerprinting.
    """
    return SUNNewMetric(
        reference_fingerprints_path=reference_fingerprints_path,
        fingerprint_source=fingerprint_source,
        **kwargs,
    )


def create_property_based_sun_metric(**kwargs) -> SUNNewMetric:
    """Create SUN metric that only uses preprocessed fingerprints from properties."""
    return create_sun_new_metric(
        fingerprint_source="property",
        fallback_to_computation=False,
        **kwargs
    )


def create_computation_based_sun_metric(**kwargs) -> SUNNewMetric:
    """Create SUN metric that computes fingerprints on-demand."""
    return create_sun_new_metric(
        fingerprint_source="compute",
        **kwargs
    )


def create_robust_sun_metric(**kwargs) -> SUNNewMetric:
    """Create SUN metric with robust settings for most use cases."""
    return create_sun_new_metric(
        fingerprint_source="auto",
        symprec=0.1,
        angle_tolerance=10.0,
        fallback_to_computation=True,
        **kwargs
    )