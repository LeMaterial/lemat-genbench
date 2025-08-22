"""SUN (Stable, Unique, Novel) metrics for evaluating material structures.

This module implements SUN and MetaSUN metrics that measure the proportion
of generated structures that are simultaneously stable (or metastable),
unique, and novel. Updated to support the new computation order:
Stability → Uniqueness → Novelty, and structure matcher support.
"""

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from pymatgen.core.structure import Structure

from lemat_genbench.metrics.base import BaseMetric, MetricConfig, MetricResult
from lemat_genbench.metrics.novelty_metric import NoveltyMetric
from lemat_genbench.metrics.uniqueness_metric import UniquenessMetric
from lemat_genbench.utils.logging import logger

warnings.filterwarnings("ignore", message="No oxidation states specified on sites!")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=r".*__array__.*copy.*"
)


@dataclass
class SUNConfig(MetricConfig):
    """Configuration for the SUN metric.

    Parameters
    ----------
    stability_threshold : float, default=0.0
        Energy above hull threshold for stability (eV/atom).
        0.0 for SUN (stable), higher values for MetaSUN (metastable).
    metastability_threshold : float, default=0.1
        Energy above hull threshold for metastability (eV/atom).
        Used for MetaSUN calculations.
    reference_dataset : str, default="LeMaterial/LeMat-Bulk"
        HuggingFace dataset name to use as reference for novelty.
    reference_config : str, default="compatible_pbe"
        Configuration/subset of the reference dataset to use.
    fingerprint_method : str, default="bawl"
        Method to use for structure fingerprinting.
        Supports: "bawl", "short-bawl", "structure-matcher", "pdd"
    cache_reference : bool, default=True
        Whether to cache the reference dataset fingerprints.
    max_reference_size : int | None, default=None
        Maximum number of structures to load from reference dataset.
    """

    stability_threshold: float = 0.0
    metastability_threshold: float = 0.1
    reference_dataset: str = "LeMaterial/LeMat-Bulk"
    reference_config: str = "compatible_pbe"
    fingerprint_method: str = "bawl"
    cache_reference: bool = True
    max_reference_size: Optional[int] = None


class SUNMetric(BaseMetric):
    """Evaluate SUN (Stable, Unique, Novel) rate of structures.

    This metric computes the proportion of structures that are simultaneously:
    1. Stable (e_above_hull <= stability_threshold) [computed first]
    2. Unique (not duplicated within the stable set) [computed second]
    3. Novel (not present in reference dataset) [computed third]

    **NEW COMPUTATION ORDER**: Stability → Uniqueness → Novelty
    
    Key changes from the original implementation:
    - **Order changed**: Now computes stability first, then uniqueness within stable/metastable sets,
      then novelty within unique stable/metastable structures
    - **Structure matcher support**: Handles both fingerprinting and structure matcher methods
    - **Hierarchical reporting**: Reports counts at each stage of the hierarchy
    - **Separate tracking**: Tracks stable and metastable structures separately through the pipeline

    The SUN rate is defined as:
    SUN = |{x ∈ S_unique | x ∉ T}| / |G|

    where:
    - G is the complete set of generated structures
    - S is the subset of stable structures from G
    - S_unique is the subset of unique structures within S
    - T is the reference dataset

    Output metrics include hierarchical counts:
    - Level 1: stable_count, metastable_count (from all structures)
    - Level 2: unique_in_stable_count, unique_in_metastable_count (from stable/metastable)  
    - Level 3: sun_count, msun_count (from unique stable/metastable structures)

    Parameters
    ----------
    stability_threshold : float, default=0.0
        Energy above hull threshold for stability.
    metastability_threshold : float, default=0.1
        Energy above hull threshold for metastability (for MetaSUN).
    reference_dataset : str, default="LeMaterial/LeMat-Bulk"
        HuggingFace dataset name to use as reference.
    reference_config : str, default="compatible_pbe"
        Configuration/subset of the reference dataset to use.
    fingerprint_method : str, default="bawl"
        Method to use for structure fingerprinting/comparison.
        Supports: "bawl", "short-bawl", "structure-matcher", "pdd"
    cache_reference : bool, default=True
        Whether to cache the reference dataset fingerprints.
    max_reference_size : int | None, default=None
        Maximum number of structures to load from reference dataset.
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
        reference_dataset: str = "LeMaterial/LeMat-Bulk",
        reference_config: str = "compatible_pbe",
        fingerprint_method: str = "bawl",
        cache_reference: bool = True,
        max_reference_size: Optional[int] = None,
        name: str | None = None,
        description: str | None = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        super().__init__(
            name=name or "SUN",
            description=description
            or "Measures proportion of structures that are Stable, Unique, and Novel",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self.config = SUNConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            stability_threshold=stability_threshold,
            metastability_threshold=metastability_threshold,
            reference_dataset=reference_dataset,
            reference_config=reference_config,
            fingerprint_method=fingerprint_method,
            cache_reference=cache_reference,
            max_reference_size=max_reference_size,
        )

        # Initialize sub-metrics
        self.uniqueness_metric = UniquenessMetric(
            fingerprint_method=fingerprint_method,
            n_jobs=1,  # We'll handle parallelization at this level
        )

        self.novelty_metric = NoveltyMetric(
            reference_dataset=reference_dataset,
            reference_config=reference_config,
            fingerprint_method=fingerprint_method,
            cache_reference=cache_reference,
            max_reference_size=max_reference_size,
            n_jobs=1,  # We'll handle parallelization at this level
            verbose=verbose,
        )

    def compute(self, structures: list[Structure]) -> MetricResult:
        """Compute the SUN metric on a batch of structures.

        New computation order: Stability → Uniqueness → Novelty

        This method efficiently computes SUN by:
        1. First identifying stable/metastable structures
        2. Then checking uniqueness within the stable/metastable sets
        3. Finally checking novelty for unique structures from stable/metastable sets

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
            # Step 1: Identify stable and metastable structures
            logger.info("Computing stability for all structures...")
            stable_indices, metastable_indices = self._compute_stability_all(structures)
            
            stable_count = len(stable_indices)
            metastable_count = len(metastable_indices)
            
            logger.info(
                f"Found {stable_count} stable and {metastable_count} metastable structures"
            )

            if not stable_indices and not metastable_indices:
                logger.info("No stable or metastable structures found")
                return self._create_result(
                    start_time,
                    n_structures,
                    [],
                    [],
                    stable_count,
                    metastable_count,
                    0,  # unique_in_stable_count
                    0,  # unique_in_metastable_count
                    [],
                    stable_indices,
                    metastable_indices,
                    [],
                    [],
                )

            # Step 2: Check uniqueness within stable structures
            stable_unique_indices = []
            if stable_indices:
                logger.info(f"Computing uniqueness within {len(stable_indices)} stable structures...")
                stable_structures = [structures[i] for i in stable_indices]
                stable_unique_relative_indices = self._compute_uniqueness_within_set(
                    stable_structures
                )
                # Map back to original indices
                stable_unique_indices = [
                    stable_indices[i] for i in stable_unique_relative_indices
                ]
                logger.info(
                    f"Found {len(stable_unique_indices)} unique structures among stable ones"
                )

            # Step 3: Check uniqueness within metastable structures
            metastable_unique_indices = []
            if metastable_indices:
                logger.info(f"Computing uniqueness within {len(metastable_indices)} metastable structures...")
                metastable_structures = [structures[i] for i in metastable_indices]
                metastable_unique_relative_indices = self._compute_uniqueness_within_set(
                    metastable_structures
                )
                # Map back to original indices
                metastable_unique_indices = [
                    metastable_indices[i] for i in metastable_unique_relative_indices
                ]
                logger.info(
                    f"Found {len(metastable_unique_indices)} unique structures among metastable ones"
                )

            # Step 4: Check novelty for unique stable structures
            sun_indices = []
            if stable_unique_indices:
                logger.info(f"Computing novelty for {len(stable_unique_indices)} unique stable structures...")
                stable_unique_structures = [structures[i] for i in stable_unique_indices]
                stable_novel_relative_indices = self._compute_novelty_within_set(
                    stable_unique_structures
                )
                # Map back to original indices
                sun_indices = [
                    stable_unique_indices[i] for i in stable_novel_relative_indices
                ]
                logger.info(f"Found {len(sun_indices)} SUN structures")

            # Step 5: Check novelty for unique metastable structures  
            msun_indices = []
            if metastable_unique_indices:
                logger.info(f"Computing novelty for {len(metastable_unique_indices)} unique metastable structures...")
                metastable_unique_structures = [structures[i] for i in metastable_unique_indices]
                metastable_novel_relative_indices = self._compute_novelty_within_set(
                    metastable_unique_structures
                )
                # Map back to original indices
                msun_indices = [
                    metastable_unique_indices[i] for i in metastable_novel_relative_indices
                ]
                logger.info(f"Found {len(msun_indices)} MetaSUN structures")

            return self._create_result(
                start_time,
                n_structures,
                sun_indices,
                msun_indices,
                stable_count,
                metastable_count,
                len(stable_unique_indices),
                len(metastable_unique_indices),
                [],  # No failed indices for this implementation
                stable_indices,
                metastable_indices,
                stable_unique_indices,
                metastable_unique_indices,
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

    def _compute_stability_all(
        self, structures: list[Structure]
    ) -> tuple[List[int], List[int]]:
        """Compute stability for all structures.

        Parameters
        ----------
        structures : list[Structure]
            All structures.

        Returns
        -------
        tuple[List[int], List[int]]
            Tuple of (stable indices, metastable indices).
        """
        stable_indices = []
        metastable_indices = []

        for idx, structure in enumerate(structures):
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
                    stable_indices.append(idx)
                # Check if metastable (MetaSUN) - note: stable structures are NOT included in MetaSUN
                elif e_above_hull_val <= self.config.metastability_threshold:
                    metastable_indices.append(idx)

            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid e_above_hull value for structure {idx}: {e}")
                continue

        return stable_indices, metastable_indices

    def _compute_uniqueness_within_set(
        self, structures: list[Structure]
    ) -> List[int]:
        """Compute uniqueness within a set of structures.

        Parameters
        ----------
        structures : list[Structure]
            Structures to check for uniqueness.

        Returns
        -------
        List[int]
            Indices (relative to input set) of unique structures.
        """
        if not structures:
            return []

        # Handle structure matcher method differently since it requires pairwise comparison
        if "structure-matcher" in self.config.fingerprint_method.lower():
            fingerprinter = self.uniqueness_metric.fingerprinter
            unique_indices = []
            
            for i, structure1 in enumerate(structures):
                is_unique = True
                # Check against all previous structures to avoid duplicating earlier ones
                for j in range(i):
                    structure2 = structures[j]
                    try:
                        is_equivalent = fingerprinter.is_equivalent(structure1, structure2)
                        if is_equivalent:
                            is_unique = False
                            break
                    except Exception as e:
                        logger.warning(
                            f"Failed to compare structures {i} and {j}: {str(e)}"
                        )
                        # Continue to next comparison on error
                        continue
                
                if is_unique:
                    unique_indices.append(i)
            
            return unique_indices
        
        # For fingerprint-based methods, use the uniqueness metric
        uniqueness_result = self.uniqueness_metric.compute(structures)

        # Get the unique structure representatives
        if hasattr(uniqueness_result, 'fingerprints') and uniqueness_result.fingerprints:
            # Use fingerprint-based approach
            seen_fingerprints = set()
            unique_indices = []
            fingerprint_idx = 0
            
            for struct_idx in range(len(structures)):
                if struct_idx not in uniqueness_result.failed_indices:
                    fingerprint = uniqueness_result.fingerprints[fingerprint_idx]
                    if fingerprint not in seen_fingerprints:
                        unique_indices.append(struct_idx)
                        seen_fingerprints.add(fingerprint)
                    fingerprint_idx += 1
            
            return unique_indices
        else:
            # Fallback: use individual values
            logger.warning("Using individual values fallback for uniqueness computation")
            seen_values = set()
            unique_indices = []
            
            for i, val in enumerate(uniqueness_result.individual_values):
                if i not in uniqueness_result.failed_indices and not np.isnan(val):
                    if val == 1.0:
                        # Unique structure
                        unique_indices.append(i)
                    elif val not in seen_values:
                        # First occurrence of this duplicate group
                        unique_indices.append(i)
                        seen_values.add(val)

            return unique_indices

    def _compute_novelty_within_set(
        self, structures: list[Structure]
    ) -> List[int]:
        """Compute novelty within a set of structures.

        Parameters
        ----------
        structures : list[Structure]
            Structures to check for novelty.

        Returns
        -------
        List[int]
            Indices (relative to input set) of novel structures.
        """
        if not structures:
            return []

        # Use the novelty metric to find novel structures
        novelty_result = self.novelty_metric.compute(structures)

        # For structure matcher, individual_values may not be 1.0/0.0
        # but the pattern should still be that novel structures have higher values
        if "structure-matcher" in self.config.fingerprint_method.lower():
            # For structure matcher, we interpret non-zero values as novel
            novel_indices = []
            for i, val in enumerate(novelty_result.individual_values):
                if (i not in novelty_result.failed_indices and 
                    not np.isnan(val) and 
                    val > 0.0):  # Novel if positive value
                    novel_indices.append(i)
            return novel_indices
        else:
            # For fingerprint-based methods, novel structures have value 1.0
            novel_indices = []
            for i, val in enumerate(novelty_result.individual_values):
                # Check if this novelty computation was successful and structure is novel
                if (i not in novelty_result.failed_indices and 
                    not np.isnan(val) and 
                    val == 1.0):
                    novel_indices.append(i)
            return novel_indices

    def _create_result(
        self,
        start_time: float,
        n_structures: int,
        sun_indices: List[int],
        msun_indices: List[int],
        stable_count: int,
        metastable_count: int,
        unique_in_stable_count: int,
        unique_in_metastable_count: int,
        failed_indices: List[int],
        stable_indices: List[int] = None,
        metastable_indices: List[int] = None,
        stable_unique_indices: List[int] = None,
        metastable_unique_indices: List[int] = None,
    ) -> MetricResult:
        """Create MetricResult from computed indices and counts.
        
        The resulting MetricResult will have the following custom attributes added:
        - sun_indices: List of indices for structures that are SUN (stable, unique, novel)
        - msun_indices: List of indices for structures that are MetaSUN (metastable, unique, novel)
        - stable_indices: List of indices for all stable structures
        - metastable_indices: List of indices for all metastable structures  
        - stable_unique_indices: List of indices for unique structures within stable set
        - metastable_unique_indices: List of indices for unique structures within metastable set
        """
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

        # Calculate additional rates
        stable_rate = stable_count / n_structures if n_structures > 0 else 0.0
        metastable_rate = metastable_count / n_structures if n_structures > 0 else 0.0
        unique_in_stable_rate = (
            unique_in_stable_count / stable_count if stable_count > 0 else 0.0
        )
        unique_in_metastable_rate = (
            unique_in_metastable_count / metastable_count if metastable_count > 0 else 0.0
        )
        
        total_sun_msun = len(sun_indices) + len(msun_indices)
        combined_rate = total_sun_msun / n_structures if n_structures > 0 else 0.0

        metrics = {
            # Primary SUN metrics
            "sun_rate": sun_rate,
            "msun_rate": msun_rate,
            "combined_sun_msun_rate": combined_rate,
            
            # Hierarchical counts and rates - Level 1: Stability
            "total_structures_evaluated": n_structures,
            "stable_count": stable_count,
            "metastable_count": metastable_count,
            "stable_rate": stable_rate,
            "metastable_rate": metastable_rate,
            
            # Hierarchical counts and rates - Level 2: Uniqueness within stable/metastable
            "unique_in_stable_count": unique_in_stable_count,
            "unique_in_metastable_count": unique_in_metastable_count,
            "unique_in_stable_rate": unique_in_stable_rate,
            "unique_in_metastable_rate": unique_in_metastable_rate,
            
            # Hierarchical counts - Level 3: Novelty (final SUN/MetaSUN)
            "sun_count": len(sun_indices),
            "msun_count": len(msun_indices),
            
            # Error tracking
            "failed_count": len(failed_indices),
        }

        result = MetricResult(
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
        
        # Add hierarchical indices as custom attributes for debugging and analysis
        result.sun_indices = sun_indices
        result.msun_indices = msun_indices
        result.stable_indices = stable_indices or []
        result.metastable_indices = metastable_indices or []
        result.stable_unique_indices = stable_unique_indices or []
        result.metastable_unique_indices = metastable_unique_indices or []
        
        return result

    def _empty_result(self, start_time: float) -> MetricResult:
        """Create result for empty structure list."""
        return MetricResult(
            metrics={
                "sun_rate": float("nan"),
                "msun_rate": float("nan"),
                "combined_sun_msun_rate": float("nan"),
                
                "total_structures_evaluated": 0,
                "stable_count": 0,
                "metastable_count": 0,
                "stable_rate": float("nan"),
                "metastable_rate": float("nan"),
                
                "unique_in_stable_count": 0,
                "unique_in_metastable_count": 0,
                "unique_in_stable_rate": float("nan"),
                "unique_in_metastable_rate": float("nan"),
                
                "sun_count": 0,
                "msun_count": 0,
                
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


class MetaSUNMetric(SUNMetric):
    """Evaluate MetaSUN (Metastable, Unique, Novel) rate of structures.

    This is a convenience class that extends SUNMetric with different default
    thresholds optimized for metastability evaluation.
    """

    def __init__(self, metastability_threshold: float = 0.1, **kwargs):
        # Set stability_threshold to metastability_threshold for primary metric
        kwargs.setdefault("stability_threshold", metastability_threshold)
        kwargs.setdefault("name", "MetaSUN")
        kwargs.setdefault(
            "description",
            "Measures proportion of structures that are Metastable, Unique, and Novel",
        )

        super().__init__(metastability_threshold=metastability_threshold, **kwargs)

    @property
    def primary_threshold(self) -> float:
        """Get the primary threshold for this metric."""
        return self.config.metastability_threshold


if __name__ == "__main__":
    import warnings

    from pymatgen.util.testing import PymatgenTest

    from lemat_genbench.metrics.sun_metric import SUNMetric
    from lemat_genbench.preprocess.universal_stability_preprocess import (
        UniversalStabilityPreprocessor,
    )

    warnings.filterwarnings("ignore", message="No oxidation states specified on sites!")
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module="pymatgen.analysis.graphs"
    )
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, message=r".*__array__.*copy.*"
    )

    test = PymatgenTest()

    structures = [
        test.get_structure("Si"),
        test.get_structure("LiFePO4"),
    ]

    mlips = ["orb", "mace"]
    for mlip in mlips:
        metric = SUNMetric()

        timeout = 60  # seconds to timeout for each MLIP run
        stability_preprocessor = UniversalStabilityPreprocessor(
            model_name=mlip,
            timeout=timeout,
            relax_structures=False,
        )

        stability_preprocessor_result = stability_preprocessor(structures)
        metric_result = metric(stability_preprocessor_result.processed_structures)
        print(mlip + " " + str(metric_result.metrics))