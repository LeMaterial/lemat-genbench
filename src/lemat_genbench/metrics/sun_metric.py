"""SUN (Stable, Unique, Novel) metrics for evaluating material structures.

This module implements SUN and MetaSUN metrics that measure the proportion
of generated structures that are simultaneously stable (or metastable),
unique, and novel.
"""

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from lemat_genbench.metrics.base import BaseMetric, MetricConfig, MetricResult
from lemat_genbench.metrics.novelty_metric import NoveltyMetric
from lemat_genbench.metrics.uniqueness_metric import UniquenessMetric
from lemat_genbench.utils.logging import logger
from pymatgen.core.structure import Structure

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
    1. Stable (e_above_hull <= stability_threshold)
    2. Unique (not duplicated within the generated set)
    3. Novel (not present in reference dataset)

    The SUN rate is defined as:
    SUN = |{x ∈ G | E_hull(x) ≤ 0, x ∉ T, x is unique}| / |G|

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
        Method to use for structure fingerprinting.
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
    ):
        super().__init__(
            name=name or "SUN",
            description=description
            or "Measures proportion of structures that are Stable, Unique, and Novel",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
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
        )

    def compute(self, structures: list[Structure]) -> MetricResult:
        """Compute the SUN metric on a batch of structures.

        This method efficiently computes SUN by:
        1. First identifying unique structures
        2. Then checking novelty only for unique structures
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
            logger.info("Computing uniqueness...")
            uniqueness_result = self.uniqueness_metric.compute(structures)

            # Identify unique structures (individual_values = 1.0 means unique)
            unique_indices = [
                i
                for i, val in enumerate(uniqueness_result.individual_values)
                if not np.isnan(val) and val == 1.0
            ]

            if not unique_indices:
                logger.info("No unique structures found")
                return self._create_result(
                    start_time,
                    n_structures,
                    [],
                    [],
                    [],
                    uniqueness_result.failed_indices,
                )

            logger.info(f"Found {len(unique_indices)} unique structures")

            # Step 2: Check novelty only for unique structures
            logger.info("Computing novelty for unique structures...")
            unique_structures = [structures[i] for i in unique_indices]
            novelty_result = self.novelty_metric.compute(unique_structures)

            # Identify which unique structures are novel (individual_values = 1.0 means novel)
            novel_among_unique_indices = [
                unique_indices[i]
                for i, val in enumerate(novelty_result.individual_values)
                if not np.isnan(val) and val == 1.0
            ]

            if not novel_among_unique_indices:
                logger.info("No novel structures among unique structures")
                return self._create_result(
                    start_time,
                    n_structures,
                    [],
                    [],
                    [],
                    uniqueness_result.failed_indices + novelty_result.failed_indices,
                )

            logger.info(
                f"Found {len(novel_among_unique_indices)} novel unique structures"
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
                set(uniqueness_result.failed_indices + novelty_result.failed_indices)
            )

            return self._create_result(
                start_time,
                n_structures,
                sun_indices,
                msun_indices,
                unique_indices,
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
        unique_indices: List[int],
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
        unique_rate = len(unique_indices) / n_structures if n_structures > 0 else 0.0
        total_sun_msun = len(sun_indices) + len(msun_indices)
        combined_rate = total_sun_msun / n_structures if n_structures > 0 else 0.0

        metrics = {
            "sun_rate": sun_rate,
            "msun_rate": msun_rate,
            "combined_sun_msun_rate": combined_rate,
            "sun_count": len(sun_indices),
            "msun_count": len(msun_indices),
            "unique_count": len(unique_indices),
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

    from lemat_genbench.metrics.sun_metric import SUNMetric
    from lemat_genbench.preprocess.universal_stability_preprocess import (
        UniversalStabilityPreprocessor,
    )
    from pymatgen.util.testing import PymatgenTest

    warnings.filterwarnings("ignore", message="No oxidation states specified on sites!")
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module="pymatgen.analysis.graphs"
    )
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, message=r".*__array__.*copy.*"
    )

    breakpoint
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
