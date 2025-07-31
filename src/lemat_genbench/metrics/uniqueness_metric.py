"""Uniqueness metrics for evaluating material structures.

This module implements the uniqueness metric that measures the fraction
of unique structures in a generated set using BAWL fingerprinting.
"""

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List

from lemat_genbench.metrics.base import BaseMetric, MetricConfig, MetricResult
from lemat_genbench.utils.logging import logger
from material_hasher.hasher.bawl import BAWLHasher
from pymatgen.analysis.local_env import EconNN
from pymatgen.core.structure import Structure

warnings.filterwarnings("ignore", message="No oxidation states specified on sites!")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=r".*__array__.*copy.*"
)


@dataclass
class UniquenessConfig(MetricConfig):
    """Configuration for the Uniqueness metric.

    Parameters
    ----------
    fingerprint_method : str, default="bawl"
        Method to use for structure fingerprinting. Currently supports "bawl".
    """

    fingerprint_method: str = "bawl"


class UniquenessMetric(BaseMetric):
    """Evaluate uniqueness of structures within a generated set.

    This metric computes the fraction of unique structures in a generated set
    using BAWL structure fingerprinting to determine uniqueness.

    The uniqueness score is defined as:
    U = |unique(G)| / |G|

    where G is the set of generated structures and unique(G) returns
    the set of unique structures based on their fingerprints.

    Parameters
    ----------
    fingerprint_method : str, default="bawl"
        Method to use for structure fingerprinting.
    name : str, optional
        Custom name for the metric.
    description : str, optional
        Description of what the metric measures.
    lower_is_better : bool, default=False
        Higher uniqueness values indicate more unique structures.
    n_jobs : int, default=1
        Number of parallel jobs to run.
    """

    def __init__(
        self,
        fingerprint_method: str = "bawl",
        name: str | None = None,
        description: str | None = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "Uniqueness",
            description=description
            or "Measures fraction of unique structures in generated set",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )

        self.config = UniquenessConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            fingerprint_method=fingerprint_method,
        )

        # Initialize fingerprinting method
        self._init_fingerprinter()

    def _init_fingerprinter(self) -> None:
        """Initialize the fingerprinting method."""
        if self.config.fingerprint_method.lower() == "bawl":
            self.fingerprinter = BAWLHasher(
                graphing_algorithm="WL",
                bonding_algorithm=EconNN,
                bonding_kwargs={
                    "tol": 0.2,
                    "cutoff": 10,
                    "use_fictive_radius": True,
                },
                include_composition=True,
                symmetry_labeling="SPGLib",
                shorten_hash=False,
            )
        else:
            raise ValueError(
                f"Unknown fingerprint method: {self.config.fingerprint_method}. "
                "Currently supported: 'bawl'"
            )

    def _get_compute_attributes(self) -> Dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {
            "fingerprinter": self.fingerprinter,
        }

    @staticmethod
    def _compute_structure_fingerprint(
        structure: Structure,
        fingerprinter: Any,
    ) -> str | None:
        """Compute the fingerprint for a structure.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to evaluate.
        fingerprinter : Any
            Fingerprinting method object.

        Returns
        -------
        str | None
            Fingerprint string if successful, None if failed.
        """
        try:
            # Get fingerprint for the structure
            fingerprint = fingerprinter.get_material_hash(structure)
            return fingerprint
        except Exception as e:
            # Log the specific error and structure details for debugging
            logger.warning(
                f"Error computing fingerprint for structure {structure.composition.reduced_formula}: {str(e)}"
            )
            return None

    def compute(
        self,
        structures: list[Structure],
    ) -> "MetricResult":
        """Compute the uniqueness metric on a batch of structures.

        This method overrides the base compute method to handle fingerprint
        collection and uniqueness calculation differently from individual
        structure scoring.

        Parameters
        ----------
        structures : list[Structure]
            List of pymatgen Structure objects to evaluate.

        Returns
        -------
        MetricResult
            Object containing the uniqueness metrics and computation metadata.
        """

        start_time = time.time()
        fingerprints = []
        failed_indices = []
        warnings = []

        compute_args = self._get_compute_attributes()

        try:
            if self.config.n_jobs <= 1:
                # Serial computation
                for idx, structure in enumerate(structures):
                    try:
                        fingerprint = self._compute_structure_fingerprint(
                            structure, **compute_args
                        )
                        if fingerprint is not None:
                            fingerprints.append(fingerprint)
                        else:
                            failed_indices.append(idx)
                            warnings.append(
                                f"Failed to compute fingerprint for structure {idx}"
                            )
                    except Exception as e:
                        failed_indices.append(idx)
                        warnings.append(
                            f"Failed to compute fingerprint for structure {idx}: {str(e)}"
                        )
                        logger.warning(
                            f"Failed to compute fingerprint for structure {idx}",
                            exc_info=True,
                        )
            else:
                # Parallel computation
                from concurrent.futures import ProcessPoolExecutor

                batch_size = max(1, len(structures) // self.config.n_jobs)
                batches = self._split_into_batches(structures, batch_size)

                with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
                    futures = [
                        executor.submit(
                            self._compute_batch_fingerprints, batch, compute_args
                        )
                        for batch in batches
                    ]

                    current_idx = 0
                    for future in futures:
                        batch_fingerprints, failed_batch_indices, batch_warnings = (
                            future.result()
                        )
                        fingerprints.extend(batch_fingerprints)
                        failed_indices.extend(
                            [current_idx + i for i in failed_batch_indices]
                        )
                        warnings.extend(batch_warnings)
                        current_idx += len(batch_fingerprints) + len(
                            failed_batch_indices
                        )

            # Calculate uniqueness metrics
            result_dict = self._calculate_uniqueness_metrics(
                fingerprints, len(structures), len(failed_indices)
            )

        except Exception as e:
            logger.error("Failed to compute uniqueness metric", exc_info=True)
            return MetricResult(
                metrics={self.name: float("nan")},
                primary_metric=self.name,
                uncertainties={},
                config=self.config,
                computation_time=time.time() - start_time,
                n_structures=len(structures),
                individual_values=[float("nan")] * len(structures),
                failed_indices=list(range(len(structures))),
                warnings=[
                    f"Global computation failure: {str(e)}"
                    for _ in range(len(structures))
                ],
            )

        # Create individual values for consistency with base class
        # For uniqueness, individual values don't make as much sense,
        # but we'll assign 1.0 to unique structures and proportional values to duplicates
        individual_values = self._assign_individual_values(
            structures, fingerprints, failed_indices
        )

        return MetricResult(
            metrics=result_dict["metrics"],
            primary_metric=result_dict["primary_metric"],
            uncertainties=result_dict["uncertainties"],
            config=self.config,
            computation_time=time.time() - start_time,
            n_structures=len(structures),
            individual_values=individual_values,
            failed_indices=failed_indices,
            warnings=warnings,
        )

    @staticmethod
    def _compute_batch_fingerprints(
        structures: list[Structure],
        compute_args: dict[str, Any],
    ) -> tuple[List[str], List[int], List[str]]:
        """Compute fingerprints for a batch of structures.

        Parameters
        ----------
        structures : list[Structure]
            Batch of structures to process.
        compute_args : dict[str, Any]
            Additional keyword arguments for the compute_structure method.

        Returns
        -------
        tuple[List[str], List[int], List[str]]
            Tuple containing:
            - List of fingerprints for successful structures.
            - List of indices of the structures that failed to compute.
            - List of warnings for failed structures.
        """
        fingerprints = []
        failed_indices = []
        warnings = []

        try:
            for idx, structure in enumerate(structures):
                try:
                    fingerprint = UniquenessMetric._compute_structure_fingerprint(
                        structure, **compute_args
                    )
                    if fingerprint is not None:
                        fingerprints.append(fingerprint)
                    else:
                        failed_indices.append(idx)
                        warnings.append("Failed to compute fingerprint")
                except Exception as e:
                    failed_indices.append(idx)
                    warnings.append(str(e))

            return fingerprints, failed_indices, warnings

        except Exception as e:
            logger.error("Failed to compute fingerprints for batch", exc_info=True)
            return (
                [],
                [i for i in range(len(structures))],
                [
                    f"Batch computation failure: {str(e)}"
                    for _ in range(len(structures))
                ],
            )

    def _calculate_uniqueness_metrics(
        self, fingerprints: List[str], total_structures: int, failed_count: int
    ) -> Dict[str, Any]:
        """Calculate uniqueness metrics from fingerprints.

        Parameters
        ----------
        fingerprints : List[str]
            List of fingerprints from successful computations.
        total_structures : int
            Total number of structures evaluated.
        failed_count : int
            Number of structures that failed fingerprinting.

        Returns
        -------
        dict
            Dictionary with calculated metrics.
        """
        if not fingerprints:
            return {
                "metrics": {
                    "uniqueness_score": float("nan"),
                    "unique_structures_count": 0,
                    "total_structures_evaluated": total_structures,
                    "duplicate_structures_count": 0,
                    "failed_fingerprinting_count": failed_count,
                },
                "primary_metric": "uniqueness_score",
                "uncertainties": {},
            }

        # Count unique fingerprints
        unique_fingerprints = set(fingerprints)
        unique_count = len(unique_fingerprints)
        total_valid = len(fingerprints)
        duplicate_count = total_valid - unique_count

        # Calculate uniqueness score
        uniqueness_score = unique_count / total_valid if total_valid > 0 else 0.0

        return {
            "metrics": {
                "uniqueness_score": uniqueness_score,
                "unique_structures_count": unique_count,
                "total_structures_evaluated": total_structures,
                "duplicate_structures_count": duplicate_count,
                "failed_fingerprinting_count": failed_count,
            },
            "primary_metric": "uniqueness_score",
            "uncertainties": {
                "uniqueness_score": {
                    "std": 0.0  # Uniqueness is deterministic given fingerprints
                }
            },
        }

    def _assign_individual_values(
        self,
        structures: list[Structure],
        fingerprints: List[str],
        failed_indices: List[int],
    ) -> List[float]:
        """Assign individual values to structures for consistency.

        For uniqueness metric, individual values represent how "unique"
        each structure is within the set.

        Parameters
        ----------
        structures : list[Structure]
            Original list of structures.
        fingerprints : List[str]
            List of successful fingerprints.
        failed_indices : List[int]
            Indices of structures that failed fingerprinting.

        Returns
        -------
        List[float]
            Individual values for each structure.
        """
        individual_values = [float("nan")] * len(structures)

        # Count occurrences of each fingerprint
        if fingerprints:
            from collections import Counter

            fingerprint_counts = Counter(fingerprints)

            # Assign values based on uniqueness
            fingerprint_idx = 0
            for struct_idx in range(len(structures)):
                if struct_idx not in failed_indices:
                    fingerprint = fingerprints[fingerprint_idx]
                    count = fingerprint_counts[fingerprint]
                    # Unique structures get 1.0, duplicates get 1/count
                    individual_values[struct_idx] = 1.0 / count
                    fingerprint_idx += 1

        return individual_values

    @staticmethod
    def compute_structure(structure: Structure, **compute_args: Any) -> float:
        """Compute metric for a single structure.

        This method is required by the base class but not used directly
        for uniqueness calculation. Instead, we override the compute method.

        For uniqueness, we need to compare against all other structures in the set,
        so individual structure evaluation doesn't make sense.

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
        for uniqueness calculation since we override the compute method.

        Parameters
        ----------
        values : list[float]
            Individual values (not used directly).

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        # This method is not used in our custom compute implementation
        # but is required by the base class
        return {
            "metrics": {"uniqueness_score": 0.0},
            "primary_metric": "uniqueness_score",
            "uncertainties": {},
        }
