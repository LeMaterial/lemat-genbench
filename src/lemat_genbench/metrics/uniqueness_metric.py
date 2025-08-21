"""Uniqueness metrics for evaluating material structures.

This module implements the uniqueness metric that measures the fraction
of unique structures in a generated set using BAWL fingerprinting.
"""

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List

import ot
from pymatgen.core import Structure
from scipy.spatial.distance import cdist
from tqdm import tqdm

from lemat_genbench.fingerprinting.utils import get_fingerprint, get_fingerprinter
from lemat_genbench.metrics.base import BaseMetric, MetricConfig, MetricResult
from lemat_genbench.utils.logging import logger

warnings.filterwarnings("ignore", message="No oxidation states specified on sites!")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=r".*__array__.*copy.*"
)


def emd_distance(pdd_Q, pdd_S):
    weights_S = pdd_S[:, 0].copy()  # First column contains the weights
    distances_S = pdd_S[
        :, 1:
    ].copy()  # The rest of the columns are the distance vectors

    # For structure Q
    weights_Q = pdd_Q[:, 0].copy()
    distances_Q = pdd_Q[:, 1:].copy()

    # --- Step 3: Compute the cost matrix M ---
    # The cost matrix M[i, j] stores the cost of moving "earth" from row i of S
    # to row j of Q. The paper specifies using the L-infinity distance.
    # L-infinity norm is also called the Chebyshev distance.

    # cdist computes the distance between each pair of rows from distances_S and distances_Q
    # metric='chebyshev' corresponds to the L-infinity norm: max(|x_i - y_i|)
    M = cdist(distances_S, distances_Q, metric="chebyshev")

    # --- Step 4: Compute the Earth Mover's Distance ---
    # ot.emd2 computes the EMD value given the weights and the cost matrix.
    # The '2' in emd2 signifies that it returns only the final distance value (cost).
    emd_distance = ot.emd2(weights_S, weights_Q, M)

    return emd_distance


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
        try:
            self.fingerprinter = get_fingerprinter(self.config.fingerprint_method)
        except ValueError as e:
            raise ValueError(
                f"Unknown fingerprint method: {self.config.fingerprint_method}. "
                "Currently supported: 'bawl', 'short-bawl', 'structure-matcher', 'pdd'"
            ) from e

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
        # Use the common get_fingerprint function that checks for cached values
        return get_fingerprint(structure, fingerprinter)

    def compute(
        self,
        structures: list[Structure],
        **kwargs,
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
            The result will have a custom 'fingerprints' attribute containing
            the computed fingerprints for successful structures.
        """

        start_time = time.time()
        fingerprints = []
        failed_indices = []
        warnings = []

        compute_args = self._get_compute_attributes()

        try:
            if "pdd" in self.config.fingerprint_method.lower():
                fingerprinter = compute_args["fingerprinter"]
                all_hashes = [
                    fingerprinter.get_material_hash(structure)
                    for structure in structures
                ]
                threshold = 0.15
                count_unique = 0

                individual_values = []
                for i, structure1 in enumerate(all_hashes):
                    is_unique = True
                    min_distance = float("inf")
                    for j, structure2 in enumerate(all_hashes):
                        if i < j and structure2.shape[0] == structure1.shape[0]:
                            distance = emd_distance(structure1, structure2)
                            if distance <= threshold:
                                is_unique = False
                            min_distance = min(min_distance, distance)
                    individual_values.append(min_distance)
                    if is_unique:
                        count_unique += 1

                return MetricResult(
                    metrics={self.name: count_unique / len(structures)},
                    primary_metric=self.name,
                    uncertainties={},
                    config=self.config,
                    computation_time=time.time() - start_time,
                    n_structures=len(structures),
                    individual_values=individual_values,
                    failed_indices=failed_indices,
                    warnings=warnings,
                )

            if "structure-matcher" in self.config.fingerprint_method.lower():
                fingerprinter = compute_args["fingerprinter"]
                count_unique = 0

                individual_values = []
                for i, structure1 in tqdm(
                    enumerate(structures),
                    total=len(structures),
                    disable=not self.verbose,
                ):
                    is_unique = True
                    # min_distance = float("inf")
                    for j, structure2 in enumerate(structures):
                        if i < j:
                            is_equivalent = fingerprinter.is_equivalent(
                                structure1, structure2
                            )
                            # similarity = fingerprinter.get_similarity_score(structure1, structure2)
                            # distance = 1 - similarity
                            # min_distance = min(min_distance, distance)
                            if is_equivalent:
                                is_unique = False
                    # individual_values.append(min_distance)
                    if is_unique:
                        count_unique += 1

                return MetricResult(
                    metrics={self.name: count_unique / len(structures)},
                    primary_metric=self.name,
                    uncertainties={},
                    config=self.config,
                    computation_time=time.time() - start_time,
                    n_structures=len(structures),
                    individual_values=individual_values,
                    failed_indices=failed_indices,
                    warnings=warnings,
                )

            else:
                for idx, structure in tqdm(
                    enumerate(structures),
                    total=len(structures),
                    disable=not self.verbose,
                ):
                    try:
                        fingerprint = self._compute_structure_fingerprint(
                            structure, compute_args["fingerprinter"]
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

                # Calculate uniqueness metrics
                result_dict = self._calculate_uniqueness_metrics(
                    fingerprints, len(structures), len(failed_indices)
                )

                # Create individual values for consistency with base class
                # For uniqueness, individual values don't make as much sense,
                # but we'll assign 1.0 to unique structures and proportional values to
                # duplicates
                individual_values = self._assign_individual_values(
                    structures, fingerprints, failed_indices
                )

                result = MetricResult(
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

                # Add fingerprints as custom attribute (no base class modification needed)
                result.fingerprints = fingerprints

                return result

        except Exception as e:
            logger.error("Failed to compute uniqueness metric", exc_info=True)
            result = MetricResult(
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
            # Add empty fingerprints on failure
            result.fingerprints = []

            return result

    @staticmethod
    def _compute_batch_fingerprints(
        structures: list[Structure],
        compute_args: dict[str, Any],
    ) -> tuple[List[str], List[int], List[str]]:
        raise NotImplementedError(
            "UniquenessMetric does not support batch fingerprint computation. "
            "Use the compute method to evaluate a set of structures."
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
        raise NotImplementedError(
            "UniquenessMetric does not support individual structure evaluation. "
            "Use the compute method to evaluate a set of structures."
        )

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
