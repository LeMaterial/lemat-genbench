"""Novelty metrics for evaluating material structures.

This module implements novelty metrics that measure how many generated
structures are not present in a reference dataset of known materials.
Uses LeMat-Bulk dataset and BAWL fingerprinting.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import numpy as np
from datasets import load_dataset
from material_hasher.hasher.bawl import BAWLHasher
from pymatgen.analysis.local_env import EconNN
from pymatgen.core.structure import Structure

from lemat_genbench.metrics.base import BaseMetric, MetricConfig
from lemat_genbench.utils.logging import logger

warnings.filterwarnings("ignore", message="No oxidation states specified on sites!")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=r".*__array__.*copy.*"
)


@dataclass
class NoveltyConfig(MetricConfig):
    """Configuration for the Novelty metric.

    Parameters
    ----------
    reference_dataset : str, default="LeMaterial/LeMat-Bulk"
        HuggingFace dataset name to use as reference for known materials.
    reference_config : str, default="compatible_pbe"
        Configuration/subset of the reference dataset to use.
    fingerprint_method : str, default="bawl"
        Method to use for structure fingerprinting. Currently supports "bawl".
    cache_reference : bool, default=True
        Whether to cache the reference dataset fingerprints in memory.
    max_reference_size : int | None, default=None
        Maximum number of structures to load from reference dataset.
        If None, loads all structures.
    """

    reference_dataset: str = "LeMaterial/LeMat-Bulk"
    reference_config: str = "compatible_pbe"
    fingerprint_method: str = "bawl"
    cache_reference: bool = True
    max_reference_size: Optional[int] = None


class NoveltyMetric(BaseMetric):
    """Evaluate novelty of structures compared to a reference dataset.

    This metric computes the fraction of generated structures that are NOT
    present in a reference dataset of known materials, using BAWL structure
    fingerprinting to determine uniqueness.

    The novelty score is defined as:
    N = |{x ∈ G | x ∉ T}| / |G|

    where G is the set of generated structures and T is the set of known materials.

    Parameters
    ----------
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
        Higher novelty values indicate more novel structures.
    n_jobs : int, default=1
        Number of parallel jobs to run.
    """

    def __init__(
        self,
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
            name=name or "Novelty",
            description=description
            or "Measures fraction of structures not present in reference dataset",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )

        self.config = NoveltyConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            reference_dataset=reference_dataset,
            reference_config=reference_config,
            fingerprint_method=fingerprint_method,
            cache_reference=cache_reference,
            max_reference_size=max_reference_size,
        )

        # Initialize fingerprinting method
        self._init_fingerprinter()

        # Cache for reference fingerprints
        self._reference_fingerprints: Optional[Set[str]] = None
        self._reference_loaded = False

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

    def _load_reference_dataset(self) -> Set[str]:
        """Load and fingerprint the reference dataset.

        Returns
        -------
        Set[str]
            Set of fingerprints from the reference dataset.
        """
        if self._reference_fingerprints is not None and self._reference_loaded:
            return self._reference_fingerprints

        logger.info(
            f"Loading reference dataset: {self.config.reference_dataset} "
            f"(config: {self.config.reference_config})"
        )

        try:
            # Load the dataset
            dataset = load_dataset(
                self.config.reference_dataset,
                self.config.reference_config,
                split="train",
            )

            # Limit dataset size if specified
            if self.config.max_reference_size is not None:
                dataset = dataset.select(
                    range(min(len(dataset), self.config.max_reference_size))
                )

            logger.info(f"Loaded {len(dataset)} structures from reference dataset")

            # Check if fingerprints are already available in the dataset
            if "entalpic_fingerprint" in dataset.column_names:
                logger.info("Using pre-computed BAWL fingerprints from dataset")
                fingerprints = set(dataset["entalpic_fingerprint"])
                # Filter out any None or empty fingerprints
                fingerprints = {fp for fp in fingerprints if fp and fp.strip()}
            else:
                logger.info("Computing fingerprints for reference dataset structures")
                fingerprints = set()

                for i, row in enumerate(dataset):
                    try:
                        # Convert dataset row to pymatgen Structure
                        structure = self._row_to_structure(row)
                        fingerprint = self.fingerprinter.get_material_hash(structure)
                        if fingerprint:
                            fingerprints.add(fingerprint)
                    except Exception as e:
                        logger.warning(
                            f"Failed to process reference structure {i}: {str(e)}"
                        )

                    if (i + 1) % 1000 == 0:
                        logger.info(
                            f"Processed {i + 1}/{len(dataset)} reference structures"
                        )

            logger.info(
                f"Loaded {len(fingerprints)} unique fingerprints from reference dataset"
            )

            if self.config.cache_reference:
                self._reference_fingerprints = fingerprints
                self._reference_loaded = True

            return fingerprints

        except Exception as e:
            logger.error(f"Failed to load reference dataset: {str(e)}")
            raise

    def _row_to_structure(self, row: Dict[str, Any]) -> Structure:
        """Convert a dataset row to a pymatgen Structure.

        Parameters
        ----------
        row : dict
            Row from the reference dataset.

        Returns
        -------
        Structure
            Pymatgen Structure object.
        """
        # Extract lattice vectors and convert to numpy array
        lattice = np.array(row["lattice_vectors"])

        # Extract species and positions
        species = row["species_at_sites"]
        positions = np.array(row["cartesian_site_positions"])

        # Create structure (positions are already in cartesian coordinates)
        structure = Structure(
            lattice=lattice,
            species=species,
            coords=positions,
            coords_are_cartesian=True,
        )

        return structure

    def _get_compute_attributes(self) -> Dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        # Load reference fingerprints once
        reference_fingerprints = self._load_reference_dataset()

        return {
            "reference_fingerprints": reference_fingerprints,
            "fingerprinter": self.fingerprinter,
        }

    @staticmethod
    def compute_structure(
        structure: Structure,
        reference_fingerprints: Set[str],
        fingerprinter: Any,
    ) -> float:
        """Check if a structure is novel compared to the reference dataset.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to evaluate.
        reference_fingerprints : Set[str]
            Set of fingerprints from the reference dataset.
        fingerprinter : Any
            Fingerprinting method object.

        Returns
        -------
        float
            1.0 if the structure is novel (not in reference), 0.0 otherwise.
        """
        try:
            # Get fingerprint for the structure
            fingerprint = fingerprinter.get_material_hash(structure)

            if not fingerprint:
                logger.warning("Could not compute fingerprint for structure")
                return float("nan")

            # Check if fingerprint is in reference set
            is_novel = fingerprint not in reference_fingerprints

            return 1.0 if is_novel else 0.0

        except Exception as e:
            # Log the specific error and structure details for debugging
            logger.warning(
                f"Error computing novelty for structure {structure.composition.reduced_formula}: {str(e)}"
            )
            # Return NaN for failed fingerprinting, but don't crash
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
            return {
                "metrics": {
                    "novelty_score": float("nan"),
                    "novel_structures_count": 0,
                    "total_structures_evaluated": 0,
                },
                "primary_metric": "novelty_score",
                "uncertainties": {},
            }

        # Calculate novelty metrics
        novel_count = sum(valid_values)
        total_count = len(valid_values)
        novelty_score = novel_count / total_count if total_count > 0 else 0.0

        return {
            "metrics": {
                "novelty_score": novelty_score,
                "novel_structures_count": int(novel_count),
                "total_structures_evaluated": total_count,
            },
            "primary_metric": "novelty_score",
            "uncertainties": {
                "novelty_score": {
                    "std": np.std(valid_values) if len(valid_values) > 1 else 0.0
                }
            },
        }
