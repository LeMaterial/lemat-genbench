"""Novelty metrics for evaluating material structures.

This module implements novelty metrics that measure how many generated
structures are not present in a reference dataset of known materials.
Uses MP-20 dataset and supports both BAWL fingerprinting and structure matching.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import numpy as np
from pymatgen.core import Structure
from tqdm import tqdm

from lemat_genbench.fingerprinting.utils import get_fingerprint, get_fingerprinter
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
    reference_dataset_path : str, default="mp-20-data/mp_20.csv"
        Path to MP-20 CSV file to use as reference for known materials.
    fingerprint_method : str, default="bawl"
        Method to use for structure fingerprinting. 
        Supports "bawl", "short-bawl", and "structure-matcher".
    cache_reference : bool, default=True
        Whether to cache the reference dataset fingerprints in memory.
    max_reference_size : int | None, default=None
        Maximum number of structures to load from reference dataset.
        If None, loads all structures.
    """

    reference_dataset_path: str = "mp-20-data/mp_20.csv"
    fingerprint_method: str = "bawl"
    cache_reference: bool = True
    max_reference_size: Optional[int] = None


class NoveltyMetric(BaseMetric):
    """Evaluate novelty of structures compared to MP-20 reference dataset.

    This metric computes the fraction of generated structures that are NOT
    present in the MP-20 reference dataset, using BAWL fingerprinting or
    structure matching to determine uniqueness.

    The novelty score is defined as:
    N = |{x ∈ G | x ∉ T}| / |G|

    where G is the set of generated structures and T is the MP-20 reference set.

    Parameters
    ----------
    reference_dataset_path : str, default="mp-20-data/mp_20.csv"
        Path to MP-20 CSV file to use as reference.
    fingerprint_method : str, default="bawl"
        Method to use for structure fingerprinting.
        Supports "bawl", "short-bawl", and "structure-matcher".
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
        reference_dataset_path: str = "mp-20-data/mp_20.csv",
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
            name=name or "Novelty",
            description=description
            or "Measures fraction of structures not present in MP-20 reference dataset",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self.config = NoveltyConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            reference_dataset_path=reference_dataset_path,
            fingerprint_method=fingerprint_method,
            cache_reference=cache_reference,
            max_reference_size=max_reference_size,
        )

        # Initialize fingerprinting method
        self._init_fingerprinter()

        # Cache for reference fingerprints
        self._dataset_information: Optional[Dict[str, Any]] = None
        self._reference_loaded = False

    def _init_fingerprinter(self) -> None:
        """Initialize the fingerprinting method."""

        try:
            self.fingerprinter = get_fingerprinter(self.config.fingerprint_method)
        except ValueError as e:
            raise ValueError(
                f"Unknown fingerprint method: {self.config.fingerprint_method}. "
                "Currently supported: 'bawl', 'short-bawl', 'structure-matcher'"
            ) from e

    def _load_reference_dataset(self) -> dict[str, Any]:
        """Load and fingerprint the MP-20 reference dataset.

        Returns
        -------
        dict[str, Any]
            Dictionary containing reference dataset information.
        """
        if self._dataset_information is not None and self._reference_loaded:
            return self._dataset_information

        logger.info(f"Loading MP-20 reference dataset from: {self.config.reference_dataset_path}")

        from pathlib import Path
        from lemat_genbench.data.mp20_loader import (
            load_mp20_dataset,
            mp20_item_to_structure,
            get_mp20_compositions,
            filter_df_by_composition,
        )

        dataset_information = {}
        try:
            # Load MP-20 dataset from CSV
            df = load_mp20_dataset(self.config.reference_dataset_path)

            # Limit dataset size if specified
            if self.config.max_reference_size is not None:
                df = df.head(self.config.max_reference_size)
                logger.info(f"Limited to {len(df)} structures (max_reference_size={self.config.max_reference_size})")

            logger.info(f"Loaded {len(df)} structures from MP-20 reference dataset")

            # BAWL fingerprint path
            if "bawl" in self.config.fingerprint_method.lower():
                # Check for pre-computed fingerprints (split-aware)
                # Create cache filename based on the dataset path
                dataset_name = Path(self.config.reference_dataset_path).stem  # e.g., "mp_20", "train", "test"
                fingerprint_cache = Path(f"data/mp20_bawl_fingerprints_{dataset_name}.pkl")
                
                if fingerprint_cache.exists():
                    import pickle
                    logger.info(f"Loading pre-computed BAWL fingerprints for {dataset_name}")
                    with open(fingerprint_cache, 'rb') as f:
                        fingerprints = pickle.load(f)
                    logger.info(f"Loaded {len(fingerprints)} pre-computed fingerprints")
                else:
                    logger.info(f"Computing BAWL fingerprints for {dataset_name} (this may take a while)")
                    logger.info(f"Consider running: scripts/precompute_mp20_reference.py --csv-path {self.config.reference_dataset_path}")
                    fingerprints = set()
                    
                    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing fingerprints"):
                        try:
                            structure = mp20_item_to_structure(row)
                            fingerprint = get_fingerprint(structure, self.fingerprinter)
                            if fingerprint:
                                fingerprints.add(fingerprint)
                        except Exception as e:
                            if idx < 10:  # Only log first 10 errors
                                logger.warning(f"Failed to process structure {idx}: {e}")
                    
                    # Cache for future use (split-aware)
                    logger.info(f"Caching {len(fingerprints)} fingerprints to {fingerprint_cache}")
                    fingerprint_cache.parent.mkdir(parents=True, exist_ok=True)
                    import pickle
                    with open(fingerprint_cache, 'wb') as f:
                        pickle.dump(fingerprints, f)
                
                # Apply short fingerprint if requested
                if "short" in self.config.fingerprint_method.lower():
                    fingerprints = {
                        f"{fp.split('_')[0]}_{fp.split('_')[2]}" for fp in fingerprints
                    }
                    logger.info("Using shortened BAWL fingerprints")
                
                dataset_information["fingerprints"] = fingerprints
                logger.info(f"Using {len(fingerprints)} unique fingerprints for novelty detection")

            # Structure matcher path
            elif self.config.fingerprint_method.lower() == "structure-matcher":
                logger.info("Setting up structure matcher with composition filtering")
                
                # Prepare dataframe for fast indexing
                df_indexed = df.set_index('material_id', drop=False)
                df_indexed['index_number'] = np.arange(len(df_indexed))
                
                # Load composition matrix for fast filtering
                all_compositions = get_mp20_compositions(self.config.reference_dataset_path)
                
                dataset_information["dataset_dataframe"] = df_indexed
                dataset_information["all_compositions"] = all_compositions
                dataset_information["dataset"] = df
                dataset_information["filter_function"] = filter_df_by_composition
                
                logger.info(f"Loaded composition matrix: {all_compositions.shape}")
                logger.info("Structure matcher ready for novelty checks")

            else:
                raise ValueError(f"Unsupported fingerprint method: {self.config.fingerprint_method}")

            if self.config.cache_reference:
                self._dataset_information = dataset_information
                self._reference_loaded = True

            return dataset_information

        except Exception as e:
            logger.error(f"Failed to load MP-20 reference dataset: {str(e)}")
            raise


    def _get_compute_attributes(self) -> Dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        # Load reference fingerprints once
        dataset_information = self._load_reference_dataset()

        return {
            "dataset_information": dataset_information,
            "fingerprinter": self.fingerprinter,
            "verbose": self.verbose,
        }

    @staticmethod
    def compute_structure(
        structure: Structure,
        dataset_information: dict[str, Any],
        fingerprinter: Any,
        verbose: bool = False,
    ) -> float:
        """Check if a structure is novel compared to the MP-20 reference dataset.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to evaluate.
        dataset_information : dict[str, Any]
            Dictionary containing reference dataset information (fingerprints or dataframe).
        fingerprinter : Any
            Fingerprinting method object.
        verbose: bool
            If True, print detailed information about the novelty check process.

        Returns
        -------
        float
            1.0 if the structure is novel (not in reference), 0.0 otherwise.
        """
        try:
            # Get fingerprint for the structure, using cached value if available
            fingerprint = get_fingerprint(structure, fingerprinter)

            if hasattr(fingerprinter, "get_material_hash"):
                # Fingerprint-based novelty (BAWL)
                if not fingerprint:
                    logger.warning("Could not compute fingerprint for structure")
                    return float("nan")

                # Check if fingerprint is in reference set
                is_novel = fingerprint not in dataset_information["fingerprints"]

            else:
                # Structure matcher path (comparison-based)
                from lemat_genbench.data.mp20_loader import mp20_item_to_structure
                
                # Filter to structures with same elements (critical optimization)
                filter_function = dataset_information.get("filter_function")
                df_filtered = filter_function(
                    dataset_information["dataset_dataframe"],
                    dataset_information["all_compositions"],
                    structure,
                )
                
                if len(df_filtered) == 0:
                    # No structures with same elements -> definitely novel
                    return 1.0
                
                if verbose:
                    logger.info(
                        f"Filtered from {len(dataset_information['dataset_dataframe'])} "
                        f"to {len(df_filtered)} structures based on composition"
                    )
                
                # Compare against filtered structures
                dataset_df = dataset_information["dataset"]
                filtered_indices = df_filtered['index_number'].values
                
                is_equivalent = False
                for idx in tqdm(filtered_indices, disable=not verbose, desc="Comparing structures"):
                    row = dataset_df.iloc[idx]
                    ref_structure = mp20_item_to_structure(row)
                    _is_equivalent = fingerprinter.is_equivalent(structure, ref_structure)
                    
                    if _is_equivalent:
                        is_equivalent = True
                        break  # Found a match, can stop

                is_novel = not is_equivalent

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
