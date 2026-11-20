"""Novelty metrics for evaluating material structures.

This module implements novelty metrics that measure how many generated
structures are not present in a reference dataset of known materials.
Uses LeMat-Bulk dataset and BAWL fingerprinting.
"""

import json
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from datasets import load_dataset
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

from lemat_genbench.fingerprinting.encode_compositions import (
    filter_df,
    get_all_compositions,
    lematbulk_item_to_structure,
)
from lemat_genbench.fingerprinting.utils import get_fingerprint, get_fingerprinter
from lemat_genbench.metrics.base import BaseMetric, MetricConfig
from lemat_genbench.utils.logging import logger

warnings.filterwarnings("ignore", message="No oxidation states specified on sites!")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=r".*__array__.*copy.*"
)


# Module-level symprec used by SpacegroupAnalyzer calls in this module.
# Can be overridden at runtime (e.g. by the symprec sweep script).
_SYMPREC = 0.01

# Coded return values from compute_structure for novel structures
NOT_NOVEL = 0.0
NOVEL_COMPOSITION = 1.0       # composition not in reference
NOVEL_SPACEGROUP = 2.0        # known composition, different spacegroup
NOVEL_STRUCTURE_ONLY = 3.0    # known composition + spacegroup, novel by fingerprint


def _get_reference_spacegroups(
    formula: str, dataset_information: Dict[str, Any]
) -> Set[int]:
    """Get the set of spacegroup numbers for *formula* in the reference dataset.

    Results are cached in ``dataset_information["_sg_cache"]`` so that each
    composition is computed at most once.

    Parameters
    ----------
    formula : str
        Normalised reduced formula (pymatgen canonical form).
    dataset_information : dict[str, Any]
        Must contain ``comp_to_indices``, ``dataset_ref``, and ``_sg_cache``.

    Returns
    -------
    set[int]
        Spacegroup numbers found for *formula* in the reference.
    """
    sg_cache = dataset_information.get("_sg_cache", {})
    if formula in sg_cache:
        return sg_cache[formula]

    indices = dataset_information.get("comp_to_indices", {}).get(formula, [])
    dataset = dataset_information.get("dataset_ref")
    if dataset is None or not indices:
        return set()

    spacegroups: Set[int] = set()
    for idx in indices:
        try:
            item = dataset[idx]
            structure = Structure(
                lattice=item["lattice_vectors"],
                species=item["species_at_sites"],
                coords=item["cartesian_site_positions"],
                coords_are_cartesian=True,
            )
            sg = SpacegroupAnalyzer(structure, symprec=_SYMPREC).get_space_group_number()
            spacegroups.add(sg)
        except Exception:
            pass

    sg_cache[formula] = spacegroups
    return spacegroups


def _classify_framework(
    structure: Structure, dataset_information: Dict[str, Any]
) -> None:
    """Classify a structure's framework novelty and append to the accumulator.

    Appends one of ``"existing_anon_known_sg"``, ``"existing_anon_novel_sg"``,
    ``"novel_anon"``, or ``None`` (on failure) to
    ``dataset_information["_framework_classifications"]``.

    No-op if the anon SG index was not loaded.

    Parameters
    ----------
    structure : Structure
        The structure to classify.
    dataset_information : dict[str, Any]
        Must contain ``anon_sg_index`` and ``_framework_classifications``.
    """
    fw_list = dataset_information.get("_framework_classifications")
    anon_sg_index = dataset_information.get("anon_sg_index")
    if fw_list is None or anon_sg_index is None:
        return

    try:
        anon = Composition(
            structure.composition.anonymized_formula
        ).reduced_composition.anonymized_formula

        if anon not in anon_sg_index:
            fw_list.append("novel_anon")
            return

        sg = SpacegroupAnalyzer(
            structure, symprec=_SYMPREC
        ).get_space_group_number()
        if sg in anon_sg_index[anon]:
            fw_list.append("existing_anon_known_sg")
        else:
            fw_list.append("existing_anon_novel_sg")
    except Exception:
        fw_list.append(None)


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
        verbose: bool = False,
    ):
        super().__init__(
            name=name or "Novelty",
            description=description
            or "Measures fraction of structures not present in reference dataset",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
            verbose=verbose,
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
        self._dataset_information: Optional[Set[str]] = None
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
        """Load and fingerprint the reference dataset.

        Returns
        -------
        dict[str, Any]
            Dictionary containing reference dataset information.
        """
        if self._dataset_information is not None and self._reference_loaded:
            return self._dataset_information

        logger.info(
            f"Loading reference dataset: {self.config.reference_dataset} "
            f"(config: {self.config.reference_config})"
        )

        dataset_information = {}
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

            # Build composition index for novelty breakdown
            # (before branching — dataset still has all columns here)
            if "chemical_formula_reduced" in dataset.column_names:
                logger.info("Building reference composition index...")
                ref_compositions: Set[str] = set()
                comp_to_indices: dict[str, list[int]] = defaultdict(list)
                formula_norm_cache: dict[str, Optional[str]] = {}
                for idx, formula in enumerate(dataset["chemical_formula_reduced"]):
                    if formula not in formula_norm_cache:
                        try:
                            formula_norm_cache[formula] = (
                                Composition(formula).reduced_formula
                            )
                        except Exception:
                            formula_norm_cache[formula] = None
                    norm = formula_norm_cache[formula]
                    if norm is not None:
                        ref_compositions.add(norm)
                        comp_to_indices[norm].append(idx)

                dataset_information["reference_compositions"] = ref_compositions
                dataset_information["comp_to_indices"] = dict(comp_to_indices)
                # Keep full-column dataset for on-demand spacegroup lookups
                dataset_information["dataset_ref"] = dataset
                dataset_information["_sg_cache"] = {}
                logger.info(
                    f"Indexed {len(ref_compositions)} unique reference compositions"
                )

            # Load pre-built anonymous formula → space group index for framework classification
            _anon_sg_path = (
                Path(__file__).resolve().parents[3]
                / "data"
                / "lematbulk_anon_sg_index.json"
            )
            if _anon_sg_path.exists():
                with open(_anon_sg_path) as f:
                    raw_index = json.load(f)
                dataset_information["anon_sg_index"] = {
                    anon: set(sgs) for anon, sgs in raw_index.items()
                }
                dataset_information["_framework_classifications"] = []
                logger.info(
                    f"Loaded framework index: {len(raw_index)} anonymous formulas"
                )
            else:
                logger.info(
                    "No framework index found at %s — skipping framework classification",
                    _anon_sg_path,
                )

            # Check if fingerprints are already available in the dataset
            if (
                "entalpic_fingerprint" in dataset.column_names
                and "bawl" in self.config.fingerprint_method.lower()
            ):
                logger.info("Using pre-computed BAWL fingerprints from dataset")
                fingerprints = set(dataset["entalpic_fingerprint"])
                # Filter out any None or empty fingerprints
                fingerprints = {fp for fp in fingerprints if fp and fp.strip()}

                if "short" in self.config.fingerprint_method.lower():
                    fingerprints = {
                        f"{fp.split('_')[0]}_{fp.split('_')[2]}" for fp in fingerprints
                    }

                dataset_information["fingerprints"] = fingerprints

                logger.info(
                    f"Loaded {len(fingerprints)} unique fingerprints from reference dataset"
                )

            elif self.config.fingerprint_method.lower() in ["structure-matcher"]:
                df = dataset.select_columns(
                    ["immutable_id", "chemical_formula_descriptive"]
                ).to_pandas()
                df = df.set_index("immutable_id")
                df["index_number"] = np.arange(len(df))

                dataset = load_dataset(
                    "LeMaterial/LeMat-Bulk",
                    "compatible_pbe",
                    split="train",
                    columns=[
                        "elements",
                        "immutable_id",
                        "chemical_formula_descriptive",
                        "energy",
                        "species_at_sites",
                        "cartesian_site_positions",
                        "lattice_vectors",
                    ],
                )

                all_compositions = get_all_compositions()

                dataset_information["dataset_dataframe"] = df
                dataset_information["all_compositions"] = all_compositions
                dataset_information["dataset"] = dataset

            elif hasattr(self.fingerprinter, "get_material_hash"):
                logger.info("Computing fingerprints for reference dataset structures")
                fingerprints = set()

                for i, row in tqdm(enumerate(dataset)):
                    try:
                        # Convert dataset row to pymatgen Structure
                        structure = self._row_to_structure(row)
                        fingerprint = get_fingerprint(structure, self.fingerprinter)
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

                dataset_information["fingerprints"] = fingerprints

                logger.info(
                    f"Loaded {len(fingerprints)} unique fingerprints from reference dataset"
                )

            if self.config.cache_reference:
                self._dataset_information = dataset_information
                self._reference_loaded = True

            return dataset_information

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
        """Check if a structure is novel compared to the reference dataset.

        For novel structures the return value encodes a category:

        - ``NOT_NOVEL`` (0.0) — structure fingerprint found in reference.
        - ``NOVEL_COMPOSITION`` (1.0) — composition not in reference at all.
        - ``NOVEL_SPACEGROUP`` (2.0) — composition exists in reference but no
          reference entry shares the same space group.
        - ``NOVEL_STRUCTURE_ONLY`` (3.0) — a reference entry with matching
          composition **and** space group exists, yet the fingerprint is unique.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to evaluate.
        dataset_information : dict[str, Any]
            Reference dataset information (fingerprints, compositions, etc.).
        fingerprinter : Any
            Fingerprinting method object.
        verbose : bool
            If True, print detailed information about the novelty check process.

        Returns
        -------
        float
            Coded novelty category (see above), or ``nan`` on failure.
        """
        try:
            # Framework classification (independent of novelty, appends to shared list)
            _classify_framework(structure, dataset_information)

            fingerprint = get_fingerprint(structure, fingerprinter)

            if hasattr(fingerprinter, "get_material_hash"):
                if not fingerprint:
                    logger.warning("Could not compute fingerprint for structure")
                    return float("nan")

                is_novel = fingerprint not in dataset_information["fingerprints"]

            else:
                # Comparison-based matchers (e.g. structure-matcher)
                df_filtered = filter_df(
                    dataset_information["dataset_dataframe"],
                    dataset_information["all_compositions"],
                    structure,
                )
                dataset_select = dataset_information["dataset"].select(
                    dataset_information["dataset_dataframe"].loc[df_filtered.index][
                        "index_number"
                    ]
                )
                is_equivalent = False

                for item in tqdm(dataset_select, disable=not verbose):
                    ref_structure = lematbulk_item_to_structure(item)
                    _is_equivalent = fingerprinter.is_equivalent(
                        structure, ref_structure
                    )
                    is_equivalent = is_equivalent or _is_equivalent

                is_novel = not is_equivalent

            if not is_novel:
                return NOT_NOVEL

            # --- Classify the novel structure ---
            ref_compositions = dataset_information.get("reference_compositions")
            if ref_compositions is None:
                # No composition index available — count as novel (unclassified)
                return NOVEL_COMPOSITION

            formula = structure.composition.reduced_formula
            if formula not in ref_compositions:
                return NOVEL_COMPOSITION

            # Composition is known — check space group
            try:
                gen_sg = SpacegroupAnalyzer(
                    structure, symprec=_SYMPREC
                ).get_space_group_number()
            except Exception:
                # Can't determine space group; conservatively classify as
                # different-spacegroup since we can't confirm a match
                return NOVEL_SPACEGROUP

            ref_sgs = _get_reference_spacegroups(formula, dataset_information)
            if gen_sg not in ref_sgs:
                return NOVEL_SPACEGROUP
            return NOVEL_STRUCTURE_ONLY

        except Exception as e:
            logger.warning(
                f"Error computing novelty for structure "
                f"{structure.composition.reduced_formula}: {str(e)}"
            )
            return float("nan")

    def aggregate_results(self, values: List[float]) -> Dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            Coded novelty values for each structure.  ``0.0`` = not novel,
            ``1.0`` = novel composition, ``2.0`` = novel spacegroup,
            ``3.0`` = novel structure only, ``nan`` = failed.

        Returns
        -------
        dict
            Dictionary with aggregated metrics including per-category counts.
        """
        valid_values = [v for v in values if not np.isnan(v)]

        if not valid_values:
            return {
                "metrics": {
                    "novelty_score": float("nan"),
                    "novel_structures_count": 0,
                    "total_structures_evaluated": 0,
                    "novel_composition_count": 0,
                    "novel_spacegroup_count": 0,
                    "novel_structure_only_count": 0,
                },
                "primary_metric": "novelty_score",
                "uncertainties": {},
            }

        # Any value > 0 counts as novel
        novel_count = sum(1 for v in valid_values if v > 0)
        total_count = len(valid_values)
        novelty_score = novel_count / total_count if total_count > 0 else 0.0

        # Per-category breakdown
        novel_composition_count = sum(
            1 for v in valid_values if v == NOVEL_COMPOSITION
        )
        novel_spacegroup_count = sum(
            1 for v in valid_values if v == NOVEL_SPACEGROUP
        )
        novel_structure_only_count = sum(
            1 for v in valid_values if v == NOVEL_STRUCTURE_ONLY
        )

        # Binary novel/not-novel for std calculation
        binary = [1.0 if v > 0 else 0.0 for v in valid_values]

        metrics = {
            "novelty_score": novelty_score,
            "novel_structures_count": int(novel_count),
            "total_structures_evaluated": total_count,
            "novel_composition_count": int(novel_composition_count),
            "novel_spacegroup_count": int(novel_spacegroup_count),
            "novel_structure_only_count": int(novel_structure_only_count),
        }

        # Framework novelty counts (accumulated during compute_structure calls)
        fw_classifications = None
        if self._dataset_information is not None:
            fw_classifications = self._dataset_information.get(
                "_framework_classifications"
            )
        if fw_classifications is not None:
            from collections import Counter

            fw_counts = Counter(c for c in fw_classifications if c is not None)
            metrics["framework_existing_anon_known_sg"] = fw_counts.get(
                "existing_anon_known_sg", 0
            )
            metrics["framework_existing_anon_novel_sg"] = fw_counts.get(
                "existing_anon_novel_sg", 0
            )
            metrics["framework_novel_anon"] = fw_counts.get("novel_anon", 0)

        return {
            "metrics": metrics,
            "primary_metric": "novelty_score",
            "uncertainties": {
                "novelty_score": {
                    "std": np.std(binary) if len(binary) > 1 else 0.0
                }
            },
        }
