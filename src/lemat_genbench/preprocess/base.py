import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, TypeVar

import pandas as pd
from pymatgen.core.structure import Structure
from tqdm import tqdm

from lemat_genbench.data.structure import format_structures
from lemat_genbench.utils.logging import logger

PreprocessorClassVar = TypeVar("PreprocessorClassVar", bound="BasePreprocessor")


@dataclass
class PreprocessorConfig:
    """Base configuration for all preprocessors.

    This class defines the common configuration parameters shared by all preprocessors.
    Specific preprocessors should inherit from this class and add their own parameters.

    Parameters
    ----------
    name : str, optional
        Custom name for the preprocessor. If None, the class name will be used.
    description : str, optional
        Description of what the preprocessor does.
    n_jobs : int, default=1
        Number of parallel jobs to run.
    """

    name: str | None = None
    description: str | None = None
    n_jobs: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert the preprocessor configuration to a dictionary for serialization.

        Returns
        -------
        dict[str, Any]
        """
        return {
            "name": self.name,
            "description": self.description,
            "n_jobs": self.n_jobs,
        }


@dataclass
class PreprocessorResult:
    """Result of a preprocessing computation.

    Parameters
    ----------
    processed_structures : list[Structure]
        List of successfully processed pymatgen Structure objects.
    config : PreprocessorConfig
        The configuration used for this preprocessing task.
    computation_time : float
        The time taken to complete the preprocessing.
    n_input_structures : int
        The total number of structures provided as input.
    failed_indices : list[int]
        The indices (from the original input list) of structures that failed processing.
    warnings : list[str]
        The warnings generated during the computation.
    """

    processed_structures: list[Structure]
    config: PreprocessorConfig
    computation_time: float
    n_input_structures: int
    failed_indices: list[int]
    warnings: list[str]

    def __post_init__(self):
        """Validate the preprocessor result."""
        if self.n_input_structures < (
            len(self.processed_structures) + len(self.failed_indices)
        ):
            # This check might be too strict if one input can result in multiple outputs or some other complex scenario
            # For now, it assumes a one-to-one mapping or failure.
            logger.warning(
                "Number of input structures is less than processed + failed. This might indicate an issue."
            )
        if self.config is None:
            raise ValueError("config cannot be None")


class BasePreprocessor(ABC):
    """Base class for all preprocessors used in materials science workflows.

    This class defines the interface for preprocessors and provides common functionality
    like parallelization.

    Parameters
    ----------
    name : str, optional
        Custom name for the preprocessor. If None, the class name will be used.
    description : str, optional
        Description of what the preprocessor does.
    n_jobs : int, default=1
        Number of parallel jobs to run.
    """

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        n_jobs: int = 1,
    ):
        self.config = PreprocessorConfig(
            name=name or self.__class__.__name__,
            description=description,
            n_jobs=n_jobs,
        )

    @property
    def name(self) -> str:
        """Get the name of the preprocessor.

        Returns
        -------
        str
        """
        return self.config.name

    @property
    def description(self) -> str:
        """Get the description of the preprocessor.

        Returns
        -------
        str
        """
        return self.config.description or "No description provided."

    @staticmethod
    @abstractmethod
    def process_structure(structure: Structure, **process_args: Any) -> Structure:
        """Process a single structure.

        This is the main method to implement in derived classes.
        It should take a pymatgen Structure object, perform operations on it
        (e.g., add properties, modify it), and return the processed Structure object.
        If processing fails, it should raise an exception.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to process.
        **process_args : Any
            Additional keyword arguments that depend on the preprocessor implementation.

        Returns
        -------
        Structure
            The processed (e.g., annotated or modified) pymatgen Structure object.

        Raises
        ------
        Exception
            If processing of the structure fails.
        """
        pass

    @staticmethod
    def _process_batch(
        structures: list[Structure],
        process_args: dict[str, Any],
        preprocessor_class: PreprocessorClassVar,
    ) -> tuple[List[Optional[Structure]], List[int], List[str]]:
        """Process a batch of structures.

        Parameters
        ----------
        structures : list[Structure]
            Batch of structures to process.
        process_args : dict[str, Any]
            Additional keyword arguments for the process_structure method.
        preprocessor_class : PreprocessorClassVar
            The preprocessor class to use for the computation.

        Returns
        -------
        tuple[List[Optional[Structure]], List[int], List[str]]
            Tuple containing:
            - List of processed Structure objects or None for failures.
            - List of indices (local to batch) of structures that failed.
            - List of warning messages for failures.
        """
        batch_results: List[Optional[Structure]] = []
        failed_indices_in_batch: List[int] = []
        warnings_for_batch: List[str] = []

        for idx, structure in enumerate(structures):
            try:
                processed_structure = preprocessor_class.process_structure(
                    structure, **process_args
                )
                batch_results.append(processed_structure)
            except Exception as e:
                batch_results.append(None)
                failed_indices_in_batch.append(idx)
                warnings_for_batch.append(
                    f"Failed to process structure at batch index {idx} (original may vary): {str(e)}"
                )
                logger.debug(
                    f"Failed to process structure in batch for {preprocessor_class.name}",
                    exc_info=True,
                )
        return batch_results, failed_indices_in_batch, warnings_for_batch

    def _split_into_batches(
        self, structures: list[Structure], batch_size: int
    ) -> list[list[Structure]]:
        """Split structures into batches.

        Parameters
        ----------
        structures : list[Structure]
            List of structures to split.
        batch_size : int
            Size of each batch.

        Returns
        -------
        list[list[Structure]]
            List of batches of structures.
        """
        batches = [
            structures[i : i + batch_size]
            for i in range(0, len(structures), batch_size)
        ]
        return batches

    def _get_process_attributes(self) -> dict[str, Any]:
        """Get additional attributes/arguments for the process_structure method.
        Subclasses can override this to pass dynamic configuration.

        Returns
        -------
        dict[str, Any]
        """
        return {}

    def run(
        self,
        structures: list[Structure],
    ) -> PreprocessorResult:
        """Run the preprocessing on a list of structures.

        Parameters
        ----------
        structures : list[Structure]
            List of pymatgen Structure objects to process.

        Returns
        -------
        PreprocessorResult
            Object containing the processed structures and computation metadata.
        """
        start_time = time.time()
        n_input = len(structures)

        # This list will hold results in the original order, with None for failures.
        all_results_with_nones: List[Optional[Structure]] = [None] * n_input
        global_failed_indices: List[int] = []
        global_warnings: List[str] = []

        process_args = self._get_process_attributes()

        try:
            if (
                self.config.n_jobs <= 1 or n_input <= 1
            ):  # Also run serially for single structure
                # Serial computation
                for idx, structure in enumerate(tqdm(structures, desc="Processing structures (serial)", unit="structure")):
                    try:
                        processed_structure = self.process_structure(
                            structure, **process_args
                        )
                        all_results_with_nones[idx] = processed_structure
                    except Exception as e:
                        global_failed_indices.append(idx)
                        global_warnings.append(
                            f"Failed to process structure {idx}: {str(e)}"
                        )
                        logger.warning(
                            f"Failed to process structure {idx} for {self.name}",
                            exc_info=True,
                        )
            else:
                # Parallel computation
                # Ensure batch_size is at least 1, and doesn't create more batches than structures or jobs
                num_workers = min(self.config.n_jobs, n_input)
                batch_size = max(
                    1, (n_input + num_workers - 1) // num_workers
                )  # Distribute as evenly as possible

                batches = self._split_into_batches(structures, batch_size)

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(
                            self._process_batch, batch, process_args, self.__class__
                        )
                        for batch in batches
                    ]

                    current_original_idx_offset = 0
                    # Add progress bar for parallel processing
                    with tqdm(total=len(batches), desc=f"Processing batches ({num_workers} workers)", unit="batch") as pbar:
                        for future in futures:
                            (
                                batch_processed_structures_or_nones,
                                failed_indices_in_batch,
                                warnings_for_batch,
                            ) = future.result()

                            for i, struct_or_none in enumerate(
                                batch_processed_structures_or_nones
                            ):
                                original_idx = current_original_idx_offset + i
                                if original_idx < n_input:  # Boundary check
                                    all_results_with_nones[original_idx] = struct_or_none

                            global_failed_indices.extend(
                                [
                                    current_original_idx_offset + i
                                    for i in failed_indices_in_batch
                                ]
                            )
                            global_warnings.extend(warnings_for_batch)
                            current_original_idx_offset += len(
                                batch_processed_structures_or_nones
                            )
                            pbar.update(1)

            # Filter out Nones to get successfully processed structures
            final_processed_structures = [
                s for s in all_results_with_nones if s is not None
            ]

        except Exception as e:
            logger.error(f"Global failure in preprocessor {self.name}", exc_info=True)
            return PreprocessorResult(
                processed_structures=[],
                config=self.config,
                computation_time=time.time() - start_time,
                n_input_structures=n_input,
                failed_indices=list(range(n_input)),  # All failed
                warnings=[f"Global preprocessing failure for {self.name}: {str(e)}"]
                * n_input,
            )

        return PreprocessorResult(
            processed_structures=final_processed_structures,
            config=self.config,
            computation_time=time.time() - start_time,
            n_input_structures=n_input,
            failed_indices=sorted(
                list(set(global_failed_indices))
            ),  # Ensure unique and sorted
            warnings=global_warnings,
        )

    def __call__(
        self,
        structures: list[Structure] | list[dict] | pd.DataFrame | str | Path,
    ) -> PreprocessorResult:
        """Convenient callable interface for running the preprocessor.

        Parameters
        ----------
        structures : list[Structure] | list[dict] | pd.DataFrame | str | Path
            Structures to process, in various supported formats.

        Returns
        -------
        PreprocessorResult
            Object containing the processed structures and computation metadata.
        """
        structures_list = format_structures(structures)
        if not isinstance(
            structures_list, list
        ):  # Ensure it's a list for consistent processing
            structures_list = list(structures_list)
        return self.run(structures_list)

    @classmethod
    def from_config(cls, config: PreprocessorConfig) -> PreprocessorClassVar:
        """Create a preprocessor from a configuration.

        Parameters
        ----------
        config : PreprocessorConfig
            Configuration for the preprocessor.
        """
        # Extract only relevant args for BasePreprocessor constructor
        # Subclasses might need to handle additional args from their specific configs
        base_args = {
            k: v
            for k, v in config.to_dict().items()
            if k in ["name", "description", "n_jobs"]
        }
        return cls(**base_args)
