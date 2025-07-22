import time
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, TypeVar

import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure

from lematerial_forgebench.data.structure import format_structures
from lematerial_forgebench.utils.logging import logger

warnings.filterwarnings("ignore", message="No oxidation states specified on sites!")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*__array__.*copy.*")

ClassVar = TypeVar("ClassVar", bound="BaseMetric")


@dataclass
class MetricConfig:
    """Base configuration for all metrics.

    This class defines the common configuration parameters shared by all metrics.
    Specific metrics should inherit from this class and add their own parameters.

    Parameters
    ----------
    name : str, optional
        Custom name for the metric. If None, the class name will be used.
    description : str, optional
        Description of what the metric measures.
    lower_is_better : bool, default=False
        Whether lower values of this metric indicate better performance.
    n_jobs : int, default=1
        Number of parallel jobs to run.
    """

    name: str | None = None
    description: str | None = None
    lower_is_better: bool = False
    n_jobs: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert the metric configuration to a dictionary for serialization.

        Returns
        -------
        dict[str, Any]
        """
        return {
            "name": self.name,
            "description": self.description,
            "lower_is_better": self.lower_is_better,
            "n_jobs": self.n_jobs,
        }


@dataclass
class MetricResult:
    """Result of a metric computation.

    Every instance of this class contains the following attributes,
    for a single metric computation on a batch of structures:

    Parameters
    ----------
    metrics : dict[str, float]
        Dictionary of all computed metric values. Each metric should have a descriptive
        name as the key and its computed value as the float value.
    primary_metric : str
        The key in metrics dict that represents the primary metric value. This is used
        for ranking and comparison purposes.
    uncertainties : dict[str, dict[str, float]]
        Dictionary mapping metric names to their uncertainty measures. For each metric,
        common uncertainty keys include:
        - 'std': Standard deviation
        - 'std_error': Standard error of the mean
        - 'confidence_95': 95% confidence interval
        - 'p_value': P-value from statistical tests
    config : MetricConfig
        The configuration used to compute this result.
    computation_time : float
        The time taken to compute the metric.
    n_structures : int
        The number of structures that were evaluated.
    individual_values : list[float]
        The individual metric values for each structure.
        This will have the same length as the number of structures
        and will be NaN for the structures that failed to compute.
    failed_indices : list[int]
        The indices of the structures that failed to compute.
    warnings : list[str]
        The warnings generated during the computation.
    """

    metrics: dict[str, float]
    primary_metric: str
    uncertainties: dict[str, dict[str, float]]
    config: MetricConfig
    computation_time: float
    n_structures: int
    individual_values: list[float]
    failed_indices: list[int]
    warnings: list[str]

    def __post_init__(self):
        """Validate the metric result and initialize empty dictionaries if None."""
        if not self.metrics:
            raise ValueError("metrics dictionary cannot be empty")
        if self.primary_metric not in self.metrics:
            raise ValueError(
                f"primary_metric '{self.primary_metric}' must be a key in metrics"
            )
        if self.uncertainties is None:
            self.uncertainties = {}

    @property
    def value(self) -> float:
        """Get the primary metric value for backward compatibility.

        Returns
        -------
        float
            The value of the primary metric.
        """
        return self.metrics[self.primary_metric]


class BaseMetric(ABC):
    """Base class for all metrics used to evaluate generative models in materials science.

    This class defines the interface for all metrics and provides common functionality
    like parallelization and metadata handling.

    Parameters
    ----------
    name : str, optional
        Custom name for the metric. If None, the class name will be used.
    description : str, optional
        Description of what the metric measures.
    lower_is_better : bool, default=False
        Whether lower values of this metric indicate better performance.
    n_jobs : int, default=1
        Number of parallel jobs to run.

    Notes
    -----
    All metrics should inherit from this base class and implement the required methods.
    Each metric should also define its own configuration class that inherits from MetricConfig.
    """

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
    ):
        self.config = MetricConfig(
            name=name or self.__class__.__name__,
            description=description,
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )

    @property
    def name(self) -> str:
        """Get the name of the metric.

        Returns
        -------
        str
            The name of the metric.
        """
        return self.config.name

    @property
    def description(self) -> str:
        """Get the description of the metric.

        Returns
        -------
        str
            The description of the metric or "No description provided" if none was set.
        """
        return self.config.description or "No description provided."

    def compute_structure(structure: Structure, **compute_args: Any) -> float:
        """Compute the metric for a single structure.

        This is the main method to implement in derived classes.
        Any configuration parameters needed should be accessed via instance attributes.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to evaluate.
        **compute_args : Any
            Additional keyword arguments that depend on the metric implementation.

        Returns
        -------
        float
            The computed metric value for this structure.
        """
        pass

    @abstractmethod
    def aggregate_results(self, values: list[float]) -> dict[str, Any]:
        """Aggregate individual structure results into final metrics.

        This method should be implemented by each metric to define how individual
        structure results are combined into final metrics.

        Parameters
        ----------
        values : list[float]
            Individual metric values for each structure. May contain NaN values
            for failed computations.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - metrics: dict[str, float] - All computed metrics
            - primary_metric: str - Key of the primary metric in metrics dict
            - uncertainties: dict[str, dict[str, float]] - Uncertainties per metric
        """
        pass

    @staticmethod
    def _compute_batch(
        structures: list[Structure],
        compute_args: dict[str, Any],
        metric_class: ClassVar,
    ) -> tuple[List[float], List[int], List[str]]:
        """Compute metric for a batch of structures.

        Parameters
        ----------
        structures : list[Structure]
            Batch of structures to process.
        compute_args : dict[str, Any]
            Additional keyword arguments for the compute_structure method.
        metric_class : ClassVar
            The metric class to use for the computation.
            This is needed to call the compute_structure method of the metric class.

        Returns
        -------
        tuple[List[float], List[int], List[str]]
            Tuple containing:
            - List of metric values for each structure.
            - List of indices of the structures that failed to compute.
            - List of warnings for each structure.
        """
        individual_values = []
        failed_indices = []
        warnings = []
        try:
            for idx, structure in enumerate(structures):
                try:
                    value = metric_class.compute_structure(structure, **compute_args)
                    individual_values.append(value)
                except Exception as e:
                    failed_indices.append(idx)
                    individual_values.append(float("nan"))
                    warnings.append(str(e))
            return individual_values, failed_indices, warnings
        except Exception as e:
            logger.error("Failed to compute metric for batch", exc_info=True)
            return (
                [float("nan")] * len(structures),
                [i for i in range(len(structures))],
                [
                    f"Batch computation failure: {str(e)}"
                    for _ in range(len(structures))
                ],
            )

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

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method.

        Returns
        -------
        dict[str, Any]
        """
        return {}

    def compute(
        self,
        structures: list[Structure],
        **compute_args: Any,
    ) -> MetricResult:
        """Compute the metric on a batch of structures.

        Parameters
        ----------
        structures : list[Structure]
            List of pymatgen Structure objects to evaluate.

        Returns
        -------
        MetricResult
            Object containing the metric values and computation metadata.
        """
        start_time = time.time()
        failed_indices = []
        warnings = []
        values = []

        compute_args = self._get_compute_attributes()

        try:
            if self.config.n_jobs <= 1:
                # Serial computation
                for idx, structure in enumerate(structures):
                    try:
                        value = self.compute_structure(structure, **compute_args)
                        values.append(value)
                    except Exception as e:
                        failed_indices.append(idx)
                        values.append(float("nan"))
                        warnings.append(
                            f"Failed to compute metric for structure {idx}: {str(e)}"
                        )
                        logger.warning(
                            f"Failed to compute metric for structure {idx}",
                            exc_info=True,
                        )
            else:
                batch_size = max(1, len(structures) // self.config.n_jobs)
                batches = self._split_into_batches(structures, batch_size)

                with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
                    futures = [
                        executor.submit(
                            self._compute_batch, batch, compute_args, self.__class__
                        )
                        for batch in batches
                    ]

                    current_idx = 0
                    for future in futures:
                        batch_values, failed_batch_indices, batch_warnings = (
                            future.result()
                        )
                        values.extend(batch_values)
                        failed_indices.extend(
                            [current_idx + i for i in failed_batch_indices]
                        )
                        warnings.extend(batch_warnings)
                        current_idx += len(batch_values)

            # Compute aggregate statistics
            try:
                if values and not all(np.isnan(v) for v in values):
                    result_dict = self.aggregate_results(values)
                else:
                    # Case where all values are NaN or empty
                    result_dict = {
                        "metrics": {self.name: float("nan")},
                        "primary_metric": self.name,
                        "uncertainties": {},
                    }
            except TypeError:
                if values[0].values() and not all(
                    np.isnan(v) for v in values[0].values()
                ):
                    result_dict = self.aggregate_results(values[0])
                else:
                    # Case where all values are NaN or empty
                    result_dict = {
                        "metrics": {self.name: float("nan")},
                        "primary_metric": self.name,
                        "uncertainties": {},
                    }

        except Exception as e:
            logger.error("Failed to compute metric", exc_info=True)
            return MetricResult(
                metrics={self.name: float("nan")},
                primary_metric=self.name,
                uncertainties={},
                config=self.config,
                computation_time=time.time() - start_time,
                n_structures=len(structures),
                individual_values=values,
                failed_indices=list(range(len(structures))),
                warnings=[
                    f"Global computation failure: {str(e)}"
                    for _ in range(len(structures))
                ],
            )

        return MetricResult(
            metrics=result_dict["metrics"],
            primary_metric=result_dict["primary_metric"],
            uncertainties=result_dict["uncertainties"],
            config=self.config,
            computation_time=time.time() - start_time,
            n_structures=len(structures),
            individual_values=values,
            failed_indices=failed_indices,
            warnings=warnings,
        )

    def __call__(
        self,
        structures: list[Structure] | list[dict] | pd.DataFrame | str | Path,
        **compute_args: Any,
    ) -> MetricResult:
        """Convenient callable interface for computing the metric.

        Parameters
        ----------
        structures : list[Structure] | list[dict] | pd.DataFrame | str | Path
            Structures to evaluate in various supported formats.

        **compute_args : Any
            Additional keyword arguments for the compute_structure method.

        Returns
        -------
        MetricResult
            Object containing the metric value and computation metadata.
        """
        structures_list = format_structures(structures)
        return self.compute(structures_list, **compute_args)

    @classmethod
    def from_config(cls, config: MetricConfig) -> ClassVar:
        """Create a metric from a configuration.

        Parameters
        ----------
        config : MetricConfig
            Configuration for the metric.
        """
        return cls(
            **config.to_dict(),
        )
