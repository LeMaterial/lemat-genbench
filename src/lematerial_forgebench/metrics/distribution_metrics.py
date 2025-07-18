"""Distribution metrics for evaluating material structures.

This module implements distribution metrics that quantify the degree of similarity
between a set of structures sampled from a generative model and a database of materials.

.. note::

    Example usage to be improved ⬇️

.. code-block:: python

    from lematerial_forgebench.metrics.distribution_metrics import JSDistance
    from lematerial_forgebench.metrics.base import MetricEvaluator

    metric = JSDistance()
    evaluator = MetricEvaluator(metric)
"""

import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from lematerial_forgebench.metrics.base import BaseMetric, MetricConfig, MetricResult
from lematerial_forgebench.utils.distribution_utils import (
    compute_frechetdist,
    compute_jensen_shannon_distance,
    compute_mmd,
)


@dataclass
class JSDistanceConfig(MetricConfig):
    """Configuration for the JSDistance metric.

    Parameters
    ----------
    reference_df : pandas dataframe
        dataframe with reference data to compare to the input sample of crystals
    """

    reference_df: pd.DataFrame | str = "LeMaterial/LeMat-Bulk"


class JSDistance(BaseMetric):
    """Calculate Jensen-Shannon distance between two distributions.

    This metric compares a set of distribution wide properties (crystal system,
    space group, elemental composition, lattice constants, and Wyckoff positions)
    between two samples of crystal structures and determines the degree of similarity
    between those two distributions for the particular structural property.

    Parameters
    ----------
    reference_df : pandas dataframe
        dataframe with reference data to compare to the input sample of crystals
        This dataframe is calculated by "src/lematerial_forgebench/preprocess/distribution_preprocess.py"
    name : str, optional
        Name of the metric
    description : str, optional
        Description of the metric
    n_jobs : int, optional
        Number of jobs to run in parallel
    """

    def __init__(
        self,
        reference_df: pd.DataFrame,
        name: str | None = None,
        description: str | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "Distribution",
            description=description
            or "Measures distance between two reference distributions",
            n_jobs=n_jobs,
        )
        self.config = JSDistanceConfig(
            name=self.config.name,
            description=self.config.description,
            n_jobs=self.config.n_jobs,
            reference_df=reference_df,
        )

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {"reference_df": self.config.reference_df}

    def compute(self, structures: list[Structure], **compute_args: Any) -> MetricResult:
        """Compute the similarity of the structure to a target distribution.

        Important
        ---------
        This metric expects a `reference_df` to be passed to the `compute_structure` method.
        The `reference_df` is a pandas dataframe that contains

        Parameters
        ----------
        structure : Structure
            Contains the values of the structural properties of interest for
            each of the structures in the distribution. This dataframe is
            calculated by "src/lematerial_forgebench/preprocess/distribution_preprocess.py"
            which specifies the format, column names etc used here for compatibility with
            the reference datasets. When changing the reference dataset, ensure the
            column names etc correspond to those found in the above script.
        **compute_args : Any
            Required: reference_df
            Optional: None
            This is used to pass the reference dataframe to the compute_structure method.

        Returns
        -------
        dict
            Jensen-Shannon Distances, where the keys are the structural property
            and the values are the JS Distances.
        """

        start_time = time.time()
        all_properties = [
            structure.properties.get("distribution_properties", {})
            for structure in structures
        ]

        df_all_properties = pd.DataFrame(all_properties)
        reference_df = compute_args.get("reference_df")
        if reference_df is None:
            raise ValueError(
                "a `reference_df` arg is required to compute the JSDistance"
            )

        quantities = list(df_all_properties.columns)
        dist_metrics = {}
        for quant in quantities:
            if quant in reference_df.columns:
                if isinstance(reference_df[quant].iloc[0], np.float64):
                    pass
                else:
                    js = compute_jensen_shannon_distance(
                        reference_df,
                        df_all_properties,
                        quant,
                        metric_type=type(reference_df[quant].iloc[0]),
                    )
                    dist_metrics[quant] = js

        for quant in ["CompositionCounts", "Composition"]:
            js = compute_jensen_shannon_distance(
                reference_df,
                df_all_properties,
                quant,
                metric_type=type(df_all_properties[quant].iloc[0]),
            )
            dist_metrics[quant] = js

        end_time = time.time()
        computation_time = end_time - start_time

        # This metric is used by default for ranking and comparison purposes
        dist_metrics["Average_Jensen_Shannon_Distance"] = np.mean(
            list(dist_metrics.values())
        )

        return MetricResult(
            metrics=dist_metrics,
            primary_metric="Average_Jensen_Shannon_Distance",
            uncertainties={},
            config=self.config,
            computation_time=computation_time,
            n_structures=len(structures),
            individual_values=None,  # Grouped metric
            failed_indices=[],
            warnings=[],
        )

    @staticmethod
    def compute_structure(structure: Structure, **compute_args: Any) -> dict:
        raise NotImplementedError(
            "This method is not supported for this metric because it is a batch metric"
        )

    def aggregate_results(self, values: dict[str, float]) -> Dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : dict[str, float]
            Jensen-Shannon Distance values for each structural property.

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        # Filter out NaN values
        valid_values = [v for v in values.values() if not np.isnan(v)]
        if not valid_values:
            return {
                "metrics": {
                    "Jensen_Shannon_Distance": float("nan"),
                },
                "primary_metric": "Jensen_Shannon_Distance",
                "uncertainties": {},
            }

        return {
            "metrics": {
                "Jensen_Shannon_Distance": values,
            },
            "primary_metric": "Jensen_Shannon_Distance",
            "uncertainties": {},
        }


@dataclass
class MMDConfig(MetricConfig):
    """Configuration for the MMD metric.

    Parameters
    ----------
    reference_df : pandas dataframe
        dataframe with reference data to compare to the input sample of crystals
    """

    reference_df: pd.DataFrame | str = "LeMaterial/LeMat-Bulk"


class MMD(BaseMetric):
    """Calculate MMD between two distributions.

    This metric compares a set of distribution wide properties (crystal system,
    space group, elemental composition, lattice constants, and wykoff positions)
    between two samples of crystal structures and determines the degree of similarity
    between those two distributions for the particular structural property.

    """

    def __init__(
        self,
        reference_df: pd.DataFrame,
        name: str | None = None,
        description: str | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "Distribution",
            description=description
            or "Measures distance between two reference distributions",
            n_jobs=n_jobs,
        )
        self.config = MMDConfig(
            name=self.config.name,
            description=self.config.description,
            n_jobs=self.config.n_jobs,
            reference_df=reference_df,
        )

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {"reference_df": self.config.reference_df}

    def compute(self, structures: list[Structure], **compute_args: Any) -> MetricResult:
        """Compute the similarity of a sample of structures to a target distribution.

        Parameters
        ----------
        structures : list[Structure]
            A list of pymatgen Structure objects to evaluate.


        Returns
        -------
        dict[str, float]
            MMD values for each structural property.
        """
        start_time = time.time()
        np.random.seed(32)

        all_properties = [
            structure.properties.get("distribution_properties", {})
            for structure in structures
        ]
        df_all_properties = pd.DataFrame(all_properties)
        reference_df = compute_args.get("reference_df")
        if reference_df is None:
            raise ValueError("a `reference_df` arg is required to compute the MMD")

        if len(reference_df) > 10000:
            ref_ints = np.random.randint(0, len(reference_df), 10000)
            ref_sample_df = reference_df.iloc[ref_ints]
        else:
            ref_sample_df = reference_df
        if len(df_all_properties) > 10000:
            strut_ints = np.random.randint(0, len(df_all_properties), 10000)
            strut_sample_df = df_all_properties.iloc[strut_ints]
        else:
            strut_sample_df = df_all_properties
        dist_metrics = {}
        quantities = strut_sample_df.columns
        for quant in quantities:
            if quant in ref_sample_df.columns:
                if isinstance(ref_sample_df[quant].iloc[0], np.int64):
                    pass
                else:
                    try:
                        mmd = compute_mmd(ref_sample_df, strut_sample_df, quant)
                        dist_metrics[quant] = mmd

                        dist_metrics[quant] = mmd
                    except ValueError:
                        pass

        end_time = time.time()

        dist_metrics["Average_MMD"] = np.mean(list(dist_metrics.values()))

        return MetricResult(
            metrics=dist_metrics,
            primary_metric="Average_MMD",
            uncertainties={},
            config=self.config,
            computation_time=end_time - start_time,
            n_structures=len(structures),
            individual_values=None,  # Grouped metric
            failed_indices=[],
            warnings=[],
        )

    @staticmethod
    def compute_structure(structure: Structure, **compute_args: Any) -> float:
        raise NotImplementedError(
            "This method is not supported for this metric because it is a batch metric"
        )

    def aggregate_results(self, values: dict[str, float]) -> Dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : dict[str, float]
            Jensen-Shannon Distance values for each structural property.

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        # Filter out NaN values
        valid_values = [v for v in values.values() if not np.isnan(v)]
        if not valid_values:
            return {
                "metrics": {
                    "MMD": float("nan"),
                },
                "primary_metric": "MMD",
                "uncertainties": {},
            }

        return {
            "metrics": {
                "MMD": values,
            },
            "primary_metric": "MMD",
            "uncertainties": {},
        }


@dataclass
class FrechetDistanceConfig(MetricConfig):
    """Configuration for the FrechetDistance metric.

    Parameters
    ----------
    reference_df : pandas dataframe
        dataframe with reference data to compare to the input sample of crystals
    """

    reference_df: pd.DataFrame | str = "LeMaterial/LeMat-Bulk"


class FrechetDistance(BaseMetric):
    def __init__(
        self,
        reference_df: pd.DataFrame,
        name: str | None = None,
        description: str | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "Distribution",
            description=description
            or "Measures distance between two reference distributions",
            n_jobs=n_jobs,
        )
        self.config = FrechetDistanceConfig(
            name=self.config.name,
            description=self.config.description,
            n_jobs=self.config.n_jobs,
            reference_df=reference_df,
        )

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {"reference_df": self.config.reference_df}

    def compute(self, structures: list[Structure], **compute_args: Any) -> MetricResult:
        """Compute the similarity of a sample of structures to a target distribution."""
        start_time = time.time()

        all_properties = [
            structure.properties.get("graph_embedding", {})
            for structure in structures
        ]
        reference_df = compute_args.get("reference_df")

        if "ORB" in structures[0].properties.get("mlip_model"):
            reference_column = "OrbGraphEmbeddings"
        if "MACE" in structures[0].properties.get("mlip_model"):
            reference_column = "MaceGraphEmbeddings"
        if "UMA" in structures[0].properties.get("mlip_model"):
            reference_column = "UmaGraphEmbeddings"
        if "Equiformer" in structures[0].properties.get("mlip_model"):
            reference_column = "EquiformerGraphEmbeddings"        

        reference_embeddings = reference_df[reference_column]


        if reference_df is None:
            raise ValueError(
                "a `reference_df` arg is required to compute the FrechetDistance"
            )

        dist_metrics = {}

        frechetdist = compute_frechetdist(
            reference_embeddings, all_properties
        )
        dist_metrics["FrechetDistance"] = frechetdist

        end_time = time.time()

        return MetricResult(
            metrics=dist_metrics,
            primary_metric="FrechetDistance",
            uncertainties={},
            config=self.config,
            computation_time=end_time - start_time,
            n_structures=len(structures),
            individual_values=None,  # Grouped metric
            failed_indices=[],
            warnings=[],
        )

    @staticmethod
    def compute_structure(structure: Structure, **compute_args: Any) -> float:
        """Compute the similarity of the structure to a target distribution."""
        raise NotImplementedError(
            "This method is not supported for this metric because it is a batch metric"
        )

    def aggregate_results(self, values: list[float]) -> Dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            Absolute deviations from charge neutrality for each structure.

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
                    "FrechetDistance": float("nan"),
                },
            }

        return (
            {
                "metrics": {
                    "FrechetDistance": values,
                },
                "primary_metric": "FrechetDistance",
            },
        )
    

if __name__ == "__main__":
    import pickle

    from pymatgen.util.testing import PymatgenTest

    from lematerial_forgebench.preprocess.distribution_preprocess import (
        DistributionPreprocessor,
    )
    from lematerial_forgebench.preprocess.universal_stability_preprocess import (
        UniversalStabilityPreprocessor,
    )

    with open("data/full_reference_df.pkl", "rb") as f:
        test_lemat = pickle.load(f)
    test = PymatgenTest()

    structures = [
        test.get_structure("Si"),
        test.get_structure("LiFePO4"),
    ]

    distribution_preprocessor = DistributionPreprocessor()
    distribution_preprocessor_result = distribution_preprocessor(structures)

    metric = JSDistance(reference_df=test_lemat) 
    default_args = metric._get_compute_attributes()
    metric_result = metric(distribution_preprocessor_result.processed_structures, **default_args)
    print(metric_result.metrics)

    metric = MMD(reference_df=test_lemat) 
    default_args = metric._get_compute_attributes()
    metric_result = metric(distribution_preprocessor_result.processed_structures, **default_args)
    print(metric_result.metrics)

    mlips = ["orb", "mace"]
    for mlip in mlips:
        metric = FrechetDistance(reference_df=test_lemat) 
        
        timeout = 60 # seconds to timeout for each MLIP run 
        stability_preprocessor = UniversalStabilityPreprocessor(
            model_name=mlip,
            timeout=timeout,
            relax_structures=False,
        )

        stability_preprocessor_result = stability_preprocessor(structures)

        default_args = metric._get_compute_attributes()
        metric_result = metric(stability_preprocessor_result.processed_structures, **default_args)
        print(mlip +" " + str(metric_result.metrics))


