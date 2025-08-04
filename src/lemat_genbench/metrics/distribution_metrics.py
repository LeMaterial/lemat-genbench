"""Distribution metrics for evaluating material structures.

This module implements distribution metrics that quantify the degree of similarity
between a set of structures sampled from a generative model and a database of materials.

.. note::

    Example usage to be improved ⬇️

.. code-block:: python

    from lemat_genbench.metrics.distribution_metrics import JSDistance
    from lemat_genbench.metrics.base import MetricEvaluator

    metric = JSDistance()
    evaluator = MetricEvaluator(metric)
"""

import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from pydantic import Field
from pymatgen.core import Structure

from lemat_genbench.metrics.base import BaseMetric, MetricConfig, MetricResult
from lemat_genbench.utils.distribution_utils import (
    compute_frechetdist_with_cache,
    compute_jensen_shannon_distance,
    compute_mmd,
    load_reference_stats_cache,
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
        This dataframe is calculated by "src/lemat_genbench/preprocess/distribution_preprocess.py"
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
            calculated by "src/lemat_genbench/preprocess/distribution_preprocess.py"
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
    cache_dir : str
        Directory containing pre-computed reference statistics (mu and sigma)
    mlips : list[str]
        List of MLIP models to compute Fréchet distance for
    """

    mlips: list[str] = Field(default_factory=lambda: ["orb", "mace", "uma"])
    cache_dir: str = "./data"


class FrechetDistance(BaseMetric):
    def __init__(
        self,
        mlips: list[str],
        cache_dir: str = "./data",
        name: str | None = None,
        description: str | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "FrechetDistance",
            description=description
            or "Measures Fréchet distance between generated and reference distributions using pre-computed statistics",
            n_jobs=n_jobs,
        )
        self.config = FrechetDistanceConfig(
            name=self.config.name,
            description=self.config.description,
            n_jobs=self.config.n_jobs,
            mlips=mlips,
            cache_dir=cache_dir,
        )
        
        # Load cached reference statistics (required)
        self.reference_stats = load_reference_stats_cache(cache_dir, mlips)
        if not self.reference_stats:
            raise ValueError(
                f"Could not load cached reference statistics from {cache_dir}. "
                f"Please run 'uv run scripts/compute_reference_stats.py --cache-dir {cache_dir}' first."
            )

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {
            "mlips": self.config.mlips,
            "reference_stats": self.reference_stats,
        }

    def compute(self, structures: list[Structure], **compute_args: Any) -> MetricResult:
        """Compute Fréchet distance using pre-computed reference statistics."""
        start_time = time.time()
        reference_stats = compute_args.get("reference_stats")
        mlips = compute_args.get("mlips", [])
        
        distances = []
        warnings = []
        
        for mlip in mlips:
            # Get generated embeddings
            all_properties = [
                structure.properties.get(f"graph_embedding_{mlip}", {})
                for structure in structures
            ]
            
            # Skip if no embeddings found
            if not all_properties or all(emb is None or (hasattr(emb, '__len__') and len(emb) == 0) for emb in all_properties):
                warnings.append(f"No embeddings found for model {mlip}")
                continue
            
            try:
                # Use cached statistics (required)
                if reference_stats and mlip in reference_stats:
                    cached_stats = reference_stats[mlip]
                    frechetdist = compute_frechetdist_with_cache(
                        cached_stats["mu"], 
                        cached_stats["sigma"], 
                        all_properties
                    )
                    distances.append(frechetdist)
                else:
                    warnings.append(f"No cached statistics found for model {mlip}")
                    continue
                
            except Exception as e:
                warnings.append(f"Failed to compute Fréchet distance for {mlip}: {str(e)}")
                continue

        if not distances:
            raise ValueError("No valid Fréchet distances computed for any model")

        dist_metrics = {}
        dist_metrics["FrechetDistanceMean"] = np.mean(distances)

        end_time = time.time()

        return MetricResult(
            metrics=dist_metrics,
            primary_metric="FrechetDistanceMean",
            uncertainties={
                "FrechetDistanceStd": np.std(distances) if len(distances) > 1 else 0.0,
                "FrechetDistancesFull": distances,
                "n_models_computed": len(distances),
            },
            config=self.config,
            computation_time=end_time - start_time,
            n_structures=len(structures),
            individual_values=None,  # Grouped metric
            failed_indices=[],
            warnings=warnings,
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

    from lemat_genbench.preprocess.distribution_preprocess import (
        DistributionPreprocessor,
    )

    # Load test data
    with open("data/full_reference_df.pkl", "rb") as f:
        test_lemat = pickle.load(f)
    
    test = PymatgenTest()
    structures = [test.get_structure("Si"), test.get_structure("LiFePO4")]

    # Test JSDistance
    preprocessor = DistributionPreprocessor()
    processed = preprocessor(structures)
    
    js_metric = JSDistance(reference_df=test_lemat)
    js_result = js_metric(processed.processed_structures, **js_metric._get_compute_attributes())
    print("JSDistance:", js_result.metrics)

    # Test MMD
    mmd_metric = MMD(reference_df=test_lemat)
    mmd_result = mmd_metric(processed.processed_structures, **mmd_metric._get_compute_attributes())
    print("MMD:", mmd_result.metrics)

    # Test FrechetDistance with cache for all 3 models
    print("\n" + "="*50)
    print("TESTING FRÉCHET DISTANCE (ALL 3 MODELS)")
    print("="*50)
    
    try:
        # Initialize FrechetDistance for all 3 models
        frechet_metric = FrechetDistance(mlips=["uma", "orb", "mace"], cache_dir="./data")
        print("✅ FrechetDistance metric created successfully")
        print(f"📊 Loaded cache for models: {list(frechet_metric.reference_stats.keys())}")
        
        # Generate real embeddings using MultiMLIPStabilityPreprocessor
        from lemat_genbench.preprocess.multi_mlip_preprocess import (
            MultiMLIPStabilityPreprocessor,
        )
        
        print("🔧 Setting up MLIP preprocessor...")
        mlip_configs = {
            "orb": {"model_type": "orb_v3_conservative_inf_omat", "device": "cpu"},
            "mace": {"model_type": "mp", "device": "cpu"},
            "uma": {"task": "omat", "device": "cpu"},
        }
        
        mlip_preprocessor = MultiMLIPStabilityPreprocessor(
            mlip_names=["uma", "orb", "mace"],
            mlip_configs=mlip_configs,
            relax_structures=False,  # Skip relaxation for faster demo
            extract_embeddings=True,
            timeout=60,
        )
        
        print(f"🧮 Computing embeddings for {len(structures)} structures...")
        print("   (This may take 1-2 minutes...)")
        
        # Generate embeddings
        mlip_result = mlip_preprocessor(structures)
        print(f"✅ Generated embeddings for {len(mlip_result.processed_structures)} structures")
        
        # Debug: Check available embeddings
        print("\n🔍 Available embeddings in processed structures:")
        for i, structure in enumerate(mlip_result.processed_structures):
            embedding_keys = [key for key in structure.properties.keys() if 'embedding' in key.lower()]
            print(f"   Structure {i}: {embedding_keys}")
        
        print("\n🔬 Computing Fréchet distance for all models...")
        
        # Compute Fréchet distance
        result = frechet_metric(mlip_result.processed_structures, **frechet_metric._get_compute_attributes())
        
        print("\n📈 FRÉCHET DISTANCE RESULTS:")
        print(f"   • Mean Distance: {result.metrics['FrechetDistanceMean']:.4f}")
        print(f"   • Std Deviation: {result.uncertainties['FrechetDistanceStd']:.4f}")
        print(f"   • Individual Distances: {[f'{d:.4f}' for d in result.uncertainties['FrechetDistancesFull']]}")
        print(f"   • Models Successfully Computed: {result.uncertainties['n_models_computed']}/3")
        print(f"   • Total Computation Time: {result.computation_time:.3f} seconds")
        
        if result.warnings:
            print("\n⚠️  Warnings:")
            for warning in result.warnings:
                print(f"     • {warning}")
        
        # Show per-model breakdown if available
        if len(result.uncertainties['FrechetDistancesFull']) > 1:
            models = ["uma", "orb", "mace"][:len(result.uncertainties['FrechetDistancesFull'])]
            print("\n📊 Per-Model Breakdown:")
            for model, distance in zip(models, result.uncertainties['FrechetDistancesFull']):
                print(f"   • {model.upper()}: {distance:.4f}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have:")
        print("   1. Computed reference statistics: uv run scripts/compute_reference_stats.py --cache-dir ./data")
        print("   2. Installed MLIP dependencies: uv add orb_models")
