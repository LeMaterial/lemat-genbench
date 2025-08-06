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
    reference_distributions_file : str
        Path to JSON file containing pre-computed reference distributions
    """

    reference_distributions_file: str = "data/lematbulk_jsdistance_distributions.json"


class JSDistance(BaseMetric):
    """Calculate Jensen-Shannon distance between two distributions.

    This metric compares a set of distribution wide properties (crystal system,
    space group, elemental composition, lattice constants, and Wyckoff positions)
    between two samples of crystal structures and determines the degree of similarity
    between those two distributions for the particular structural property.

    Parameters
    ----------
    reference_distributions_file : str, optional
        Path to JSON file containing pre-computed reference distributions
    name : str, optional
        Name of the metric
    description : str, optional
        Description of the metric
    n_jobs : int, optional
        Number of jobs to run in parallel
    """

    def __init__(
        self,
        reference_distributions_file: str = "data/lematbulk_jsdistance_distributions.json",
        name: str | None = None,
        description: str | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "JSDistance",
            description=description
            or "Measures Jensen-Shannon distance between generated and reference distributions",
            n_jobs=n_jobs,
        )
        self.config = JSDistanceConfig(
            name=self.config.name,
            description=self.config.description,
            n_jobs=self.config.n_jobs,
            reference_distributions_file=reference_distributions_file,
        )

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {"reference_distributions_file": self.config.reference_distributions_file}

    def compute(self, structures: list[Structure], **compute_args: Any) -> MetricResult:
        """Compute Jensen-Shannon distance using pre-computed reference distributions.

        Parameters
        ----------
        structures : list[Structure]
            List of structures with distribution_properties in their properties dict
        **compute_args : Any
            Optional: reference_distributions_file
            Path to JSON file containing pre-computed reference distributions

        Returns
        -------
        MetricResult
            Jensen-Shannon distances for each property and average
        """

        start_time = time.time()
        all_properties = [
            structure.properties.get("distribution_properties", {})
            for structure in structures
        ]

        df_all_properties = pd.DataFrame(all_properties)
        reference_distributions_file = compute_args.get(
            "reference_distributions_file", 
            self.config.reference_distributions_file
        )

        # Define properties that JSDistance processes (non-float64 types)
        js_properties = {
            "SpaceGroup": np.int64,
            "CrystalSystem": np.int64, 
            "CompositionCounts": np.ndarray,
            "Composition": np.ndarray
        }

        dist_metrics = {}
        warnings = []
        
        # Process each property that JSDistance handles
        for prop, metric_type in js_properties.items():
            if prop in df_all_properties.columns:
                try:
                    js = compute_jensen_shannon_distance(
                        df_all_properties,
                        prop,
                        metric_type,
                        reference_distributions_file
                    )
                    dist_metrics[prop] = js
                except Exception as e:
                    warnings.append(f"Failed to compute JSDistance for {prop}: {str(e)}")

        if not dist_metrics:
            raise ValueError("No valid Jensen-Shannon distances computed for any property")

        end_time = time.time()
        computation_time = end_time - start_time

        # Compute average Jensen-Shannon distance
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
            warnings=warnings,
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
    reference_values_file : str
        Path to pickle file containing 15K sampled reference values
    """

    reference_values_file: str = "data/lematbulk_mmd_values_15k.pkl"


class MMD(BaseMetric):
    """Calculate MMD between two distributions.

    This metric compares continuous distribution properties (volume, densities)
    between two samples of crystal structures and determines the degree of similarity
    between those two distributions using kernel methods.

    Uses a fixed 15K sample from the LeMat-Bulk dataset for fast, reproducible results.
    The sample was pre-computed using seed=42 and stored in a lightweight pickle file.

    Parameters
    ----------
    reference_values_file : str, optional
        Path to pickle file containing 15K sampled reference values
    name : str, optional
        Name of the metric
    description : str, optional
        Description of the metric
    n_jobs : int, optional
        Number of jobs to run in parallel
    """

    def __init__(
        self,
        reference_values_file: str = "data/lematbulk_mmd_values_15k.pkl",
        name: str | None = None,
        description: str | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "MMD",
            description=description
            or "Measures Maximum Mean Discrepancy between generated and reference distributions",
            n_jobs=n_jobs,
        )
        self.config = MMDConfig(
            name=self.config.name,
            description=self.config.description,
            n_jobs=self.config.n_jobs,
            reference_values_file=reference_values_file,
        )

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {"reference_values_file": self.config.reference_values_file}

    def compute(self, structures: list[Structure], **compute_args: Any) -> MetricResult:
        """Compute MMD using 15K sampled reference values.

        Parameters
        ----------
        structures : list[Structure]
            List of structures with distribution_properties in their properties dict
        **compute_args : Any
            Optional: reference_values_file
            Path to pickle file containing 15K sampled reference values

        Returns
        -------
        MetricResult
            MMD values for each continuous property and average
        """
        start_time = time.time()
        np.random.seed(32)

        all_properties = [
            structure.properties.get("distribution_properties", {})
            for structure in structures
        ]
        df_all_properties = pd.DataFrame(all_properties)
        reference_values_file = compute_args.get(
            "reference_values_file", 
            self.config.reference_values_file
        )

        # Define properties that MMD processes (continuous/non-int64 types)
        mmd_properties = ["Volume", "Density(g/cm^3)", "Density(atoms/A^3)"]

        # Sample generated data if too large (for computational efficiency)
        if len(df_all_properties) > 15000:
            strut_ints = np.random.randint(0, len(df_all_properties), 15000)
            df_sample = df_all_properties.iloc[strut_ints]
        else:
            df_sample = df_all_properties

        dist_metrics = {}
        warnings = []
        
        # Process each property that MMD handles
        for prop in mmd_properties:
            if prop in df_sample.columns:
                try:
                    mmd = compute_mmd(
                        df_sample,
                        prop,
                        reference_values_file
                    )
                    dist_metrics[prop] = mmd
                except Exception as e:
                    warnings.append(f"Failed to compute MMD for {prop}: {str(e)}")

        if not dist_metrics:
            raise ValueError("No valid MMD values computed for any property")

        end_time = time.time()

        # Compute average MMD
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
            warnings=warnings,
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
            MMD values for each structural property.

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

    from pymatgen.util.testing import PymatgenTest

    from lemat_genbench.preprocess.distribution_preprocess import (
        DistributionPreprocessor,
    )

    test = PymatgenTest()
    structures = [test.get_structure("Si"), test.get_structure("LiFePO4")]

    # Test JSDistance with lightweight reference files
    preprocessor = DistributionPreprocessor()
    processed = preprocessor(structures)
    
    js_metric = JSDistance()  # Uses default lightweight reference file
    js_result = js_metric(processed.processed_structures, **js_metric._get_compute_attributes())
    print("JSDistance:", js_result.metrics)

    # Test MMD with 15K sampled reference values
    mmd_metric = MMD()  # Uses default 15K sample file
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
            timeout=300,
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
