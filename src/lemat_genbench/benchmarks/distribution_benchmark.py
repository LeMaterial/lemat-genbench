"""Distribution benchmark for material structures.

This module implements a benchmark that compares two distributions of crystal structures.
"""

from typing import Any, Dict

import numpy as np

from lemat_genbench.benchmarks.base import BaseBenchmark
from lemat_genbench.evaluator import EvaluationResult, EvaluatorConfig
from lemat_genbench.metrics.distribution_metrics import (
    MMD,
    FrechetDistance,
    JSDistance,
)
from lemat_genbench.utils.distribution_utils import safe_float


class DistributionBenchmark(BaseBenchmark):
    """Benchmark for evaluating quantitative similarity of two distributions of
    materials structures."""

    def __init__(
        self,
        mlips: list[str],
        cache_dir: str = "./data",
        js_distributions_file: str = "data/lematbulk_jsdistance_distributions.json",
        mmd_values_file: str = "data/lematbulk_mmd_values_15k.pkl",
        name: str = "DistributionBenchmark",
        description: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ):
        """Initialize the distribution benchmark.

        Parameters
        ----------
        mlips : list[str]
            List of MLIP models to use for Fréchet distance
        cache_dir : str
            Directory containing pre-computed reference statistics for Fréchet distance
        js_distributions_file : str
            Path to JSON file containing pre-computed JSDistance reference distributions
        mmd_values_file : str
            Path to pickle file containing 15K sampled MMD reference values
        name : str
            Name of the benchmark.
        description : str, optional
            Description of the benchmark.
        metadata : dict, optional
            Additional metadata for the benchmark.
        """
        if description is None:
            description = (
                "Compares the distribution of structural parameters from a sample of "
                "crystals to a reference distribution."
            )

        # Initialize the JSDistance metric with lightweight reference file
        JSDistance_metric = JSDistance(reference_distributions_file=js_distributions_file)
        # Set up evaluator configs
        evaluator_configs = {
            "JSDistance": EvaluatorConfig(
                name="JSDistance",
                description="Calculates the JS Distance between two distributions",
                metrics={"JSDistance": JSDistance_metric},
                weights={"JSDistance": 1.0},
                aggregation_method="weighted_mean",
            )
        }

        # Initialize the MMD metric with 15K sampled reference file
        MMD_metric = MMD(reference_values_file=mmd_values_file)

        # add to evaluator config
        evaluator_configs["MMD"] = EvaluatorConfig(
            name="MMD Analysis",
            description="Calculates the MMD between two distributions",
            metrics={"MMD": MMD_metric},
            weights={"MMD": 1.0},
            aggregation_method="weighted_mean",
        )

        # Initialize the FrechetDistance metric
        FrechetDistance_metric = FrechetDistance(
            mlips=mlips,
            cache_dir=cache_dir,
        )

        # add to evaluator config
        evaluator_configs["FrechetDistance"] = EvaluatorConfig(
            name="FrechetDistance Analysis",
            description="Calculates the Frechet Distance between two distributions",
            metrics={"FrechetDistance": FrechetDistance_metric},
            weights={"FrechetDistance": 1.0},
            aggregation_method="weighted_mean",
        )

        # Create benchmark metadata
        benchmark_metadata = {
            "version": "0.1.0",
            "category": "distribution",
            **(metadata or {}),
        }

        super().__init__(
            name=name,
            description=description,
            evaluator_configs=evaluator_configs,
            metadata=benchmark_metadata,
        )

    def aggregate_evaluator_results(
        self, evaluator_results: Dict[str, EvaluationResult]
    ) -> Dict[str, float]:
        """Aggregate results from multiple evaluators into final scores.

        Parameters
        ----------
        evaluator_results : dict[str, EvaluationResult]
            Results from each evaluator.

        Returns
        -------
        dict[str, float]
            Final aggregated scores.
        """

        final_scores = {
            "JSDistance": np.nan,
            "MMD": np.nan,
            "FrechetDistance": np.nan,
        }

        # Extract JSDistance results
        JSDistance_results = evaluator_results.get("JSDistance")
        if JSDistance_results:
            final_scores["JSDistance"] = JSDistance_results.get("combined_value")

        # Extract MMD results
        MMD_results = evaluator_results.get("MMD")
        if MMD_results:
            final_scores["MMD"] = safe_float(MMD_results.get("combined_value"))

        # Extract FrechetDistance results
        FrechetDistance_results = evaluator_results.get("FrechetDistance")
        if FrechetDistance_results:
            final_scores["FrechetDistance"] = safe_float(
                FrechetDistance_results.get("combined_value")
            )

        return final_scores


if __name__ == "__main__":
    from pymatgen.util.testing import PymatgenTest

    from lemat_genbench.preprocess.base import PreprocessorResult
    from lemat_genbench.preprocess.distribution_preprocess import (
        DistributionPreprocessor,
    )
    from lemat_genbench.preprocess.multi_mlip_preprocess import (
        MultiMLIPStabilityPreprocessor,
    )
    test = PymatgenTest()

    structures = [
        test.get_structure("Si"),
        test.get_structure("LiFePO4"),
    ]

    mlips = ["orb", "mace", "uma"]

    distribution_preprocessor = DistributionPreprocessor()
    dist_preprocessor_result = distribution_preprocessor(structures)
    timeout = 60
    output_dfs = {}

    mlip_configs = {
        "orb": {
            "model_type": "orb_v3_conservative_inf_omat",  # Default
            "device": "cpu",
        },
        "mace": {
            "model_type": "mp",  # Default
            "device": "cpu",
        },
        "uma": {
            "task": "omat",  # Default
            "device": "cpu",
        },
    }

    preprocessor = MultiMLIPStabilityPreprocessor(
        mlip_names=["orb", "mace", "uma"],
        mlip_configs=mlip_configs,
        relax_structures=True,
        relaxation_config={"fmax": 0.01, "steps": 300},  # Tighter convergence
        calculate_formation_energy=True,
        calculate_energy_above_hull=True,
        extract_embeddings=True,
        timeout=120,  # Longer timeout
    )

    stability_preprocessor_result = preprocessor(structures)
    final_processed_structures = []

    for ind in range(0, len(dist_preprocessor_result.processed_structures)):
        combined_structure = dist_preprocessor_result.processed_structures[ind]
        for entry in stability_preprocessor_result.processed_structures[
            ind
        ].properties.keys():
            combined_structure.properties[entry] = (
                stability_preprocessor_result.processed_structures[ind].properties[
                    entry
                ]
            )
        final_processed_structures.append(combined_structure)

    preprocessor_result = PreprocessorResult(
        processed_structures=final_processed_structures,
        config={
            "stability_preprocessor_config": stability_preprocessor_result.config,
            "distribution_preprocessor_config": dist_preprocessor_result.config,
        },
        computation_time={
            "stability_preprocessor_computation_time": stability_preprocessor_result.computation_time,
            "distribution_preprocessor_computation_time": dist_preprocessor_result.computation_time,
        },
        n_input_structures=stability_preprocessor_result.n_input_structures,
        failed_indices={
            "stability_preprocessor_failed_indices": stability_preprocessor_result.failed_indices,
            "distribution_preprocessor_failed_indices": dist_preprocessor_result.failed_indices,
        },
        warnings={
            "stability_preprocessor_warnings": stability_preprocessor_result.warnings,
            "distribution_preprocessor_warnings": dist_preprocessor_result.warnings,
        },
    )

    benchmark = DistributionBenchmark(mlips=mlips, cache_dir="./data")
    benchmark_result = benchmark.evaluate(preprocessor_result.processed_structures)

    print("JSDistance")
    print(
        benchmark_result.evaluator_results["JSDistance"]["metric_results"][
            "JSDistance"
        ].metrics
    )
    print(
        "Average JSDistance: "
        + str(benchmark_result.evaluator_results["JSDistance"]["JSDistance_value"])
    )
    print("MMD")
    print(benchmark_result.evaluator_results["MMD"]["metric_results"]["MMD"].metrics)
    print("Average MMD: " + str(benchmark_result.evaluator_results["MMD"]["MMD_value"]))
    print("Avg FrechetDistance")
    print(
        benchmark_result.evaluator_results["FrechetDistance"]["metric_results"][
            "FrechetDistance"
        ].metrics
    )
    print(
        "Uncertianty Frechet Distance: "
        + str(
            benchmark_result.evaluator_results["FrechetDistance"]["metric_results"][
                "FrechetDistance"
            ].uncertainties
        )
    )
