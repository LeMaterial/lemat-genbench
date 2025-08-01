"""Tests for validity benchmark."""

import pickle

import pytest
from pymatgen.util.testing import PymatgenTest

from lemat_genbench.benchmarks.distribution_benchmark import (
    DistributionBenchmark,
)
from lemat_genbench.preprocess.base import PreprocessorResult
from lemat_genbench.preprocess.distribution_preprocess import (
    DistributionPreprocessor,
)
from lemat_genbench.preprocess.multi_mlip_preprocess import (
    MultiMLIPStabilityPreprocessor,
)


@pytest.fixture
def valid_structures():
    """Create valid test structures."""
    test = PymatgenTest()
    structures = [
        test.get_structure("Si"),  # Silicon
        test.get_structure("LiFePO4"),  # Lithium iron phosphate
    ]
    return structures


@pytest.fixture
def reference_data():
    "create reference dataset"
    with open("data/full_reference_df.pkl", "rb") as f:
        reference_df = pickle.load(f)

    return reference_df


@pytest.fixture
def mlips():
    "create MLIP list"
    # mlips = ["orb", "mace", "uma"]
    mlips = ["orb", "mace"]

    return mlips


def test_initialization(reference_data, mlips):
    """Test initialization with default parameters."""

    benchmark = DistributionBenchmark(reference_df=reference_data, mlips=mlips)

    # Check name and properties
    assert benchmark.config.name == "DistributionBenchmark"
    assert "version" in benchmark.config.metadata

    # Check correct evaluators
    assert len(benchmark.evaluators) == 3
    assert "JSDistance" in benchmark.evaluators
    assert "MMD" in benchmark.evaluators
    assert "FrechetDistance" in benchmark.evaluators


def test_custom_initialization(reference_data, mlips):
    """Test initialization with custom parameters."""

    benchmark = DistributionBenchmark(
        reference_df=reference_data,
        mlips=mlips,
        name="Custom Benchmark",
        description="Custom description",
        metadata={"test_key": "test_value"},
    )

    # Check custom values
    assert benchmark.config.name == "Custom Benchmark"
    assert benchmark.config.description == "Custom description"
    assert benchmark.config.metadata["test_key"] == "test_value"


def test_evaluate(valid_structures, reference_data, mlips):
    """Test benchmark evaluation on structures."""

    distribution_preprocessor = DistributionPreprocessor()
    dist_preprocessor_result = distribution_preprocessor(valid_structures)

    mlip_configs = {
        "orb": {
            "model_type": "orb_v3_conservative_inf_omat",  # Default
            "device": "cpu",
        },
        "mace": {
            "model_type": "mp",  # Default
            "device": "cpu",
        },
        # "uma": {
        #     "task": "omat",  # Default
        #     "device": "cpu",
        # },
    }

    preprocessor = MultiMLIPStabilityPreprocessor(
        mlip_names=["orb", "mace"],
        mlip_configs=mlip_configs,
        relax_structures=True,
        relaxation_config={"fmax": 0.01, "steps": 300},  # Tighter convergence
        calculate_formation_energy=True,
        calculate_energy_above_hull=True,
        extract_embeddings=True,
        timeout=120,  # Longer timeout
    )

    stability_preprocessor_result = preprocessor(valid_structures)
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

    benchmark = DistributionBenchmark(reference_df=reference_data, mlips=mlips)
    result = benchmark.evaluate(preprocessor_result.processed_structures)

    # Check result format
    assert len(result.evaluator_results) == 3
    assert "JSDistance" in result.final_scores
    assert "MMD" in result.final_scores
    assert "FrechetDistance" in result.final_scores

    # Check score ranges
    for name, score in result.final_scores.items():
        if "JSDistance" in name or "MMD" in name:
            assert 0 <= score <= 1.0, f"{name} should be between 0 and 1"
        if "FrechetDistance" in name:
            assert 0 <= score, f"{name} should be greater than 0"
