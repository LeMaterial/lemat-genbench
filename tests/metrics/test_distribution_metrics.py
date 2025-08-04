"""Tests for validity metrics implementation."""

import numpy as np
import pytest
from pymatgen.util.testing import PymatgenTest

from lemat_genbench.metrics.distribution_metrics import (
    MMD,
    FrechetDistance,
    JSDistance,
)
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


def test_JSDistance_metric(valid_structures):
    """Test JSDistance_metric on valid structures."""
    distribution_preprocessor = DistributionPreprocessor()
    preprocessor_result = distribution_preprocessor(valid_structures)

    metric = JSDistance()  # Uses default lightweight reference file
    result = metric.compute(
        preprocessor_result.processed_structures, **metric._get_compute_attributes()
    )

    # Check computation didn't fail
    assert len(result.failed_indices) == 0

    # Check result structure
    assert "Average_Jensen_Shannon_Distance" in result.metrics

    # Check values
    values = [val for key, val in result.metrics.items() if "Average" not in key]
    assert not np.any(np.isnan(values))
    for val in values:
        assert 0.0 <= val <= 1.0


def test_MMD_metric(valid_structures):
    """Test MMD_metric on valid structures."""
    distribution_preprocessor = DistributionPreprocessor()
    preprocessor_result = distribution_preprocessor(valid_structures)

    metric = MMD()  # Uses default 15K sample file
    result = metric.compute(
        preprocessor_result.processed_structures, **metric._get_compute_attributes()
    )
    # Check computation didn't fail
    assert len(result.failed_indices) == 0

    # Check result structure
    assert "Average_MMD" in result.metrics

    # Check values
    values = [val for key, val in result.metrics.items() if "Average" not in key]
    assert not np.any(np.isnan(values))
    for val in values:
        assert 0.0 <= val <= 1.0


def test_FrechetDistance_metric(valid_structures):
    """Test MMD_metric on valid structures."""
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

    # metric = FrechetDistance(mlips=["orb", "mace", "uma"], cache_dir="./data")
    metric = FrechetDistance(mlips=["orb", "mace"], cache_dir="./data")

    stability_preprocessor_result = preprocessor(valid_structures)

    default_args = metric._get_compute_attributes()
    result = metric(stability_preprocessor_result.processed_structures, **default_args)

    # Check computation didn't fail
    assert len(result.failed_indices) == 0

    # Check result structure
    assert "FrechetDistanceMean" in result.metrics
    assert "FrechetDistanceStd" in result.uncertainties
    assert "FrechetDistancesFull" in result.uncertainties

    assert isinstance(result.metrics["FrechetDistanceMean"], float)
    assert isinstance(result.uncertainties["FrechetDistanceStd"], float)
    assert isinstance(result.uncertainties["FrechetDistancesFull"], list)

    # Check values
    values = [val for key, val in result.metrics.items() if "Average" not in key]
    assert not np.any(np.isnan(values))
    for val in values:
        assert val >= 0
