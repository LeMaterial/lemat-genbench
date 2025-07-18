"""Tests for validity metrics implementation."""

import pickle

import numpy as np
import pytest
from pymatgen.util.testing import PymatgenTest

from lematerial_forgebench.metrics.distribution_metrics import (
    MMD,
    FrechetDistance,
    JSDistance,
)
from lematerial_forgebench.preprocess.distribution_preprocess import (
    DistributionPreprocessor,
)
from lematerial_forgebench.preprocess.universal_stability_preprocess import (
    UniversalStabilityPreprocessor,
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

# @pytest.fixture
# def reference_data_embeddings():
#     "create reference dataset for embedding matching"
#     output_dfs = {}
#     for mlip in ["orb", "mace", "uma", "equiformer"]:
#         try:
#             with open("data/test_small_lematbulk/"+mlip+"_full_embedding_df.pkl", "rb") as f:
#                 sample_embeddings_df = pickle.load(f)
#             output_dfs[mlip] = sample_embeddings_df
#         except FileNotFoundError:
#             pass

#     return output_dfs


def test_JSDistance_metric(valid_structures, reference_data):
    """Test JSDistance_metric on valid structures."""
    distribution_preprocessor = DistributionPreprocessor()
    preprocessor_result = distribution_preprocessor(valid_structures)

    metric = JSDistance(reference_data)
    result = metric.compute(
        preprocessor_result.processed_structures, reference_df=reference_data
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


def test_MMD_metric(valid_structures, reference_data):
    """Test MMD_metric on valid structures."""
    distribution_preprocessor = DistributionPreprocessor()
    preprocessor_result = distribution_preprocessor(valid_structures)

    metric = MMD(reference_data)
    result = metric.compute(
        preprocessor_result.processed_structures, reference_df=reference_data
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

def test_FrechetDistance_metric(valid_structures, reference_data):
    """Test MMD_metric on valid structures."""

    for mlip in ["orb", "mace", "uma", "equiformer"]:
        try: 
            timeout = 60 # seconds to timeout for each MLIP run 
            stability_preprocessor = UniversalStabilityPreprocessor(
                model_name=mlip,
                timeout=timeout,
                relax_structures=False,
            )

            stability_preprocessor_result = stability_preprocessor(valid_structures)
            metric = FrechetDistance(reference_df=reference_data)
            default_args = metric._get_compute_attributes()
            result = metric.compute(
                stability_preprocessor_result.processed_structures, **default_args
                    )
            # Check computation didn't fail
            assert len(result.failed_indices) == 0

            # Check result structure
            assert "FrechetDistance" in result.metrics

            # Check values
            values = [val for key, val in result.metrics.items() if "Average" not in key]
            assert not np.any(np.isnan(values))
            for val in values:
                assert val >= 0

        
        except (ImportError, ValueError):
            pass
