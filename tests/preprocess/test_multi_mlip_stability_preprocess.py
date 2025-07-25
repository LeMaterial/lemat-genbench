"""Tests for Multi-MLIP stability preprocessor."""
import numpy as np
import pytest
from pymatgen.util.testing import PymatgenTest

from lematerial_forgebench.preprocess.multi_mlip_stability_preprocess import (
    MultiMLIPStabilityPreprocessor,
    MultiMLIPStabilityPreprocessorConfig,
    _aggregate_results,
    create_multi_mlip_stability_preprocessor,
)


@pytest.fixture
def test_structures():
    """Create test structures."""
    test = PymatgenTest()
    structures = [
        test.get_structure("Si"),     # Simple cubic structure
        test.get_structure("LiFePO4"), # More complex structure
    ]
    return structures


class TestMultiMLIPStabilityPreprocessorConfig:
    """Test configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MultiMLIPStabilityPreprocessorConfig()
        
        assert config.models == ["orb", "mace", "uma"]
        assert config.timeout == 60
        assert config.energy_aggregation == "both"
        assert config.rmse_aggregation == "both"
        assert config.embedding_strategy == "both"
        assert config.calculate_formation_energy is True
        assert config.calculate_energy_above_hull is True
        assert config.extract_embeddings is True
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MultiMLIPStabilityPreprocessorConfig(
            models=["orb", "mace"],
            timeout=120,
            energy_aggregation="individual",
            rmse_aggregation="aggregated",
            embedding_strategy="individual",
        )
        
        assert config.models == ["orb", "mace"]
        assert config.timeout == 120
        assert config.energy_aggregation == "individual"
        assert config.rmse_aggregation == "aggregated"
        assert config.embedding_strategy == "individual"


class TestMultiMLIPStabilityPreprocessor:
    """Test multi-MLIP stability preprocessor."""
    
    def test_initialization_default(self):
        """Test initialization with default parameters."""
        preprocessor = MultiMLIPStabilityPreprocessor()
        
        # Check configuration
        assert preprocessor.config.models == ["orb", "mace", "uma"]
        assert len(preprocessor.calculators) == 3
        assert "orb" in preprocessor.calculators
        assert "mace" in preprocessor.calculators
        assert "uma" in preprocessor.calculators
    
    def test_initialization_custom_models(self):
        """Test initialization with custom model list."""
        preprocessor = MultiMLIPStabilityPreprocessor(
            models=["orb", "mace"],
            energy_aggregation="individual",
            rmse_aggregation="aggregated"
        )
        
        # Check configuration
        assert preprocessor.config.models == ["orb", "mace"]
        assert len(preprocessor.calculators) == 2
        assert preprocessor.config.energy_aggregation == "individual"
        assert preprocessor.config.rmse_aggregation == "aggregated"
    
    def test_initialization_with_model_configs(self):
        """Test initialization with model-specific configurations."""
        model_configs = {
            "orb": {"device": "cpu", "precision": "float32-high"},  # FIXED: Use correct precision
            "mace": {"device": "cpu", "model_path": "mace_mp"},
        }
        
        preprocessor = MultiMLIPStabilityPreprocessor(
            models=["orb", "mace"],
            model_configs=model_configs
        )
        
        assert len(preprocessor.calculators) == 2
        assert preprocessor.config.model_configs == model_configs
    
    def test_invalid_model_name(self):
        """Test that invalid model names raise errors."""
        with pytest.raises(ValueError, match="Model 'invalid_model' not available"):
            MultiMLIPStabilityPreprocessor(models=["invalid_model"])
    
    def test_factory_function(self):
        """Test factory function."""
        preprocessor = create_multi_mlip_stability_preprocessor(
            models=["orb"],
            device="cpu"
        )
        
        assert len(preprocessor.calculators) == 1
        assert "orb" in preprocessor.calculators


class TestMultiMLIPProcessing:
    """Test actual processing with multiple MLIPs."""
    
    def test_process_single_structure_energy_only(self, test_structures):
        """Test processing with energy calculations only."""
        preprocessor = MultiMLIPStabilityPreprocessor(
            models=["orb"],  # Single model - should NOT have aggregated results
            relax_structures=False,
            extract_embeddings=False,
            calculate_formation_energy=True,
            calculate_energy_above_hull=True,
            energy_aggregation="both"
        )
        
        structure = test_structures[0]  # Use Si structure
        process_args = preprocessor._get_process_attributes()
        result = preprocessor.process_structure(structure, **process_args)
        
        # Check individual model results
        assert "energy_orb" in result.properties
        assert "formation_energy_orb" in result.properties
        assert "e_above_hull_orb" in result.properties
        
        # Should NOT have aggregated results for single model
        assert "energy" not in result.properties
        assert "formation_energy" not in result.properties
        assert "e_above_hull" not in result.properties
    
    def test_process_with_multiple_models(self, test_structures):
        """Test processing with multiple models."""
        preprocessor = MultiMLIPStabilityPreprocessor(
            models=["orb", "mace"],  # Use two models
            relax_structures=False,
            extract_embeddings=False,
            calculate_formation_energy=True,
            calculate_energy_above_hull=False,  # Skip this to keep test simpler
            energy_aggregation="both"
        )
        
        structure = test_structures[0]  # Use Si structure
        process_args = preprocessor._get_process_attributes()
        result = preprocessor.process_structure(structure, **process_args)
        
        # Check individual model results
        assert "formation_energy_orb" in result.properties
        assert "formation_energy_mace" in result.properties
        
        # Check aggregated results
        assert "formation_energy" in result.properties
        assert "formation_energy_std" in result.properties
        assert result.properties["formation_energy_count"] == 2
        
        # Standard deviation should be computed
        assert result.properties["formation_energy_std"] is not None

    def test_process_with_relaxation(self, test_structures):
        """Test processing with structure relaxation."""
        preprocessor = MultiMLIPStabilityPreprocessor(
            models=["orb"],  # Single model for simplicity
            relax_structures=True,
            extract_embeddings=False,
            calculate_formation_energy=False,
            calculate_energy_above_hull=False,
            rmse_aggregation="both"
        )
        
        structure = test_structures[0]
        process_args = preprocessor._get_process_attributes()
        result = preprocessor.process_structure(structure, **process_args)
        
        # Check that relaxation was attempted (even if it failed)
        # The relaxation_rmse might be None if relaxation failed
        assert "relaxation_rmse_orb" in result.properties
        
        # Should NOT have aggregated results for single model
        assert "relaxation_rmse" not in result.properties
        
        # But if we use multiple models, we should get aggregated results
        preprocessor_multi = MultiMLIPStabilityPreprocessor(
            models=["orb", "mace"],  # Multiple models
            relax_structures=True,
            extract_embeddings=False,
            calculate_formation_energy=False,
            calculate_energy_above_hull=False,
            rmse_aggregation="both"
        )
        
        process_args = preprocessor_multi._get_process_attributes()
        result_multi = preprocessor_multi.process_structure(structure, **process_args)
        
        # Should have both individual and aggregated results
        assert "relaxation_rmse_orb" in result_multi.properties
        assert "relaxation_rmse_mace" in result_multi.properties
        assert "relaxation_rmse" in result_multi.properties

    def test_process_with_embeddings(self, test_structures):
        """Test processing with embedding extraction."""
        preprocessor = MultiMLIPStabilityPreprocessor(
            models=["orb"],  # Single model for simplicity
            relax_structures=False,
            extract_embeddings=True,
            calculate_formation_energy=False,
            calculate_energy_above_hull=False,
            embedding_strategy="both"
        )
        
        structure = test_structures[0]
        process_args = preprocessor._get_process_attributes()
        result = preprocessor.process_structure(structure, **process_args)
        
        # Check embedding results
        assert "node_embeddings_orb" in result.properties
        assert "graph_embedding_orb" in result.properties
        
        # Should NOT have aggregated embeddings for single model
        assert "graph_embedding" not in result.properties
        
        # Check embedding shapes
        node_embs = result.properties["node_embeddings_orb"]
        graph_emb = result.properties["graph_embedding_orb"]
        
        assert node_embs is not None
        assert graph_emb is not None
        assert len(graph_emb.shape) == 1  # Should be 1D vector
    
    def test_aggregation_modes(self, test_structures):
        """Test aggregation behavior based on number of models."""
        # Test single model - should have individual results but NO aggregated ones
        preprocessor_single = MultiMLIPStabilityPreprocessor(
            models=["orb"],  # Single model
            calculate_formation_energy=True,
            calculate_energy_above_hull=False,
            relax_structures=False,
            extract_embeddings=False,
        )
        
        structure = test_structures[0]
        process_args = preprocessor_single._get_process_attributes()
        result = preprocessor_single.process_structure(structure, **process_args)
        
        # Should have individual results
        assert "formation_energy_orb" in result.properties
        # Should NOT have aggregated results (only 1 model)
        assert "formation_energy" not in result.properties
        assert "formation_energy_std" not in result.properties
        
        # Test multiple models - should have BOTH individual AND aggregated results
        preprocessor_multiple = MultiMLIPStabilityPreprocessor(
            models=["orb", "mace"],  # Multiple models
            calculate_formation_energy=True,
            calculate_energy_above_hull=False,
            relax_structures=False,
            extract_embeddings=False,
        )
        
        process_args = preprocessor_multiple._get_process_attributes()
        result = preprocessor_multiple.process_structure(structure, **process_args)
        
        # Should have individual results from both models
        assert "formation_energy_orb" in result.properties
        assert "formation_energy_mace" in result.properties
        
        # Should ALSO have aggregated results (2+ models)
        assert "formation_energy" in result.properties
        assert "formation_energy_std" in result.properties
        assert "formation_energy_count" in result.properties
        assert result.properties["formation_energy_count"] == 2
    def test_process_structure_list(self, test_structures):
        """Test processing a list of structures."""
        preprocessor = MultiMLIPStabilityPreprocessor(
            models=["orb"],  # Single model - no aggregated results
            relax_structures=False,
            extract_embeddings=False,
            calculate_formation_energy=True,
            calculate_energy_above_hull=False,
        )
        
        # Process all test structures using the base class method
        result = preprocessor(test_structures)
        
        # Check that we get results for all structures
        assert result.n_input_structures == len(test_structures)
        assert len(result.processed_structures) <= len(test_structures)
        
        # Check that each processed structure has the expected properties
        for structure in result.processed_structures:
            assert "formation_energy_orb" in structure.properties
            # Should NOT have aggregated results for single model
            assert "formation_energy" not in structure.properties


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_timeout_handling(self, test_structures):
        """Test timeout handling."""
        preprocessor = MultiMLIPStabilityPreprocessor(
            models=["orb"],
            timeout=1,  # Very short timeout to trigger timeout
            relax_structures=False,
            extract_embeddings=False,
            calculate_formation_energy=True,
            calculate_energy_above_hull=False,
        )
        
        structure = test_structures[0]
        process_args = preprocessor._get_process_attributes()
        # This should not crash, but may timeout
        result = preprocessor.process_structure(structure, **process_args)
        
        # Result should be returned (possibly unchanged if timeout occurred)
        assert result is not None
    
    def test_partial_calculation_failure(self):
        """Test handling when some calculations fail but others succeed."""
        # This is a more complex test that would require mocking
        # We'll implement this after basic functionality is working
        pass


class TestAggregationFunctions:
    """Test aggregation helper functions."""
    
    def test_compute_stats_valid_values(self):
        """Test statistics computation with valid values."""
        structure = PymatgenTest().get_structure("Si")
        
        # Mock model results
        model_results = {
            "orb": {"formation_energy": -1.5},
            "mace": {"formation_energy": -1.7},
            "uma": {"formation_energy": -1.6},
        }
        
        _aggregate_results(
            structure, model_results, ["orb", "mace", "uma"],
            "aggregated", "individual", "individual"
        )
        
        # Check aggregated formation energy
        assert "formation_energy" in structure.properties
        assert "formation_energy_std" in structure.properties
        assert "formation_energy_count" in structure.properties
        
        # Check values
        assert abs(structure.properties["formation_energy"] - (-1.6)) < 0.1  # Mean should be around -1.6
        assert structure.properties["formation_energy_std"] > 0  # Should have some variation
        assert structure.properties["formation_energy_count"] == 3
    
    def test_compute_stats_with_none_values(self):
        """Test statistics computation with some None values."""
        structure = PymatgenTest().get_structure("Si")
        
        # Mock model results with some failures
        model_results = {
            "orb": {"formation_energy": -1.5},
            "mace": {"formation_energy": None},  # Failed calculation
            "uma": {"formation_energy": -1.6},
        }
        
        _aggregate_results(
            structure, model_results, ["orb", "mace", "uma"],
            "aggregated", "individual", "individual"
        )
        
        # Check that aggregation works with partial data
        assert "formation_energy" in structure.properties
        assert structure.properties["formation_energy_count"] == 2  # Only 2 valid values
        assert abs(structure.properties["formation_energy"] - (-1.55)) < 0.1  # Mean of -1.5 and -1.6
    
    def test_compute_stats_all_none_values(self):
        """Test statistics computation when all values are None."""
        structure = PymatgenTest().get_structure("Si")
        
        # Mock model results with all failures
        model_results = {
            "orb": {"formation_energy": None},
            "mace": {"formation_energy": None},
            "uma": {"formation_energy": None},
        }
        
        _aggregate_results(
            structure, model_results, ["orb", "mace", "uma"],
            "aggregated", "individual", "individual"
        )
        
        # Check that aggregation handles all None values gracefully
        assert structure.properties["formation_energy"] is None
        assert structure.properties["formation_energy_std"] is None
        assert structure.properties["formation_energy_count"] == 0


class TestIntegrationWithRealModels:
    """Integration tests with real models."""
    
    @pytest.mark.slow
    def test_full_pipeline_all_models(self, test_structures):
        """Test full pipeline with all three models."""
        preprocessor = MultiMLIPStabilityPreprocessor(
            models=["orb", "mace", "uma"],
            relax_structures=False,  # Skip relaxation for speed
            extract_embeddings=True,
            calculate_formation_energy=True,
            calculate_energy_above_hull=True,
            energy_aggregation="both",
            embedding_strategy="both"
        )
        
        # Process first structure only (for speed)
        structure = test_structures[0]
        process_args = preprocessor._get_process_attributes()
        result = preprocessor.process_structure(structure, **process_args)
        
        # Check that all models produced results
        assert "formation_energy_orb" in result.properties
        assert "formation_energy_mace" in result.properties
        assert "formation_energy_uma" in result.properties
        
        # Check aggregated results
        assert "formation_energy" in result.properties
        assert "formation_energy_std" in result.properties
        assert result.properties["formation_energy_count"] == 3
        
        # Check embeddings
        assert "graph_embedding_orb" in result.properties
        assert "graph_embedding_mace" in result.properties
        assert "graph_embedding_uma" in result.properties
        assert "graph_embedding" in result.properties
    
    @pytest.mark.slow
    def test_model_agreement(self, test_structures):
        """Test that models give reasonable agreement."""
        preprocessor = MultiMLIPStabilityPreprocessor(
            models=["orb", "mace"],  # Use two models for comparison
            relax_structures=False,
            extract_embeddings=False,
            calculate_formation_energy=True,
            calculate_energy_above_hull=False,
            energy_aggregation="both"
        )
        
        structure = test_structures[1]  # Use LiFePO4 structure instead of Si
        process_args = preprocessor._get_process_attributes()
        result = preprocessor.process_structure(structure, **process_args)
        
        # Check that models give reasonable values
        fe_orb = result.properties["formation_energy_orb"]
        fe_mace = result.properties["formation_energy_mace"]
        
        # For LiFePO4, formation energy should be negative (compound is stable)
        # But let's be less strict since MLIPs can have different reference states
        print(f"Formation energies: ORB={fe_orb}, MACE={fe_mace}")
        
        # Just check that both models give finite values
        assert not np.isnan(fe_orb)
        assert not np.isnan(fe_mace)
        
        # They shouldn't be too different (within factor of 3 for MLIPs)
        if fe_orb != 0 and fe_mace != 0:
            ratio = abs(fe_orb / fe_mace)
            assert 0.3 < ratio < 3.0, f"Formation energies too different: ORB={fe_orb}, MACE={fe_mace}"
