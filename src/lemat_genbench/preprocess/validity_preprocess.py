"""Validity preprocessor for crystal structure analysis.

This module implements a preprocessor that adds validity metadata to Pymatgen 
structures by running fundamental validity checks. The preprocessor evaluates
charge neutrality, interatomic distances, and physical plausibility, then
adds comprehensive validity information as properties to the Structure objects.
"""

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict

from pymatgen.core import Structure
from tqdm import tqdm

from lemat_genbench.metrics.validity_metrics import (
    ChargeNeutralityMetric,
    MinimumInteratomicDistanceMetric,
    PhysicalPlausibilityMetric,
)
from lemat_genbench.preprocess.base import (
    BasePreprocessor,
    PreprocessorConfig,
    PreprocessorResult,
)
from lemat_genbench.utils.logging import logger

# Suppress common warnings
warnings.filterwarnings("ignore", message="No oxidation states specified on sites!")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class ValidityPreprocessorConfig(PreprocessorConfig):
    """Configuration for validity preprocessor.
    
    Parameters
    ----------
    name : str
        Name of the preprocessor.
    description : str
        Description of the preprocessor.
    n_jobs : int
        Number of parallel jobs.
    charge_tolerance : float
        Tolerance for charge neutrality deviations.
    charge_strict : bool
        Whether to require determinable oxidation states for all atoms.
    distance_scaling_factor : float
        Factor to scale minimum interatomic distances.
    plausibility_min_density : float
        Minimum plausible density in g/cm³.
    plausibility_max_density : float
        Maximum plausible density in g/cm³.
    plausibility_check_format : bool
        Whether to check crystallographic format validity.
    plausibility_check_symmetry : bool
        Whether to check space group symmetry validity.
    """
    name: str
    description: str
    n_jobs: int
    charge_tolerance: float = 0.1
    charge_strict: bool = False
    distance_scaling_factor: float = 0.5
    plausibility_min_density: float = 0.01
    plausibility_max_density: float = 25.0
    plausibility_check_format: bool = True
    plausibility_check_symmetry: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert the preprocessor configuration to a dictionary for serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the configuration.
        """
        base_dict = super().to_dict()
        base_dict.update({
            "charge_tolerance": self.charge_tolerance,
            "charge_strict": self.charge_strict,
            "distance_scaling_factor": self.distance_scaling_factor,
            "plausibility_min_density": self.plausibility_min_density,
            "plausibility_max_density": self.plausibility_max_density,
            "plausibility_check_format": self.plausibility_check_format,
            "plausibility_check_symmetry": self.plausibility_check_symmetry,
        })
        return base_dict


class ValidityPreprocessor(BasePreprocessor):
    """Preprocessor that adds validity metadata to Pymatgen structures.
    
    This preprocessor runs fundamental validity checks on crystal structures
    and adds comprehensive validity information as properties. The checks include:
    
    1. Charge Neutrality: Ensures structures are electrically neutral
    2. Interatomic Distances: Validates minimum distances between atoms
    3. Physical Plausibility: Checks density, format, and symmetry
    
    The preprocessor adds the following properties to each structure:
    - overall_valid: bool (True if passes ALL validity checks)
    - charge_valid: bool (True if charge neutral within tolerance)
    - charge_deviation: float (absolute deviation from neutrality)
    - distance_valid: bool (True if all interatomic distances are reasonable)
    - distance_score: float (0.0 or 1.0 indicating distance validity)
    - plausibility_valid: bool (True if physically plausible)
    - plausibility_score: float (0.0 or 1.0 indicating plausibility)
    - validity_details: dict (detailed breakdown of validity checks)
    
    Parameters
    ----------
    charge_tolerance : float, default=0.1
        Tolerance for charge neutrality deviations.
    charge_strict : bool, default=False
        Whether to require determinable oxidation states for all atoms.
    distance_scaling_factor : float, default=0.5
        Factor to scale minimum interatomic distances.
    plausibility_min_density : float, default=0.01
        Minimum plausible density in g/cm³.
    plausibility_max_density : float, default=25.0
        Maximum plausible density in g/cm³.
    plausibility_check_format : bool, default=True
        Whether to check crystallographic format validity.
    plausibility_check_symmetry : bool, default=True
        Whether to check space group symmetry validity.
    name : str, optional
        Custom name for the preprocessor.
    description : str, optional
        Description of the preprocessor.
    n_jobs : int, default=1
        Number of parallel jobs to run.
    
    Examples
    --------
    >>> from pymatgen.util.testing import PymatgenTest
    >>> test = PymatgenTest()
    >>> structures = [test.get_structure("Si"), test.get_structure("LiFePO4")]
    >>> 
    >>> preprocessor = ValidityPreprocessor()
    >>> result = preprocessor(structures)
    >>> 
    >>> # Check validity of first structure
    >>> structure = result.processed_structures[0]
    >>> print(f"Overall valid: {structure.properties['overall_valid']}")
    >>> print(f"Charge valid: {structure.properties['charge_valid']}")
    >>> print(f"Distance valid: {structure.properties['distance_valid']}")
    >>> print(f"Plausibility valid: {structure.properties['plausibility_valid']}")
    """
    
    def __init__(
        self,
        charge_tolerance: float = 0.1,
        charge_strict: bool = False,
        distance_scaling_factor: float = 0.5,
        plausibility_min_density: float = 0.01,
        plausibility_max_density: float = 25.0,
        plausibility_check_format: bool = True,
        plausibility_check_symmetry: bool = True,
        name: str = None,
        description: str = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "ValidityPreprocessor",
            description=description or "Adds validity metadata to crystal structures",
            n_jobs=n_jobs,
        )
        
        self.config = ValidityPreprocessorConfig(
            name=self.config.name,
            description=self.config.description,
            n_jobs=self.config.n_jobs,
            charge_tolerance=charge_tolerance,
            charge_strict=charge_strict,
            distance_scaling_factor=distance_scaling_factor,
            plausibility_min_density=plausibility_min_density,
            plausibility_max_density=plausibility_max_density,
            plausibility_check_format=plausibility_check_format,
            plausibility_check_symmetry=plausibility_check_symmetry,
        )
        
        # Initialize validity metrics for configuration - these will be used to get compute attributes
        self.charge_metric = ChargeNeutralityMetric(
            tolerance=charge_tolerance,
            strict=charge_strict,
        )
        self.distance_metric = MinimumInteratomicDistanceMetric(
            scaling_factor=distance_scaling_factor,
        )
        self.plausibility_metric = PhysicalPlausibilityMetric(
            min_density=plausibility_min_density,
            max_density=plausibility_max_density,
            check_format=plausibility_check_format,
            check_symmetry=plausibility_check_symmetry,
        )

    def _get_process_attributes(self) -> Dict[str, Any]:
        """Get additional attributes for the process_structure method.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing the compute attributes for all metrics and configuration.
        """
        return {
            "charge_compute_args": self.charge_metric._get_compute_attributes(),
            "distance_compute_args": self.distance_metric._get_compute_attributes(),
            "plausibility_compute_args": self.plausibility_metric._get_compute_attributes(),
            "charge_tolerance": self.config.charge_tolerance,
        }

    def run(self, structures: list[Structure], structure_sources: list[str] = None) -> PreprocessorResult:
        """Run the validity preprocessor on a list of structures.
        
        This override allows passing source information for ID tracking.
        
        Parameters
        ----------
        structures : list[Structure]
            List of structures to process.
        structure_sources : list[str], optional
            List of source identifiers (filenames, row IDs, etc.) corresponding to each structure.
            If None, will use indices as identifiers.
            
        Returns
        -------
        PreprocessorResult
            Result containing processed structures with validity metadata.
        """
        n_input = len(structures)
        start_time = time.time()
        
        # Prepare source tracking
        if structure_sources is None:
            structure_sources = [f"structure_{i}" for i in range(n_input)]
        elif len(structure_sources) != n_input:
            logger.warning(
                f"Structure sources length ({len(structure_sources)}) != structures length ({n_input}). "
                "Using indices as fallback."
            )
            structure_sources = [f"structure_{i}" for i in range(n_input)]
        
        try:
            # Process structures with ID tracking
            processed_structures = []
            failed_indices = []
            warnings = []
            
            # Get process attributes once
            process_args = self._get_process_attributes()
            
            with tqdm(total=n_input, desc=f"Processing {self.name}") as pbar:
                for i, structure in enumerate(structures):
                    try:
                        # Add index and source to process args
                        process_args_with_id = {
                            **process_args,
                            "structure_index": i,
                            "original_source": structure_sources[i],
                        }
                        
                        processed_structure = self.process_structure(
                            structure, **process_args_with_id
                        )
                        processed_structures.append(processed_structure)
                        
                    except Exception as e:
                        failed_indices.append(i)
                        warnings.append(f"Failed to process structure {i} ({structure_sources[i]}): {str(e)}")
                        logger.debug(
                            f"Failed to process structure {i} for {self.name}",
                            exc_info=True,
                        )
                    
                    pbar.update(1)
        
        except Exception as e:
            logger.error(f"Global failure in preprocessor {self.name}", exc_info=True)
            return PreprocessorResult(
                processed_structures=[],
                config=self.config,
                computation_time=time.time() - start_time,
                n_input_structures=n_input,
                failed_indices=list(range(n_input)),
                warnings=[f"Global preprocessing failure for {self.name}: {str(e)}"] * n_input,
            )
        
        return PreprocessorResult(
            processed_structures=processed_structures,
            config=self.config,
            computation_time=time.time() - start_time,
            n_input_structures=n_input,
            failed_indices=failed_indices,
            warnings=warnings,
        )

    @staticmethod
    def process_structure(
        structure: Structure,
        charge_compute_args: Dict[str, Any],
        distance_compute_args: Dict[str, Any],
        plausibility_compute_args: Dict[str, Any],
        charge_tolerance: float,
        structure_index: int = None,
        original_source: str = None,
        **kwargs: Any,
    ) -> Structure:
        """Process a single structure to add validity metadata.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to process.
        charge_compute_args : Dict[str, Any]
            Compute arguments for the charge neutrality metric.
        distance_compute_args : Dict[str, Any]
            Compute arguments for the interatomic distance metric.
        plausibility_compute_args : Dict[str, Any]
            Compute arguments for the physical plausibility metric.
        charge_tolerance : float
            Tolerance for charge neutrality.
        structure_index : int, optional
            Index of this structure in the original input list.
        original_source : str, optional
            Original source identifier (filename, row ID, etc.).
        **kwargs : Any
            Additional keyword arguments (ignored).

        Returns
        -------
        Structure
            The structure with added validity properties.

        Raises
        ------
        Exception
            If any validity check fails to execute.
        """
        # Create a copy to avoid modifying the original
        processed_structure = structure.copy()
        
        try:
            # Run individual validity checks using class static methods
            charge_deviation = ChargeNeutralityMetric.compute_structure(
                structure, **charge_compute_args
            )
            distance_score = MinimumInteratomicDistanceMetric.compute_structure(
                structure, **distance_compute_args
            )
            plausibility_score = PhysicalPlausibilityMetric.compute_structure(
                structure, **plausibility_compute_args
            )
            
            # Determine validity for each check
            # Note: charge_deviation should be <= tolerance for validity
            charge_valid = charge_deviation <= charge_tolerance
            
            # Distance and plausibility metrics return exactly 1.0 for valid, 0.0 for invalid
            distance_valid = distance_score == 1.0
            plausibility_valid = plausibility_score == 1.0
            
            # Overall validity requires ALL checks to pass
            overall_valid = charge_valid and distance_valid and plausibility_valid
            
            # Add comprehensive validity metadata
            processed_structure.properties.update({
                # Structure identification
                "structure_id": structure_index if structure_index is not None else "unknown",
                "original_source": original_source or f"structure_{structure_index}" if structure_index is not None else "unknown",
                
                # Primary validity flags
                "overall_valid": overall_valid,
                "charge_valid": charge_valid,
                "distance_valid": distance_valid,
                "plausibility_valid": plausibility_valid,
                
                # Detailed scores and deviations
                "charge_deviation": charge_deviation,
                "distance_score": distance_score,
                "plausibility_score": plausibility_score,
                
                # Summary for easy access
                "validity_details": {
                    "charge_neutrality": {
                        "valid": charge_valid,
                        "deviation": charge_deviation,
                        "tolerance": charge_tolerance,
                    },
                    "interatomic_distance": {
                        "valid": distance_valid,
                        "score": distance_score,
                    },
                    "physical_plausibility": {
                        "valid": plausibility_valid,
                        "score": plausibility_score,
                    },
                    "overall": {
                        "valid": overall_valid,
                        "checks_passed": sum([charge_valid, distance_valid, plausibility_valid]),
                        "total_checks": 3,
                    },
                },
                
                # Metadata for tracking
                "validity_preprocessor_version": "1.1.0",  # Updated version
                "validity_timestamp": str(int(time.time())),
            })
            
            logger.debug(
                f"Validity check for {structure.formula} (ID: {structure_index}, Source: {original_source}): "
                f"overall={overall_valid}, charge={charge_valid}, "
                f"distance={distance_valid}, plausibility={plausibility_valid}"
            )
            
            return processed_structure
            
        except Exception as e:
            logger.error(
                f"Failed to process validity for structure {structure.formula}: {str(e)}",
                exc_info=True,
            )
            raise


# Convenience functions for common configurations
def create_strict_validity_preprocessor(n_jobs: int = 1) -> ValidityPreprocessor:
    """Create a strict validity preprocessor with tight tolerances.
    
    Parameters
    ----------
    n_jobs : int, default=1
        Number of parallel jobs to run.
        
    Returns
    -------
    ValidityPreprocessor
        Configured preprocessor with strict validity criteria.
    """
    return ValidityPreprocessor(
        charge_tolerance=0.01,  # Very strict charge neutrality
        charge_strict=True,     # Require determinable oxidation states
        distance_scaling_factor=0.8,  # Stricter distance requirements
        plausibility_check_format=True,
        plausibility_check_symmetry=True,
        name="StrictValidityPreprocessor",
        description="Strict validity preprocessing with tight tolerances",
        n_jobs=n_jobs,
    )


def create_lenient_validity_preprocessor(n_jobs: int = 1) -> ValidityPreprocessor:
    """Create a lenient validity preprocessor with relaxed tolerances.
    
    Parameters
    ----------
    n_jobs : int, default=1
        Number of parallel jobs to run.
        
    Returns
    -------
    ValidityPreprocessor
        Configured preprocessor with relaxed validity criteria.
    """
    return ValidityPreprocessor(
        charge_tolerance=0.5,   # More lenient charge neutrality
        charge_strict=False,    # Don't require determinable oxidation states
        distance_scaling_factor=0.3,  # More lenient distance requirements
        plausibility_check_format=False,  # Skip format checks
        plausibility_check_symmetry=False,  # Skip symmetry checks
        name="LenientValidityPreprocessor",
        description="Lenient validity preprocessing with relaxed tolerances",
        n_jobs=n_jobs,
    )


def create_fast_validity_preprocessor(n_jobs: int = 4) -> ValidityPreprocessor:
    """Create a fast validity preprocessor optimized for speed.
    
    Parameters
    ----------
    n_jobs : int, default=4
        Number of parallel jobs to run.
        
    Returns
    -------
    ValidityPreprocessor
        Configured preprocessor optimized for speed.
    """
    return ValidityPreprocessor(
        charge_tolerance=0.1,
        charge_strict=False,    # Skip expensive oxidation state determination
        distance_scaling_factor=0.5,
        plausibility_check_format=False,  # Skip expensive format checks
        plausibility_check_symmetry=False,  # Skip expensive symmetry checks
        name="FastValidityPreprocessor",
        description="Fast validity preprocessing with minimal expensive checks",
        n_jobs=n_jobs,
    )


def create_validity_preprocessor_from_sources(
    structures: list[Structure], 
    sources: list[str], 
    **kwargs
) -> tuple[ValidityPreprocessor, PreprocessorResult]:
    """Convenience function to create and run validity preprocessor with source tracking.
    
    Parameters
    ----------
    structures : list[Structure]
        List of structures to process.
    sources : list[str]
        List of source identifiers corresponding to each structure.
    **kwargs
        Additional arguments passed to ValidityPreprocessor constructor.
        
    Returns
    -------
    tuple[ValidityPreprocessor, PreprocessorResult]
        The preprocessor instance and the processing result.
    """
    preprocessor = ValidityPreprocessor(**kwargs)
    result = preprocessor.run(structures, structure_sources=sources)
    return preprocessor, result


# Test the preprocessor when run directly
if __name__ == "__main__":
    try:
        from pymatgen.util.testing import MatSciTest as PymatgenTest
    except ImportError:
        from pymatgen.util.testing import PymatgenTest
    
    test = PymatgenTest()
    structures = [test.get_structure("Si"), test.get_structure("LiFePO4")]
    sources = ["test_Si.cif", "test_LiFePO4.cif"]
    
    # Test default preprocessor with source tracking
    print("Testing ValidityPreprocessor with source tracking...")
    preprocessor = ValidityPreprocessor()
    result = preprocessor.run(structures, structure_sources=sources)
    
    print(f"Processed {len(result.processed_structures)} structures")
    print(f"Failed indices: {result.failed_indices}")
    print(f"Computation time: {result.computation_time:.2f}s")
    
    # Display validity results with IDs
    for i, structure in enumerate(result.processed_structures):
        formula = structure.formula
        structure_id = structure.properties.get("structure_id")
        original_source = structure.properties.get("original_source")
        overall_valid = structure.properties.get("overall_valid")
        charge_valid = structure.properties.get("charge_valid")
        distance_valid = structure.properties.get("distance_valid")
        plausibility_valid = structure.properties.get("plausibility_valid")
        
        print(f"\nStructure {i+1} ({formula}):")
        print(f"  ID: {structure_id}")
        print(f"  Source: {original_source}")
        print(f"  Overall valid: {overall_valid}")
        print(f"  Charge valid: {charge_valid}")
        print(f"  Distance valid: {distance_valid}")
        print(f"  Plausibility valid: {plausibility_valid}")
        
        if "validity_details" in structure.properties:
            details = structure.properties["validity_details"]
            print(f"  Charge deviation: {details['charge_neutrality']['deviation']:.3f}")
            print(f"  Distance score: {details['interatomic_distance']['score']:.3f}")
            print(f"  Plausibility score: {details['physical_plausibility']['score']:.3f}")
    
    # Test convenience functions
    print("\n" + "="*50)
    print("Testing convenience functions...")
    
    strict_preprocessor = create_strict_validity_preprocessor()
    print(f"Strict preprocessor: {strict_preprocessor.name}")
    
    lenient_preprocessor = create_lenient_validity_preprocessor()
    print(f"Lenient preprocessor: {lenient_preprocessor.name}")
    
    fast_preprocessor = create_fast_validity_preprocessor()
    print(f"Fast preprocessor: {fast_preprocessor.name}")
    
    # Test convenience function with source tracking
    print("\n" + "="*50)
    print("Testing convenience function with source tracking...")
    preprocessor_conv, result_conv = create_validity_preprocessor_from_sources(
        structures, sources, charge_tolerance=0.05
    )
    print(f"Convenience preprocessor: {preprocessor_conv.name}")
    print(f"Processed {len(result_conv.processed_structures)} structures with sources")