"""Augmented fingerprint preprocessor for crystal structure analysis.

This module implements a preprocessor that adds augmented fingerprint strings
to Pymatgen structures using the new augmented fingerprinting logic. The 
preprocessor extracts crystallographic metadata and generates highly specific
fingerprints based on space group, elements, and Wyckoff position information.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict

from pymatgen.core import Structure

from lemat_genbench.fingerprinting.augmented_fingerprint import (
    AugmentedFingerprinter,
    get_augmented_fingerprint,
)
from lemat_genbench.fingerprinting.crystallographic_analyzer import (
    structure_to_crystallographic_dict,
)
from lemat_genbench.preprocess.base import BasePreprocessor, PreprocessorConfig
from lemat_genbench.utils.logging import logger

# Suppress common warnings
warnings.filterwarnings("ignore", message="No oxidation states specified on sites!")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class AugmentedFingerprintPreprocessorConfig(PreprocessorConfig):
    """Configuration for augmented fingerprint preprocessor.
    
    Parameters
    ----------
    name : str
        Name of the preprocessor.
    description : str
        Description of the preprocessor.
    n_jobs : int
        Number of parallel jobs.
    symprec : float
        Symmetry precision for crystallographic analysis.
    angle_tolerance : float
        Angle tolerance in degrees for crystallographic analysis.
    include_fallback_properties : bool
        Whether to include fallback properties when fingerprinting fails.
    """
    name: str
    description: str
    n_jobs: int
    symprec: float = 0.01
    angle_tolerance: float = 5.0
    include_fallback_properties: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert the preprocessor configuration to a dictionary for serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the configuration.
        """
        base_dict = super().to_dict()
        base_dict.update({
            "symprec": self.symprec,
            "angle_tolerance": self.angle_tolerance,
            "include_fallback_properties": self.include_fallback_properties,
        })
        return base_dict


class AugmentedFingerprintPreprocessor(BasePreprocessor):
    """Preprocessor that adds augmented fingerprints to Pymatgen structures.
    
    This preprocessor uses the augmented fingerprinting approach to generate
    highly specific structure fingerprints based on crystallographic metadata.
    The fingerprints are added as properties to the Structure objects for use
    in downstream analysis and comparison tasks.
    
    Parameters
    ----------
    symprec : float, default=0.01
        Symmetry precision for crystallographic analysis.
    angle_tolerance : float, default=5.0
        Angle tolerance in degrees for crystallographic analysis.
    include_fallback_properties : bool, default=True
        Whether to include basic structural properties when fingerprinting fails.
    name : str, optional
        Custom name for the preprocessor.
    description : str, optional
        Description of what the preprocessor does.
    n_jobs : int, default=1
        Number of parallel jobs to run.
    """

    def __init__(
        self,
        symprec: float = 0.01,
        angle_tolerance: float = 5.0,
        include_fallback_properties: bool = True,
        name: str | None = None,
        description: str | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "AugmentedFingerprintPreprocessor",
            description=description or "Adds augmented fingerprints to crystal structures",
            n_jobs=n_jobs,
        )
        
        self.config = AugmentedFingerprintPreprocessorConfig(
            name=self.config.name,
            description=self.config.description,
            n_jobs=self.config.n_jobs,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            include_fallback_properties=include_fallback_properties,
        )

    def _get_process_attributes(self) -> Dict[str, Any]:
        """Get the attributes for the process_structure method."""
        return {
            "symprec": self.config.symprec,
            "angle_tolerance": self.config.angle_tolerance,
            "include_fallback_properties": self.config.include_fallback_properties,
        }

    @staticmethod
    def process_structure(
        structure: Structure,
        symprec: float = 0.01,
        angle_tolerance: float = 5.0,
        include_fallback_properties: bool = True,
    ) -> Structure:
        """Process a structure by adding augmented fingerprint properties.

        This method extracts crystallographic metadata from the structure and
        generates an augmented fingerprint string. The fingerprint and related
        properties are added to the structure's properties dictionary.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to process.
        symprec : float, default=0.01
            Symmetry precision for crystallographic analysis.
        angle_tolerance : float, default=5.0
            Angle tolerance in degrees for crystallographic analysis.
        include_fallback_properties : bool, default=True
            Whether to include fallback properties when fingerprinting fails.

        Returns
        -------
        Structure
            The processed structure with augmented fingerprint properties added.
        """
        try:
            # Generate augmented fingerprint using the convenience function
            fingerprint = get_augmented_fingerprint(
                structure, 
                symprec=symprec, 
                angle_tolerance=angle_tolerance
            )
            
            # Prepare fingerprint properties
            fingerprint_properties = {
                "augmented_fingerprint": fingerprint,
                "augmented_fingerprint_success": fingerprint is not None,
                "augmented_fingerprint_symprec": symprec,
                "augmented_fingerprint_angle_tolerance": angle_tolerance,
            }
            
            # If fingerprint generation succeeded, extract additional metadata
            if fingerprint is not None:
                try:
                    # Create fingerprinter instance for additional analysis
                    _ = AugmentedFingerprinter(
                        symprec=symprec, 
                        angle_tolerance=angle_tolerance
                    )
                    
                    # Get crystallographic data for additional properties
                    
                    crystal_data = structure_to_crystallographic_dict(
                        structure, 
                        symprec=symprec, 
                        angle_tolerance=angle_tolerance
                    )
                    
                    if crystal_data["success"]:
                        fingerprint_properties.update({
                            "augmented_fingerprint_spacegroup": crystal_data.get("spacegroup_number"),
                            "augmented_fingerprint_elements": crystal_data.get("elements"),
                            "augmented_fingerprint_site_symmetries": crystal_data.get("site_symmetries"),
                            "augmented_fingerprint_multiplicity": crystal_data.get("multiplicity"),
                            "augmented_fingerprint_n_enumerations": len(
                                crystal_data.get("sites_enumeration_augmented", [])
                            ),
                        })
                    
                except Exception as e:
                    logger.debug(f"Failed to extract additional fingerprint metadata: {e}")
                    
            else:
                # Fingerprint generation failed
                logger.debug(f"Augmented fingerprint generation failed for structure: {structure.formula}")
                
                # Add fallback properties if requested
                if include_fallback_properties:
                    try:
                        # Basic structural properties as fallback
                        space_group_info = structure.get_space_group_info()
                        fingerprint_properties.update({
                            "augmented_fingerprint_fallback_spacegroup": space_group_info[1],
                            "augmented_fingerprint_fallback_formula": structure.formula,
                            "augmented_fingerprint_fallback_composition": str(structure.composition),
                        })
                    except Exception as e:
                        logger.debug(f"Failed to extract fallback properties: {e}")
                        
            # Add all fingerprint properties to the structure
            if "augmented_fingerprint_properties" not in structure.properties:
                structure.properties["augmented_fingerprint_properties"] = {}
            
            structure.properties["augmented_fingerprint_properties"].update(fingerprint_properties)
            
            # Also add the main fingerprint as a top-level property for easy access
            structure.properties["augmented_fingerprint"] = fingerprint
            
            return structure
            
        except Exception as e:
            logger.warning(f"Augmented fingerprint preprocessing failed for {structure.formula}: {e}")
            
            # Return structure with error information
            error_properties = {
                "augmented_fingerprint": None,
                "augmented_fingerprint_success": False,
                "augmented_fingerprint_error": str(e),
                "augmented_fingerprint_symprec": symprec,
                "augmented_fingerprint_angle_tolerance": angle_tolerance,
            }
            
            if "augmented_fingerprint_properties" not in structure.properties:
                structure.properties["augmented_fingerprint_properties"] = {}
                
            structure.properties["augmented_fingerprint_properties"].update(error_properties)
            structure.properties["augmented_fingerprint"] = None
            
            return structure


# Factory functions for common configurations
def create_augmented_fingerprint_preprocessor(
    symprec: float = 0.01,
    angle_tolerance: float = 5.0,
    include_fallback_properties: bool = True,
    **kwargs
) -> AugmentedFingerprintPreprocessor:
    """Factory function to create augmented fingerprint preprocessor with common configurations.

    Parameters
    ----------
    symprec : float, default=0.01
        Symmetry precision for crystallographic analysis.
    angle_tolerance : float, default=5.0
        Angle tolerance in degrees for crystallographic analysis.
    include_fallback_properties : bool, default=True
        Whether to include fallback properties when fingerprinting fails.
    **kwargs
        Additional arguments for the preprocessor.

    Returns
    -------
    AugmentedFingerprintPreprocessor
        Configured augmented fingerprint preprocessor.
    """
    return AugmentedFingerprintPreprocessor(
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        include_fallback_properties=include_fallback_properties,
        **kwargs,
    )


def create_high_precision_fingerprint_preprocessor(**kwargs) -> AugmentedFingerprintPreprocessor:
    """Create preprocessor with high precision settings for fingerprinting."""
    return create_augmented_fingerprint_preprocessor(
        symprec=0.001,
        angle_tolerance=1.0,
        **kwargs
    )


def create_robust_fingerprint_preprocessor(**kwargs) -> AugmentedFingerprintPreprocessor:
    """Create preprocessor with robust settings that work well for most structures."""
    return create_augmented_fingerprint_preprocessor(
        symprec=0.1,
        angle_tolerance=10.0,
        include_fallback_properties=True,
        **kwargs
    )