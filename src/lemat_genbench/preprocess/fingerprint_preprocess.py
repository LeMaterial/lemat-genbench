import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List

from pymatgen.core import Structure
from tqdm import tqdm

from lemat_genbench.fingerprinting.utils import get_fingerprinter
from lemat_genbench.preprocess.base import (
    BasePreprocessor,
    PreprocessorConfig,
    PreprocessorResult,
)
from lemat_genbench.utils.logging import logger

warnings.filterwarnings("ignore", message="No oxidation states specified on sites!")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class FingerprintPreprocessorConfig(PreprocessorConfig):
    """Configuration for fingerprint preprocessor."""

    fingerprint_method: str = "short-bawl"

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({"fingerprint_method": self.fingerprint_method})
        return base_dict


class FingerprintPreprocessor(BasePreprocessor):
    """Preprocessor that adds a fingerprint (short-bawl) to each structure."""

    def __init__(
        self,
        fingerprint_method: str = "short-bawl",
        name: str = None,
        description: str = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "FingerprintPreprocessor",
            description=description
            or "Adds short-bawl fingerprint to crystal structures",
            n_jobs=n_jobs,
        )

        self.config = FingerprintPreprocessorConfig(
            name=self.config.name,
            description=self.config.description,
            n_jobs=self.config.n_jobs,
            fingerprint_method=fingerprint_method,
        )

        try:
            self.fingerprinter = get_fingerprinter(fingerprint_method)
        except ValueError:
            self.fingerprinter = None

    def process_structure(
        self, structure: Structure, structure_id: int, source: str
    ) -> Structure:
        """Process a single structure by adding a fingerprint to it.

        Args:
            structure: The structure to process
            structure_id: The ID of the structure
            source: The source identifier for the structure

        Returns:
            Structure: The processed structure with added fingerprint
        """
        if self.fingerprinter is None:
            return structure

        from lemat_genbench.fingerprinting.utils import get_fingerprint

        processed_structure = structure.copy()

        # Check if fingerprint is already in properties
        if "fingerprint" not in processed_structure.properties:
            fingerprint = get_fingerprint(processed_structure, self.fingerprinter)
            processed_structure.properties["fingerprint"] = fingerprint

        processed_structure.properties["structure_id"] = structure_id
        processed_structure.properties["original_source"] = source
        return processed_structure

    def run(
        self, structures: List[Structure], structure_sources: List[str] = None
    ) -> PreprocessorResult:
        n_input = len(structures)
        start_time = time.time()

        if self.fingerprinter is None:
            return PreprocessorResult(
                processed_structures=structures,
                config=self.config,
                computation_time=time.time() - start_time,
                n_input_structures=n_input,
                failed_indices=[],
                warnings=[],
            )

        if structure_sources is None:
            structure_sources = [f"structure_{i}" for i in range(n_input)]
        elif len(structure_sources) != n_input:
            logger.warning(
                f"Structure sources length ({len(structure_sources)}) != structures length ({n_input}). Using indices as fallback."
            )
            structure_sources = [f"structure_{i}" for i in range(n_input)]
        processed_structures = []
        failed_indices = []
        warnings_list = []
        with tqdm(total=n_input, desc=f"Processing {self.name}") as pbar:
            for i, structure in enumerate(structures):
                try:
                    processed_structure = self.process_structure(
                        structure=structure, structure_id=i, source=structure_sources[i]
                    )
                    processed_structures.append(processed_structure)
                except Exception as e:
                    failed_indices.append(i)
                    warnings_list.append(
                        f"Failed to compute fingerprint for structure {i} ({structure_sources[i]}): {str(e)}"
                    )
                    logger.debug(
                        f"Failed to compute fingerprint for structure {i}",
                        exc_info=True,
                    )
                pbar.update(1)
        return PreprocessorResult(
            processed_structures=processed_structures,
            config=self.config,
            computation_time=time.time() - start_time,
            n_input_structures=n_input,
            failed_indices=failed_indices,
            warnings=warnings_list,
        )


# Example usage
if __name__ == "__main__":
    try:
        from pymatgen.util.testing import MatSciTest as PymatgenTest
    except ImportError:
        from pymatgen.util.testing import PymatgenTest
    test = PymatgenTest()
    breakpoint()
    structures = [test.get_structure("Si"), test.get_structure("LiFePO4")]
    sources = ["test_Si.cif", "test_LiFePO4.cif"]
    preprocessor = FingerprintPreprocessor()
    result = preprocessor.run(structures, structure_sources=sources)
    print(f"Processed {len(result.processed_structures)} structures")
    print(f"Failed indices: {result.failed_indices}")
    print(f"Computation time: {result.computation_time:.2f}s")
    for i, structure in enumerate(result.processed_structures):
        print(
            f"Structure {i + 1} ({structure.formula}): Fingerprint: {structure.properties.get('fingerprint')}"
        )
