from dataclasses import dataclass

from pymatgen.core import Structure

from lematerial_forgebench.preprocess.base import BasePreprocessor, PreprocessorConfig
from lematerial_forgebench.utils.distribution_utils import (
    map_space_group_to_crystal_system,
    one_hot_encode_composition,
)


@dataclass
class DistributionPreprocessorConfig(PreprocessorConfig):
    name: str
    description: str
    n_jobs: int


class DistributionPreprocessor(BasePreprocessor):
    def __init__(
        self,
        distribution_metric: str = "all",
        crystal_descriptor: str = "SpaceGroup",
        name: str | None = None,
        description: str | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "DistributionPreprocessor",
            description=description
            or "Preprocesses structures for distribution analysis",
            n_jobs=n_jobs,
        )
        self.config = DistributionPreprocessorConfig(
            name=self.config.name,
            description=self.config.description,
            n_jobs=self.config.n_jobs,
        )

    @staticmethod
    def process_structure(
        structure: Structure,
    ) -> Structure:
        """Process a sample of structures by turning them into a dataframe
        built of labeled structural properties for comparison to other distributions.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to process.


        Returns
        -------
        Structure
            The processed structure with the distribution properties added to the properties dictionary.

        """

        one_hot_vectors = one_hot_encode_composition(structure.composition)

        distribution_properties = {
            "Volume": structure.volume,
            "Density(g/cm^3)": structure.density,
            "Density(atoms/A^3)": structure.num_sites / structure.volume,
            "SpaceGroup": structure.get_space_group_info()[1],
            "CrystalSystem": map_space_group_to_crystal_system(
                structure.get_space_group_info()[1]
            ),
            "CompositionCounts": one_hot_vectors[0],
            "Composition": one_hot_vectors[1],
        }

        structure.properties["distribution_properties"] = distribution_properties

        return structure
