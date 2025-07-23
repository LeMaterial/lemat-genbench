"""
Diversity: distribution of space group, element species, lattice params, etc. And potential comparison with
training set. Ensuring the generative process explores a wide range of chemical and structural space, rather
than collapsing to a few known motifs.

Metrics include: 
(i) Comparing distributions of elements, stoichiometry,
space groups, or lattice parameters between generated and training/reference sets[Kazeev et al., 2025]. 

(ii) Calculating pairwise similarities (e.g., using structure matchers or adapted molecular metrics like Tanimoto
similarity[Gubina et al., 2024]) within the generated set (internal diversity) and between generated and
known structures (novelty/external diversity).

(iii) Techniques like Density and Coverage[Naeem et al.,2020], originally from image generation, aim to 
separately quantify sample fidelity and the extent to which the generated distribution covers the true
distribution. The Fréchet Wrenformer Distance (FWD) has been proposed specifically for crystals to account
 for symmetry[Kelvinius et al., 2025].

(iv) Algorithms and metrics like the moqd-score explicitly reward finding diverse sets of
high-performing solutions across different feature dimensions (e.g., different conductivity
or deformation resistance values)[Janmohamed et al., 2024

"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from pymatgen.core.structure import Structure
from scipy.special import rel_entr

from lematerial_forgebench.metrics import BaseMetric
from lematerial_forgebench.metrics.base import MetricConfig
from lematerial_forgebench.utils.diversity_utils import (
    compute_packing_factor,
    compute_vendi_score_with_uncertainty,
)
from lematerial_forgebench.utils.logging import logger

"""
-------------------------------------------------------------------------------
Elemental Diversity
-------------------------------------------------------------------------------
"""

@dataclass
class ElementComponentConfig(MetricConfig):
    """Configuration for the Elemental Diversity metric.

    This configuration extends the base MetricConfig to include
    weights for evaluating Elemental components of structural diversity.

    Parameters
    ----------
    reference_element_space : int
        The number of Reference Elements to compute Coverage across. Set to 118 as a default
    """

    reference_element_space:int = 118

class ElementDiversityMetric(BaseMetric):
    """
    Calculates a scalar score capturing elemental diversity across the structures compared to uniform distribution
    """
    def __init__(
        self,
        name: str | None = "Element Diversity",
        description: str | None = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
        reference_element_space = 118,
    ):
        super().__init__(
            name=name or "Element Diversity",
            description=description 
            or "Scalar Score of Elemental Diversity in Generated Set",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )
        self.config = ElementComponentConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            reference_element_space=reference_element_space,
            )
        
        # Initialize element Histogram
        self._init_element_histogram()
    
    def _init_element_histogram(self):
        """
        Initialize an empty dictionary to function as a histogram counter for element.
        Dictionary mapping is Element -> total count across all structure
        Note: Before compute, Dictionary Values are normalized 
        """

        self.element_histogram = defaultdict(int)


    def _get_compute_attributes(self) -> dict[str, Any]:
        return {
            "elemental_histogram" : self.element_histogram
        }

    @staticmethod
    def compute_structure(structure: Structure, elemental_histogram: Dict[str, int]) -> float:
        """
        Retrieves all elements present in Structure and adds count to internal elemental distribution
        Parameters
        ----------
        structure: Structure
            A pymatgen Structure object to evaluate.
        elemental_histogram: dict[str:int]
            Class variable for storing the current histogram/distribution of elements across all structures

        Returns:
        -------
        float:
            This value serves as a Binary Indicator representing if the structure was successfully evaluated or not. 

        """
        try:
            all_sites_in_structure = structure.sites
            for site in all_sites_in_structure:
                atom = site.species_string
                elemental_histogram[atom] += 1
            return 0.0
    
        except Exception as e:
            logger.debug(f"Could not determine Elements in {structure.formula} : {str(e)}")
            return 1.0

    def aggregate_results(self, values: list[float]) -> dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            1 or 0 indicators showcasing if the compute function was able to parse through a structure
            - values of 1 represent erroneous calculations and is used for debugging only

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        invalid_computations = sum(values)
        elemental_diversity_metric = compute_vendi_score_with_uncertainty(self.element_histogram)
        coverage_ratio = len(self.element_histogram.keys()) / self.config.reference_element_space

        return {
            "metrics": {
                "element_diversity_vendi_score": elemental_diversity_metric["vendi_score"],
                "element_diversity_shannon_entropy": elemental_diversity_metric["shannon_entropy"],
                "invalid_computations_in_batch":invalid_computations,
                "element_coverage_ratio" : coverage_ratio
            },
            "primary_metric": "element_coverage_ratio",
            "uncertainties": {
                "shannon_entropy_std" : elemental_diversity_metric["entropy_std"],
                "shannon_entropy_variance": elemental_diversity_metric["entropy_variance"],
            }
        }

"""
-------------------------------------------------------------------------------
SpaceGroup Diversity
-------------------------------------------------------------------------------
"""

@dataclass
class SpaceGroupComponentConfig(MetricConfig):
    """Configuration for the DiversityScore metric.

    This configuration extends the base MetricConfig to include
    weights for evaluating different components of structural diversity.
    
    Parameters
    ----------
    reference_space_group_space : int
        The number of Reference Elements to compute Coverage across. Set to 230 as a default
    """
    reference_space_group_space: int = 230

class SpaceGroupDiversityMetric(BaseMetric):
    """
    Calculates a scalar score capturing Spacegroup Diversity across the structures
    """

    def __init__(
        self,
        name: str | None = "Space Group Diversity",
        description: str | None = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
        reference_space_group_space: int = 230

    ):
        super().__init__(
            name=name or "Space Group Diversity",
            description=description 
            or "Scalar Score of Space Group Diversity in Generated Set",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )
        self.config = SpaceGroupComponentConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            reference_space_group_space=reference_space_group_space
            )
        
        self._init_spacegroup_histogram()

    def _init_spacegroup_histogram(self):
        """
        Initialize an empty dictionary to function as a histogram counter for element.
        Dictionary mapping is Element -> total count across all structure
        Note: Before compute, Dictionary Values are normalized 
        """

        self.spacegroup_histogram = defaultdict(int)
    

    def _get_compute_attributes(self) -> dict[str, Any]:
        return {
            "spacegroup_histogram" : self.spacegroup_histogram
        }
    
    @staticmethod
    def compute_structure(structure: Structure, spacegroup_histogram: Dict[str, int]) -> float:
        """
        Retrieves all elements present in Structure and adds count to internal elemental distribution
        Parameters
        ----------
        structure: Structure
            A pymatgen Structure object to evaluate.
        elemental_histogram: dict[str:int]
            Class variable for storing the current histogram/distribution of elements across all structures

        Returns:
        -------
        float:
            This value serves as a Binary Indicator representing if the structure was successfully evaluated or not. 

        """
        try:
            spacegroup_symbol, spacegroup_number = structure.get_space_group_info()
            spacegroup_histogram[spacegroup_symbol] += 1
            return spacegroup_number/230
    
        except Exception as e:
            logger.debug(f"Could not determine Spacegroup in {structure.formula} : {str(e)}")
            return 0
        
    def aggregate_results(self, values: list[float]) -> dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            the structure-wise symmetry score computed based off spacegroup number / 230. 
            This normalized score gives a sense of low vs high symmetry

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        mean_symmetry_rating = np.mean(values)
        spacegroup_diversity_metric = compute_vendi_score_with_uncertainty(self.spacegroup_histogram)
        space_group_coverage = len(self.spacegroup_histogram.keys()) / self.config.reference_space_group_space

        return {
            "metrics": {
                "spacegroup_diversity_vendi_score": spacegroup_diversity_metric["vendi_score"],
                "spacegroup_diversity_shannon_entropy": spacegroup_diversity_metric["shannon_entropy"],
                "mean_symmetry_rating":mean_symmetry_rating,
                "space_group_coverage": space_group_coverage
            },
            "primary_metric": "space_group_coverage",
            "uncertainties": {
                "shannon_entropy_std" : spacegroup_diversity_metric["entropy_std"],
                "shannon_entropy_variance": spacegroup_diversity_metric["entropy_variance"],
            }
        }

        
"""
-------------------------------------------------------------------------------
Density Diversity
-------------------------------------------------------------------------------
"""

@dataclass
class PhysicalSizeComponentConfig(MetricConfig):
    """Configuration for the Density, Volume, and Lattice Parameter Size sub-component of Diversity metric.

    This configuration extends the base MetricConfig to include
    weights for evaluating different components of structural diversity.
    Parameters
    ----------
        density_bin_size : Float
            Float value for bin size and effective precision for density histogram
        lattice_bin_size : Float
            Float value for bin size and effective precision for lattice histograms
        packing_factor_bin_size : Float
            Float value for bin size and effective precision for packing factor histogram

    """
    density_bin_size: float = 0.5
    lattice_bin_size: float = 0.5
    packing_factor_bin_size: float = 0.05

class PhysicalSizeComponentMetric(BaseMetric):
    """
    Calculates a scalar score capturing Physical Aspects of the Structure to measure Diversity across the structures
    """

    def __init__(
        self,
        name: str | None = "Physical Diversity",
        description: str | None = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
        density_bin_size:float = 0.5,
        lattice_bin_size:float = 0.5,
        packing_factor_bin_size:float = 0.05,
    ):
        super().__init__(
            name=name or "Physical Diversity",
            description=description 
            or "Scalar Score of Physical Diversity in Generated Set",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )
        self.config = PhysicalSizeComponentConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            density_bin_size= density_bin_size,
            lattice_bin_size=lattice_bin_size,
            packing_factor_bin_size=packing_factor_bin_size,
            )
        
        # Initializing Generated Dataset Histograms 
        self._init_density_histogram(
            smallest_density_bucket=0.5,
            largest_density_bucket=25.0,
            )

        self._init_lattice_param_histogram(
            smallest_lattice_size=1.0,
            largest_lattice_size=100.0,
        )

        self._init_packing_factor_histogram(
            smallest_packing_factor=0.0,
            largest_packing_factor=1.0
        )
        
        # Initializing Reference Dataset Histograms 
        self._init_reference_density_histogram(
            smallest_density_bucket=0.5,
            largest_density_bucket=25.0,
        )
        self._init_reference_lattice_params_histogram(
            smallest_lattice_size=1.0,
            largest_lattice_size=100.0,
        )
        self._init_packing_factor_histogram(
            smallest_packing_factor=0.0,
            largest_packing_factor=1.0,
        )


    def _init_density_histogram(
            self,
            smallest_density_bucket:float = 0.5,
            largest_density_bucket:float = 25.0,
            ):
        """
        Instantiates a bucket-based histogram capturing diversity and corresponding frequency
        of materials in the generated set.
        Parameters
        ----------
        smallest_density_bucket: Float
            Float value for the smallest sized bucket to capture
        largest_density_bucket: Float
            Float value for the largest sized bucket to capture
        """

        bins = np.arange(smallest_density_bucket, largest_density_bucket, self.config.density_bin_size )
        bucket_dict = {lower_ceiling :0 for lower_ceiling in bins }
        self.density_histogram = bucket_dict
    
    def _init_lattice_param_histogram(
            self,
            smallest_lattice_size:float = 0.0,
            largest_lattice_size:float = 50.0,
            ):
        """
        Instantiates a bucket-based histogram capturing diversity and corresponding frequency
        of materials in the generated set.
        Parameters
        ----------
        smallest_lattice_size: Float
            Float value for the smallest sized bucket to capture
        largest_lattice_size: Float
            Float value for the largest sized bucket to capture
        """

        bins = np.arange(smallest_lattice_size, largest_lattice_size, self.config.lattice_bin_size )
        bucket_dict = {lower_ceiling :0 for lower_ceiling in bins }
        self.lattice_a_histogram = bucket_dict
        self.lattice_b_histogram = bucket_dict
        self.lattice_c_histogram = bucket_dict

    def _init_packing_factor_histogram(
            self,
            smallest_packing_factor:float = 0.0,
            largest_packing_factor:float = 1.0,
            ):
        """
        Instantiates a bucket-based histogram capturing diversity and corresponding frequency
        of materials in the generated set.
        Parameters
        ----------
        smallest_density_bucket: Float
            Float value for the smallest sized bucket to capture
        largest_density_bucket: Float
            Float value for the largest sized bucket to capture
        """

        bins = np.arange(smallest_packing_factor, largest_packing_factor, self.config.packing_factor_bin_size )
        bucket_dict = {lower_ceiling :0 for lower_ceiling in bins }
        self.packing_factor_histogram = bucket_dict

    def _init_reference_density_histogram(
            self,
            smallest_density_bucket:float = 0.5,
            largest_density_bucket:float = 25.0,
            ):
        """
        Instantiates a bucket-based histogram capturing diversity and corresponding frequency
        of materials in the generated set.
        Parameters
        ----------
        smallest_density_bucket: Float
            Float value for the smallest sized bucket to capture
        largest_density_bucket: Float
            Float value for the largest sized bucket to capture
        """

        bins = np.arange(smallest_density_bucket, largest_density_bucket, self.config.density_bin_size )
        bucket_dict = {lower_ceiling :1 for lower_ceiling in bins } # setting this as a uniform dist 
        self.reference_density_histogram = bucket_dict

    def _init_reference_lattice_params_histogram(
            self,
            smallest_lattice_size:float = 0.0,
            largest_lattice_size:float = 50.0,
            ):
        """
        Instantiates a bucket-based histogram capturing diversity and corresponding frequency
        of materials in the generated set.
        Parameters
        ----------
        smallest_density_bucket: Float
            Float value for the smallest sized bucket to capture
        largest_density_bucket: Float
            Float value for the largest sized bucket to capture
        """

        bins = np.arange(smallest_lattice_size, largest_lattice_size, self.config.density_bin_size )
        bucket_dict = {lower_ceiling:1 for lower_ceiling in bins } # setting as a uniform dist
        self.reference_lattice_a = bucket_dict
        self.reference_lattice_b = bucket_dict
        self.reference_lattice_c = bucket_dict

    def _init_reference_packing_factor_histogram(
            self,
            smallest_packing_factor:float = 0.0,
            largest_packing_factor:float = 1.0,
            ):
        """
        Instantiates a bucket-based histogram capturing diversity and corresponding frequency
        of materials in the generated set.
        Parameters
        ----------
        smallest_density_bucket: Float
            Float value for the smallest sized bucket to capture
        largest_density_bucket: Float
            Float value for the largest sized bucket to capture
        """

        bins = np.arange(smallest_packing_factor, largest_packing_factor, self.config.packing_factor_bin_size )
        bucket_dict = {lower_ceiling:1 for lower_ceiling in bins }
        self.reference_packing_factor = bucket_dict

    def _compute_diversity_with_kl(  
        self,
        actual_histogram,
        reference_histogram
    ) -> dict[str, float]:
        """
        Compute Shannon entropy and KL divergence between the actual 
        distribution of the current dataset and a uniform distribution.

        Assumes both distributions are defined over the same bins, and are class objects

        Returns
        -------
        dict[str, float]
            Dictionary containing:
            - shannon_entropy: Entropy of the current dataset
            - entropy_variance: Multinomial variance estimate of entropy
            - entropy_std: Standard deviation of entropy
            - kl_divergence: D_KL(P || Q) from current to reference

        References
        ----------
        - Shannon, C. E. (1948). 
        *A Mathematical Theory of Communication*.  
        [https://ieeexplore.ieee.org/document/6773024](https://ieeexplore.ieee.org/document/6773024)

        - Kullback, S., & Leibler, R. A. (1951). 
        *On Information and Sufficiency*.  
        [https://projecteuclid.org/euclid.aoms/1177729694](https://projecteuclid.org/euclid.aoms/1177729694)
        """
        values = np.array(list(actual_histogram.values()), dtype=float)
        ref_values = np.array([reference_histogram.get(k, 1) for k in actual_histogram], dtype=float)

        total = np.sum(values)
        ref_total = np.sum(ref_values)

        if total == 0 or ref_total == 0:
            return {
                "shannon_entropy": 0.0,
                "entropy_variance": 0.0,
                "entropy_std": 0.0,
                "kl_divergence": 0.0,
            }

        p = values / total
        q = ref_values / ref_total

        # Add small epsilon to avoid log(0)
        epsilon = 1e-12
        p = np.clip(p, epsilon, 1)
        q = np.clip(q, epsilon, 1)

        entropy = -np.sum(p * np.log(p))
        second_moment = np.sum(p * (np.log(p)) ** 2)
        entropy_variance = (1 / total) * (second_moment - entropy**2)
        entropy_std = np.sqrt(entropy_variance)

        kl_divergence = np.sum(rel_entr(p, q))  # D_KL(P || Q)

        return {
            "shannon_entropy": entropy,
            "entropy_variance": entropy_variance,
            "entropy_std": entropy_std,
            "kl_divergence": kl_divergence,
        }


    def _get_compute_attributes(self) -> dict[str, Any]:
        return {
            "packing_factor_function": compute_packing_factor,
            "density_bin_size": self.config.density_bin_size,
            "lattice_bin_size": self.config.lattice_bin_size,
            "packing_factor_bin_size": self.config.packing_factor_bin_size,
        }
    
    @staticmethod
    def compute_structure(
        structure: Structure,
        packing_factor_function: callable,
        density_bin_size: float,
        lattice_bin_size: float,
        packing_factor_bin_size: float,

        ) -> float:
        """
        Retrieves all elements present in Structure and adds count to internal elemental distribution
        Parameters
        ----------
        structure: Structure
            A pymatgen Structure object to evaluate.
        packing_factor_function: Callable
            A private function to calculate the packing factor of a structure to then save to the distribution
        density_bin_size : Float
            Float value for bin size and effective precision for density histogram
        lattice_bin_size : Float
            Float value for bin size and effective precision for lattice histograms
        packing_factor_bin_size : Float
            Float value for bin size and effective precision for packing factor histogram
        Returns:
        -------
        float:
            This value is the Volume of the Cell Calculated by A x B x C 

        """
        return_dict = defaultdict(lambda: defaultdict(int)) 
        # try:
        # Capture Density
        density = structure.density
        density_bin_index = int(density // density_bin_size)
        return_dict['density_histogram'][density_bin_index] += 1

        # Capture Lattice A
        lattice_a = structure.lattice.a
        lattice_a_bin_index = int(lattice_a // lattice_bin_size)
        return_dict['lattice_a_histogram'][lattice_a_bin_index] += 1

        # Capture Lattice B
        lattice_b = structure.lattice.b
        lattice_b_bin_index = int(lattice_b // lattice_bin_size)
        return_dict['lattice_b_histogram'][lattice_b_bin_index] += 1

        # Capture Lattice A
        lattice_c = structure.lattice.c
        lattice_c_bin_index = int(lattice_c // lattice_bin_size)
        return_dict['lattice_c_histogram'][lattice_c_bin_index] += 1

        # Capture Packing Factor
        packing_factor = packing_factor_function(structure)
        packing_factor_bin_index = int(packing_factor // packing_factor_bin_size)
        return_dict['packing_factor_histogram'][packing_factor_bin_index] += 1

        return_dict['volume'][0] +=  lattice_a * lattice_b * lattice_c # returning the Volume
        return_dict["counts"][0] += 1 
        return return_dict
        
        # except Exception as e:
        #     logger.debug(f"Could not determine Physical Properties of {structure.formula} : {str(e)}")
        #     return defaultdict(lambda: defaultdict(int)) 
     
    def aggregate_results(self, values: list[float]) -> dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            Binary indicator to indicate if the density is correctly  the value was correctly 

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """

        # Calculate the Mean Volume of the Generated Set
        values_dict = defaultdict(lambda: defaultdict(int))

        for d in values:
            for outer_key, subdict in d.items():
                for inner_key, value in subdict.items():
                    values_dict[outer_key][inner_key] += value

        # Convert to regular nested dicts
        values_dict = {k: dict(v) for k, v in values_dict.items()}

        mean_volume = values_dict["volume"][0]/values_dict["counts"][0]

        density_metrics = self._compute_diversity_with_kl(
            actual_histogram=values_dict["density_histogram"],
            reference_histogram= self.reference_density_histogram
            )
        lattice_a_metrics = self._compute_diversity_with_kl(
            actual_histogram=values_dict["lattice_a_histogram"],
            reference_histogram= self.reference_lattice_a
            )
        lattice_b_metrics = self._compute_diversity_with_kl(
            actual_histogram=values_dict["lattice_b_histogram"],
            reference_histogram= self.reference_lattice_b
            )
        lattice_c_metrics = self._compute_diversity_with_kl(
            actual_histogram=values_dict["lattice_c_histogram"],
            reference_histogram= self.reference_lattice_c
            )
        packing_factor_metrics = self._compute_diversity_with_kl(
            actual_histogram=values_dict["packing_factor_histogram"],
            reference_histogram=self.reference_packing_factor
        )
        
        norm_sum_shannon_entropy = (
            density_metrics["shannon_entropy"] + 
            lattice_a_metrics["shannon_entropy"] + 
            lattice_b_metrics["shannon_entropy"] + 
            lattice_c_metrics["shannon_entropy"] + 
            packing_factor_metrics["shannon_entropy"] 
            )/np.log2(len(values))
        
        avg_norm_shannon_entropy = norm_sum_shannon_entropy / 5


        print()

        return {
            "metrics": {
                "density_diversity_shannon_entropy": density_metrics["shannon_entropy"],
                "density_diversity_kl_divergence_from_uniform": density_metrics["kl_divergence"],
                "lattice_a_diversity_shannon_entropy": lattice_a_metrics["shannon_entropy"],
                "lattice_a_diversity_kl_divergence_from_uniform": lattice_a_metrics["kl_divergence"],
                "lattice_b_diversity_shannon_entropy": lattice_b_metrics["shannon_entropy"],
                "lattice_b_diversity_kl_divergence_from_uniform": lattice_b_metrics["kl_divergence"],
                "lattice_c_diversity_shannon_entropy": lattice_c_metrics["shannon_entropy"],
                "lattice_c_diversity_kl_divergence_from_uniform": lattice_c_metrics["kl_divergence"],
                "packing_factor_diversity_shannon_entropy": packing_factor_metrics["shannon_entropy"],
                "packing_factor_diversity_kl_divergence_from_uniform": packing_factor_metrics["kl_divergence"],
                "mean_volume": mean_volume,
                "avg_norm_shannon_entropy": avg_norm_shannon_entropy,
            },
            "primary_metric": "avg_norm_shannon_entropy",
            "uncertainties": {
                "density_shannon_entropy_std" : density_metrics["entropy_std"],
                "density_shannon_entropy_variance": density_metrics["entropy_variance"],
                "lattice_a_shannon_entropy_std" : lattice_a_metrics["entropy_std"],
                "lattice_a_shannon_entropy_variance": lattice_a_metrics["entropy_variance"],
                "lattice_b_shannon_entropy_std" : lattice_b_metrics["entropy_std"],
                "lattice_b_shannon_entropy_variance": lattice_b_metrics["entropy_variance"],
                "lattice_c_shannon_entropy_std" : lattice_c_metrics["entropy_std"],
                "lattice_c_shannon_entropy_variance": lattice_c_metrics["entropy_variance"],
                "packing_factor_shannon_entropy_std" : packing_factor_metrics["entropy_std"],
                "packing_factor_shannon_entropy_variance": packing_factor_metrics["entropy_variance"],
            }
        }
    
    
"""
-------------------------------------------------------------------------------
Atom Number Diversity
-------------------------------------------------------------------------------
"""

#TODO: Update Tests
@dataclass
class SiteNumberComponentConfig(MetricConfig):
    """Configuration for the Number of Sites Diversity metric.

    This configuration extends the base MetricConfig to include
    weights for evaluating Elemental components of structural diversity.
    """

class SiteNumberComponentMetric(BaseMetric):
    """
    Calculates a scalar score capturing Number of Sites diversity across the structures compared to uniform distribution
    """

    # TODO should this be number of symmetrically unique sites? or does that just become a 
    # proxy of Space Group etc? 

    def __init__(
        self,
        name: str | None = "Site Number Diversity",
        description: str | None = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "Site Number Diversity",
            description=description 
            or "Scalar Score of Site Number Diversity in Generated Set",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )
        self.config = SiteNumberComponentConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            )
        
        # Initialize site Histogram
        self._init_site_histogram()
    
    def _init_site_histogram(self):
        """
        Initialize an empty dictionary to function as a histogram counter for number of sites.
        Dictionary mapping is Number of Sites in unit Cell -> total count across all structure
        Note: Before compute, Dictionary Values are normalized 
        """

        self.site_number = defaultdict(int)

    
    def _get_compute_attributes(self) -> dict[str, Any]:
        return {
            "site_number_histogram" : self.site_number
        }

    @staticmethod
    def compute_structure(structure: Structure, site_number_histogram: Dict[int, int]) -> float:
        """
        Retrieves all elements present in Structure and adds count to internal elemental distribution
        Parameters
        ----------
        structure: Structure
            A pymatgen Structure object to evaluate.
        site_number_histogram: dict[str:int]
            Class variable for storing the current histogram/distribution of elements across all structures

        Returns:
        -------
        float:
            This value serves as a Binary Indicator representing if the structure was successfully evaluated or not. 

        """
        try:
            number_of_sites_in_structure = len(structure.sites)
            site_number_histogram[number_of_sites_in_structure] += 1

            return number_of_sites_in_structure
    
        except Exception as e:
            logger.debug(f"Could not determine number of sites in {structure.formula} : {str(e)}")
            return 0.0

    def aggregate_results(self, values: list[float]) -> dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            1 or 0 indicators showcasing if the compute function was able to parse through a structure
            - values of 1 represent erroneous calculations and is used for debugging only

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        non_zero_values = [v for v in values if v != 0.0]
        if non_zero_values:
            mean_atoms_per_structure = np.mean(values)
        else:
            mean_atoms_per_structure = 0.0
        site_diversity = compute_vendi_score_with_uncertainty(self.site_number)

        return {
            "metrics": {
                "site_number_diversity_vendi_score": site_diversity["vendi_score"],
                "site_number_diversity_shannon_entropy": site_diversity["shannon_entropy"],
                "mean_atoms_per_structure" : mean_atoms_per_structure,
            },
            "primary_metric": "site_number_diversity_vendi_score",
            "uncertainties": {
                "shannon_entropy_std" : site_diversity["entropy_std"],
                "shannon_entropy_variance": site_diversity["entropy_variance"],
            }
        }


if __name__ == "__main__":
    from pymatgen.util.testing import PymatgenTest

    test = PymatgenTest()

    structures = [
        test.get_structure("Si"),
        test.get_structure("Si"),
        test.get_structure("Si"),

        test.get_structure("LiFePO4"),
    ]

    metric = ElementDiversityMetric()
    metric_result = metric(structures)
    print("ElementDiversityMetric")
    print(metric_result)
    print("")

    metric = SpaceGroupDiversityMetric()
    metric_result = metric(structures)
    print("SpaceGroupDiversityMetric")
    print(metric_result)
    print("")

    metric = PhysicalSizeComponentMetric()
    metric._init_reference_packing_factor_histogram()
    metric_result = metric(structures)
    print("PhysicalSizeComponentMetric")
    print(metric_result.metrics)
    print("")

    metric = SiteNumberComponentMetric()
    metric_result = metric(structures)
    print("SiteNumberComponentMetric")
    print(metric_result)
    print("")
