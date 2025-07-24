import numpy as np
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core.structure import Structure

from lematerial_forgebench.utils.oxidation_state import (
    get_inequivalent_site_info,
)


def compute_vendi_score_with_uncertainty(site_number) -> dict[str, float]:
    """
    Compute the Vendi score (effective diversity) from an #of species distribution,
    along with Shannon entropy, variance, and standard deviation.

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - vendi_score: Effective number of categories
        - shannon_entropy: Raw entropy in nats
        - entropy_variance: Estimated variance of entropy (multi-nomial approx.)
        - entropy_std: Standard deviation (sqrt of variance)
    
    References
    ----------
    Friedman, D., & Dieng, A. B. (2023). 
    The Vendi Score: A Diversity Evaluation Metric for Machine Learning. 
    Transactions on Machine Learning Research. https://openreview.net/forum?id=aNVLfhU9pH

    """
    values = np.array(list(site_number.values()), dtype=float)
    total = np.sum(values)

    if total == 0:
        return {
            "vendi_score": 0.0,
            "shannon_entropy": 0.0,
            "entropy_variance": 0.0,
            "entropy_std": 0.0,
        }

    # Normalize to probability distribution
    probs = values / total

    # Shannon entropy (in nats)
    entropy = -np.sum(probs * np.log(probs + 1e-12))  # add epsilon to avoid log(0)

    # Vendi score
    vendi_score = np.exp(entropy)

    # Variance of entropy estimate (asymptotic approximation)
    second_moment = np.sum(probs * (np.log(probs + 1e-12)) ** 2)
    entropy_variance = (1 / total) * (second_moment - entropy ** 2)
    entropy_std = np.sqrt(entropy_variance)

    return {
        "vendi_score": vendi_score,
        "shannon_entropy": entropy,
        "entropy_variance": entropy_variance,
        "entropy_std": entropy_std,
    }

def compute_packing_factor(structure: Structure) -> float:
    """
    Approximate the atomic packing factor (APF) of a structure
    using covalent radii to estimate atomic volumes.

    Parameters
    ----------
    structure : pymatgen Structure
        The crystal structure to analyze.

    Returns
    -------
    float
        Estimated packing factor (0 to ~0.74 typical).
    """
    total_atomic_volume = 0.0
    structure = structure.remove_oxidation_states()
    sites = get_inequivalent_site_info(structure)

    for site_index in range(0, len(sites['sites'])):            
        radius = CovalentRadius().radius[sites["species"][site_index]]
        atom_volume = (4/3) * np.pi * (radius ** 3)
        total_atomic_volume += atom_volume * sites["multiplicities"][site_index]

    packing_factor = total_atomic_volume / structure.volume
    return min(packing_factor, 1.0)  # Clamp to 1.0 max
