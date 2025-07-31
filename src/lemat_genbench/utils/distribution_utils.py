import json

import numpy as np
from pymatgen.core import Element
from scipy import linalg
from scipy.spatial.distance import cdist, jensenshannon


def safe_float(value):
    """Currently a no-op function.

    This is a placeholder for a function that will safely convert a value to a 
    float, handling None and NaN.
    """
    return value


def map_space_group_to_crystal_system(space_group: int):
    if space_group <= 2 and space_group > 0:
        return 1  # "triclinic"
    elif space_group <= 15 and space_group > 2:
        return 2  # "monoclinic"
    elif space_group <= 74 and space_group > 15:
        return 3  # "orthorhombic"
    elif space_group <= 142 and space_group > 74:
        return 4  # "tetragonal"
    elif space_group <= 167 and space_group > 142:
        return 5  # "trigonal"
    elif space_group <= 194 and space_group > 167:
        return 6  # "hexagonal"
    elif space_group <= 230 and space_group > 194:
        return 7  # "cubic"
    else:
        raise ValueError


def one_hot_encode_composition(composition):
    one_hot_counts = np.zeros(118)
    one_hot_bool = np.zeros(118)
    for element in composition.elements:
        one_hot_bool[int(Element(element).number) - 1] = 1
        one_hot_counts[int(Element(element).number) - 1] = composition.as_dict()[
            element.as_dict()["element"]
        ]
    return [one_hot_counts, one_hot_bool]


def generate_probabilities(df, metric, metric_type=np.int64, return_2d_array=False):
    # create an empty list of space groups/crystal systems/compositions and fill in proportions/counts
    # depending on the application (as some samples will have zero of space group 1 etc)

    if metric_type == np.int64:
        if metric == "SpaceGroup":
            prob_dict = {}
            for i in range(0, 230):
                prob_dict[str(i + 1)] = 0
        if metric == "CrystalSystem":
            prob_dict = {}
            for i in range(0, 7):
                prob_dict[str(i + 1)] = 0

        probs = np.asarray(df.value_counts(metric) / len(df))
        indices = np.asarray(df.value_counts(metric).index)
        strut_list = np.concatenate(([indices], [probs]), axis=0).T
        strut_list = strut_list[strut_list[:, 0].argsort()]
        if return_2d_array:
            return strut_list

        for row in strut_list:
            prob_dict[str(int(row[0]))] = row[1]

    if metric_type == np.ndarray:
        prob_dict = {}
        for i in range(0, 118):
            prob_dict[str(i + 1)] = 0

        one_hots = np.zeros(118)
        for i in range(0, len(df)):
            one_hots += df.iloc[i][metric]

        one_hots = one_hots / sum(one_hots)

        for i in range(0, 118):
            prob_dict[str(i + 1)] = one_hots[i]

    return prob_dict


def compute_shannon_entropy(probability_vals):
    H = 0
    for i in range(len(probability_vals)):
        val = probability_vals[i]
        if val < 10**-14:
            pass
        else:
            H += val * np.log(val)
    H = -H
    return H


def compute_jensen_shannon_distance(
    reference_data, generated_crystals, crystal_param, metric_type
):
    """
    reference_data - letmatbulk dataframe
    generated_crystals - dataframe of generated crystals
    crystal_param - CrystalSystem, SpaceGroup, or LatticeConstant
    """
    generated_crystals_dist = generate_probabilities(
        generated_crystals, metric=crystal_param, metric_type=metric_type
    )
    if crystal_param not in ["CompositionCounts", "Composition"]:
        reference_data_dist = generate_probabilities(
            reference_data, metric=crystal_param, metric_type=metric_type
        )

    elif crystal_param == "CompositionCounts":
        with open("data/lematbulk_composition_counts_distribution.json", "r") as file:
            reference_data_dist = json.load(file)
    elif crystal_param == "Composition":
        with open("data/lematbulk_composition_distribution.json", "r") as file:
            reference_data_dist = json.load(file)
    gen_vals = np.array(list(generated_crystals_dist.values()))
    ref_vals = np.array(list(reference_data_dist.values()))

    return jensenshannon(gen_vals, ref_vals)


def gaussian_kernel(x, y, sigma=1.0):
    pairwise_dists = cdist(x, y, "sqeuclidean")
    return np.exp(-pairwise_dists / (2 * sigma**2))


def compute_mmd(reference_data, generated_crystals, crystal_param, sigma=1.0):
    generated_crystals_dist = np.atleast_2d(
        generated_crystals[crystal_param].to_numpy()
    ).T
    reference_data_dist = np.atleast_2d(reference_data[crystal_param].to_numpy()).T

    k_xx = gaussian_kernel(generated_crystals_dist, generated_crystals_dist, sigma)
    k_yy = gaussian_kernel(reference_data_dist, reference_data_dist, sigma)
    k_xy = gaussian_kernel(generated_crystals_dist, reference_data_dist, sigma)

    mmd = np.mean(k_xx) + np.mean(k_yy) - 2 * np.mean(k_xy)
    return mmd

def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Implemented from https://github.com/bioinf-jku/FCD/blob/master/fcd/utils.py
    
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1:     The mean of the activations of preultimate layer of the Graph 
                embedding of the MLIP for the first distribution. 
    -- mu2:     The mean of the activations of preultimate layer of the Graph 
                embedding of the MLIP for the second distribution.
    -- sigma1:  The covariance matrix of the activations of of the Graph 
                embedding of the MLIP for the first distribution.
    -- sigma2:  The covariance matrix of the activations of of the Graph 
                embedding of the MLIP for the first distribution.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    is_real = np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3)

    if not np.isfinite(covmean).all() or not is_real:
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    assert isinstance(covmean, np.ndarray)
    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def compute_frechetdist(reference_data, generated_crystals):

    X = np.stack(reference_data, axis = 0)
    mu1 = np.mean(X, axis = 0) # mean generated distribution 

    Y = np.stack(generated_crystals, axis = 0)
    mu2 = np.mean(Y, axis = 0) # mean reference distribution

    sigma1 = np.cov(X, rowvar = False) # covariance matrix generated distribution 
    sigma2 = np.cov(Y, rowvar = False) # covariance matrix reference distribution


    distance = frechet_distance(mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2)
    return distance
