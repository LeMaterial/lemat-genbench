import numpy as np


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
