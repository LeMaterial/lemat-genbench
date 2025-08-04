import json
from pathlib import Path

import numpy as np

# REPRODUCIBLE MMD SAMPLING:
# For MMD computation, we use a fixed 15K sample from the full 5.3M LeMat-Bulk dataset.
# Sample indices are pre-computed (seed=42) and stored in data/lematbulk_mmd_sample_indices_15k.npy
# This ensures consistent, reproducible results across all MMD computations while maintaining
# statistical validity and optimal performance (1.7GB memory, ~4-5s computation time).
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
    """Map space group number to crystal system integer.
    
    Parameters
    ----------
    space_group : int
        Space group number (1-230)
    
    Returns
    -------
    int
        Crystal system integer (1-7)
        
    Crystal System Mapping:
    ----------------------
    1 = Triclinic      (space groups 1-2)
    2 = Monoclinic     (space groups 3-15)  
    3 = Orthorhombic   (space groups 16-74)
    4 = Tetragonal     (space groups 75-142)
    5 = Trigonal       (space groups 143-167)
    6 = Hexagonal      (space groups 168-194)
    7 = Cubic          (space groups 195-230)
    """
    if space_group <= 2 and space_group > 0:
        return 1  # Triclinic
    elif space_group <= 15 and space_group > 2:
        return 2  # Monoclinic
    elif space_group <= 74 and space_group > 15:
        return 3  # Orthorhombic
    elif space_group <= 142 and space_group > 74:
        return 4  # Tetragonal
    elif space_group <= 167 and space_group > 142:
        return 5  # Trigonal
    elif space_group <= 194 and space_group > 167:
        return 6  # Hexagonal
    elif space_group <= 230 and space_group > 194:
        return 7  # Cubic
    else:
        raise ValueError(f"Invalid space group number: {space_group}. Must be 1-230.")


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
    generated_crystals, crystal_param, metric_type, reference_distributions_file="data/lematbulk_jsdistance_distributions.json"
):
    """
    Compute Jensen-Shannon distance using pre-computed reference distributions.
    
    Parameters
    ----------
    generated_crystals : DataFrame
        Dataframe of generated crystals with distribution properties
    crystal_param : str
        Property name (CrystalSystem, SpaceGroup, CompositionCounts, Composition)
    metric_type : type
        Data type for the metric (np.int64, np.ndarray, etc.)
    reference_distributions_file : str
        Path to JSON file containing pre-computed reference distributions
        
    Returns
    -------
    float
        Jensen-Shannon distance between generated and reference distributions
    """
    # Generate distribution for the input crystals
    generated_crystals_dist = generate_probabilities(
        generated_crystals, metric=crystal_param, metric_type=metric_type
    )
    
    # Load pre-computed reference distributions
    try:
        with open(reference_distributions_file, "r") as file:
            all_reference_distributions = json.load(file)
        
        if crystal_param not in all_reference_distributions:
            raise ValueError(f"Property '{crystal_param}' not found in reference distributions")
            
        reference_data_dist = all_reference_distributions[crystal_param]
        
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Reference distributions file not found: {reference_distributions_file}. "
            f"Please run the distribution reference statistics computation first."
        )
    
    # Convert to numpy arrays for JS distance calculation
    gen_vals = np.array(list(generated_crystals_dist.values()))
    ref_vals = np.array(list(reference_data_dist.values()))

    return jensenshannon(gen_vals, ref_vals)


def gaussian_kernel(x, y, sigma=1.0):
    pairwise_dists = cdist(x, y, "sqeuclidean")
    return np.exp(-pairwise_dists / (2 * sigma**2))


def compute_mmd(generated_crystals, crystal_param, reference_values_file="data/lematbulk_mmd_values.pkl", sigma=1.0, max_reference_samples=15000):
    """
    Compute MMD using pre-computed reference values with reproducible sampling.
    
    Uses a fixed 15K sample (created with seed=42) from the full 5.3M LeMat-Bulk dataset
    for reproducible and efficient MMD computation. The sample indices are stored in
    data/lematbulk_mmd_sample_indices_15k.npy for consistency across runs.
    
    Parameters
    ----------
    generated_crystals : DataFrame
        Dataframe of generated crystals with distribution properties
    crystal_param : str
        Property name (Volume, Density(g/cm^3), Density(atoms/A^3))
    reference_values_file : str
        Path to pickle file containing pre-computed reference values
    sigma : float
        Gaussian kernel bandwidth parameter
    max_reference_samples : int
        Maximum number of reference samples to use. Default 15,000 uses the
        pre-computed reproducible sample for optimal accuracy and performance.
        
    Returns
    -------
    float
        Maximum Mean Discrepancy between generated and reference distributions
        
    Notes
    -----
    For reproducibility, this function uses a fixed 15K sample (indices stored in
    data/lematbulk_mmd_sample_indices_15k.npy) created with numpy seed=42 from the
    full 5,335,299 LeMat-Bulk samples. This ensures consistent results across runs
    while maintaining statistical validity.
    """
    import pickle
    
    # Extract generated values
    generated_crystals_dist = np.atleast_2d(
        generated_crystals[crystal_param].to_numpy()
    ).T
    
    # Load pre-computed reference values
    try:
        with open(reference_values_file, "rb") as file:
            all_reference_values = pickle.load(file)
        
        if crystal_param not in all_reference_values:
            raise ValueError(f"Property '{crystal_param}' not found in reference values")
            
        reference_values = all_reference_values[crystal_param]
        
        # Use reproducible sampling for default 15K samples
        if len(reference_values) > max_reference_samples and max_reference_samples == 15000:
            try:
                # Load pre-computed reproducible sample indices (created with seed=42)
                sample_indices_file = "data/lematbulk_mmd_sample_indices_15k.npy"
                ref_indices = np.load(sample_indices_file)
                reference_values = reference_values[ref_indices]
            except FileNotFoundError:
                # Fallback to random sampling if indices file not found
                np.random.seed(42)  # Use same seed as pre-computed indices
                ref_indices = np.random.choice(len(reference_values), max_reference_samples, replace=False)
                reference_values = reference_values[ref_indices]
        elif len(reference_values) > max_reference_samples:
            # For non-default sample sizes, use random sampling
            np.random.seed(42)  # For reproducibility
            ref_indices = np.random.choice(len(reference_values), max_reference_samples, replace=False)
            reference_values = reference_values[ref_indices]
        
        reference_data_dist = np.atleast_2d(reference_values).T
        
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Reference values file not found: {reference_values_file}. "
            f"Please run the distribution reference statistics computation first."
        )

    # Compute MMD using Gaussian kernels
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




def compute_reference_stats_direct(
    dataset_name="LeMaterial/LeMat-GenBench-embeddings",
    model_names=None,
    cache_dir=None
):
    """
    Compute reference statistics for each model using direct computation.
    
    This function loads the full dataset and computes mean and covariance matrices
    for each model's embeddings using memory-efficient processing for large models.
    
    Parameters
    ----------
    dataset_name : str
        HuggingFace dataset name
    model_names : list[str], optional
        List of model names to compute stats for. Defaults to ["mace", "orb", "uma"]
    cache_dir : str, optional
        Directory to save computed statistics
        
    Returns
    -------
    dict
        Dictionary with model_name -> {"mu": array, "sigma": array}
    """
    try:
        import gc
        import os
        import time

        import numpy as np
        import psutil
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError as e:
        raise ImportError(f"Required library not found: {e}. Install with: uv add datasets psutil")
    
    if model_names is None:
        model_names = ["mace", "orb", "uma"]
    
    print("🧮 Computing reference statistics using direct computation")
    print(f"📡 Dataset: {dataset_name}")
    print(f"🎯 Models: {model_names}")
    
    # Load dataset (disable multiprocessing to avoid semaphore leaks)
    print("📊 Loading complete dataset...")
    dataset = load_dataset(dataset_name, split="train", streaming=False, num_proc=1)
    total_samples = len(dataset)
    print(f"📊 Total samples in dataset: {total_samples:,}")
    
    # Memory monitoring
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
    print(f"💾 Initial memory usage: {initial_memory:.1f} GB")
    
    # First pass: determine embedding dimensions
    print("🔍 Determining embedding dimensions from first sample...")
    first_sample = dataset[0]
    embedding_dims = {}
    for model in model_names:
        key = f"{model}_embeddings"
        if key in first_sample and first_sample[key] is not None:
            embedding_dims[model] = len(first_sample[key])
            print(f"   📐 {model.upper()}: {embedding_dims[model]} dimensions")
        else:
            raise ValueError(f"Missing {key} in first sample")
    
    # Pre-allocate numpy arrays (memory efficient)
    print("🔄 Pre-allocating numpy arrays...")
    embeddings = {}
    for model in model_names:
        dims = embedding_dims[model]
        size_gb = total_samples * dims * 4 / 1024 / 1024 / 1024  # float32 = 4 bytes
        print(f"   📋 {model.upper()}: allocating {total_samples:,} × {dims} = {size_gb:.1f} GB")
        
        embeddings[model] = np.empty((total_samples, dims), dtype=np.float32)
    
    print("✅ Arrays pre-allocated! Now filling with data...")
    start_time = time.time()
    
    valid_samples = 0
    skipped_samples = 0
    
    for i, sample in enumerate(tqdm(dataset, desc="Filling arrays")):
        try:
            sample_valid = True
            
            # Extract and validate embeddings for each model
            sample_embeddings = {}
            for model in model_names:
                key = f"{model}_embeddings"
                if key in sample and sample[key] is not None:
                    emb = sample[key]
                    
                    # Convert to numpy array if it's a list
                    if isinstance(emb, list):
                        emb = np.array(emb, dtype=np.float32)
                    
                    # Check for NaN/Inf
                    if np.isnan(emb).any() or np.isinf(emb).any():
                        sample_valid = False
                        break
                    
                    # Check dimensions match
                    if len(emb) != embedding_dims[model]:
                        sample_valid = False
                        break
                    
                    sample_embeddings[model] = emb
                else:
                    sample_valid = False
                    break
            
            if sample_valid and len(sample_embeddings) == len(model_names):
                # Directly fill pre-allocated arrays (no memory growth!)
                for model in model_names:
                    embeddings[model][valid_samples] = sample_embeddings[model]
                valid_samples += 1
            else:
                skipped_samples += 1
                
        except Exception as e:
            print(f"⚠️  Error processing sample {i}: {e}")
            skipped_samples += 1
        
        # Periodic memory monitoring and cleanup
        if (i + 1) % 100000 == 0:
            gc.collect()  # Force garbage collection
            current_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
            if (i + 1) % 500000 == 0:  # Print less frequently
                print(f"   💾 Memory at {i+1:,} samples: {current_memory:.1f} GB")
    
    load_time = time.time() - start_time
    
    # Final garbage collection
    gc.collect()
    final_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
    
    print("✅ Loading complete!")
    print(f"   ⏱️  Time: {load_time:.1f} seconds")
    print(f"   ✅ Valid samples: {valid_samples:,}")
    print(f"   ⚠️  Skipped samples: {skipped_samples:,}")
    print(f"   💾 Final memory usage: {final_memory:.1f} GB")
    
    # Trim arrays to actual valid samples (if some were skipped)
    if valid_samples < total_samples:
        print(f"🔧 Trimming arrays from {total_samples:,} to {valid_samples:,} samples...")
        for model in model_names:
            embeddings[model] = embeddings[model][:valid_samples]
    
    # Compute statistics for each model
    results = {}
    model_order = ['uma', 'orb', 'mace']  # Process from smallest to largest
    
    for model in model_order:
        if model not in model_names:
            continue
            
        print(f"\n🔄 Processing {model.upper()}...")
        print(f"   📊 Input shape: {embeddings[model].shape}")
        print(f"   💾 Data size: {embeddings[model].nbytes/1024/1024/1024:.1f} GB")
        
        # Memory monitoring
        _ = process.memory_info().rss / 1024 / 1024 / 1024
        gc.collect()
        
        model_start_time = time.time()
        
        if model == 'mace':
            # Use memory-efficient computation for MACE
            print("   🔧 Using memory-efficient chunked computation...")
            
            # Compute mean in chunks
            print("   🧮 Computing mean in chunks...")
            chunk_size = 25000
            n_samples, n_dims = embeddings[model].shape
            n_chunks = (n_samples + chunk_size - 1) // chunk_size
            
            mean_accumulator = np.zeros(n_dims, dtype=np.float64)
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n_samples)
                chunk = embeddings[model][start_idx:end_idx]
                mean_accumulator += np.sum(chunk, axis=0, dtype=np.float64)
                if i % 20 == 0:
                    gc.collect()
            
            mu_direct = (mean_accumulator / n_samples).astype(np.float32)
            del mean_accumulator
            gc.collect()
            
            # Compute covariance in chunks
            print("   🧮 Computing covariance in chunks...")
            cov_accumulator = np.zeros((n_dims, n_dims), dtype=np.float64)
            
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n_samples)
                chunk = embeddings[model][start_idx:end_idx]
                chunk_centered = chunk - mu_direct
                cov_accumulator += np.dot(chunk_centered.T, chunk_centered).astype(np.float64)
                del chunk, chunk_centered
                if i % 20 == 0:
                    gc.collect()
            
            sigma_direct = (cov_accumulator / (n_samples - 1)).astype(np.float32)
            del cov_accumulator
            gc.collect()
            
        else:
            # Standard computation for smaller models
            print("   🧮 Computing mean...")
            mu_direct = np.mean(embeddings[model], axis=0)
            
            gc.collect()
            
            print("   🧮 Computing covariance matrix...")
            sigma_direct = np.cov(embeddings[model], rowvar=False)
        
        model_time = time.time() - model_start_time
        post_memory = process.memory_info().rss / 1024 / 1024 / 1024
        
        results[model] = {"mu": mu_direct, "sigma": sigma_direct}
        
        print(f"   ✅ Results: mu {mu_direct.shape}, sigma {sigma_direct.shape}")
        print(f"   ⏱️  Computation time: {model_time:.1f} seconds")
        print(f"   💾 Memory after computation: {post_memory:.1f} GB")
        
        # Clear embeddings from memory after each model (except last)
        if model != model_order[-1] or model not in model_names[-1:]:
            print(f"   🧹 Clearing {model.upper()} embeddings from memory...")
            del embeddings[model]
            gc.collect()
    
    # Save to cache if specified
    if cache_dir:
        print("\n💾 Saving results to cache...")
        save_reference_stats_cache(results, cache_dir)
    
    return results


def save_reference_stats_cache(stats_dict, cache_dir, dataset_version="LeMat-GenBench-5335K-samples"):
    """Save computed reference statistics to cache directory."""
    import time
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "dataset_version": dataset_version,
        "models": list(stats_dict.keys()),
        "computed_at": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "method": "direct_computation"
    }
    
    for model_name, stats in stats_dict.items():
        # Save mu and sigma as separate files
        mu_path = cache_path / f"{model_name}_mu.npy"
        sigma_path = cache_path / f"{model_name}_sigma.npy"
        
        np.save(mu_path, stats["mu"])
        np.save(sigma_path, stats["sigma"])
        
        print(f"Saved {model_name} stats: mu={mu_path}, sigma={sigma_path}")
    
    # Save metadata
    metadata_path = cache_path / "reference_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Cache saved to {cache_path}")
    print(f"Metadata saved to: {metadata_path}")


def load_reference_stats_cache(cache_dir, model_names=None):
    """Load cached reference statistics."""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        return None
    
    # Try new metadata file name first, then fall back to old one
    metadata_path = cache_path / "reference_metadata.json"
    if not metadata_path.exists():
        metadata_path = cache_path / "metadata.json"
        if not metadata_path.exists():
            return None
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    if model_names is None:
        model_names = metadata["models"]
    
    stats = {}
    for model_name in model_names:
        mu_path = cache_path / f"{model_name}_mu.npy"
        sigma_path = cache_path / f"{model_name}_sigma.npy"
        
        if mu_path.exists() and sigma_path.exists():
            stats[model_name] = {
                "mu": np.load(mu_path),
                "sigma": np.load(sigma_path)
            }
        else:
            return None
    
    return stats


def compute_frechetdist_with_cache(mu1_cached, sigma1_cached, generated_crystals):
    """Compute Fréchet distance using pre-computed reference statistics."""
    Y = np.stack(generated_crystals, axis=0)
    mu2 = np.mean(Y, axis=0)
    sigma2 = np.cov(Y, rowvar=False)
    
    distance = frechet_distance(mu1=mu1_cached, sigma1=sigma1_cached, mu2=mu2, sigma2=sigma2)
    return distance