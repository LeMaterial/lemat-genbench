# Fréchet Distance Optimization: Reference Statistics Caching

The Fréchet distance metric compares embeddings between generated and reference structures. With 4.9M reference structures, computing mean and covariance matrices every time is prohibitively expensive. This document explains the optimization using cached reference statistics.

## Problem

**Original inefficiency:**
- **Dataset size**: 4.9M structures with embeddings (MACE: 640D, ORB: 256D, UMA: 128D)
- **Memory requirement**: ~20GB just to load embeddings
- **Computation time**: Hours to compute covariance matrices
- **Repeated work**: Same statistics computed for every evaluation

## Solution: Streaming Computation + Caching

### 1. Pre-computed Statistics (Included)

**Good news**: Pre-computed reference statistics are included in the repository! Users get instant performance benefits without any setup.

### 2. Optional: Recompute Statistics

If needed, you can regenerate statistics using HuggingFace streaming:

```bash
# Recompute and cache reference statistics
uv run python scripts/compute_reference_stats.py --cache-dir ./data --models mace orb uma
```

**Implementation highlights:**
- **Welford's algorithm** for streaming mean/covariance computation
- **Memory efficient**: Processes 1000 samples at a time
- **Progress tracking**: Reports progress every 10k samples  
- **Automatic caching**: Saves results as `.npy` files

### 3. Instant Fast Evaluation

The repository includes cached statistics, so you get instant performance:

```python
from lemat_genbench.metrics.distribution_metrics import FrechetDistance

# Use cached statistics (fast)
metric = FrechetDistance(
    reference_df=reference_df,
    mlips=["mace", "orb", "uma"],
    cache_dir="./data",
    use_cache=True
)

# Automatic fallback to reference_df if cache not available
```

## Performance Comparison

| Approach | Memory | Time | Reusable |
|----------|--------|------|----------|
| **No cache** | ~20GB | Hours | ❌ |
| **With cache** | <1GB | Seconds | ✅ |

**Cache storage:**
- MACE: ~1.6MB (640×640 covariance)
- ORB: ~256KB (256×256 covariance)  
- UMA: ~64KB (128×128 covariance)
- **Total: <3MB** for all models

## Usage Patterns

### Option 1: Use Pre-included Cache (Recommended)
```python
# Works immediately - cache is included in repo!
benchmark = DistributionBenchmark(
    reference_df=df, 
    mlips=["mace", "orb", "uma"],
    cache_dir="./data"  # Pre-computed cache available
)
```

### Option 2: Automatic Fallback
```python
# Tries cache first, falls back to reference_df
metric = FrechetDistance(
    reference_df=large_reference_df,
    mlips=["mace"],
    cache_dir="./data",  # May not exist
    use_cache=True
)
```

### Option 3: Disable Caching
```python
# Force computation from reference_df (slow but always works)
metric = FrechetDistance(
    reference_df=small_reference_df,
    mlips=["mace"],
    use_cache=False
)
```

## Cache Structure

```
data/
├── mace_mu.npy          # Mean vector (640,)
├── mace_sigma.npy       # Covariance matrix (640, 640)
├── orb_mu.npy           # Mean vector (256,)
├── orb_sigma.npy        # Covariance matrix (256, 256)
├── uma_mu.npy           # Mean vector (128,)
├── uma_sigma.npy        # Covariance matrix (128, 128)
└── metadata.json        # Cache metadata
```

## Algorithm Details

### Streaming Statistics (Welford's Algorithm)
```python
# Incremental mean update
delta = sample - current_mean
current_mean += delta / count

# Incremental covariance update  
delta2 = sample - updated_mean
M2 += outer(delta, delta2)
final_covariance = M2 / (count - 1)
```

**Benefits:**
- **O(1) memory** regardless of dataset size
- **Numerically stable** for large datasets
- **Single pass** through the data

### HuggingFace Streaming
```python
dataset = load_dataset("LeMaterial/LeMat-GenBench-embeddings", streaming=True)

for batch in batched(dataset, batch_size=1000):
    for model in ["mace", "orb", "uma"]:
        embeddings = [example[f"{model}_embeddings"] for example in batch]
        model_stats[model].update(embeddings)
```

## Troubleshooting

### Common Issues

**1. Missing datasets library**
```bash
uv add datasets
```

**2. Internet connectivity**
```bash
# Dataset streams from HuggingFace - requires stable internet
# Consider running overnight for full dataset
```

**3. Cache corruption**
```bash
# Recompute if files are corrupted
uv run python scripts/compute_reference_stats.py --cache-dir ./data --models mace
```

**4. Memory issues**
```bash
# Reduce batch size for lower-memory systems
uv run python scripts/compute_reference_stats.py --batch-size 100
```

### Validation

Verify cached statistics match reference computation:
```python
# Load cached stats
cached_stats = load_reference_stats_cache("./data", ["uma"])

# Compare with small reference sample
sample_embeddings = reference_df["UmaGraphEmbeddings"][:1000]
mu_ref = np.mean(sample_embeddings, axis=0)
sigma_ref = np.cov(sample_embeddings, rowvar=False)

# Should be close (within sampling variance)
print(f"Mean difference: {np.mean(np.abs(cached_stats['uma']['mu'] - mu_ref))}")
```

## Integration with Benchmarks

The distribution benchmark automatically uses cached statistics when available:

```python
from lemat_genbench.benchmarks.distribution_benchmark import DistributionBenchmark

benchmark = DistributionBenchmark(
    reference_df=reference_df,
    mlips=["mace", "orb", "uma"],
    cache_dir="./data",       # Use cached stats
    use_cache=True,           # Enable caching
)

# Benchmark will automatically:
# 1. Try to load cached statistics
# 2. Fall back to reference_df if cache missing
# 3. Report which method was used
```

This optimization makes the Fréchet distance practical for large-scale evaluation while maintaining mathematical correctness.