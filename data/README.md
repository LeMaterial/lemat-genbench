# Data Directory

This directory contains reference data and cached statistics for lemat-genbench.

## Structure

```
data/
├── README.md                     # This file
├── [existing reference data]     # Original reference datasets
└── [cached statistics]           # Computed reference statistics (optional)
```

## Cached Reference Statistics

When using the Fréchet distance metric, reference statistics (mean and covariance matrices) can be pre-computed and cached here for significant performance improvements.

**Note**: Pre-computed cache files are included in this repository so users can immediately benefit from optimized performance without needing to compute statistics themselves.

### Pre-computed Cache (Included)

This repository includes pre-computed cached statistics:
- `mace_mu.npy` & `mace_sigma.npy` - MACE model statistics  
- `orb_mu.npy` & `orb_sigma.npy` - ORB model statistics
- `uma_mu.npy` & `uma_sigma.npy` - UMA model statistics
- `metadata.json` - Cache metadata

**Users get instant performance benefits without any setup!**

### Recomputing Cache (Optional)

If needed, you can regenerate cached statistics from the LeMat-GenBench-embeddings dataset:

```bash
uv run python scripts/compute_reference_stats.py --cache-dir ./data --models mace orb uma
```

### Using Cache

The metrics will automatically use cached statistics when available:

```python
from lemat_genbench.metrics.distribution_metrics import FrechetDistance

metric = FrechetDistance(
    reference_df=reference_df,
    mlips=["mace", "orb", "uma"],
    cache_dir="./data",
    use_cache=True
)
```

### Performance Impact

- **Without cache**: ~20GB memory, hours of computation
- **With cache**: <1GB memory, seconds of computation  
- **Cache size**: <3MB total for all models

See `docs/frechet_distance_optimization.md` for detailed documentation.