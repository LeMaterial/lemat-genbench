# MMD Reference Sampling Documentation

## Overview
This directory contains pre-computed reference statistics for Maximum Mean Discrepancy (MMD) computation in the LeMat-GenBench distribution metrics.

## Files

### 1. `lematbulk_mmd_values.pkl` (122 MB)
- **Content**: Raw property values for 5,335,299 structures from LeMat-Bulk dataset
- **Properties**: Volume, Density(g/cm³), Density(atoms/A³)
- **Format**: Python pickle file with dictionary structure

### 2. `lematbulk_mmd_sample_indices_15k.npy` (117 KB)
- **Content**: Fixed indices for reproducible 15K sampling
- **Purpose**: Ensures identical MMD results across runs
- **Creation**: Generated with `numpy.random.seed(42)` from full 5.3M dataset
- **Format**: NumPy array of 15,000 integer indices

### 3. `lematbulk_mmd_sample_metadata.json` (960 B)
- **Content**: Metadata about the sampling process
- **Includes**: Creation date, random seed, sampling method, statistics

## Reproducible Sampling Strategy

### Why Fixed Sampling?
1. **Reproducibility**: Identical results across different runs
2. **Performance**: Avoids memory explosion (212TB → 1.7GB)
3. **Statistical Validity**: 15K random sample preserves distribution properties
4. **Optimal Balance**: Best accuracy vs performance trade-off

### Sample Selection Process
```python
import numpy as np

# Fixed seed for reproducibility
np.random.seed(42)

# Sample 15,000 indices from 5,335,299 total samples
sample_indices = np.random.choice(5335299, 15000, replace=False)
sample_indices.sort()  # Deterministic ordering

# Save for consistent reuse
np.save('lematbulk_mmd_sample_indices_15k.npy', sample_indices)
```

### Usage in MMD Computation
The `compute_mmd()` function automatically uses the fixed sample when `max_reference_samples=15000` (default):

```python
from lemat_genbench.utils.distribution_utils import compute_mmd

# Uses reproducible 15K sample automatically
mmd_result = compute_mmd(generated_data, 'Volume')

# Custom sample size uses random sampling
mmd_result = compute_mmd(generated_data, 'Volume', max_reference_samples=10000)
```

## Performance Characteristics

| Sample Size | Memory | Computation Time | MMD Accuracy |
|-------------|--------|------------------|--------------|
| 15,000      | 1.7 GB | ~4-5 seconds     | Optimal      |
| 10,000      | 0.75 GB| ~1.7 seconds     | Good         |
| 25,000      | 4.7 GB | ~175 seconds     | Marginal gain|

## Regeneration
To regenerate the reproducible sample (should not be necessary):

```bash
uv run python -c "
import numpy as np
np.random.seed(42)
indices = np.random.choice(5335299, 15000, replace=False)
np.save('data/lematbulk_mmd_sample_indices_15k.npy', indices)
"
```

**Note**: Only regenerate if you understand the implications for reproducibility of existing results.
