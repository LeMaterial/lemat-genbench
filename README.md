![](assets/lematerial-logo.png)
# LeMat-GenBench: Benchmark for generative models for materials

A comprehensive benchmarking framework for evaluating material generation models across multiple metrics including validity, distribution, diversity, novelty, uniqueness, and stability.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/LeMaterial/lemat-genbench.git
cd lemat-genbench

# Install dependencies
uv sync

# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Set up UMA access (required for stability and distribution benchmarks)
huggingface-cli login

# Run a quick benchmark
uv run scripts/run_benchmarks.py --cifs notebooks --config comprehensive --name quick_test
```

## 📦 Installation

### Prerequisites

- **Python 3.11+**
- **uv** package manager (recommended)
- **HuggingFace account** (for UMA model access)

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LeMaterial/lemat-genbench.git
   cd lemat-genbench
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Activate the virtual environment:**
   ```bash
   # On macOS/Linux:
   source .venv/bin/activate
   
   # On Windows:
   .venv\Scripts\activate
   ```

4. **Set up UMA model access (required for stability and distribution benchmarks):**
   ```bash
   # Request access to UMA model on HuggingFace
   # Visit: https://huggingface.co/facebook/UMA
   # Click "Request access" and wait for approval
   
   # Login to HuggingFace CLI
   huggingface-cli login
   # Enter your HuggingFace token when prompted
   ```

## 🔧 Setup

### UMA Model Access Setup

The UMA model is gated and requires special access. Follow these steps:

1. **Request Access:**
   - Visit [UMA model page](https://huggingface.co/facebook/UMA)
   - Click "Request access" button
   - Wait for approval (usually within 24 hours)

2. **Get HuggingFace Token:**
   - Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
   - Create a new token with "read" permissions
   - Copy the token

3. **Login via CLI:**
   ```bash
   huggingface-cli login
   # Enter your token when prompted
   ```

4. **Verify Access:**
   ```bash
   # Test UMA access
   uv run scripts/run_benchmarks.py --cifs notebooks --config comprehensive --name uma_test --families stability
   ```

## 📊 Benchmark Metrics

### 1. **Validity Metrics**
- **Charge Neutrality**: Ensures structures are charge-balanced using oxidation state analysis and bond valence calculations
- **Minimum Interatomic Distance**: Validates that atomic distances exceed minimum thresholds based on atomic radii
- **Coordination Environment**: Checks if coordination numbers match expected values for each element
- **Physical Plausibility**: Validates density, lattice parameters, crystallographic format, and symmetry

### 2. **Distribution Metrics**
- **Jensen-Shannon Distance (JSD)**: Measures similarity of categorical properties (space groups, crystal systems, elemental compositions) between generated and reference materials
- **Maximum Mean Discrepancy (MMD)**: Measures similarity of continuous properties (volume, density) between generated and reference materials using kernel methods
- **Fréchet Distance**: Measures similarity of learned structural representations (embeddings) from MLIPs (ORB, MACE, UMA) between generated and reference materials

**⚠️ Important Caveat**: MMD calculations use a **15K sample** from LeMat-Bulk dataset due to computational complexity. The full dataset contains ~5M structures, but using the entire dataset would be computationally infeasible.

### 3. **Diversity Metrics**
- **Element Diversity**: Measures variety of chemical elements used across generated structures using Vendi scores and Shannon entropy
- **Space Group Diversity**: Measures variety of crystal symmetries (space groups) present in generated structures
- **Site Number Diversity**: Measures variety in the number of atomic sites per structure
- **Physical Size Diversity**: Measures variety in physical properties (density, lattice parameters, packing factor) compared to uniform distribution

### 4. **Novelty Metrics**
- **Novelty Ratio**: Fraction of generated structures NOT present in LeMat-Bulk reference dataset
- **BAWL Fingerprinting**: Uses BAWL structure hashing to efficiently compare against ~5M known materials
- **Structure Matcher**: Alternative method using pymatgen StructureMatcher for structural comparison
- **Reference Comparison**: Measures how many structures are truly novel vs. known materials

### 5. **Uniqueness Metrics**
- **Uniqueness Ratio**: Fraction of unique structures within the generated set (internal diversity)
- **BAWL Fingerprinting**: Uses BAWL structure hashing to identify duplicate structures efficiently
- **Structure Matcher**: Alternative method using pymatgen StructureMatcher for structural comparison
- **Duplicate Detection**: Counts and reports duplicate structures within the generated set

### 6. **Stability Metrics**
- **Stability Ratio**: Fraction of structures with energy above hull ≤ 0 eV/atom (thermodynamically stable)
- **Metastability Ratio**: Fraction of structures with energy above hull ≤ 0.1 eV/atom (metastable)
- **Mean E_Above_Hull**: Average energy above hull across multiple MLIPs (ORB, MACE, UMA)
- **Formation Energy**: Average formation energy across multiple MLIPs (ORB, MACE, UMA)
- **Relaxation Stability**: RMSE between original and relaxed atomic positions
- **Ensemble Statistics**: Mean and standard deviation across MLIP predictions for uncertainty quantification

**⚠️ Important Caveat**: Energy above hull calculations may fail for charged species (e.g., Cs+, Br-) as phase diagrams expect neutral compounds.

### 7. **HHI (Herfindahl-Hirschman Index)**
- **Production HHI**: Measures supply risk based on concentration of element production sources (market concentration)
- **Reserve HHI**: Measures long-term supply risk based on concentration of element reserves (geographic distribution)

### 8. **SUN (Stability, Uniqueness, Novelty)**
- **SUN Rate**: Fraction of structures that are simultaneously stable (e_above_hull ≤ 0), unique, and novel
- **MetaSUN Rate**: Fraction of structures that are simultaneously metastable (e_above_hull ≤ 0.1), unique, and novel
- **Combined Rate**: Fraction of structures that are either stable or metastable, unique, and novel
- **Efficient Computation**: Uses hierarchical filtering (uniqueness → novelty → stability) for optimal performance

## 🏃‍♂️ Usage

### Basic Usage

```bash
# Run all benchmark families on CIF files in a directory
uv run scripts/run_benchmarks.py --cifs /path/to/cif/directory --config comprehensive --name my_benchmark

# Run specific benchmark families
uv run scripts/run_benchmarks.py --cifs structures.txt --config comprehensive --families validity novelty --name custom_run

# Use a file list instead of directory
uv run scripts/run_benchmarks.py --cifs my_structures.txt --config comprehensive --name file_list_run

# Load structures from CSV file
uv run scripts/run_benchmarks.py --csv my_structures.csv --config comprehensive --name csv_benchmark

# Run specific families on CSV input
uv run scripts/run_benchmarks.py --csv structures.csv --config comprehensive --families validity diversity --name csv_quick_test

# Use structure-matcher for fingerprinting (alternative to BAWL)
uv run scripts/run_benchmarks.py \
  --cifs submissions/test \
  --config comprehensive_structure_matcher \
  --name test_run_structure_matcher \
  --fingerprint-method structure-matcher
```

### Input Formats

#### Option 1: Directory of CIF Files
```bash
# Point to a directory containing CIF files
uv run scripts/run_benchmarks.py --cifs /path/to/cif/directory --config comprehensive --name my_run
```

#### Option 2: File List
Create a text file with CIF paths:
```txt
# my_structures.txt
path/to/structure1.cif
path/to/structure2.cif
path/to/structure3.cif
```

Then run:
```bash
uv run scripts/run_benchmarks.py --cifs my_structures.txt --config comprehensive --name my_run
```

#### Option 3: CSV File with Structures
Load structures directly from a CSV file containing structure data:

```bash
# Load from CSV file
uv run scripts/run_benchmarks.py --csv my_structures.csv --config comprehensive --name my_csv_run
```

**CSV Format Requirements:**
- Must contain a column named `structure`, `LeMatStructs`, or `cif_string`
- The structure column should contain either:
  - **JSON strings** (pymatgen Structure dictionaries) - recommended
  - **CIF strings** (CIF format text)

**Example CSV format:**
```csv
material_id,structure,other_metadata
0,"{""@module"": ""pymatgen.core.structure"", ""@class"": ""Structure"", ""lattice"": {...}, ""sites"": [...]}",metadata1
1,"{""@module"": ""pymatgen.core.structure"", ""@class"": ""Structure"", ""lattice"": {...}, ""sites"": [...]}",metadata2
```

**Note:** You can only use one input method at a time (`--cifs` OR `--csv`, not both).

### Configuration Options

- **`--cifs`**: Path to directory or file list (use with `--cifs` OR `--csv`)
- **`--csv`**: Path to CSV file containing structures (use with `--cifs` OR `--csv`)
- **`--config`**: Configuration name (default: `comprehensive`)
- **`--name`**: Name for this benchmark run (required)
- **`--families`**: Specific benchmark families to run (optional, defaults to all)
- **`--fingerprint-method`**: Fingerprinting method to use (`bawl`, `short-bawl`, `structure-matcher`, `pdd`)

### Available Benchmark Families

| Family | Description | Computational Cost |
|--------|-------------|-------------------|
| `validity` | Fundamental structure validation (charge, distance, plausibility) | Low |
| `distribution` | Distribution similarity (JSD, MMD, Fréchet distance) | Medium |
| `diversity` | Structural diversity (element, space group, site number, physical) | Low |
| `novelty` | Novelty vs. LeMat-Bulk reference dataset | Medium |
| `uniqueness` | Internal uniqueness within generated set | Low |
| `stability` | Thermodynamic stability (formation energy, e_above_hull) | High |
| `hhi` | Supply risk assessment (production/reserve concentration) | Low |
| `sun` | Composite metric (Stability + Uniqueness + Novelty) | High |

### Available Configurations

- `comprehensive.yaml` - All benchmark families using BAWL fingerprinting (default)
- `comprehensive_structure_matcher.yaml` - All benchmark families using structure-matcher
- `comprehensive_new.yaml` - Enhanced benchmarks with augmented fingerprinting
- `validity.yaml` - Validity metrics only
- `distribution.yaml` - Distribution metrics only
- `diversity.yaml` - Diversity metrics only
- `novelty.yaml` - Novelty metrics only
- `uniqueness.yaml` - Uniqueness metrics only
- `stability.yaml` - Stability metrics only
- `hhi.yaml` - HHI metrics only
- `sun.yaml` - SUN metrics only

### Fingerprinting Methods

| Method | Description | Speed | Memory Usage |
|--------|-------------|-------|--------------|
| `bawl` | Full BAWL fingerprinting | Fast | Low |
| `short-bawl` | Shortened BAWL fingerprinting (default) | Fast | Low |
| `structure-matcher` | PyMatGen StructureMatcher comparison | Slow | High |
| `pdd` | Packing density descriptor | Medium | Medium |

### Running Specific Benchmark Families

#### Single Family
```bash
# Run only validity checks
uv run scripts/run_benchmarks.py --cifs structures/ --config validity --families validity --name validity_only

# Run only stability analysis
uv run scripts/run_benchmarks.py --cifs structures/ --config stability --families stability --name stability_only
```

#### Multiple Families (2-3 families)
```bash
# Run validity and novelty (low + medium cost)
uv run scripts/run_benchmarks.py --cifs structures/ --config comprehensive --families validity novelty --name validity_novelty

# Run diversity, uniqueness, and HHI (all low cost)
uv run scripts/run_benchmarks.py --cifs structures/ --config comprehensive --families diversity uniqueness hhi --name diversity_analysis

# Run distribution and stability (medium + high cost)
uv run scripts/run_benchmarks.py --cifs structures/ --config comprehensive --families distribution stability --name distribution_stability

# Run novelty, uniqueness, and SUN (medium + low + high cost)
uv run scripts/run_benchmarks.py --cifs structures/ --config comprehensive --families novelty uniqueness sun --name novelty_sun
```

#### All Families (Default)
```bash
# Run all benchmark families
uv run scripts/run_benchmarks.py --cifs structures/ --config comprehensive --name full_analysis
# or explicitly specify all families
uv run scripts/run_benchmarks.py --cifs structures/ --config comprehensive --families validity distribution diversity novelty uniqueness stability hhi sun --name explicit_full
```

#### Using Structure-Matcher for Better Accuracy
```bash
# Use structure-matcher instead of BAWL fingerprinting for more accurate structural comparison
uv run scripts/run_benchmarks.py \
  --cifs submissions/test \
  --config comprehensive_structure_matcher \
  --name test_run_structure_matcher \
  --fingerprint-method structure-matcher

# Run specific families with structure-matcher
uv run scripts/run_benchmarks.py \
  --cifs structures/ \
  --config comprehensive \
  --families novelty uniqueness \
  --name novelty_uniqueness_matcher \
  --fingerprint-method structure-matcher
```

## 📁 Output

Results are saved to `results/` directory with format:
```
{run_name}_{config_name}_{timestamp}.json
```

Example: `my_benchmark_comprehensive_20241204_143022.json`

### Output Structure
```json
{
  "run_info": {
    "run_name": "my_benchmark",
    "config_name": "comprehensive",
    "timestamp": "20241204_143022",
    "n_structures": 100,
    "benchmark_families": ["validity", "distribution", "diversity", ...]
  },
  "results": {
    "validity": { ... },
    "distribution": { ... },
    "diversity": { ... },
    ...
  }
}
```

## 🔍 Examples

### Quick Validation Check
```bash
uv run scripts/run_benchmarks.py --cifs notebooks --config validity --name quick_validity
```

### Full Stability Analysis
```bash
uv run scripts/run_benchmarks.py --cifs my_structures/ --config stability --name stability_analysis
```

### Custom Benchmark Selection
```bash
uv run scripts/run_benchmarks.py --cifs structures.txt --config comprehensive --families validity novelty uniqueness --name custom_analysis
```

### CSV Input Examples
```bash
# Quick validation of CSV structures
uv run scripts/run_benchmarks.py --csv my_structures.csv --config validity --name csv_validity

# Full analysis of CSV structures
uv run scripts/run_benchmarks.py --csv generated_structures.csv --config comprehensive --name csv_full_analysis

# Distribution analysis only
uv run scripts/run_benchmarks.py --csv structures.csv --config distribution --families distribution --name csv_distribution
```

### High-Performance SSH Examples
```bash
# Use SSH-optimized script for large datasets
uv run scripts/run_benchmarks_ssh.py --cifs large_dataset/ --config comprehensive --name large_run

# Structure-matcher with SSH optimization
uv run scripts/run_benchmarks_ssh.py \
  --cifs submissions/large_test \
  --config comprehensive_structure_matcher \
  --name large_test_structure_matcher \
  --fingerprint-method structure-matcher
```

## ⚠️ Important Notes

### Computational Requirements
- **MMD Reference Sample**: Uses 15K samples from LeMat-Bulk for computational efficiency
- **MLIP Models**: Requires significant computational resources for stability benchmarks
- **Memory Usage**: Large structure sets may require substantial RAM
- **Structure-Matcher**: More accurate but computationally expensive than BAWL fingerprinting

### Model Access
- **UMA Model**: Requires HuggingFace access approval
- **ORB Models**: Automatically downloaded on first use
- **MACE Models**: Cached locally after first download

### Charged Species Handling
- **Formation Energy**: Works with charged species (Cs+, Br-, etc.)
- **E_above_hull**: May fail for charged species (expected behavior)
- **Warnings**: Some warnings are informational and expected

### Performance Tips
- **Small Sets**: Use `--families` to run only needed benchmarks
- **Large Sets**: Consider running benchmarks separately for memory efficiency
- **Caching**: Models are cached locally for faster subsequent runs
- **SSH Optimization**: Use `run_benchmarks_ssh.py` for high-core environments
- **Fingerprinting**: Use `structure-matcher` for accuracy, `short-bawl` for speed

## 🛠 Troubleshooting

### Common Issues

1. **UMA Access Denied:**
   ```bash
   # Ensure you're logged in
   huggingface-cli login
   
   # Check access status
   huggingface-cli whoami
   ```

2. **Memory Issues:**
   ```bash
   # Run fewer families at once
   uv run scripts/run_benchmarks.py --cifs structures/ --config validity --families validity --name memory_test
   ```

3. **Timeout Errors:**
   - Reduce structure count
   - Use faster MLIP models (ORB instead of UMA)
   - Increase timeout in configuration

4. **Private Dataset Access Error:**
   ```bash
   # Error: 'Entalpic/LeMaterial-Above-Hull-dataset' doesn't exist on the Hub
   # Solution: Download datasets locally (one-time setup)
   uv run scripts/download_above_hull_datasets.py
   ```
   
   This downloads the required datasets to `data/` folder for local access.

5. **Structure-Matcher Performance:**
   - Structure-matcher is more accurate but much slower than BAWL
   - Consider using for smaller datasets or when accuracy is critical
   - Use SSH-optimized script for large datasets

### Getting Help

- Check the [scripts documentation](scripts/README_benchmark_runner.md)
- Review example configurations in `src/config/`
- Examine test files for usage patterns

## 📚 References

### Datasets
- **LeMat-Bulk Dataset**: [HuggingFace](https://huggingface.co/datasets/LeMaterial/LeMat-Bulk) - Siron, Martin, et al. "LeMat-Bulk: aggregating, and de-duplicating quantum chemistry materials databases." AI for Accelerated Materials Design-ICLR 2025.

### MLIP Models
- **UMA Model**: [HuggingFace](https://huggingface.co/facebook/UMA) - Wood, Brandon M., et al. "UMA: A Family of Universal Models for Atoms." arXiv preprint arXiv:2506.23971 (2025).
- **ORB Models**: [GitHub](https://github.com/orbital-materials/orb-models) - Rhodes, Benjamin, et al. "Orb-v3: atomistic simulation at scale." arXiv preprint arXiv:2504.06231 (2025).
- **MACE Models**: [GitHub](https://github.com/ACEsuit/mace) - Batatia, Ilyes, et al. "MACE: Higher order equivariant message passing neural networks for fast and accurate force fields." Advances in neural information processing systems 35 (2022): 11423-11436.

### Core Metrics and Methods

#### Distribution Metrics
- **Fréchet Distance**: [FCD Implementation](https://github.com/bioinf-jku/FCD/blob/master/fcd/utils.py) - Measures similarity between embedding distributions
- **Maximum Mean Discrepancy (MMD)**: [Gretton et al. (2012)](https://jmlr.org/papers/v13/gretton12a.html) - Gretton, Arthur, et al. "A kernel two-sample test." The journal of machine learning research 13.1 (2012): 723-773.
- **Jensen-Shannon Distance**: [Lin (1991)](https://ieeexplore.ieee.org/document/86638) - Lin, Jianhua. "Divergence measures based on the Shannon entropy." IEEE Transactions on Information theory 37.1 (2002): 145-151.

#### Diversity Metrics
- **Vendi Score**: [Friedman & Dieng (2023)](https://arxiv.org/abs/2210.02410) - Friedman, Dan, and Adji Bousso Dieng. "The vendi score: A diversity evaluation metric for machine learning." arXiv preprint arXiv:2210.02410 (2022).

#### Supply Risk Metrics
- **Herfindahl-Hirschman Index (HHI)**: [Mansouri Tehrani, Aria, et al](https://link.springer.com/article/10.1007/s40192-017-0085-4) - Mansouri Tehrani, Aria, et al. "Balancing mechanical properties and sustainability in the search for superhard materials." Integrating materials and manufacturing innovation 6.1 (2017): 1-8.

## 📄 License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
