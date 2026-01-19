![](assets/lematerial-logo.png)

# LeMat-GenBench: A Unified Evaluation Framework for Crystal Generative Models

A comprehensive benchmarking framework for evaluating material generation models across multiple metrics including validity, distribution, diversity, novelty, uniqueness, and stability. [[NeurIPS AI4Mat 2025 Spotlight](https://openreview.net/forum?id=ZfPGcTfDWn)]

[![Paper](https://img.shields.io/badge/arXiv-2512.04562-b31b1b.svg)](https://arxiv.org/abs/2512.04562)
[![Leaderboard](https://img.shields.io/badge/🤗%20HuggingFace-Leaderboard-yellow)](https://huggingface.co/spaces/LeMaterial/LeMat-GenBench)

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/LeMaterial/lemat-genbench.git
cd lemat-genbench

# Install dependencies
uv sync

# Activate the virtual environment (macOS/Linux)
source .venv/bin/activate

# Set up UMA access (required for stability and distribution benchmarks)
huggingface-cli login

# Run a quick benchmark
uv run scripts/run_benchmarks.py --cifs cif_folder --config comprehensive_multi_mlip_hull --name quick_test
```

---

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

4. **Set up UMA model access** (required for stability and distribution benchmarks):
   ```bash
   # Request access to UMA model on HuggingFace
   # Visit: https://huggingface.co/facebook/UMA
   # Click "Request access" and wait for approval
   
   # Login to HuggingFace CLI
   huggingface-cli login
   ```

---

## 📊 Benchmark Metrics

| Family | Description | Cost |
|--------|-------------|------|
| `validity` | Structure validation (charge, distance, plausibility) | Low |
| `distribution` | Distribution similarity (JSD, MMD, Fréchet distance) | Medium |
| `diversity` | Structural diversity (element, space group, site number) | Low |
| `novelty` | Novelty vs. LeMat-Bulk reference dataset | Medium |
| `uniqueness` | Internal uniqueness within generated set | Low |
| `stability` | Thermodynamic stability (formation energy, e_above_hull) | High |
| `hhi` | Supply risk assessment (production/reserve concentration) | Low |
| `sun` | Composite metric (Stability + Uniqueness + Novelty) | High |

<details>
<summary><strong>📖 Detailed Metric Descriptions</strong></summary>

### 1. Validity Metrics
- **Charge Neutrality**: Ensures structures are charge-balanced using oxidation state analysis and bond valence calculations
- **Minimum Interatomic Distance**: Validates that atomic distances exceed minimum thresholds based on atomic radii
- **Coordination Environment**: Checks if coordination numbers match expected values for each element
- **Physical Plausibility**: Validates density, lattice parameters, crystallographic format, and symmetry

### 2. Distribution Metrics
- **Jensen-Shannon Distance (JSD)**: Measures similarity of categorical properties (space groups, crystal systems, elemental compositions) between generated and reference materials
- **Maximum Mean Discrepancy (MMD)**: Measures similarity of continuous properties (volume, density) between generated and reference materials using kernel methods
- **Fréchet Distance**: Measures similarity of learned structural representations (embeddings) from MLIPs (ORB, MACE, UMA) between generated and reference materials

> ⚠️ **Note**: MMD calculations use a **15K sample** from LeMat-Bulk dataset due to computational complexity.

### 3. Diversity Metrics
- **Element Diversity**: Measures variety of chemical elements using Vendi scores and Shannon entropy
- **Space Group Diversity**: Measures variety of crystal symmetries present in generated structures
- **Site Number Diversity**: Measures variety in the number of atomic sites per structure
- **Physical Size Diversity**: Measures variety in physical properties (density, lattice parameters, packing factor)

### 4. Novelty Metrics
- **Novelty Ratio**: Fraction of generated structures NOT present in LeMat-Bulk reference dataset
- **BAWL Fingerprinting**: Uses BAWL structure hashing to efficiently compare against ~5M known materials
- **Structure Matcher**: Alternative method using pymatgen StructureMatcher for structural comparison

### 5. Uniqueness Metrics
- **Uniqueness Ratio**: Fraction of unique structures within the generated set (internal diversity)
- **BAWL Fingerprinting**: Uses BAWL structure hashing to identify duplicate structures efficiently
- **Duplicate Detection**: Counts and reports duplicate structures within the generated set

### 6. Stability Metrics
- **Stability Ratio**: Fraction of structures with energy above hull ≤ 0 eV/atom (thermodynamically stable)
- **Metastability Ratio**: Fraction of structures with energy above hull ≤ 0.1 eV/atom (metastable)
- **Mean E_Above_Hull**: Average energy above hull across multiple MLIPs (ORB, MACE, UMA)
- **Formation Energy**: Average formation energy across multiple MLIPs
- **Relaxation Stability**: RMSE between original and relaxed atomic positions
- **Ensemble Statistics**: Mean and standard deviation across MLIP predictions for uncertainty quantification

> ⚠️ **Note**: Energy above hull calculations may fail for charged species (e.g., Cs+, Br-) as phase diagrams expect neutral compounds.

### 7. HHI (Herfindahl-Hirschman Index)
- **Production HHI**: Measures supply risk based on concentration of element production sources
- **Reserve HHI**: Measures long-term supply risk based on concentration of element reserves

### 8. SUN (Stability, Uniqueness, Novelty)
- **SUN Rate**: Fraction of structures that are simultaneously stable (e_above_hull ≤ 0), unique, and novel
- **MetaSUN Rate**: Fraction of structures that are simultaneously metastable (e_above_hull ≤ 0.1), unique, and novel
- **Combined Rate**: Fraction of structures that are either stable or metastable, unique, and novel

</details>

---

## 🏃‍♂️ Usage

### Basic Usage

```bash
# Run all benchmark families on CIF files in a directory
uv run scripts/run_benchmarks.py --cifs /path/to/cif/directory --config comprehensive_multi_mlip_hull --name my_benchmark

# Run specific benchmark families
uv run scripts/run_benchmarks.py --cifs structures.txt --config comprehensive_multi_mlip_hull --families validity novelty --name custom_run

# Load structures from CSV file
uv run scripts/run_benchmarks.py --csv my_structures.csv --config comprehensive_multi_mlip_hull --name csv_benchmark
```

> 💡 **Tip**: Use configuration `comprehensive_multi_mlip_hull` for results comparable to the [leaderboard](https://huggingface.co/spaces/LeMaterial/LeMat-GenBench).

### Command Line Options

| Option | Description |
|--------|-------------|
| `--cifs` | Path to directory or file list containing CIF files |
| `--csv` | Path to CSV file containing structures |
| `--config` | Configuration name (default: `comprehensive`) |
| `--name` | Name for this benchmark run (required) |
| `--families` | Specific benchmark families to run (optional) |
| `--fingerprint-method` | Method: `bawl`, `short-bawl`, `structure-matcher`, `pdd` |

<details>
<summary><strong>📥 Input Format Details</strong></summary>

#### Option 1: Directory of CIF Files
```bash
uv run scripts/run_benchmarks.py --cifs /path/to/cif/directory --config comprehensive_multi_mlip_hull --name my_run
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
uv run scripts/run_benchmarks.py --cifs my_structures.txt --config comprehensive_multi_mlip_hull --name my_run
```

#### Option 3: CSV File with Structures
```bash
uv run scripts/run_benchmarks.py --csv my_structures.csv --config comprehensive_multi_mlip_hull --name my_csv_run
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

> **Note:** You can only use one input method at a time (`--cifs` OR `--csv`, not both).

</details>

<details>
<summary><strong>🔧 Fingerprinting Methods</strong></summary>

| Method | Description | Speed | Memory |
|--------|-------------|-------|--------|
| `bawl` | Full BAWL fingerprinting | Fast | Low |
| `short-bawl` | Shortened BAWL fingerprinting (default) | Fast | Low |
| `structure-matcher` | PyMatGen StructureMatcher comparison | Slow | High |
| `pdd` | Packing density descriptor | Medium | Medium |

```bash
# Use structure-matcher for better accuracy (slower)
uv run scripts/run_benchmarks.py \
  --cifs submissions/test \
  --config comprehensive_structure_matcher \
  --name test_run \
  --fingerprint-method structure-matcher
```

</details>

<details>
<summary><strong>🎯 Running Specific Benchmark Families</strong></summary>

#### Single Family
```bash
# Run only validity checks
uv run scripts/run_benchmarks.py --cifs structures/ --config validity --families validity --name validity_only

# Run only stability analysis
uv run scripts/run_benchmarks.py --cifs structures/ --config stability --families stability --name stability_only
```

#### Multiple Families
```bash
# Run validity and novelty (low + medium cost)
uv run scripts/run_benchmarks.py --cifs structures/ --config comprehensive_multi_mlip_hull --families validity novelty --name validity_novelty

# Run diversity, uniqueness, and HHI (all low cost)
uv run scripts/run_benchmarks.py --cifs structures/ --config comprehensive_multi_mlip_hull --families diversity uniqueness hhi --name diversity_analysis

# Run distribution and stability (medium + high cost)
uv run scripts/run_benchmarks.py --cifs structures/ --config comprehensive_multi_mlip_hull --families distribution stability --name distribution_stability
```

#### All Families (Default)
```bash
uv run scripts/run_benchmarks.py --cifs structures/ --config comprehensive_multi_mlip_hull --name full_analysis
```

</details>

---

## 📁 Output

Results are saved to `results/` directory:

```
{run_name}_{config_name}_{timestamp}.json
```

<details>
<summary><strong>📄 Output Structure</strong></summary>

```json
{
  "run_info": {
    "run_name": "my_benchmark",
    "config_name": "comprehensive",
    "timestamp": "20241204_143022",
    "n_structures": 100,
    "benchmark_families": ["validity", "distribution", "diversity", "..."]
  },
  "results": {
    "validity": { "..." },
    "distribution": { "..." },
    "diversity": { "..." }
  }
}
```

</details>

<details>
<summary><strong>📊 Extract Metrics from JSON</strong></summary>

```bash
# Process a single file
uv run scripts/extract_benchmark_metrics.py results/my_run_comprehensive.json

# Process all JSON files in a directory
uv run scripts/extract_benchmark_metrics.py results_new/ --directory

# Specify custom output directory
uv run scripts/extract_benchmark_metrics.py results/my_run.json --output-dir custom_output/

# Process directory with custom pattern
uv run scripts/extract_benchmark_metrics.py final_results/ --directory --pattern "*comprehensive*.json"
```

</details>

---

<details>
<summary><strong>🔍 More Examples</strong></summary>

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
uv run scripts/run_benchmarks.py --cifs structures.txt --config comprehensive_multi_mlip_hull --families validity novelty uniqueness --name custom_analysis
```

### CSV Input Examples
```bash
# Quick validation of CSV structures
uv run scripts/run_benchmarks.py --csv my_structures.csv --config validity --name csv_validity

# Full analysis of CSV structures
uv run scripts/run_benchmarks.py --csv generated_structures.csv --config comprehensive_multi_mlip_hull --name csv_full_analysis
```

### High-Performance SSH Examples
```bash
# Use SSH-optimized script for large datasets
uv run scripts/run_benchmarks_ssh.py --cifs large_dataset/ --config comprehensive_multi_mlip_hull --name large_run
```

</details>

<details>
<summary><strong>⚠️ Important Notes</strong></summary>

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

### Performance Tips
- **Small Sets**: Use `--families` to run only needed benchmarks
- **Large Sets**: Consider running benchmarks separately for memory efficiency
- **Caching**: Models are cached locally for faster subsequent runs
- **SSH Optimization**: Use `run_benchmarks_ssh.py` for high-core environments

</details>

<details>
<summary><strong>🛠 Troubleshooting</strong></summary>

### Common Issues

**1. UMA Access Denied:**
```bash
# Ensure you're logged in
huggingface-cli login

# Check access status
huggingface-cli whoami
```

**2. Memory Issues:**
```bash
# Run fewer families at once
uv run scripts/run_benchmarks.py --cifs structures/ --config validity --families validity --name memory_test
```

**3. Timeout Errors:**
- Reduce structure count
- Use faster MLIP models (ORB instead of UMA)
- Increase timeout in configuration

**4. Private Dataset Access Error:**
```bash
# Error: 'Entalpic/LeMaterial-Above-Hull-dataset' doesn't exist on the Hub
# Solution: Download datasets locally (one-time setup)
uv run scripts/download_above_hull_datasets.py
```

**5. Structure-Matcher Performance:**
- Structure-matcher is more accurate but much slower than BAWL
- Consider using for smaller datasets or when accuracy is critical
- Use SSH-optimized script for large datasets

### Getting Help
- Check the [scripts documentation](scripts/README_benchmark_runner.md)
- Review example configurations in `src/config/`
- Examine test files for usage patterns

</details>

<details>
<summary><strong>📚 References</strong></summary>

### Datasets
- **LeMat-Bulk Dataset**: [HuggingFace](https://huggingface.co/datasets/LeMaterial/LeMat-Bulk)  
  Siron, Martin, et al. "LeMat-Bulk: aggregating, and de-duplicating quantum chemistry materials databases." *AI for Accelerated Materials Design-ICLR 2025*.

### MLIP Models
- **UMA Model**: [HuggingFace](https://huggingface.co/facebook/UMA)  
  Wood, Brandon M., et al. "UMA: A Family of Universal Models for Atoms." *arXiv preprint arXiv:2506.23971* (2025).

- **ORB Models**: [GitHub](https://github.com/orbital-materials/orb-models)  
  Rhodes, Benjamin, et al. "Orb-v3: atomistic simulation at scale." *arXiv preprint arXiv:2504.06231* (2025).

- **MACE Models**: [GitHub](https://github.com/ACEsuit/mace)  
  Batatia, Ilyes, et al. "MACE: Higher order equivariant message passing neural networks for fast and accurate force fields." *Advances in Neural Information Processing Systems 35* (2022): 11423-11436.

### Core Metrics and Methods

**Distribution Metrics:**
- **Fréchet Distance**: [FCD Implementation](https://github.com/bioinf-jku/FCD/blob/master/fcd/utils.py)
- **Maximum Mean Discrepancy (MMD)**: [Gretton et al. (2012)](https://jmlr.org/papers/v13/gretton12a.html)  
  Gretton, Arthur, et al. "A kernel two-sample test." *JMLR 13.1* (2012): 723-773.
- **Jensen-Shannon Distance**: [Lin (1991)](https://ieeexplore.ieee.org/document/86638)  
  Lin, Jianhua. "Divergence measures based on the Shannon entropy." *IEEE Transactions on Information Theory 37.1* (2002): 145-151.

**Diversity Metrics:**
- **Vendi Score**: [Friedman & Dieng (2023)](https://arxiv.org/abs/2210.02410)  
  Friedman, Dan, and Adji Bousso Dieng. "The vendi score: A diversity evaluation metric for machine learning." *arXiv preprint arXiv:2210.02410* (2022).

**Supply Risk Metrics:**
- **Herfindahl-Hirschman Index (HHI)**: [Mansouri Tehrani et al.](https://link.springer.com/article/10.1007/s40192-017-0085-4)  
  Mansouri Tehrani, Aria, et al. "Balancing mechanical properties and sustainability in the search for superhard materials." *Integrating Materials and Manufacturing Innovation 6.1* (2017): 1-8.

</details>

---

## 📄 License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

---

## 📝 Citation

If you use LeMat-GenBench in your research, please cite:

```bibtex
@article{betala2025lemat,
  title     = {LeMat-GenBench: A Unified Evaluation Framework for Crystal Generative Models},
  author    = {Betala, Siddharth and Gleason, Samuel P. and Ramlaoui, Ali and Xu, Andy and 
               Channing, Georgia and Levy, Daniel and Fourrier, Cl{\'e}mentine and Kazeev, Nikita and 
               Joshi, Chaitanya K. and Kaba, S{\'e}kou-Oumar and Therrien, F{\'e}lix and 
               Hernandez-Garcia, Alex and Mercado, Roc{\'\i}o and Krishnan, N. M. Anoop and 
               Duval, Alexandre},
  journal   = {arXiv preprint arXiv:2512.04562},
  year      = {2025}
}
```
