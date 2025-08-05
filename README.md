# LeMat-GenBench: Material Generation Benchmarking Framework

A comprehensive benchmarking framework for evaluating material generation models across multiple metrics including validity, distribution, diversity, novelty, uniqueness, and stability.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/LeMaterial/lemat-genbench.git
cd lemat-genbench

# Install dependencies
uv sync

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

3. **Set up UMA model access (required for stability and distribution benchmarks):**
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
- **Charge Neutrality**: Ensures structures are charge-balanced
- **Distance Checks**: Validates atomic distances and coordination
- **Physical Plausibility**: Checks for realistic bond lengths and angles

### 2. **Distribution Metrics**
- **Jensen-Shannon Distance (JSD)**: Compares categorical distributions (space groups, crystal systems)
- **Maximum Mean Discrepancy (MMD)**: Compares continuous distributions (volume, density, etc.)
- **Fréchet Distance**: Compares embedding distributions from MLIPs

**⚠️ Important Caveat**: MMD calculations use a **15K sample** from LeMat-Bulk dataset due to computational complexity. The full dataset contains ~5M structures, but using the entire dataset would be computationally infeasible.

### 3. **Diversity Metrics**
- **Embedding-based diversity**: Measures structural diversity using MLIP embeddings
- **Composition diversity**: Analyzes chemical composition variety

### 4. **Novelty Metrics**
- **BAWL fingerprinting**: Compares against LeMat-Bulk reference dataset
- **Structural novelty**: Identifies structures not present in training data

### 5. **Uniqueness Metrics**
- **Structural uniqueness**: Measures fraction of unique structures in generated set
- **Fingerprint-based**: Uses BAWL hashing for efficient comparison

### 6. **Stability Metrics**
- **Formation Energy**: Calculated using multiple MLIPs (ORB, MACE, UMA)
- **Energy Above Hull**: Thermodynamic stability assessment
- **Ensemble Predictions**: Combines multiple MLIP predictions for robustness

**⚠️ Important Caveat**: Energy above hull calculations may fail for charged species (e.g., Cs+, Br-) as phase diagrams expect neutral compounds.

### 7. **HHI (Herfindahl-Hirschman Index)**
- **Concentration analysis**: Measures diversity in chemical compositions

### 8. **SUN (Stability, Uniqueness, Novelty)**
- **Composite metric**: Combines stability, uniqueness, and novelty scores
- **MetaSUN**: Advanced version with additional weighting

## 🏃‍♂️ Usage

### Basic Usage

```bash
# Run all benchmark families on CIF files in a directory
uv run scripts/run_benchmarks.py --cifs /path/to/cif/directory --config comprehensive --name my_benchmark

# Run specific benchmark families
uv run scripts/run_benchmarks.py --cifs structures.txt --config comprehensive --families validity novelty --name custom_run

# Use a file list instead of directory
uv run scripts/run_benchmarks.py --cifs my_structures.txt --config comprehensive --name file_list_run
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

### Configuration Options

- **`--cifs`**: Path to directory or file list (required)
- **`--config`**: Configuration name (default: `comprehensive`)
- **`--name`**: Name for this benchmark run (required)
- **`--families`**: Specific benchmark families to run (optional, defaults to all)

### Available Configurations

- `comprehensive.yaml` - All benchmark families (default)
- `validity.yaml` - Validity metrics only
- `distribution.yaml` - Distribution metrics only
- `diversity.yaml` - Diversity metrics only
- `novelty.yaml` - Novelty metrics only
- `uniqueness.yaml` - Uniqueness metrics only
- `stability.yaml` - Stability metrics only
- `hhi.yaml` - HHI metrics only
- `sun.yaml` - SUN metrics only

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

## ⚠️ Important Notes

### Computational Requirements
- **MMD Reference Sample**: Uses 15K samples from LeMat-Bulk for computational efficiency
- **MLIP Models**: Requires significant computational resources for stability benchmarks
- **Memory Usage**: Large structure sets may require substantial RAM

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

## 🐛 Troubleshooting

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
   uv run scripts/run_benchmarks.py --cifs structures/ --families validity --name memory_test
   ```

3. **Timeout Errors:**
   - Reduce structure count
   - Use faster MLIP models (ORB instead of UMA)
   - Increase timeout in configuration

### Getting Help

- Check the [scripts documentation](scripts/README_benchmark_runner.md)
- Review example configurations in `src/config/`
- Examine test files for usage patterns

## 📚 References

- **LeMat-Bulk Dataset**: [HuggingFace](https://huggingface.co/datasets/LeMaterial/LeMat-Bulk)
- **UMA Model**: [HuggingFace](https://huggingface.co/facebook/UMA)
- **ORB Models**: [GitHub](https://github.com/Open-Catalyst-Project/Open-Catalyst-Project)
- **MACE Models**: [GitHub](https://github.com/ACEsuit/mace)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

