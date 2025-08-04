# Benchmark Runner Script

A comprehensive script for running material generation benchmarks on CIF files.

## Features

- **Multiple benchmark families**: Run validity, distribution, diversity, novelty, uniqueness, HHI, SUN, and stability benchmarks
- **Smart preprocessing**: Automatically runs appropriate preprocessors based on benchmark requirements
- **Flexible configuration**: Use existing config files or specify custom benchmark families
- **JSON output**: Results saved to timestamped JSON files in `results/` directory
- **Error handling**: Continues processing even if individual benchmarks fail

## Usage

### Basic Usage

```bash
# Run validity benchmark using validity config
uv run scripts/run_benchmarks.py --cifs my_structures.txt --config validity --name my_run

# Run distribution benchmark using distribution config  
uv run scripts/run_benchmarks.py --cifs my_structures.txt --config distribution --name test_run

# Run all families using comprehensive config (default behavior)
uv run scripts/run_benchmarks.py --cifs my_structures.txt --config comprehensive --name full_eval
```

### Advanced Usage

```bash
# Run specific benchmark families only
uv run scripts/run_benchmarks.py --cifs structures.txt --config comprehensive --families validity novelty uniqueness --name custom_run

# Run all available families (default behavior)
uv run scripts/run_benchmarks.py --cifs structures.txt --config comprehensive --name complete_eval
```

## Input Format

### CIF Files List
Create a text file with one CIF file path per line:

```txt
# Example: my_structures.txt
path/to/structure1.cif
path/to/structure2.cif
path/to/structure3.cif
```

Lines starting with `#` are ignored as comments.

## Configuration Files

The script uses YAML configuration files from `src/config/`:

- `validity.yaml` - Validity benchmark settings
- `distribution.yaml` - Distribution benchmark settings  
- `diversity.yaml` - Diversity benchmark settings
- `novelty.yaml` - Novelty benchmark settings
- `uniqueness.yaml` - Uniqueness benchmark settings
- `hhi.yaml` - HHI benchmark settings
- `sun.yaml` - SUN benchmark settings
- `multi_mlip_stability.yaml` - Multi-MLIP stability settings
- `comprehensive.yaml` - Settings for all benchmark families

## Preprocessor Logic

The script automatically determines which preprocessors to run:

| Benchmark Family | Preprocessors Required |
|------------------|------------------------|
| `validity` | None |
| `distribution` | Distribution preprocessor |
| `diversity` | Multi-MLIP (embeddings only) |
| `novelty` | Multi-MLIP (embeddings only) |
| `uniqueness` | Multi-MLIP (embeddings only) |
| `hhi` | None |
| `sun` | Multi-MLIP (stability + embeddings) |
| `multi_mlip_stability` | Multi-MLIP (stability + embeddings) |

## Output

Results are saved to `results/` directory with filename format:
```
{run_name}_{config_name}_{timestamp}.json
```

Example: `my_run_validity_20241204_143022.json`

### Output Structure

```json
{
  "run_info": {
    "run_name": "my_run",
    "config_name": "validity", 
    "timestamp": "20241204_143022",
    "n_structures": 100,
    "benchmark_families": ["validity"]
  },
  "results": {
    "validity": {
      "final_scores": {...},
      "evaluator_results": {...},
      "metadata": {...}
    }
  }
}
```

## Available Benchmark Families

- **`validity`**: Charge neutrality, interatomic distances, coordination environments, physical plausibility
- **`distribution`**: Jensen-Shannon distance, MMD, Fréchet distance
- **`diversity`**: Element diversity, space group diversity, physical size diversity, site number diversity
- **`novelty`**: Novelty compared to reference dataset using BAWL fingerprints
- **`uniqueness`**: Uniqueness within generated structures using BAWL fingerprints
- **`hhi`**: Herfindahl-Hirschman Index for supply risk assessment
- **`sun`**: Stable, Unique, and Novel structures evaluation
- **`multi_mlip_stability`**: Multi-MLIP ensemble stability predictions

## Examples

### Quick Validity Check
```bash
uv run scripts/run_benchmarks.py --cifs quick_test.txt --config validity --name quick_validity
```

### Full Distribution Analysis
```bash
uv run scripts/run_benchmarks.py --cifs generated_structures.txt --config distribution --name distribution_analysis
```

### Complete Evaluation
```bash
uv run scripts/run_benchmarks.py --cifs final_structures.txt --config comprehensive --name complete_evaluation
```

## Error Handling

- **Missing CIF files**: Script validates all files exist before processing
- **Failed benchmarks**: Individual benchmark failures don't stop the entire run
- **Preprocessor errors**: Detailed logging of preprocessing steps
- **Results**: All results (including errors) are saved to JSON output

## Performance Notes

- **Distribution preprocessing**: Fast, adds structural properties
- **Multi-MLIP preprocessing**: Can be slow, runs MLIP models for embeddings/stability
- **Memory usage**: Scales with number of structures and benchmark complexity
- **Parallel processing**: Some benchmarks support parallel computation

## Troubleshooting

### Common Issues

1. **Missing CIF files**: Check file paths in your CIF list
2. **Config not found**: Ensure config file exists in `src/config/`
3. **Preprocessor failures**: Check MLIP model availability and dependencies
4. **Memory errors**: Reduce number of structures or use sampling

### Debug Mode

Add verbose logging by setting environment variable:
```bash
export LOG_LEVEL=DEBUG
uv run scripts/run_benchmarks.py --cifs test.txt --config validity --name debug_run
``` 