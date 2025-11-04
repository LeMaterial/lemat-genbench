#!/bin/bash
# Convenience script to run baseline dataset sampling

echo "================================================================"
echo "Baseline Dataset Sampling Script"
echo "================================================================"
echo ""
echo "This will fetch and sample structures from 5 databases:"
echo "  - Materials Project (~170K structures)"
echo "  - Alexandria (~5M structures)"
echo "  - OQMD (~1.2M structures)"
echo "  - AFLOW (~3.5M structures)"
echo "  - NOMAD (~9M structures)"
echo ""
echo "Expected runtime: 6-12 hours"
echo "Output directory: baseline_data/"
echo ""
echo "The script supports resuming from checkpoints if interrupted."
echo ""
echo "================================================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null
then
    echo "Error: Python not found. Please ensure Python 3.11+ is installed."
    exit 1
fi

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "Warning: No virtual environment detected."
    echo "Attempting to activate .venv..."
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        echo "Virtual environment activated."
    else
        echo "Error: .venv not found. Please run 'uv sync' first."
        exit 1
    fi
fi

# Run the sampling script
echo "Starting sampling script..."
echo ""

python scripts/sample_baseline_datasets.py "$@"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo "Sampling completed successfully!"
    echo "================================================================"
    echo ""
    echo "Results saved to: baseline_data/"
    echo ""
    echo "Directory structure:"
    echo "  baseline_data/mp/          - Materials Project (2500 CIFs)"
    echo "  baseline_data/alexandria/  - Alexandria (2500 CIFs)"
    echo "  baseline_data/oqmd/        - OQMD (2500 CIFs)"
    echo "  baseline_data/aflow/       - AFLOW (2500 CIFs)"
    echo "  baseline_data/nomad/       - NOMAD (2500 CIFs)"
    echo ""
    echo "See baseline_data/sampling_summary.json for details."
else
    echo ""
    echo "================================================================"
    echo "Sampling failed with exit code: $exit_code"
    echo "================================================================"
    echo ""
    echo "Check baseline_sampling.log for error details."
    echo "You can resume by running this script again (checkpoints are saved)."
fi

exit $exit_code

