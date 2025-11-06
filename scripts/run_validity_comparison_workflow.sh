#!/bin/bash

# Validity Comparison Workflow Runner
# This script runs the complete validity comparison workflow:
# 1. Compare SMACT and our validity checks on sampled structures
# 2. Analyze and visualize the results

set -e  # Exit on error

echo "=========================================="
echo "Validity Comparison Workflow"
echo "=========================================="
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project root: $PROJECT_ROOT"
echo ""

# Step 1: Run the comparison
echo "Step 1: Running validity comparison..."
echo "This will sample 1000 structures from LeMat-Bulk for 5 different seeds"
echo "and compare SMACT validity with our custom validity checks."
echo ""

cd "$PROJECT_ROOT"
uv run scripts/compare_validity_checks.py

if [ $? -ne 0 ]; then
    echo "Error: Comparison failed. Exiting."
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 1 complete!"
echo "=========================================="
echo ""

# Step 2: Analyze the results
echo "Step 2: Analyzing results and generating visualizations..."
echo ""

uv run scripts/analyze_validity_comparison.py

if [ $? -ne 0 ]; then
    echo "Error: Analysis failed. Exiting."
    exit 1
fi

echo ""
echo "=========================================="
echo "Workflow complete!"
echo "=========================================="
echo ""
echo "Results saved to: validity_comparison_results/"
echo ""
echo "Output files:"
echo "  - validity_comparison_results/seed_*/               (CIF files and per-seed statistics)"
echo "  - validity_comparison_results/aggregate_statistics.json"
echo "  - validity_comparison_results/analysis/             (visualizations and analysis)"
echo ""
echo "You can now explore the results!"

