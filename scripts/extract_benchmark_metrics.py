#!/usr/bin/env python3
"""
Extract key metrics from LeMat-GenBench benchmark results JSON files.

This script parses benchmark result files and extracts important metrics
into a clean CSV format for easy analysis.
"""

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


def convert_numpy_string_to_value(value_str: str) -> Any:
    """
    Convert numpy string representation to Python value.
    
    Args:
        value_str: String like "np.float64(0.75)" or "np.int64(5)"
        
    Returns:
        Extracted Python value
    """
    if value_str.startswith('np.'):
        # Extract the value between parentheses
        match = re.search(r'\(([^)]+)\)', value_str)
        if match:
            try:
                return ast.literal_eval(match.group(1))
            except (ValueError, SyntaxError):
                pass
    return value_str


def parse_benchmark_string(benchmark_str: str) -> Dict[str, Any]:
    """
    Parse the string representation of BenchmarkResult to extract final_scores.
    
    Args:
        benchmark_str: String representation of BenchmarkResult
        
    Returns:
        Dictionary containing final_scores
    """
    # Try to extract final_scores dictionary from the string
    # Use a more flexible pattern to capture the entire final_scores dict
    final_scores_pattern = r"final_scores=(\{[^}]*(?:\{[^}]*\}[^}]*)*\})"
    match = re.search(final_scores_pattern, benchmark_str)
    
    if not match:
        return {}
    
    scores = {}
    content = match.group(1)
    
    # Try to convert numpy types to regular Python types in the string
    # Replace np.float64(...) with just the number
    content_clean = re.sub(r'np\.(float64|int64|float32|int32)\(([^)]+)\)', r'\2', content)
    
    # Try to evaluate it as a Python literal
    try:
        scores = ast.literal_eval(content_clean)
        return scores
    except (ValueError, SyntaxError):
        # Fall back to regex parsing if literal_eval fails
        pass
    
    # Fallback: Extract key-value pairs with regex
    pattern = r"'([^']+)':\s*([^,}]+)"
    matches = re.findall(pattern, content)
    
    for key, value in matches:
        value = value.strip()
        
        # Try to convert to appropriate type
        if value.startswith('np.'):
            value = convert_numpy_string_to_value(value)
        elif value.startswith('['):
            # Skip lists for now
            continue
        else:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # Keep as string if can't convert
                pass
        
        scores[key] = value
    
    return scores


def extract_metrics_from_results(results_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract all required metrics from benchmark results.
    
    Args:
        results_data: Loaded JSON data from benchmark results file
        
    Returns:
        Dictionary containing all extracted metrics
    """
    metrics = {}
    
    # Extract run info
    run_info = results_data.get('run_info', {})
    metrics['run_name'] = run_info.get('run_name', '')
    metrics['timestamp'] = run_info.get('timestamp', '')
    metrics['n_structures'] = run_info.get('n_structures', 0)
    
    # Extract validity filtering info
    validity_filtering = results_data.get('validity_filtering', {})
    metrics['valid_count'] = validity_filtering.get('valid_structures', 0)
    metrics['validity_rate'] = validity_filtering.get('validity_rate', 0.0)
    
    # Parse each benchmark result
    results = results_data.get('results', {})
    
    # Validity Benchmark
    if 'validity' in results:
        validity_scores = parse_benchmark_string(results['validity'])
        metrics['charge_neutral_count'] = validity_scores.get('charge_neutrality_count', 0)
        metrics['distance_valid_count'] = validity_scores.get('interatomic_distance_count', 0)
        metrics['plausibility_valid_count'] = validity_scores.get('physical_plausibility_count', 0)
        metrics['overall_valid_count'] = validity_scores.get('overall_validity_count', 0)
    
    # Distribution Benchmark
    if 'distribution' in results:
        dist_scores = parse_benchmark_string(results['distribution'])
        metrics['JSDistance'] = dist_scores.get('JSDistance', None)
        metrics['MMD'] = dist_scores.get('MMD', None)
        metrics['FrechetDistance'] = dist_scores.get('FrechetDistance', None)
    
    # Diversity Benchmark
    if 'diversity' in results:
        div_scores = parse_benchmark_string(results['diversity'])
        metrics['element_diversity'] = div_scores.get('element_diversity', None)
        metrics['space_group_diversity'] = div_scores.get('space_group_diversity', None)
        metrics['site_diversity'] = div_scores.get('site_number_diversity', None)
        metrics['physical_size_diversity'] = div_scores.get('physical_size_diversity', None)
    
    # Novelty Benchmark (handle both 'novelty' and 'novelty_new')
    novelty_key = 'novelty_new' if 'novelty_new' in results else 'novelty'
    if novelty_key in results:
        novelty_scores = parse_benchmark_string(results[novelty_key])
        metrics['novel_count'] = novelty_scores.get('novel_structures_count', 0)
    
    # Uniqueness Benchmark (handle both 'uniqueness' and 'uniqueness_new')
    uniqueness_key = 'uniqueness_new' if 'uniqueness_new' in results else 'uniqueness'
    if uniqueness_key in results:
        unique_scores = parse_benchmark_string(results[uniqueness_key])
        metrics['unique_count'] = unique_scores.get('unique_structures_count', 0)
    
    # HHI Benchmark
    if 'hhi' in results:
        hhi_scores = parse_benchmark_string(results['hhi'])
        metrics['hhi_production_mean'] = hhi_scores.get('hhi_production_mean', None)
        metrics['hhi_reserve_mean'] = hhi_scores.get('hhi_reserve_mean', None)
        metrics['hhi_combined_mean'] = hhi_scores.get('hhi_combined_mean', None)
    
    # SUN Benchmark (handle both 'sun' and 'sun_new')
    sun_key = 'sun_new' if 'sun_new' in results else 'sun'
    if sun_key in results:
        sun_scores = parse_benchmark_string(results[sun_key])
        metrics['stable_count'] = sun_scores.get('stable_count', 0)
        metrics['metastable_count'] = sun_scores.get('metastable_count', 0)
        metrics['unique_in_stable_count'] = sun_scores.get('unique_in_stable_count', 0)
        metrics['unique_in_metastable_count'] = sun_scores.get('unique_in_metastable_count', 0)
        metrics['sun_count'] = sun_scores.get('sun_count', 0)
        metrics['msun_count'] = sun_scores.get('msun_count', 0)
    
    # Stability Benchmark
    if 'stability' in results:
        stab_scores = parse_benchmark_string(results['stability'])
        metrics['stability_mean_above_hull'] = stab_scores.get('mean_e_above_hull', None)
        metrics['stability_std_e_above_hull'] = stab_scores.get('e_hull_std', None)
        metrics['stability_mean_ensemble_std'] = stab_scores.get('stability_mean_ensemble_std', None)
        metrics['mean_formation_energy'] = stab_scores.get('mean_formation_energy', None)
        metrics['formation_energy_std'] = stab_scores.get('formation_energy_std', None)
        metrics['mean_relaxation_RMSD'] = stab_scores.get('mean_relaxation_RMSE', None)
        metrics['relaxation_RMSE_std'] = stab_scores.get('relaxation_RMSE_std', None)
    
    return metrics


def process_results_file(input_file: Path, output_dir: Optional[Path] = None) -> Path:
    """
    Process a single benchmark results file and save extracted metrics.
    
    Args:
        input_file: Path to input JSON file
        output_dir: Optional output directory (default: extracted_results in same dir)
        
    Returns:
        Path to output CSV file
    """
    # Load the JSON file
    with open(input_file, 'r') as f:
        results_data = json.load(f)
    
    # Extract metrics
    metrics = extract_metrics_from_results(results_data)
    
    # Define the desired column order
    column_order = [
        'run_name', 'timestamp', 'n_structures', 
        'overall_valid_count', 'charge_neutral_count', 'distance_valid_count', 'plausibility_valid_count',
        'unique_count', 'novel_count',
        'mean_formation_energy', 'formation_energy_std',
        'stability_mean_above_hull', 'stability_std_e_above_hull', 'stability_mean_ensemble_std',
        'mean_relaxation_RMSD', 'relaxation_RMSE_std',
        'stable_count', 'unique_in_stable_count', 'sun_count',
        'metastable_count', 'unique_in_metastable_count', 'msun_count',
        'JSDistance', 'MMD', 'FrechetDistance',
        'element_diversity', 'space_group_diversity', 'site_diversity', 'physical_size_diversity',
        'hhi_production_mean', 'hhi_reserve_mean', 'hhi_combined_mean'
    ]
    
    # Determine output directory and file
    if output_dir is None:
        output_dir = input_file.parent / 'extracted_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Change extension to .csv
    output_file = output_dir / input_file.with_suffix('.csv').name
    
    # Write to CSV with specified column order
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=column_order, extrasaction='ignore')
        writer.writeheader()
        writer.writerow(metrics)
    
    print(f"Extracted metrics saved to: {output_file}")
    return output_file


def process_directory(input_dir: Path, pattern: str = "*.json", 
                      output_dir: Optional[Path] = None) -> list[Path]:
    """
    Process all matching JSON files in a directory.
    
    Args:
        input_dir: Directory containing JSON files
        pattern: Glob pattern for matching files
        output_dir: Optional output directory (default: extracted_results in input_dir)
        
    Returns:
        List of output file paths
    """
    input_path = Path(input_dir)
    json_files = list(input_path.glob(pattern))
    
    if not json_files:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return []
    
    print(f"Found {len(json_files)} file(s) to process")
    
    output_files = []
    for json_file in json_files:
        try:
            output_file = process_results_file(json_file, output_dir)
            output_files.append(output_file)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    return output_files


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract key metrics from LeMat-GenBench benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python extract_benchmark_metrics.py results/my_run_comprehensive.json
  
  # Process all JSON files in a directory
  python extract_benchmark_metrics.py results_new/ --directory
  
  # Specify custom output directory
  python extract_benchmark_metrics.py results/my_run.json --output-dir custom_output/
  
  # Process directory with custom pattern
  python extract_benchmark_metrics.py final_results/ --directory --pattern "*comprehensive*.json"
        """
    )
    
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to JSON file or directory containing JSON files'
    )
    
    parser.add_argument(
        '--directory', '-d',
        action='store_true',
        help='Process all JSON files in the input directory'
    )
    
    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='*.json',
        help='Glob pattern for matching files when processing directory (default: *.json)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory (default: extracted_results in input directory)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return 1
    
    if args.directory:
        if not input_path.is_dir():
            print(f"Error: --directory specified but input is not a directory: {input_path}")
            return 1
        output_files = process_directory(input_path, args.pattern, output_dir)
        print(f"\nProcessed {len(output_files)} file(s) successfully")
    else:
        if not input_path.is_file():
            print(f"Error: Input path is not a file: {input_path}")
            return 1
        process_results_file(input_path, output_dir)
    
    return 0


if __name__ == '__main__':
    exit(main())

