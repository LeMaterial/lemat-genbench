"""Analyze and visualize validity comparison results.

This script reads the output from compare_validity_checks.py and creates
visualizations and detailed analysis of the differences between SMACT and
our custom validity checks.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pymatgen.core import Composition

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_aggregate_statistics(results_dir: Path) -> Dict:
    """Load aggregate statistics from comparison results.
    
    Parameters
    ----------
    results_dir : Path
        Directory containing comparison results.
        
    Returns
    -------
    dict
        Aggregate statistics.
    """
    stats_file = results_dir / "aggregate_statistics.json"
    with open(stats_file, 'r') as f:
        return json.load(f)


def load_seed_statistics(results_dir: Path, seed: int) -> Dict:
    """Load statistics for a specific seed.
    
    Parameters
    ----------
    results_dir : Path
        Directory containing comparison results.
    seed : int
        Seed number to load.
        
    Returns
    -------
    dict
        Statistics for the specified seed.
    """
    stats_file = results_dir / f"seed_{seed}" / "statistics.json"
    with open(stats_file, 'r') as f:
        return json.load(f)


def plot_validity_comparison(aggregate_stats: Dict, output_dir: Path):
    """Create bar plot comparing validity rates.
    
    Parameters
    ----------
    aggregate_stats : dict
        Aggregate statistics across all seeds.
    output_dir : Path
        Directory to save plots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Validity rates comparison
    methods = ['SMACT', 'Our Checks']
    means = [
        aggregate_stats['smact_valid_rate_mean'] * 100,
        aggregate_stats['ours_valid_rate_mean'] * 100
    ]
    stds = [
        aggregate_stats['smact_valid_rate_std'] * 100,
        aggregate_stats['ours_valid_rate_std'] * 100
    ]
    
    x = np.arange(len(methods))
    bars = ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=['#3498db', '#e74c3c'])
    ax1.set_ylabel('Validity Rate (%)', fontsize=12)
    ax1.set_title('Validity Rates: SMACT vs Our Checks', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=11)
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}±{std:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Agreement breakdown
    categories = ['Both\nValid', 'Both\nInvalid', 'Ours Valid\nSMACT Invalid', 
                  'SMACT Valid\nOurs Invalid']
    counts_mean = [
        aggregate_stats['both_valid_mean'],
        aggregate_stats['both_invalid_mean'],
        aggregate_stats['ours_valid_smact_invalid_mean'],
        aggregate_stats['smact_valid_ours_invalid_mean'],
    ]
    counts_std = [
        aggregate_stats['both_valid_std'],
        aggregate_stats['both_invalid_std'],
        aggregate_stats['ours_valid_smact_invalid_std'],
        aggregate_stats['smact_valid_ours_invalid_std'],
    ]
    
    colors = ['#2ecc71', '#95a5a6', '#f39c12', '#9b59b6']
    x = np.arange(len(categories))
    bars = ax2.bar(x, counts_mean, yerr=counts_std, capsize=5, 
                   alpha=0.7, color=colors)
    ax2.set_ylabel('Number of Structures', fontsize=12)
    ax2.set_title('Agreement/Disagreement Breakdown', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, counts_mean, counts_std):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.0f}±{std:.0f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'validity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved validity comparison plot to {output_dir / 'validity_comparison.png'}")


def plot_seed_variation(aggregate_stats: Dict, output_dir: Path):
    """Plot variation across different seeds.
    
    Parameters
    ----------
    aggregate_stats : dict
        Aggregate statistics across all seeds.
    output_dir : Path
        Directory to save plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    seeds = aggregate_stats['seeds']
    
    # Plot 1: Ours valid, SMACT invalid
    ax = axes[0, 0]
    counts = aggregate_stats['ours_valid_smact_invalid_counts']
    ax.bar(range(len(seeds)), counts, alpha=0.7, color='#f39c12')
    ax.set_xlabel('Seed', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Ours Valid, SMACT Invalid', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seeds)
    ax.axhline(y=np.mean(counts), color='r', linestyle='--', 
               label=f'Mean: {np.mean(counts):.1f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: SMACT valid, ours invalid
    ax = axes[0, 1]
    counts = aggregate_stats['smact_valid_ours_invalid_counts']
    ax.bar(range(len(seeds)), counts, alpha=0.7, color='#9b59b6')
    ax.set_xlabel('Seed', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('SMACT Valid, Ours Invalid', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seeds)
    ax.axhline(y=np.mean(counts), color='r', linestyle='--',
               label=f'Mean: {np.mean(counts):.1f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Both valid
    ax = axes[1, 0]
    counts = aggregate_stats['both_valid_counts']
    ax.bar(range(len(seeds)), counts, alpha=0.7, color='#2ecc71')
    ax.set_xlabel('Seed', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Both Valid', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seeds)
    ax.axhline(y=np.mean(counts), color='r', linestyle='--',
               label=f'Mean: {np.mean(counts):.1f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Both invalid
    ax = axes[1, 1]
    counts = aggregate_stats['both_invalid_counts']
    ax.bar(range(len(seeds)), counts, alpha=0.7, color='#95a5a6')
    ax.set_xlabel('Seed', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Both Invalid', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seeds)
    ax.axhline(y=np.mean(counts), color='r', linestyle='--',
               label=f'Mean: {np.mean(counts):.1f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Variation Across Seeds', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'seed_variation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved seed variation plot to {output_dir / 'seed_variation.png'}")


def analyze_compositions(results_dir: Path, output_dir: Path):
    """Analyze composition patterns in disagreement cases.
    
    Parameters
    ----------
    results_dir : Path
        Directory containing comparison results.
    output_dir : Path
        Directory to save analysis results.
    """
    all_ours_valid_smact_invalid = []
    all_smact_valid_ours_invalid = []
    
    # Collect all compositions from all seeds
    for seed_dir in results_dir.glob("seed_*"):
        if not seed_dir.is_dir():
            continue
        
        stats_file = seed_dir / "statistics.json"
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        all_ours_valid_smact_invalid.extend(stats['ours_valid_smact_invalid'])
        all_smact_valid_ours_invalid.extend(stats['smact_valid_ours_invalid'])
    
    # Analyze ours valid, SMACT invalid
    print("\n" + "="*80)
    print("COMPOSITION ANALYSIS: Ours Valid, SMACT Invalid")
    print("="*80)
    
    if all_ours_valid_smact_invalid:
        compositions = [s['composition'] for s in all_ours_valid_smact_invalid]
        
        # Count elements
        element_counter = Counter()
        n_elements_counter = Counter()
        
        for comp_str in compositions:
            comp = Composition(comp_str)
            element_counter.update(comp.elements)
            n_elements_counter[len(comp.elements)] += 1
        
        print(f"\nTotal structures: {len(compositions)}")
        print("\nNumber of elements distribution:")
        for n_elem in sorted(n_elements_counter.keys()):
            print(f"  {n_elem} elements: {n_elements_counter[n_elem]} structures")
        
        print("\nMost common elements:")
        for elem, count in element_counter.most_common(15):
            print(f"  {elem}: {count} structures ({count/len(compositions)*100:.1f}%)")
        
        # Check for metallics
        metal_count = 0
        for comp_str in compositions:
            comp = Composition(comp_str)
            # Check if all elements are metals
            all_metal = all(el.is_metal or el.is_transition_metal or el.is_post_transition_metal 
                          or el.is_alkali or el.is_alkaline or el.is_lanthanoid or el.is_actinoid
                          for el in comp.elements)
            if all_metal:
                metal_count += 1
        
        print(f"\nMetallic/alloy structures: {metal_count} ({metal_count/len(compositions)*100:.1f}%)")
    
    # Analyze SMACT valid, ours invalid
    print("\n" + "="*80)
    print("COMPOSITION ANALYSIS: SMACT Valid, Ours Invalid")
    print("="*80)
    
    if all_smact_valid_ours_invalid:
        compositions = [s['composition'] for s in all_smact_valid_ours_invalid]
        
        # Count elements
        element_counter = Counter()
        n_elements_counter = Counter()
        
        for comp_str in compositions:
            comp = Composition(comp_str)
            element_counter.update(comp.elements)
            n_elements_counter[len(comp.elements)] += 1
        
        print(f"\nTotal structures: {len(compositions)}")
        print("\nNumber of elements distribution:")
        for n_elem in sorted(n_elements_counter.keys()):
            print(f"  {n_elem} elements: {n_elements_counter[n_elem]} structures")
        
        print("\nMost common elements:")
        for elem, count in element_counter.most_common(15):
            print(f"  {elem}: {count} structures ({count/len(compositions)*100:.1f}%)")
    
    # Save detailed composition lists
    analysis_file = output_dir / "composition_analysis.txt"
    with open(analysis_file, 'w') as f:
        f.write("COMPOSITION ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Ours Valid, SMACT Invalid:\n")
        f.write("-"*80 + "\n")
        for item in all_ours_valid_smact_invalid:
            f.write(f"{item['composition']}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        f.write("SMACT Valid, Ours Invalid:\n")
        f.write("-"*80 + "\n")
        for item in all_smact_valid_ours_invalid:
            f.write(f"{item['composition']}\n")
    
    print(f"\nSaved detailed composition lists to {analysis_file}")


def create_summary_report(results_dir: Path, output_dir: Path):
    """Create a comprehensive summary report.
    
    Parameters
    ----------
    results_dir : Path
        Directory containing comparison results.
    output_dir : Path
        Directory to save report.
    """
    aggregate_stats = load_aggregate_statistics(results_dir)
    
    report_file = output_dir / "summary_report.txt"
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("VALIDITY COMPARISON SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total samples processed: {aggregate_stats['total_samples']}\n")
        f.write(f"Number of seeds: {aggregate_stats['n_seeds']}\n")
        f.write(f"Samples per seed: {aggregate_stats['n_samples_per_seed']}\n")
        f.write(f"Seeds: {aggregate_stats['seeds']}\n\n")
        
        f.write("="*80 + "\n")
        f.write("CATEGORY BREAKDOWN\n")
        f.write("="*80 + "\n\n")
        
        categories = [
            ("Ours Valid, SMACT Invalid", "ours_valid_smact_invalid"),
            ("SMACT Valid, Ours Invalid", "smact_valid_ours_invalid"),
            ("Both Valid", "both_valid"),
            ("Both Invalid", "both_invalid"),
        ]
        
        for label, key in categories:
            mean = aggregate_stats[f'{key}_mean']
            std = aggregate_stats[f'{key}_std']
            total = aggregate_stats[f'{key}_total']
            percentage = (total / aggregate_stats['total_samples']) * 100
            
            f.write(f"{label}:\n")
            f.write(f"  Mean per seed: {mean:.1f} ± {std:.1f}\n")
            f.write(f"  Total: {total}\n")
            f.write(f"  Percentage: {percentage:.2f}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("VALIDITY RATES\n")
        f.write("="*80 + "\n\n")
        
        smact_rate = aggregate_stats['smact_valid_rate_mean'] * 100
        smact_std = aggregate_stats['smact_valid_rate_std'] * 100
        ours_rate = aggregate_stats['ours_valid_rate_mean'] * 100
        ours_std = aggregate_stats['ours_valid_rate_std'] * 100
        
        f.write(f"SMACT Validity Rate: {smact_rate:.2f}% ± {smact_std:.2f}%\n")
        f.write(f"Our Validity Rate: {ours_rate:.2f}% ± {ours_std:.2f}%\n")
        f.write(f"Difference: {abs(ours_rate - smact_rate):.2f}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("AGREEMENT METRICS\n")
        f.write("="*80 + "\n\n")
        
        agreement = aggregate_stats['agreement_rate_mean'] * 100
        agreement_std = aggregate_stats['agreement_rate_std'] * 100
        disagreement = aggregate_stats['disagreement_rate_mean'] * 100
        disagreement_std = aggregate_stats['disagreement_rate_std'] * 100
        
        f.write(f"Agreement Rate: {agreement:.2f}% ± {agreement_std:.2f}%\n")
        f.write(f"Disagreement Rate: {disagreement:.2f}% ± {disagreement_std:.2f}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("ERROR COUNTS\n")
        f.write("="*80 + "\n\n")
        
        smact_errors = aggregate_stats['smact_error_counts']
        ours_errors = aggregate_stats['ours_error_counts']
        
        f.write(f"SMACT errors: {smact_errors} (per seed)\n")
        f.write(f"Our checks errors: {ours_errors} (per seed)\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"\nCreated summary report: {report_file}")


def main():
    """Main function to analyze and visualize comparison results."""
    # Paths
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "validity_comparison_results"
    output_dir = results_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found at {results_dir}")
        print("Please run compare_validity_checks.py first.")
        return
    
    print("Loading results...")
    aggregate_stats = load_aggregate_statistics(results_dir)
    
    print("\nGenerating visualizations...")
    plot_validity_comparison(aggregate_stats, output_dir)
    plot_seed_variation(aggregate_stats, output_dir)
    
    print("\nAnalyzing compositions...")
    analyze_compositions(results_dir, output_dir)
    
    print("\nCreating summary report...")
    create_summary_report(results_dir, output_dir)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"All outputs saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

