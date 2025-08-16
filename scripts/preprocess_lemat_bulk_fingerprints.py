#!/usr/bin/env python3
"""Preprocess LeMat-Bulk dataset to generate augmented fingerprints for reference.

This script processes the entire LeMat-Bulk dataset to generate augmented fingerprints
and saves both fingerprints and structure information needed for novelty detection
with StructureMatcher fallback support.
"""

import os
import pickle
import time
import warnings
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_dataset

from lemat_genbench.fingerprinting.augmented_fingerprint import (
    AugmentedFingerprinter,
    get_augmented_fingerprint,
    record_to_augmented_fingerprint,
)
from lemat_genbench.fingerprinting.crystallographic_analyzer import (
    analyze_lematbulk_item,
    lematbulk_item_to_structure,
    structure_to_crystallographic_dict,
)
from lemat_genbench.utils.logging import logger

# Suppress pymatgen deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*dict interface is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pymatgen")


class LematBulkFingerprintProcessor:
    """Processor for generating LeMat-Bulk reference fingerprints with enhanced robustness."""
    
    def __init__(
        self, 
        output_dir: str = "data/reference_fingerprints_new",
        symprec: float = 0.01,
        angle_tolerance: float = 5.0,
        save_crystallographic_data: bool = True
    ):
        """Initialize the processor.
        
        Parameters
        ----------
        output_dir : str, default="data/reference_fingerprints"
            Directory to save reference fingerprint files.
        symprec : float, default=0.01
            Symmetry precision for crystallographic analysis.
        angle_tolerance : float, default=5.0
            Angle tolerance in degrees for crystallographic analysis.
        save_crystallographic_data : bool, default=True
            Whether to save detailed crystallographic analysis data.
        """
        self.output_dir = output_dir
        self.dataset_name = "LeMaterial/LeMat-Bulk"
        self.config_name = "compatible_pbe"
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
        self.save_crystallographic_data = save_crystallographic_data
        
        # Initialize fingerprinter with custom parameters
        self.fingerprinter = AugmentedFingerprinter(
            symprec=symprec, 
            angle_tolerance=angle_tolerance
        )
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Statistics tracking
        self.processed_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.crystallographic_analysis_failures = 0
        self.fingerprint_generation_failures = 0
        self.duplicate_fingerprints = 0
        self.fingerprint_set = set()
        self.error_categories = {}
        
        # Processing method tracking
        self.processing_methods = {
            'direct_fingerprinting': 0,
            'enhanced_analysis': 0,
            'failed': 0
        }
        
        # Enhanced enumeration statistics
        self.enumeration_stats = {
            'total_enumerations': 0,
            'avg_enumerations_per_structure': 0,
            'max_enumerations': 0,
            'min_enumerations': float('inf')
        }
        
    def process_dataset_streaming(self, chunk_size: int = 10000, max_structures: Optional[int] = None) -> None:
        """Process LeMat-Bulk dataset in streaming chunks with enhanced error handling."""
        print("🚀 Starting Enhanced LeMat-Bulk fingerprint preprocessing...")
        print(f"   Dataset: {self.dataset_name}")
        print(f"   Config: {self.config_name}")
        print(f"   Chunk size: {chunk_size}")
        print(f"   Symmetry precision: {self.symprec}")
        print(f"   Angle tolerance: {self.angle_tolerance}")
        print(f"   Save crystallographic data: {self.save_crystallographic_data}")
        print(f"   Output directory: {self.output_dir}")
        if max_structures:
            print(f"   Max structures: {max_structures}")
        
        # Load dataset in streaming mode
        print("\n📁 Loading LeMat-Bulk dataset...")
        try:
            ds_stream = load_dataset(
                self.dataset_name, 
                name=self.config_name, 
                split="train", 
                streaming=True
            )
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            print(f"❌ Failed to load dataset: {e}")
            return
        
        start_time = time.time()
        chunk_num = 0
        
        # Convert to iterator for proper consumption
        ds_iter = iter(ds_stream)
        
        # Process in chunks
        while True:
            chunk_num += 1
            
            # Get next chunk by manually collecting items
            chunk_data = []
            current_chunk_size = chunk_size
            
            if max_structures:
                remaining = max_structures - self.processed_count
                if remaining <= 0:
                    break
                current_chunk_size = min(chunk_size, remaining)
            
            # Collect chunk_size items from iterator
            for _ in range(current_chunk_size):
                try:
                    item = next(ds_iter)
                    chunk_data.append(item)
                except StopIteration:
                    # End of dataset reached
                    break
                except Exception as e:
                    logger.warning(f"Error reading dataset item: {e}")
                    continue
            
            if not chunk_data:
                print("\n✅ Reached end of dataset")
                break
            
            print(f"\n⚙️  Processing chunk {chunk_num} ({len(chunk_data)} structures)...")
            
            # Process chunk
            chunk_results = self._process_chunk(chunk_data, chunk_num)
            
            # Save chunk results
            self._save_chunk(chunk_results, chunk_num)
            
            # Update statistics
            self.processed_count += len(chunk_data)
            elapsed_time = time.time() - start_time
            avg_time_per_structure = elapsed_time / self.processed_count
            
            print(f"   Chunk {chunk_num} complete: {self.success_count}/{self.processed_count} successful")
            print(f"   Average time per structure: {avg_time_per_structure:.4f}s")
            
            # Progress update every 10 chunks
            if chunk_num % 10 == 0:
                self._print_progress_summary(elapsed_time)
        
        # Final processing
        self._finalize_processing(start_time)
    
    def _process_chunk(self, chunk_data: List[Dict], chunk_num: int) -> List[Dict[str, Any]]:
        """Process a single chunk of data with enhanced error categorization.
        
        Parameters
        ----------
        chunk_data : List[Dict]
            List of LeMat-Bulk items to process.
        chunk_num : int
            Chunk number for logging.
            
        Returns
        -------
        List[Dict[str, Any]]
            List of processed results.
        """
        results = []
        
        for i, item in enumerate(chunk_data):
            try:
                # Process single structure
                result = self._process_single_structure(item, chunk_num, i)
                results.append(result)
                
                if result['success']:
                    self.success_count += 1
                    
                    # Track processing method
                    method = result.get('processing_method', 'unknown')
                    self.processing_methods[method] = self.processing_methods.get(method, 0) + 1
                    
                    # Track enumeration statistics
                    if 'enumeration_count' in result:
                        enum_count = result['enumeration_count']
                        self.enumeration_stats['total_enumerations'] += enum_count
                        self.enumeration_stats['max_enumerations'] = max(
                            self.enumeration_stats['max_enumerations'], enum_count
                        )
                        self.enumeration_stats['min_enumerations'] = min(
                            self.enumeration_stats['min_enumerations'], enum_count
                        )
                    
                    # Track fingerprint uniqueness
                    fingerprint = result['fingerprint_string']
                    if fingerprint in self.fingerprint_set:
                        self.duplicate_fingerprints += 1
                    else:
                        self.fingerprint_set.add(fingerprint)
                else:
                    self.failed_count += 1
                    
                    # Track processing method for failures too
                    method = result.get('processing_method', 'unknown')
                    self.processing_methods[method] = self.processing_methods.get(method, 0) + 1
                    
                    # Categorize errors
                    error_type = result.get('error_type', 'unknown')
                    self.error_categories[error_type] = self.error_categories.get(error_type, 0) + 1
                
            except Exception as e:
                logger.error(f"Chunk {chunk_num}, item {i}: Unexpected error: {e}")
                results.append({
                    'immutable_id': item.get('immutable_id', 'unknown'),
                    'success': False,
                    'error': str(e),
                    'error_type': 'unexpected_exception',
                    'processing_method': 'failed',
                    'chunk_num': chunk_num,
                    'item_index': i
                })
                self.failed_count += 1
                self.error_categories['unexpected_exception'] = self.error_categories.get('unexpected_exception', 0) + 1
                self.processing_methods['failed'] = self.processing_methods.get('failed', 0) + 1
        
        return results
    
    def _process_single_structure(self, item: Dict, chunk_num: int, item_index: int) -> Dict[str, Any]:
        """Process a single LeMat-Bulk structure with enhanced analysis.
        
        This method uses multiple approaches for robustness:
        1. Primary: Direct fingerprinting with get_augmented_fingerprint()
        2. Fallback: Enhanced crystallographic analysis with analyze_lematbulk_item()
        
        Parameters
        ----------
        item : Dict
            LeMat-Bulk item.
        chunk_num : int
            Chunk number.
        item_index : int
            Index within chunk.
            
        Returns
        -------
        Dict[str, Any]
            Processing result with all necessary information.
        """
        immutable_id = item.get('immutable_id', 'unknown')
        
        try:
            # Method 1: Try direct fingerprinting (most efficient)
            try:
                structure = lematbulk_item_to_structure(item)
                
                # Use the convenience function for direct fingerprinting
                fingerprint_string = get_augmented_fingerprint(
                    structure, 
                    symprec=self.symprec, 
                    angle_tolerance=self.angle_tolerance
                )
                
                if fingerprint_string is not None and not fingerprint_string.startswith("AUG_FAILED"):
                    # Success with direct method - also get detailed crystallographic data if needed
                    crystal_data = None
                    if self.save_crystallographic_data:
                        crystal_data = structure_to_crystallographic_dict(
                            structure,
                            symprec=self.symprec,
                            angle_tolerance=self.angle_tolerance
                        )
                    
                    # Prepare structure data for potential StructureMatcher use
                    structure_data = {
                        'lattice_matrix': structure.lattice.matrix.tolist(),
                        'species': [str(site.specie) for site in structure],
                        'coords': structure.frac_coords.tolist(),
                        'num_sites': len(structure)
                    }
                    
                    # Extract composition metadata
                    composition = structure.composition
                    
                    # Prepare result with direct fingerprinting
                    result = {
                        'immutable_id': immutable_id,
                        'success': True,
                        'fingerprint_string': fingerprint_string,
                        'structure_data': structure_data,
                        'chunk_num': chunk_num,
                        'item_index': item_index,
                        'error': None,
                        'error_type': None,
                        'processing_method': 'direct_fingerprinting',
                        
                        # Basic structure metadata
                        'formula': composition.reduced_formula,
                        'elements': [str(el) for el in composition.element_composition.keys()],
                        'num_elements': len(composition.elements),
                        'num_sites': len(structure),
                        'density': structure.density,
                        'volume': structure.volume,
                    }
                    
                    # Add detailed crystallographic data if available and requested
                    if crystal_data and self.save_crystallographic_data:
                        result.update({
                            'spacegroup_number': crystal_data.get('spacegroup_number'),
                            'site_symmetries': crystal_data.get('site_symmetries', []),
                            'multiplicity': crystal_data.get('multiplicity', []),
                            'enumeration_count': len(crystal_data.get('sites_enumeration_augmented', [])),
                            'fingerprint_tuple': record_to_augmented_fingerprint(crystal_data),
                            'crystallographic_data': {
                                key: value for key, value in crystal_data.items() 
                                if key not in ['immutable_id', 'original_item']
                            }
                        })
                    
                    return result
                
            except Exception as e:
                logger.debug(f"Direct fingerprinting failed for {immutable_id}: {e}")
                # Fall through to enhanced method
            
            # Method 2: Enhanced crystallographic analysis (fallback)
            logger.debug(f"Using enhanced crystallographic analysis for {immutable_id}")
            
            crystal_data = analyze_lematbulk_item(
                item, 
                symprec=self.symprec, 
                angle_tolerance=self.angle_tolerance
            )
            
            if not crystal_data.get('success', False):
                self.crystallographic_analysis_failures += 1
                return {
                    'immutable_id': immutable_id,
                    'success': False,
                    'error': crystal_data.get('error', 'Crystallographic analysis failed'),
                    'error_type': 'crystallographic_analysis',
                    'processing_method': 'enhanced_analysis',
                    'chunk_num': chunk_num,
                    'item_index': item_index
                }
            
            # Generate augmented fingerprint from crystallographic data
            fingerprint_tuple = record_to_augmented_fingerprint(crystal_data)
            fingerprint_string = self.fingerprinter._fingerprint_to_string(fingerprint_tuple)
            
            if fingerprint_string is None or fingerprint_string.startswith("AUG_FAILED"):
                self.fingerprint_generation_failures += 1
                return {
                    'immutable_id': immutable_id,
                    'success': False,
                    'error': 'Fingerprint generation failed',
                    'error_type': 'fingerprint_generation',
                    'processing_method': 'enhanced_analysis',
                    'chunk_num': chunk_num,
                    'item_index': item_index
                }
            
            # Get structure metadata
            try:
                structure = lematbulk_item_to_structure(item)
                
                # Prepare structure data for potential StructureMatcher use
                structure_data = {
                    'lattice_matrix': structure.lattice.matrix.tolist(),
                    'species': [str(site.specie) for site in structure],
                    'coords': structure.frac_coords.tolist(),
                    'num_sites': len(structure)
                }
                
                # Extract additional metadata
                composition = structure.composition
                density = structure.density
                volume = structure.volume
                
            except Exception as e:
                logger.warning(f"Failed to extract structure metadata for {immutable_id}: {e}")
                structure_data = None
                density = None
                volume = None
                composition = None
            
            # Prepare result with enhanced analysis
            result = {
                'immutable_id': immutable_id,
                'success': True,
                'fingerprint_string': fingerprint_string,
                'fingerprint_tuple': fingerprint_tuple,
                'structure_data': structure_data,
                'chunk_num': chunk_num,
                'item_index': item_index,
                'error': None,
                'error_type': None,
                'processing_method': 'enhanced_analysis',
                
                # Crystallographic analysis results
                'spacegroup_number': crystal_data.get('spacegroup_number'),
                'elements': crystal_data.get('elements', []),
                'site_symmetries': crystal_data.get('site_symmetries', []),
                'multiplicity': crystal_data.get('multiplicity', []),
                'enumeration_count': len(crystal_data.get('sites_enumeration_augmented', [])),
                
                # Structure metadata (if available)
                'formula': composition.reduced_formula if composition else crystal_data.get('formula', 'Unknown'),
                'num_elements': len(crystal_data.get('elements', [])),
                'num_sites': crystal_data.get('num_sites', 0),
                'density': density,
                'volume': volume,
            }
            
            # Optionally include detailed crystallographic data
            if self.save_crystallographic_data:
                result['crystallographic_data'] = {
                    key: value for key, value in crystal_data.items() 
                    if key not in ['immutable_id', 'original_item']  # Exclude redundant data
                }
            
            return result
            
        except Exception as e:
            return {
                'immutable_id': immutable_id,
                'success': False,
                'error': str(e),
                'error_type': 'processing_exception',
                'processing_method': 'failed',
                'chunk_num': chunk_num,
                'item_index': item_index
            }
    
    def _save_chunk(self, chunk_results: List[Dict], chunk_num: int) -> None:
        """Save chunk results to disk with enhanced data organization.
        
        Parameters
        ----------
        chunk_results : List[Dict]
            Results from processing a chunk.
        chunk_num : int
            Chunk number.
        """
        # Save as both pickle (for fast loading) and CSV (for inspection)
        
        # Pickle file (includes full structure data and crystallographic analysis)
        pickle_file = os.path.join(self.output_dir, f"lemat_bulk_fingerprints_chunk_{chunk_num:04d}.pkl")
        with open(pickle_file, 'wb') as f:
            pickle.dump(chunk_results, f)
        
        # CSV file (for inspection, without complex nested data)
        csv_data = []
        for result in chunk_results:
            csv_row = {k: v for k, v in result.items() 
                      if k not in ['structure_data', 'crystallographic_data', 'fingerprint_tuple']}
            
            # Convert lists to strings for CSV
            if 'elements' in csv_row and isinstance(csv_row['elements'], list):
                csv_row['elements'] = '|'.join(csv_row['elements'])
            if 'site_symmetries' in csv_row and isinstance(csv_row['site_symmetries'], list):
                csv_row['site_symmetries'] = '|'.join(csv_row['site_symmetries'])
            if 'multiplicity' in csv_row and isinstance(csv_row['multiplicity'], list):
                csv_row['multiplicity'] = '|'.join(map(str, csv_row['multiplicity']))
                
            csv_data.append(csv_row)
        
        csv_file = os.path.join(self.output_dir, f"lemat_bulk_fingerprints_chunk_{chunk_num:04d}.csv")
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        
        # Save fingerprints only (for fast novelty detection)
        successful_results = [r for r in chunk_results if r['success']]
        if successful_results:
            fingerprints_only = {
                r['immutable_id']: r['fingerprint_string'] 
                for r in successful_results
            }
            fp_file = os.path.join(self.output_dir, f"fingerprints_only_chunk_{chunk_num:04d}.pkl")
            with open(fp_file, 'wb') as f:
                pickle.dump(fingerprints_only, f)
        
        # Log successful save
        successful_in_chunk = sum(1 for r in chunk_results if r['success'])
        logger.info(f"Saved chunk {chunk_num}: {successful_in_chunk}/{len(chunk_results)} successful to {pickle_file}")
    
    def _print_progress_summary(self, elapsed_time: float) -> None:
        """Print enhanced progress summary.
        
        Parameters
        ----------
        elapsed_time : float
            Elapsed time since start.
        """
        print("\n📊 Enhanced Progress Summary:")
        print(f"   Processed: {self.processed_count:,} structures")
        print(f"   Successful: {self.success_count:,} ({self.success_count/self.processed_count*100:.1f}%)")
        print(f"   Failed: {self.failed_count:,} ({self.failed_count/self.processed_count*100:.1f}%)")
        print(f"     - Crystallographic analysis failures: {self.crystallographic_analysis_failures:,}")
        print(f"     - Fingerprint generation failures: {self.fingerprint_generation_failures:,}")
        print(f"   Unique fingerprints: {len(self.fingerprint_set):,}")
        print(f"   Duplicate fingerprints: {self.duplicate_fingerprints:,}")
        
        # Processing method breakdown
        if self.processing_methods:
            print("   Processing method breakdown:")
            for method, count in self.processing_methods.items():
                if count > 0:
                    percentage = count / self.processed_count * 100 if self.processed_count > 0 else 0
                    print(f"     - {method}: {count:,} ({percentage:.1f}%)")
        
        # Enhanced enumeration statistics
        if self.success_count > 0:
            avg_enums = self.enumeration_stats['total_enumerations'] / self.success_count
            print("   Enumeration statistics:")
            print(f"     - Average per structure: {avg_enums:.1f}")
            print(f"     - Max enumerations: {self.enumeration_stats['max_enumerations']}")
            print(f"     - Min enumerations: {self.enumeration_stats['min_enumerations']}")
        
        print(f"   Elapsed time: {elapsed_time/3600:.2f} hours")
        if self.processed_count > 0:
            est_total_time = elapsed_time / self.processed_count * 5_400_000 / 3600
            print(f"   Estimated total completion: {est_total_time:.1f} hours")
        
        # Error breakdown
        if self.error_categories:
            print("   Error breakdown:")
            for error_type, count in self.error_categories.items():
                print(f"     - {error_type}: {count:,}")
    
    def _finalize_processing(self, start_time: float) -> None:
        """Finalize processing with enhanced summary and analysis.
        
        Parameters
        ----------
        start_time : float
            Start time timestamp.
        """
        total_time = time.time() - start_time
        
        # Calculate enhanced statistics
        avg_enumerations = (
            self.enumeration_stats['total_enumerations'] / self.success_count 
            if self.success_count > 0 else 0
        )
        
        print("\n🎯 Enhanced Final Processing Summary:")
        print(f"   Total structures processed: {self.processed_count:,}")
        print(f"   Successful: {self.success_count:,} ({self.success_count/self.processed_count*100:.1f}%)")
        print(f"   Failed: {self.failed_count:,} ({self.failed_count/self.processed_count*100:.1f}%)")
        print(f"     - Crystallographic analysis failures: {self.crystallographic_analysis_failures:,}")
        print(f"     - Fingerprint generation failures: {self.fingerprint_generation_failures:,}")
        print(f"   Unique fingerprints: {len(self.fingerprint_set):,}")
        print(f"   Duplicate fingerprints: {self.duplicate_fingerprints:,}")
        print(f"   Average enumerations per structure: {avg_enumerations:.1f}")
        print(f"   Total processing time: {total_time/3600:.2f} hours")
        print(f"   Average time per structure: {total_time/self.processed_count:.4f}s")
        
        # Processing method analysis
        if self.processing_methods:
            print("\n⚙️ Processing Method Analysis:")
            for method, count in sorted(self.processing_methods.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    percentage = count / self.processed_count * 100 if self.processed_count > 0 else 0
                    print(f"     {method}: {count:,} ({percentage:.1f}% of total)")
        
        # Error analysis
        if self.error_categories:
            print("\n❌ Error Analysis:")
            for error_type, count in sorted(self.error_categories.items(), key=lambda x: x[1], reverse=True):
                percentage = count / self.failed_count * 100 if self.failed_count > 0 else 0
                print(f"     {error_type}: {count:,} ({percentage:.1f}% of failures)")
        
        # Create enhanced summary metadata
        summary = {
            'dataset_name': self.dataset_name,
            'config_name': self.config_name,
            'processing_parameters': {
                'symprec': self.symprec,
                'angle_tolerance': self.angle_tolerance,
                'save_crystallographic_data': self.save_crystallographic_data,
            },
            'total_processed': self.processed_count,
            'successful_count': self.success_count,
            'failed_count': self.failed_count,
            'crystallographic_analysis_failures': self.crystallographic_analysis_failures,
            'fingerprint_generation_failures': self.fingerprint_generation_failures,
            'unique_fingerprints': len(self.fingerprint_set),
            'duplicate_fingerprints': self.duplicate_fingerprints,
            'success_rate': self.success_count / self.processed_count if self.processed_count > 0 else 0,
            'uniqueness_rate': len(self.fingerprint_set) / self.success_count if self.success_count > 0 else 0,
            'processing_time_hours': total_time / 3600,
            'avg_time_per_structure': total_time / self.processed_count if self.processed_count > 0 else 0,
            'processing_methods': self.processing_methods,
            'enumeration_statistics': {
                'average_per_structure': avg_enumerations,
                'max_enumerations': self.enumeration_stats['max_enumerations'],
                'min_enumerations': self.enumeration_stats['min_enumerations'] if self.enumeration_stats['min_enumerations'] != float('inf') else 0,
                'total_enumerations': self.enumeration_stats['total_enumerations'],
            },
            'error_categories': self.error_categories,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'output_directory': self.output_dir
        }
        
        # Save enhanced summary
        summary_file = os.path.join(self.output_dir, "processing_summary.pkl")
        with open(summary_file, 'wb') as f:
            pickle.dump(summary, f)
        
        # Save summary as human-readable JSON
        import json
        summary_json_file = os.path.join(self.output_dir, "processing_summary.json")
        with open(summary_json_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save fingerprint set for fast novelty detection
        fingerprint_set_file = os.path.join(self.output_dir, "unique_fingerprints.pkl")
        with open(fingerprint_set_file, 'wb') as f:
            pickle.dump(self.fingerprint_set, f)
        
        # Save fingerprint mapping (id -> fingerprint) for detailed analysis
        if self.success_count > 0:
            fingerprint_mapping_file = os.path.join(self.output_dir, "fingerprint_mapping.pkl")
            # This would require collecting mappings during processing - for now just note it
            logger.info(f"Consider creating fingerprint mapping file at {fingerprint_mapping_file}")
        
        print(f"\n💾 Saved processing summary to: {summary_file}")
        print(f"💾 Saved summary JSON to: {summary_json_file}")
        print(f"💾 Saved unique fingerprints to: {fingerprint_set_file}")
        print(f"📁 All chunk files saved in: {self.output_dir}")
        
        print("\n✅ Enhanced LeMat-Bulk fingerprint preprocessing completed!")


def main():
    """Main function for enhanced preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced LeMat-Bulk fingerprint preprocessing")
    parser.add_argument("--output-dir", default="data/reference_fingerprints_new",
                       help="Output directory for fingerprint files")
    parser.add_argument("--chunk-size", type=int, default=10000,
                       help="Number of structures per chunk")
    parser.add_argument("--max-structures", type=int, default=None,
                       help="Maximum structures to process (for testing)")
    parser.add_argument("--symprec", type=float, default=0.01,
                       help="Symmetry precision for crystallographic analysis")
    parser.add_argument("--angle-tolerance", type=float, default=5.0,
                       help="Angle tolerance in degrees for crystallographic analysis")
    parser.add_argument("--no-crystallographic-data", action="store_true",
                       help="Don't save detailed crystallographic analysis data")
    parser.add_argument("--test-run", action="store_true",
                       help="Run test with 1000 structures")
    
    args = parser.parse_args()
    
    if args.test_run:
        args.max_structures = 1000
        args.output_dir = "data/test_reference_fingerprints_new"
        print("🧪 Running in test mode with 1000 structures")
    
    # Initialize enhanced processor
    processor = LematBulkFingerprintProcessor(
        output_dir=args.output_dir,
        symprec=args.symprec,
        angle_tolerance=args.angle_tolerance,
        save_crystallographic_data=not args.no_crystallographic_data
    )
    
    # Process dataset
    processor.process_dataset_streaming(
        chunk_size=args.chunk_size,
        max_structures=args.max_structures
    )


if __name__ == "__main__":
    main()