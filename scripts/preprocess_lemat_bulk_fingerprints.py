#!/usr/bin/env python3
"""Preprocess LeMat-Bulk dataset to generate augmented fingerprints for reference.

This script processes the entire LeMat-Bulk dataset to generate augmented fingerprints
and saves both fingerprints and structure information needed for novelty detection
with StructureMatcher fallback support.
"""

import os
import pickle
import time
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset

from lemat_genbench.fingerprinting.augmented_fingerprint import (
    get_augmented_fingerprint,
)
from lemat_genbench.fingerprinting.crystallographic_analyzer import (
    lematbulk_item_to_structure,
)
from lemat_genbench.utils.logging import logger


class LematBulkFingerprintProcessor:
    """Processor for generating LeMat-Bulk reference fingerprints."""
    
    def __init__(self, output_dir: str = "data/reference_fingerprints"):
        """Initialize the processor.
        
        Parameters
        ----------
        output_dir : str, default="data/reference_fingerprints"
            Directory to save reference fingerprint files.
        """
        self.output_dir = output_dir
        self.dataset_name = "LeMaterial/LeMat-Bulk"
        self.config_name = "compatible_pbe"
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Statistics tracking
        self.processed_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.duplicate_fingerprints = 0
        self.fingerprint_set = set()
        
    def process_dataset_streaming(self, chunk_size: int = 10000, max_structures: int = None) -> None:
        """Process LeMat-Bulk dataset in streaming chunks."""
        print("🚀 Starting LeMat-Bulk fingerprint preprocessing...")
        print(f"   Dataset: {self.dataset_name}")
        print(f"   Config: {self.config_name}")
        print(f"   Chunk size: {chunk_size}")
        print(f"   Output directory: {self.output_dir}")
        if max_structures:
            print(f"   Max structures: {max_structures}")
        
        # Load dataset in streaming mode
        print("\n📁 Loading LeMat-Bulk dataset...")
        ds_stream = load_dataset(
            self.dataset_name, 
            name=self.config_name, 
            split="train", 
            streaming=True
        )
        
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
        """Process a single chunk of data.
        
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
                    
                    # Track fingerprint uniqueness
                    fingerprint = result['fingerprint_string']
                    if fingerprint in self.fingerprint_set:
                        self.duplicate_fingerprints += 1
                    else:
                        self.fingerprint_set.add(fingerprint)
                else:
                    self.failed_count += 1
                
            except Exception as e:
                logger.error(f"Chunk {chunk_num}, item {i}: Unexpected error: {e}")
                results.append({
                    'immutable_id': item.get('immutable_id', 'unknown'),
                    'success': False,
                    'error': str(e),
                    'chunk_num': chunk_num,
                    'item_index': i
                })
                self.failed_count += 1
        
        return results
    
    def _process_single_structure(self, item: Dict, chunk_num: int, item_index: int) -> Dict[str, Any]:
        """Process a single LeMat-Bulk structure.
        
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
        try:
            # Extract basic info
            immutable_id = item.get('immutable_id', 'unknown')
            
            # Convert to pymatgen Structure
            structure = lematbulk_item_to_structure(item)
            
            # Generate augmented fingerprint
            fingerprint_string = get_augmented_fingerprint(structure)
            
            if fingerprint_string is None:
                return {
                    'immutable_id': immutable_id,
                    'success': False,
                    'error': 'Fingerprint generation failed',
                    'chunk_num': chunk_num,
                    'item_index': item_index
                }
            
            # Prepare structure data for potential StructureMatcher use
            # We'll store the structure in a compact format
            structure_data = {
                'lattice_matrix': structure.lattice.matrix.tolist(),
                'species': [str(site.specie) for site in structure],
                'coords': structure.frac_coords.tolist(),
                'num_sites': len(structure)
            }
            
            # Extract additional metadata
            composition = structure.composition
            
            return {
                'immutable_id': immutable_id,
                'success': True,
                'fingerprint_string': fingerprint_string,
                'structure_data': structure_data,
                'formula': composition.reduced_formula,
                'elements': [str(el) for el in composition.element_composition.keys()],
                'num_elements': len(composition.elements),
                'num_sites': len(structure),
                'density': structure.density,
                'volume': structure.volume,
                'chunk_num': chunk_num,
                'item_index': item_index,
                'error': None
            }
            
        except Exception as e:
            return {
                'immutable_id': item.get('immutable_id', 'unknown'),
                'success': False,
                'error': str(e),
                'chunk_num': chunk_num,
                'item_index': item_index
            }
    
    def _save_chunk(self, chunk_results: List[Dict], chunk_num: int) -> None:
        """Save chunk results to disk.
        
        Parameters
        ----------
        chunk_results : List[Dict]
            Results from processing a chunk.
        chunk_num : int
            Chunk number.
        """
        # Save as both pickle (for fast loading) and CSV (for inspection)
        
        # Pickle file (includes full structure data)
        pickle_file = os.path.join(self.output_dir, f"lemat_bulk_fingerprints_chunk_{chunk_num:04d}.pkl")
        with open(pickle_file, 'wb') as f:
            pickle.dump(chunk_results, f)
        
        # CSV file (for inspection, without structure data)
        csv_data = []
        for result in chunk_results:
            csv_row = {k: v for k, v in result.items() if k != 'structure_data'}
            # Convert lists to strings for CSV
            if 'elements' in csv_row and isinstance(csv_row['elements'], list):
                csv_row['elements'] = '|'.join(csv_row['elements'])
            csv_data.append(csv_row)
        
        csv_file = os.path.join(self.output_dir, f"lemat_bulk_fingerprints_chunk_{chunk_num:04d}.csv")
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        
        # Log successful save
        successful_in_chunk = sum(1 for r in chunk_results if r['success'])
        logger.info(f"Saved chunk {chunk_num}: {successful_in_chunk}/{len(chunk_results)} successful to {pickle_file}")
    
    def _print_progress_summary(self, elapsed_time: float) -> None:
        """Print progress summary.
        
        Parameters
        ----------
        elapsed_time : float
            Elapsed time since start.
        """
        print("\n📊 Progress Summary:")
        print(f"   Processed: {self.processed_count:,} structures")
        print(f"   Successful: {self.success_count:,} ({self.success_count/self.processed_count*100:.1f}%)")
        print(f"   Failed: {self.failed_count:,} ({self.failed_count/self.processed_count*100:.1f}%)")
        print(f"   Unique fingerprints: {len(self.fingerprint_set):,}")
        print(f"   Duplicate fingerprints: {self.duplicate_fingerprints:,}")
        print(f"   Elapsed time: {elapsed_time/3600:.2f} hours")
        print(f"   Estimated completion: {elapsed_time/self.processed_count * 5_400_000 / 3600:.1f} hours")
    
    def _finalize_processing(self, start_time: float) -> None:
        """Finalize processing and create summary files.
        
        Parameters
        ----------
        start_time : float
            Start time timestamp.
        """
        total_time = time.time() - start_time
        
        print("\n🎯 Final Processing Summary:")
        print(f"   Total structures processed: {self.processed_count:,}")
        print(f"   Successful: {self.success_count:,} ({self.success_count/self.processed_count*100:.1f}%)")
        print(f"   Failed: {self.failed_count:,} ({self.failed_count/self.processed_count*100:.1f}%)")
        print(f"   Unique fingerprints: {len(self.fingerprint_set):,}")
        print(f"   Duplicate fingerprints: {self.duplicate_fingerprints:,}")
        print(f"   Total processing time: {total_time/3600:.2f} hours")
        print(f"   Average time per structure: {total_time/self.processed_count:.4f}s")
        
        # Create summary metadata
        summary = {
            'dataset_name': self.dataset_name,
            'config_name': self.config_name,
            'total_processed': self.processed_count,
            'successful_count': self.success_count,
            'failed_count': self.failed_count,
            'unique_fingerprints': len(self.fingerprint_set),
            'duplicate_fingerprints': self.duplicate_fingerprints,
            'success_rate': self.success_count / self.processed_count,
            'uniqueness_rate': len(self.fingerprint_set) / self.success_count if self.success_count > 0 else 0,
            'processing_time_hours': total_time / 3600,
            'avg_time_per_structure': total_time / self.processed_count,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'output_directory': self.output_dir
        }
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "processing_summary.pkl")
        with open(summary_file, 'wb') as f:
            pickle.dump(summary, f)
        
        # Save fingerprint set for fast novelty detection
        fingerprint_set_file = os.path.join(self.output_dir, "unique_fingerprints.pkl")
        with open(fingerprint_set_file, 'wb') as f:
            pickle.dump(self.fingerprint_set, f)
        
        print(f"\n💾 Saved processing summary to: {summary_file}")
        print(f"💾 Saved unique fingerprints to: {fingerprint_set_file}")
        print(f"📁 All chunk files saved in: {self.output_dir}")
        
        print( "\n✅ LeMat-Bulk fingerprint preprocessing completed!")


def main():
    """Main function for preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess LeMat-Bulk fingerprints")
    parser.add_argument("--output-dir", default="data/reference_fingerprints",
                       help="Output directory for fingerprint files")
    parser.add_argument("--chunk-size", type=int, default=10000,
                       help="Number of structures per chunk")
    parser.add_argument("--max-structures", type=int, default=None,
                       help="Maximum structures to process (for testing)")
    parser.add_argument("--test-run", action="store_true",
                       help="Run test with 1000 structures")
    
    args = parser.parse_args()
    
    if args.test_run:
        args.max_structures = 1000
        args.output_dir = "data/test_reference_fingerprints"
        print("🧪 Running in test mode with 1000 structures")
    
    # Initialize processor
    processor = LematBulkFingerprintProcessor(output_dir=args.output_dir)
    
    # Process dataset
    processor.process_dataset_streaming(
        chunk_size=args.chunk_size,
        max_structures=args.max_structures
    )


if __name__ == "__main__":
    main()