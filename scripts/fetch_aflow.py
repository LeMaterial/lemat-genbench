#!/usr/bin/env python3
"""Fetch structures from AFLOW database.

Fetches exactly 2500 structures from AFLOW's ICSD catalog and saves them as CIF files
in baseline_data/aflow.
"""

import logging
import pickle
import requests
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

from pymatgen.core import Composition, Structure
from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aflow_fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def is_inorganic(elements: List[str], formula: str) -> bool:
    """Check if a composition is inorganic."""
    # Exclude organic compounds (those with C-H bonds)
    has_carbon = 'C' in elements
    has_hydrogen = 'H' in elements
    
    if has_carbon and has_hydrogen:
        # Allow carbides and hydrides (single C or H with metals)
        if len(elements) == 2:
            return True
        # Exclude organic compounds
        return False
    
    return True


def save_checkpoint(data: Dict, checkpoint_name: str, checkpoint_dir: Path):
    """Save progress checkpoint."""
    checkpoint_path = checkpoint_dir / f"{checkpoint_name}.pkl"
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)


def load_checkpoint(checkpoint_name: str, checkpoint_dir: Path) -> Dict:
    """Load progress checkpoint."""
    checkpoint_path = checkpoint_dir / f"{checkpoint_name}.pkl"
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return None


def fetch_aflow_structures(target_count: int = 2500) -> Tuple[List[Structure], List[Dict]]:
    """Fetch structures from AFLOW.
    
    Parameters
    ----------
    target_count : int
        Number of structures to fetch (default: 2500)
        
    Returns
    -------
    structures : list
        List of pymatgen Structure objects
    metadata : list
        List of metadata dictionaries
    """
    # Setup directories
    project_root = Path(__file__).parent.parent
    checkpoint_dir = project_root / "baseline_data" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint("aflow_structures_2500", checkpoint_dir)
    if checkpoint:
        logger.info(f"Loaded {len(checkpoint['structures'])} AFLOW structures from checkpoint")
        return checkpoint['structures'], checkpoint['metadata']
    
    logger.info(f"Fetching {target_count} structures from AFLOW...")
    logger.info("Estimated time: 2-3 hours...")
    
    structures = []
    metadata = []
    
    # Noble gas filter
    noble_gases = {'He', 'Ne', 'Ar', 'Kr', 'Rn'}
    
    # Track statistics
    total_processed = 0
    filtered_by_api = 0
    
    # AFLOW API settings
    base_url = "http://aflow.org/API/aflux/"
    page = 0
    max_pages = 1000
    
    pbar = tqdm(total=target_count, desc="Fetching AFLOW")
    
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    try:
        while len(structures) < target_count and page < max_pages:
            try:
                # Query AFLOW
                query = f"catalog(ICSD),paging({page})"
                url = f"{base_url}?{query}"
                
                response = requests.get(url, timeout=30)
                
                if response.status_code != 200:
                    logger.warning(f"Page {page} returned status {response.status_code}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("Too many consecutive failures. Stopping.")
                        break
                    page += 1
                    continue
                
                consecutive_failures = 0
                
                # Parse JSON response
                import json
                try:
                    data = json.loads(response.text)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from page {page}: {e}")
                    page += 1
                    continue
                
                if not data or len(data) == 0:
                    logger.warning(f"Page {page} returned no entries")
                    page += 1
                    continue
                
                logger.info(f"Page {page}: Processing {len(data)} entries...")
                
                # AFLOW returns {"1 of N": {...}, "2 of N": {...}, ...}
                for key, entry in data.items():
                    try:
                        total_processed += 1
                        
                        # Get compound/formula
                        compound = entry.get('compound', '')
                        if not compound:
                            continue
                        
                        # Parse composition
                        comp = Composition(compound)
                        elements = [str(el) for el in comp.elements]
                        
                        # Check if inorganic
                        if not is_inorganic(elements, compound):
                            continue
                        
                        # Filter out exotic compositions
                        # Exclude noble gases
                        if any(el in noble_gases for el in elements):
                            filtered_by_api += 1
                            logger.debug(f"Filtered noble gas compound: {compound}")
                            continue
                        
                        # Exclude pure hydrogen structures
                        if set(elements) == {'H'}:
                            filtered_by_api += 1
                            logger.debug(f"Filtered pure H compound: {compound}")
                            continue
                        
                        # Get AURL and fetch CONTCAR
                        aurl = entry.get('aurl', '')
                        if not aurl:
                            continue
                        
                        # Fix AURL
                        aurl_fixed = aurl.replace('aflowlib.duke.edu:AFLOWDATA', 'aflow.org')
                        
                        # Fetch CONTCAR.relax
                        contcar_url = f"http://{aurl_fixed}/CONTCAR.relax"
                        contcar_response = requests.get(contcar_url, timeout=8)
                        
                        if contcar_response.status_code != 200:
                            # Try POSCAR as fallback
                            poscar_url = f"http://{aurl_fixed}/POSCAR"
                            poscar_response = requests.get(poscar_url, timeout=8)
                            if poscar_response.status_code != 200:
                                continue
                            contcar_text = poscar_response.text
                        else:
                            contcar_text = contcar_response.text
                        
                        # Parse POSCAR to Structure
                        # Strategy: Try auto-detection first (VASP5), then provide element names
                        try:
                            # First try: let pymatgen auto-detect (works for VASP5 format)
                            poscar = Poscar.from_str(contcar_text)
                            structure = poscar.structure
                            
                            # Verify composition matches what AFLOW reports
                            parsed_formula = structure.composition.reduced_formula
                            expected_formula = comp.reduced_formula
                            
                            if parsed_formula != expected_formula:
                                # Auto-detection gave wrong elements, provide them explicitly
                                # AFLOW typically uses alphabetically sorted element order
                                elements_alphabetical = sorted(elements)
                                poscar = Poscar.from_str(contcar_text, default_names=elements_alphabetical)
                                structure = poscar.structure
                                
                                # Verify again
                                if structure.composition.reduced_formula != expected_formula:
                                    logger.warning(f"Composition mismatch for {compound}: "
                                                 f"expected {expected_formula}, got {structure.composition.reduced_formula}")
                                    continue
                                    
                        except Exception as e:
                            logger.debug(f"Failed to parse POSCAR for {compound}: {e}")
                            continue
                        
                        # Get metadata
                        auid = entry.get('auid', f'aflow_{len(structures)}')
                        spacegroup = entry.get('spacegroup_relax', entry.get('sg', 'unknown'))
                        
                        structures.append(structure)
                        metadata.append({
                            'material_id': auid,
                            'formula': compound,
                            'elements': ','.join(elements),
                            'nsites': len(structure),
                            'density': structure.density,
                            'spacegroup': spacegroup
                        })
                        
                        if len(structures) <= 5:
                            logger.info(f"Successfully added structure {len(structures)}: {compound}")
                        
                        pbar.update(1)
                        
                        # Log progress every 100 structures
                        if len(structures) % 100 == 0:
                            save_checkpoint(
                                {'structures': structures, 'metadata': metadata},
                                "aflow_structures_2500_partial",
                                checkpoint_dir
                            )
                            logger.info(f"Progress: {len(structures)}/{target_count} structures "
                                      f"({total_processed} processed, {filtered_by_api} filtered)")
                        
                        # Check if we reached target
                        if len(structures) >= target_count:
                            break
                    
                    except Exception as e:
                        logger.debug(f"Error processing entry: {e}")
                        continue
                
                page += 1
                
            except Exception as e:
                logger.error(f"Error fetching page {page}: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Too many consecutive failures. Stopping.")
                    break
                page += 1
                continue
        
        pbar.close()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        pbar.close()
    
    logger.info(f"AFLOW: {len(structures)} structures fetched")
    if total_processed > 0:
        logger.info(f"Filtering stats - Total processed: {total_processed}, "
                   f"Filtered: {filtered_by_api}, "
                   f"Success rate: {len(structures)/total_processed*100:.1f}%")
    
    if len(structures) == 0:
        logger.warning("AFLOW returned 0 structures. API may be unavailable.")
    
    # Save final checkpoint
    save_checkpoint(
        {'structures': structures, 'metadata': metadata},
        "aflow_structures_2500",
        checkpoint_dir
    )
    
    return structures, metadata


def save_structures(structures: List[Structure], metadata: List[Dict], output_dir: Path):
    """Save structures as CIF files.
    
    Parameters
    ----------
    structures : list
        List of Structure objects
    metadata : list
        List of metadata dictionaries
    output_dir : Path
        Directory to save CIF files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(structures)} structures to {output_dir}...")
    
    # Save metadata CSV
    import csv
    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['material_id', 'formula', 'elements', 'nsites', 'density', 'spacegroup'])
        writer.writeheader()
        writer.writerows(metadata)
    
    logger.info(f"Saved metadata to {metadata_path}")
    
    # Save CIF files
    for i, (structure, meta) in enumerate(tqdm(zip(structures, metadata), total=len(structures), desc="Saving CIFs")):
        try:
            material_id = meta['material_id']
            cif_filename = f"aflow_{material_id}.cif"
            cif_path = output_dir / cif_filename
            
            writer = CifWriter(structure)
            writer.write_file(str(cif_path))
            
        except Exception as e:
            logger.error(f"Failed to save structure {i}: {e}")
            continue
    
    logger.info(f"Saved {len(structures)} CIF files to {output_dir}")


def main():
    """Main function."""
    logger.info("="*80)
    logger.info("AFLOW Structure Fetcher")
    logger.info("="*80)
    
    # Fetch structures
    structures, metadata = fetch_aflow_structures(target_count=2500)
    
    if len(structures) == 0:
        logger.error("No structures fetched. Exiting.")
        return
    
    # Save structures
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "baseline_data" / "aflow"
    save_structures(structures, metadata, output_dir)
    
    logger.info("="*80)
    logger.info("DONE!")
    logger.info(f"Fetched and saved {len(structures)} AFLOW structures")
    logger.info("="*80)


if __name__ == "__main__":
    main()

