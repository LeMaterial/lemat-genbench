#!/usr/bin/env python
"""
Sample inorganic crystal structures from multiple databases.

This script:
1. Fetches all inorganic structures from Materials Project (smallest dataset)
2. Samples equal number from Alexandria, OQMD, AFLOW, NOMAD
3. From each equalized dataset, samples 2500 structures with replacement
4. Saves CIF files and metadata

Usage:
    python scripts/sample_baseline_datasets.py --mp-api-key YOUR_KEY
"""

import argparse
import json
import logging
import pickle
import random
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from datasets import load_dataset
from pymatgen.core import Composition, Structure
from tqdm import tqdm

# Suppress pymatgen POSCAR warnings (old VASP format without element names)
warnings.filterwarnings('ignore', message='.*Elements in POSCAR cannot be determined.*')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('baseline_sampling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_SEED = 42
FINAL_SAMPLE_SIZE = 2500
CHECKPOINT_DIR = Path("baseline_data/checkpoints")
OUTPUT_DIR = Path("baseline_data")


def set_seed(seed: int = RANDOM_SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to: {seed}")


def is_inorganic(elements: List[str], formula: str = "") -> bool:
    """
    Check if material is inorganic.
    
    Rules:
    - Both C and H present -> organic (exclude)
    - Only metals/metalloids/common nonmetals -> inorganic
    - Carbides (C without H) -> inorganic (include)
    - Hydrides (H without C) -> inorganic (include)
    """
    has_carbon = 'C' in elements
    has_hydrogen = 'H' in elements
    
    # Exclude organic materials (both C and H)
    if has_carbon and has_hydrogen:
        return False
    
    # Check for high carbon content (might be organic precursor)
    if formula and has_carbon:
        try:
            comp = Composition(formula)
            c_fraction = comp.get_atomic_fraction('C')
            if c_fraction > 0.6:  # >60% carbon likely organic
                return False
        except Exception:
            pass
    
    return True


def save_checkpoint(data: Any, checkpoint_name: str):
    """Save checkpoint data."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_name}.pkl"
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_name: str) -> Optional[Any]:
    """Load checkpoint data if exists."""
    checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_name}.pkl"
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return data
    return None


class MaterialsFetcher:
    """Base class for fetching materials from databases."""
    
    def __init__(self, database_name: str):
        self.database_name = database_name
        self.structures = []
        self.metadata = []
    
    def fetch_all_inorganic(self, target_count: Optional[int] = None) -> Tuple[List[Structure], List[Dict]]:
        """Fetch inorganic structures. Override in subclasses."""
        raise NotImplementedError
    
    def save_structures_as_cif(self, structures: List[Structure], metadata: List[Dict], output_dir: Path):
        """Save structures as CIF files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        failed_count = 0
        
        for struct, meta in tqdm(zip(structures, metadata), total=len(structures), 
                                  desc=f"Saving {self.database_name} CIFs"):
            try:
                material_id = meta['material_id']
                filename = f"{self.database_name}_{material_id}.cif"
                filepath = output_dir / filename
                
                # Save CIF
                struct.to(filename=str(filepath), fmt="cif")
                saved_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to save {meta.get('material_id', 'unknown')}: {e}")
                failed_count += 1
        
        logger.info(f"{self.database_name}: Saved {saved_count} CIFs, {failed_count} failed")
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(output_dir / "metadata.csv", index=False)
        logger.info(f"Metadata saved: {output_dir / 'metadata.csv'}")


class MaterialsProjectFetcher(MaterialsFetcher):
    """Fetch from Materials Project."""
    
    def __init__(self, api_key: str):
        super().__init__("mp")
        self.api_key = api_key
    
    def fetch_all_inorganic(self, target_count: Optional[int] = None) -> Tuple[List[Structure], List[Dict]]:
        """Fetch all inorganic structures from Materials Project."""
        checkpoint = load_checkpoint("mp_structures")
        if checkpoint:
            logger.info(f"Loaded {len(checkpoint['structures'])} MP structures from checkpoint")
            return checkpoint['structures'], checkpoint['metadata']
        
        try:
            from mp_api.client import MPRester
        except ImportError:
            raise ImportError("Please install mp-api: pip install mp-api")
        
        logger.info("Fetching from Materials Project...")
        structures = []
        metadata = []
        
        with MPRester(self.api_key) as mpr:
            # Fetch all materials excluding hydrogen (to avoid most organics)
            docs = mpr.materials.summary.search(
                num_elements=(1, 10),
                fields=[
                    "material_id",
                    "formula_pretty",
                    "structure",
                    "elements",
                    "nsites",
                    "density",
                    "symmetry"
                ]
            )
            
            logger.info(f"Downloaded {len(docs)} materials from MP")
            
            # Filter for inorganic
            for doc in tqdm(docs, desc="Filtering MP inorganic"):
                try:
                    elements = [str(el) for el in doc.elements]
                    formula = doc.formula_pretty
                    
                    if is_inorganic(elements, formula):
                        structures.append(doc.structure)
                        metadata.append({
                            'material_id': doc.material_id,
                            'formula': formula,
                            'elements': ','.join(elements),
                            'nsites': doc.nsites,
                            'density': doc.density,
                            'spacegroup': doc.symmetry.symbol if doc.symmetry else 'unknown'
                        })
                except Exception as e:
                    logger.warning(f"Error processing MP entry: {e}")
                    continue
        
        logger.info(f"Materials Project: {len(structures)} inorganic structures")
        
        # Save checkpoint
        save_checkpoint({'structures': structures, 'metadata': metadata}, "mp_structures")
        
        return structures, metadata


class AlexandriaFetcher(MaterialsFetcher):
    """Fetch from Alexandria via LeMat-Bulk."""
    
    def __init__(self):
        super().__init__("alexandria")
    
    def fetch_all_inorganic(self, target_count: Optional[int] = None) -> Tuple[List[Structure], List[Dict]]:
        """Fetch inorganic structures from Alexandria."""
        checkpoint = load_checkpoint(f"alexandria_structures_{target_count}")
        if checkpoint:
            logger.info(f"Loaded {len(checkpoint['structures'])} Alexandria structures from checkpoint")
            return checkpoint['structures'], checkpoint['metadata']
        
        logger.info("Fetching from Alexandria (via LeMat-Bulk)...")
        
        # Load full dataset
        dataset = load_dataset(
            "LeMaterial/LeMat-Bulk",
            "compatible_pbe",
            split="train",
            streaming=False
        )
        
        logger.info(f"Loaded LeMat-Bulk with {len(dataset)} structures")
        
        # Filter for Alexandria sources
        structures = []
        metadata = []
        
        for item in tqdm(dataset, desc="Filtering Alexandria structures"):
            try:
                # Filter for Alexandria by immutable_id prefix
                # Alexandria uses 'agm' (no dash) followed by numbers
                immutable_id = item.get('immutable_id', '')
                if not immutable_id.startswith('agm'):
                    continue
                
                elements = item.get('elements', [])
                formula = item.get('chemical_formula_reduced', '')
                
                if not is_inorganic(elements, formula):
                    continue
                
                # Convert to Structure
                structure = Structure(
                    lattice=item['lattice_vectors'],
                    species=item['species_at_sites'],
                    coords=item['cartesian_site_positions'],
                    coords_are_cartesian=True
                )
                
                structures.append(structure)
                metadata.append({
                    'material_id': item.get('immutable_id', f'alex_{len(structures)}'),
                    'formula': formula,
                    'elements': ','.join(elements),
                    'nsites': len(item['species_at_sites']),
                    'density': structure.density,
                    'spacegroup': 'unknown'
                })
                
                # Stop if we have enough
                if target_count and len(structures) >= target_count:
                    break
                    
            except Exception as e:
                logger.warning(f"Error processing Alexandria entry: {e}")
                continue
        
        # Random sample if we have more than needed
        if target_count and len(structures) > target_count:
            indices = random.sample(range(len(structures)), target_count)
            structures = [structures[i] for i in indices]
            metadata = [metadata[i] for i in indices]
        
        logger.info(f"Alexandria: {len(structures)} inorganic structures")
        
        # Save checkpoint
        save_checkpoint({'structures': structures, 'metadata': metadata}, 
                       f"alexandria_structures_{target_count}")
        
        return structures, metadata


class OQMDFetcher(MaterialsFetcher):
    """Fetch from OQMD."""
    
    def __init__(self):
        super().__init__("oqmd")
        self.base_url = "http://oqmd.org/oqmdapi/formationenergy"
    
    def fetch_all_inorganic(self, target_count: Optional[int] = None) -> Tuple[List[Structure], List[Dict]]:
        """Fetch inorganic structures from OQMD."""
        checkpoint = load_checkpoint(f"oqmd_structures_{target_count}")
        if checkpoint:
            logger.info(f"Loaded {len(checkpoint['structures'])} OQMD structures from checkpoint")
            return checkpoint['structures'], checkpoint['metadata']
        
        # Note: OQMD API doesn't directly provide structures in a simple format
        # We'll use a workaround: use LeMat-Bulk's OQMD subset
        logger.info("OQMD direct API doesn't provide structures easily. Using LeMat-Bulk OQMD subset...")
        
        # Fallback to LeMat-Bulk filtering for OQMD
        return self._fetch_from_lematbulk_oqmd(target_count)
    
    def _fetch_from_lematbulk_oqmd(self, target_count: int) -> Tuple[List[Structure], List[Dict]]:
        """Fetch OQMD structures from LeMat-Bulk."""
        logger.info("Fetching OQMD structures from LeMat-Bulk...")
        
        dataset = load_dataset(
            "LeMaterial/LeMat-Bulk",
            "compatible_pbe",
            split="train",
            streaming=False
        )
        
        structures = []
        metadata = []
        
        for item in tqdm(dataset, desc="Filtering OQMD from LeMat-Bulk"):
            try:
                # Filter for OQMD by immutable_id prefix
                immutable_id = item.get('immutable_id', '')
                if not immutable_id.startswith('oqmd-'):
                    continue
                
                elements = item.get('elements', [])
                formula = item.get('chemical_formula_reduced', '')
                
                if not is_inorganic(elements, formula):
                    continue
                
                structure = Structure(
                    lattice=item['lattice_vectors'],
                    species=item['species_at_sites'],
                    coords=item['cartesian_site_positions'],
                    coords_are_cartesian=True
                )
                
                structures.append(structure)
                metadata.append({
                    'material_id': item.get('immutable_id', f'oqmd_{len(structures)}'),
                    'formula': formula,
                    'elements': ','.join(elements),
                    'nsites': len(item['species_at_sites']),
                    'density': structure.density,
                    'spacegroup': 'unknown'
                })
                
                if target_count and len(structures) >= target_count:
                    break
                    
            except Exception as e:
                logger.warning(f"Error processing OQMD entry: {e}")
                continue
        
        # Random sample if needed
        if target_count and len(structures) > target_count:
            indices = random.sample(range(len(structures)), target_count)
            structures = [structures[i] for i in indices]
            metadata = [metadata[i] for i in indices]
        
        logger.info(f"OQMD: {len(structures)} inorganic structures")
        
        save_checkpoint({'structures': structures, 'metadata': metadata},
                       f"oqmd_structures_{target_count}")
        
        return structures, metadata


class AFLOWFetcher(MaterialsFetcher):
    """Fetch from AFLOW using AFLUX REST API."""
    
    def __init__(self):
        super().__init__("aflow")
        self.base_url = "http://aflow.org/API/aflux/"
    
    def fetch_all_inorganic(self, target_count: Optional[int] = None) -> Tuple[List[Structure], List[Dict]]:
        """Fetch inorganic structures from AFLOW using direct AFLUX API."""
        checkpoint = load_checkpoint(f"aflow_structures_{target_count}")
        if checkpoint:
            logger.info(f"Loaded {len(checkpoint['structures'])} AFLOW structures from checkpoint")
            return checkpoint['structures'], checkpoint['metadata']
        
        logger.info("Fetching from AFLOW using AFLUX REST API...")
        logger.info(f"Target: {target_count} structures. Estimated time: ~30-60 minutes...")
        
        structures = []
        metadata = []
        
        # Use AFLUX API with pagination
        # Query for bulk 3D structures from the ICSD library (most reliable)
        page = 0
        page_size = 100
        max_pages = (target_count // page_size) + 100 if target_count else 5000
        
        pbar = tqdm(total=target_count or 100000, desc="Fetching AFLOW")
        
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        try:
            while len(structures) < (target_count or float('inf')) and page < max_pages:
                try:
                    # Construct AFLUX query
                    # Query for ICSD structures (high quality) with paging
                    query = f"catalog(ICSD),paging({page})"
                    url = f"{self.base_url}?{query}"
                    
                    response = requests.get(url, timeout=30)
                    
                    if response.status_code != 200:
                        logger.warning(f"AFLOW API returned status {response.status_code} for page {page}")
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            logger.error("Too many consecutive failures. Stopping AFLOW fetch.")
                            break
                        page += 1
                        continue
                    
                    consecutive_failures = 0  # Reset on success
                    
                    # Parse JSON response
                    data = response.json()
                    
                    # AFLUX returns a dict with keys like "1 of N", "2 of N", etc.
                    # Convert to list of entries
                    if not data or len(data) == 0:
                        logger.info(f"No more data from AFLOW at page {page}")
                        break
                    
                    entries = list(data.values()) if isinstance(data, dict) else data
                    
                    for entry in entries:
                        try:
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
                            
                            # Get POSCAR geometry
                            aurl = entry.get('aurl', '')
                            if not aurl:
                                continue
                            
                            # Fix AFLOW URL format: "aflowlib.duke.edu:AFLOWDATA/..." 
                            # The colon should be replaced with slash
                            aurl_fixed = aurl.replace(':', '/', 1)  # Replace first colon only
                            
                            # Fetch CONTCAR.relax (relaxed structure)
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
                            structure = Structure.from_str(contcar_text, fmt='poscar')
                            
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
                            
                            pbar.update(1)
                            
                            # Save intermediate checkpoint every 1000 structures
                            if len(structures) % 1000 == 0:
                                save_checkpoint({'structures': structures, 'metadata': metadata},
                                               f"aflow_structures_{target_count}_partial")
                                logger.info(f"Checkpoint: {len(structures)} AFLOW structures fetched")
                            
                            if target_count and len(structures) >= target_count:
                                break
                                
                        except Exception as e:
                            logger.debug(f"Error processing AFLOW entry: {e}")
                            continue
                    
                    page += 1
                    # No sleep needed - HTTP requests provide natural rate limiting
                    
                    if target_count and len(structures) >= target_count:
                        break
                    
                except Exception as e:
                    logger.error(f"Error fetching AFLOW page {page}: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        break
                    page += 1
                    continue
            
            pbar.close()
            
        except Exception as e:
            logger.error(f"Fatal error in AFLOW fetch: {e}")
            pbar.close()
        
        logger.info(f"AFLOW: {len(structures)} inorganic structures fetched")
        
        if len(structures) == 0:
            logger.warning("AFLOW returned 0 structures. API may be unavailable or all structures filtered out.")
        
        save_checkpoint({'structures': structures, 'metadata': metadata},
                       f"aflow_structures_{target_count}")
        
        return structures, metadata


class NOMADFetcher(MaterialsFetcher):
    """Fetch from NOMAD."""
    
    def __init__(self):
        super().__init__("nomad")
        self.base_url = "https://nomad-lab.eu/prod/v1/api/v1"
    
    def fetch_all_inorganic(self, target_count: Optional[int] = None) -> Tuple[List[Structure], List[Dict]]:
        """Fetch inorganic structures from NOMAD."""
        checkpoint = load_checkpoint(f"nomad_structures_{target_count}")
        if checkpoint:
            logger.info(f"Loaded {len(checkpoint['structures'])} NOMAD structures from checkpoint")
            return checkpoint['structures'], checkpoint['metadata']
        
        logger.info("Fetching from NOMAD API...")
        logger.info(f"Target: {target_count} structures. Estimated time: ~30-90 minutes...")
        
        structures = []
        metadata = []
        
        page_size = 10  # NOMAD can be slow, use smaller pages
        page_after_value = None  # NOMAD uses cursor-based pagination
        
        pbar = tqdm(total=target_count or 100000, desc="Fetching NOMAD")
        
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        while len(structures) < (target_count or float('inf')):
            try:
                # Query NOMAD - use simpler query structure
                # Just query for entries with results (most have structures)
                query_url = f"{self.base_url}/entries/query"
                
                # Simplified query - only filter for bulk (3D) structures
                query_payload = {
                    "owner": "public",  # Only public entries
                    "page_size": page_size
                }
                
                if page_after_value:
                    query_payload["page_after_value"] = page_after_value
                
                response = requests.post(query_url, json=query_payload, timeout=90)
                
                if response.status_code != 200:
                    logger.warning(f"NOMAD API returned status {response.status_code}: {response.text[:200]}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("Too many consecutive NOMAD failures. Stopping.")
                        break
                    time.sleep(2)
                    continue
                
                consecutive_failures = 0  # Reset on success
                
                data = response.json()
                pagination = data.get('pagination', {})
                page_after_value = pagination.get('next_page_after_value')
                
                entries = data.get('data', [])
                if not entries:
                    logger.info("No more data from NOMAD")
                    break
                
                for entry in entries:
                    try:
                        # Get entry ID
                        entry_id = entry.get('entry_id')
                        if not entry_id:
                            continue
                        
                        # Get basic material info from entry
                        results = entry.get('results', {})
                        material = results.get('material', {})
                        
                        # Check if it's a bulk 3D structure
                        dimensionality = material.get('dimensionality', '3D')
                        if dimensionality != '3D':
                            continue
                        
                        elements = material.get('elements', [])
                        formula = material.get('chemical_formula_reduced', '')
                        
                        if not elements:
                            continue
                        
                        # Check if inorganic
                        if not is_inorganic(elements, formula):
                            continue
                        
                        # Fetch structure from archive
                        archive_url = f"{self.base_url}/entries/{entry_id}/archive"
                        archive_response = requests.get(archive_url, timeout=20)
                        
                        if archive_response.status_code != 200:
                            continue
                        
                        archive_data = archive_response.json()
                        
                        # Navigate through NOMAD's nested structure
                        # Structure: data -> archive -> run -> system
                        run_data = archive_data.get('data', {}).get('archive', {}).get('run', [])
                        if not run_data:
                            continue
                        
                        # Get the last run (usually the final calculation)
                        last_run = run_data[-1] if isinstance(run_data, list) else run_data
                        system_data = last_run.get('system', [])
                        
                        if not system_data:
                            continue
                        
                        # Get the last system (optimized structure)
                        final_system = system_data[-1] if isinstance(system_data, list) else system_data
                        
                        # Extract atomic information
                        atoms_data = final_system.get('atoms', {})
                        if not atoms_data:
                            continue
                        
                        # Extract lattice vectors (in NOMAD, it's in atoms, not simulation_cell)
                        lattice_vecs = atoms_data.get('lattice_vectors', None)
                        species = atoms_data.get('labels', [])
                        positions = atoms_data.get('positions', [])
                        
                        if not lattice_vecs or not species or not positions or len(species) != len(positions):
                            continue
                        
                        # Create pymatgen Structure
                        # NOMAD positions are in Cartesian coordinates (Angstrom)
                        structure = Structure(
                            lattice=lattice_vecs,
                            species=species,
                            coords=positions,
                            coords_are_cartesian=True
                        )
                        
                        structures.append(structure)
                        metadata.append({
                            'material_id': entry_id,
                            'formula': formula,
                            'elements': ','.join(elements),
                            'nsites': len(species),
                            'density': structure.density,
                            'spacegroup': 'unknown'
                        })
                        
                        pbar.update(1)
                        
                        if target_count and len(structures) >= target_count:
                            break
                        
                        # Rate limiting (be nice to NOMAD API)
                        time.sleep(0.1)
                        
                    except Exception as e:
                        logger.debug(f"Error processing NOMAD entry: {e}")
                        continue
                
                if target_count and len(structures) >= target_count:
                    break
                
                # Check if there's a next page
                if not page_after_value:
                    logger.info("No more pages from NOMAD")
                    break
                
                # Save intermediate checkpoint every 500 structures
                if len(structures) % 500 == 0 and len(structures) > 0:
                    save_checkpoint({'structures': structures, 'metadata': metadata},
                                   f"nomad_structures_{target_count}_partial")
                    logger.info(f"Checkpoint: {len(structures)} NOMAD structures fetched")
                
            except Exception as e:
                logger.warning(f"Error fetching NOMAD batch: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Too many consecutive NOMAD batch failures. Stopping.")
                    break
                time.sleep(5)
                continue
        
        pbar.close()
        
        logger.info(f"NOMAD: {len(structures)} inorganic structures")
        
        save_checkpoint({'structures': structures, 'metadata': metadata},
                       f"nomad_structures_{target_count}")
        
        return structures, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Sample inorganic crystals from multiple databases"
    )
    parser.add_argument(
        "--mp-api-key",
        type=str,
        default="4jCV4ZdYAATVPbM4GOKQnf2YleaxtAeH",
        help="Materials Project API key"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})"
    )
    parser.add_argument(
        "--final-sample-size",
        type=int,
        default=FINAL_SAMPLE_SIZE,
        help=f"Final sample size per database (default: {FINAL_SAMPLE_SIZE})"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetching, use existing checkpoints"
    )
    parser.add_argument(
        "--skip-databases",
        type=str,
        nargs='+',
        default=[],
        help="Databases to skip (e.g., nomad aflow)"
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save metadata about the run
    run_metadata = {
        'random_seed': args.seed,
        'final_sample_size': args.final_sample_size,
        'timestamp': pd.Timestamp.now().isoformat(),
        'databases': ['mp', 'alexandria', 'oqmd', 'aflow', 'nomad']
    }
    
    with open(OUTPUT_DIR / 'run_metadata.json', 'w') as f:
        json.dump(run_metadata, f, indent=2)
    
    # Step 1: Fetch Materials Project (smallest dataset)
    logger.info("=" * 80)
    logger.info("STEP 1: Fetching Materials Project structures")
    logger.info("=" * 80)
    
    mp_fetcher = MaterialsProjectFetcher(args.mp_api_key)
    mp_structures, mp_metadata = mp_fetcher.fetch_all_inorganic()
    
    target_count = len(mp_structures)
    logger.info(f"\nTarget count determined from Materials Project: {target_count}")
    
    # Update run metadata
    run_metadata['target_count'] = target_count
    with open(OUTPUT_DIR / 'run_metadata.json', 'w') as f:
        json.dump(run_metadata, f, indent=2)
    
    # Dictionary to store all datasets
    all_datasets = {
        'mp': (mp_structures, mp_metadata)
    }
    
    if not args.skip_fetch:
        # Step 2: Fetch from other databases
        # For AFLOW and NOMAD, use smaller target (10k) since they're slow
        # We sample with replacement anyway, so 10k provides good diversity
        aflow_nomad_target = 10000
        
        databases = [
            ('alexandria', AlexandriaFetcher(), target_count),
            ('oqmd', OQMDFetcher(), target_count),
            ('aflow', AFLOWFetcher(), aflow_nomad_target),
            ('nomad', NOMADFetcher(), aflow_nomad_target)
        ]
        
        for i, (db_name, fetcher, db_target) in enumerate(databases):
            # Skip if requested
            if db_name in args.skip_databases:
                logger.info("=" * 80)
                logger.info(f"STEP 2.{i + 1}: Skipping {db_name.upper()} (--skip-databases)")
                logger.info("=" * 80)
                continue
            
            logger.info("=" * 80)
            logger.info(f"STEP 2.{i + 1}: Fetching {db_name.upper()} structures")
            logger.info("=" * 80)
            
            if db_target != target_count:
                logger.info(f"Using reduced target of {db_target} structures for {db_name.upper()}")
            
            structures, metadata = fetcher.fetch_all_inorganic(target_count=db_target)
            all_datasets[db_name] = (structures, metadata)
            
            logger.info(f"\n{db_name.upper()}: {len(structures)} structures fetched\n")
    
    # Step 3: Sample 2500 structures with replacement from each
    logger.info("=" * 80)
    logger.info(f"STEP 3: Sampling {args.final_sample_size} structures from each database")
    logger.info("=" * 80)
    
    summary = {
        'seed': args.seed,
        'target_count': target_count,
        'final_sample_size': args.final_sample_size,
        'databases': {}
    }
    
    for db_name, (structures, metadata) in all_datasets.items():
        logger.info(f"\nProcessing {db_name.upper()}...")
        
        # Check if we have structures
        n_available = len(structures)
        if n_available == 0:
            logger.warning(f"{db_name.upper()}: No structures available! Skipping...")
            summary['databases'][db_name] = {
                'total_fetched': 0,
                'sampled': 0,
                'unique_samples': 0,
                'output_dir': str(OUTPUT_DIR / db_name),
                'error': 'No structures fetched'
            }
            continue
        
        # Sample with replacement
        sample_indices = np.random.choice(n_available, size=args.final_sample_size, replace=True)
        
        sampled_structures = [structures[i] for i in sample_indices]
        sampled_metadata = [metadata[i] for i in sample_indices]
        
        # Create database-specific output directory
        db_output_dir = OUTPUT_DIR / db_name
        db_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CIFs
        fetcher = MaterialsFetcher(db_name)
        fetcher.save_structures_as_cif(sampled_structures, sampled_metadata, db_output_dir)
        
        # Update summary
        summary['databases'][db_name] = {
            'total_fetched': n_available,
            'sampled': args.final_sample_size,
            'unique_samples': len(set(sample_indices)),
            'output_dir': str(db_output_dir)
        }
        
        logger.info(f"{db_name.upper()}: Saved {args.final_sample_size} CIFs")
        logger.info(f"  Unique samples: {len(set(sample_indices))}")
    
    # Save summary
    summary_path = OUTPUT_DIR / 'sampling_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info(f"CIF files saved to: {OUTPUT_DIR}")
    logger.info("\nDatabase summary:")
    for db_name, info in summary['databases'].items():
        logger.info(f"  {db_name.upper()}: {info['sampled']} CIFs in {info['output_dir']}")


if __name__ == "__main__":
    main()

