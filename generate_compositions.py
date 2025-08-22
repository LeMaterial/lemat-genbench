#!/usr/bin/env python3
"""
Generate the missing all_compositions.npz file.

This script downloads the LeMat-Bulk dataset and generates the one-hot 
encoded composition matrix needed by the novelty benchmarks.
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def main():
    """Generate the compositions data file."""
    print("🚀 Generating all_compositions.npz file...")
    print("📥 This will download ~5M structures from LeMat-Bulk dataset")
    print("⏱️  This may take several minutes...")
    
    try:
        # Import the module with better error handling
        from lemat_genbench.fingerprinting.encode_compositions import get_all_compositions
        
        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Generate compositions with single process to avoid multiprocessing issues
        compositions = get_all_compositions(num_processes=1)
        
        print(f"✅ Successfully generated compositions matrix with shape: {compositions.shape}")
        print("💾 Saved to: data/all_compositions.npz")
        print("🎉 You can now run the benchmarks!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 This might be due to missing dependencies or import chain issues.")
        print("🔧 Try running: uv sync --extra all")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error generating compositions: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()