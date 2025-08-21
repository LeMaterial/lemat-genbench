import gc
import hashlib
import pickle
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import numpy as np


class ShapeGroupedArrayAnalyzer:
    def __init__(self, directory_path: str, index_db_path: str = "array_index.db"):
        self.directory_path = Path(directory_path)
        self.index_db_path = index_db_path
        self.setup_database()

    def setup_database(self):
        """Create SQLite database for efficient indexing with array storage"""
        self.conn = sqlite3.connect(self.index_db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS array_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT,
                file_chunk_num INTEGER,
                array_idx INTEGER,
                shape TEXT,
                dtype TEXT,
                array_hash TEXT,
                file_size INTEGER,
                array_data BLOB
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_shape ON array_index(shape)")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hash ON array_index(array_hash)"
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_id ON array_index(id)")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunk ON array_index(file_chunk_num)"
        )
        self.conn.commit()

    def _extract_chunk_number(self, filename: str) -> int:
        """Extract numeric chunk number from filename for proper sorting"""
        import re

        # Extract number from patterns like "pdd_hashes_chunk_666913_0.pkl"
        match = re.search(r"chunk_(\d+)_(\d+)", filename)
        if match:
            chunk_base = int(match.group(1))
            chunk_sub = int(match.group(2))
            return chunk_base * 1000 + chunk_sub  # Ensure proper ordering
        return 0

    def build_index(self, force_rebuild: bool = False, store_arrays: bool = True):
        """
        Build index of all arrays by shape across all files.
        If store_arrays=True, stores actual array data in database for instant access.
        """
        if not force_rebuild and self._index_exists():
            print("Index already exists. Use force_rebuild=True to recreate.")
            return

        # Clear existing index
        self.conn.execute("DELETE FROM array_index")
        self.conn.commit()

        pickle_files = list(self.directory_path.glob("*.pkl"))
        # Sort by chunk number for consistent ordering
        pickle_files.sort(key=lambda x: self._extract_chunk_number(x.name))

        total_arrays = 0
        total_size_gb = 0

        print(f"Building index for {len(pickle_files)} files...")
        if store_arrays:
            print("Storing array data in database for instant future access...")

        for file_idx, file_path in enumerate(pickle_files):
            file_size_gb = file_path.stat().st_size / (1024**3)
            total_size_gb += file_size_gb
            print(
                f"[{file_idx + 1}/{len(pickle_files)}] Indexing {file_path.name} ({file_size_gb:.1f}GB)"
            )

            try:
                import time

                load_start = time.time()

                with open(file_path, "rb") as f:
                    arrays = pickle.load(f)

                load_time = time.time() - load_start
                print(
                    f"  -> Loaded pickle in {load_time:.2f}s ({file_size_gb / load_time:.1f} GB/s)"
                )

                if not isinstance(arrays, list):
                    print(f"  -> Warning: {file_path.name} is not a list, skipping...")
                    continue

                chunk_num = self._extract_chunk_number(file_path.name)
                batch_data = []

                process_start = time.time()
                for array_idx, array in enumerate(arrays):
                    if isinstance(array, np.ndarray):
                        shape_str = str(array.shape)
                        dtype_str = str(array.dtype)

                        # Create a hash for the array content (for duplicate detection)
                        array_hash = hashlib.md5(array.tobytes()).hexdigest()

                        # Serialize array data for storage
                        array_blob = None
                        if store_arrays:
                            array_blob = array.tobytes()

                        batch_data.append(
                            (
                                str(file_path),
                                chunk_num,
                                array_idx,
                                shape_str,
                                dtype_str,
                                array_hash,
                                file_path.stat().st_size,
                                array_blob,
                            )
                        )
                        total_arrays += 1

                # Batch insert for efficiency
                if store_arrays:
                    self.conn.executemany(
                        """
                        INSERT INTO array_index (file_path, file_chunk_num, array_idx, shape, dtype, array_hash, file_size, array_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        batch_data,
                    )
                else:
                    # Store without array data for smaller database
                    batch_data_no_arrays = [
                        (item[:-1]) for item in batch_data
                    ]  # Remove array_blob
                    self.conn.executemany(
                        """
                        INSERT INTO array_index (file_path, file_chunk_num, array_idx, shape, dtype, array_hash, file_size)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        batch_data_no_arrays,
                    )

                process_time = time.time() - process_start
                print(f"  -> Processed {len(batch_data)} arrays in {process_time:.2f}s")

                # Clean up memory
                del arrays
                gc.collect()

            except Exception as e:
                print(f"  -> Error processing {file_path.name}: {e}")
                continue

        self.conn.commit()
        print("\n=== INDEX BUILD COMPLETE ===")
        print(f"Total arrays indexed: {total_arrays}")
        print(f"Total data processed: {total_size_gb:.1f} GB")

        # Get database size
        db_size_gb = Path(self.index_db_path).stat().st_size / (1024**3)
        print(f"Database size: {db_size_gb:.1f} GB")

        if store_arrays:
            print("✅ Arrays stored in database - no more pickle loading needed!")

        self._print_shape_summary()

    def _index_exists(self) -> bool:
        """Check if index already exists and has data"""
        cursor = self.conn.execute("SELECT COUNT(*) FROM array_index")
        return cursor.fetchone()[0] > 0

    def _print_shape_summary(self):
        """Print summary of shapes found"""
        cursor = self.conn.execute("""
            SELECT shape, COUNT(*) as count
            FROM array_index 
            GROUP BY shape 
            ORDER BY count DESC 
            LIMIT 20
        """)

        print("\nTop 20 most common shapes:")
        for shape, count in cursor.fetchall():
            print(f"  {shape}: {count} arrays")

    def get_arrays_by_shape(
        self, target_shape: tuple
    ) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields all arrays with the specified shape.
        Super fast if arrays are stored in database!
        """
        shape_str = str(target_shape)

        # Check if we have array data stored
        cursor = self.conn.execute(
            """
            SELECT array_data, dtype, shape
            FROM array_index 
            WHERE shape = ? AND array_data IS NOT NULL
            ORDER BY file_chunk_num, array_idx
            LIMIT 1
        """,
            (shape_str,),
        )

        has_array_data = cursor.fetchone() is not None

        if has_array_data:
            # Fast path: load directly from database
            cursor = self.conn.execute(
                """
                SELECT array_data, dtype, shape
                FROM array_index 
                WHERE shape = ?
                ORDER BY file_chunk_num, array_idx
            """,
                (shape_str,),
            )

            for array_data, dtype_str, shape_str in cursor:
                if array_data:
                    # Reconstruct array from stored bytes
                    array = np.frombuffer(array_data, dtype=np.dtype(dtype_str))
                    array = array.reshape(eval(shape_str))
                    yield array
        else:
            # Fallback: load from pickle files
            print("Array data not in database, falling back to pickle loading...")
            cursor = self.conn.execute(
                """
                SELECT file_path, array_idx, array_hash
                FROM array_index 
                WHERE shape = ?
                ORDER BY file_chunk_num, array_idx
            """,
                (shape_str,),
            )

            current_file = None
            current_arrays = None

            for file_path, array_idx, array_hash in cursor:
                # Load new file if needed
                if current_file != file_path:
                    if current_arrays is not None:
                        del current_arrays
                        gc.collect()

                    with open(file_path, "rb") as f:
                        current_arrays = pickle.load(f)
                    current_file = file_path

                # Yield the specific array
                if array_idx < len(current_arrays):
                    array = current_arrays[array_idx]
                    if isinstance(array, np.ndarray) and array.shape == target_shape:
                        yield array

            # Clean up
            if current_arrays is not None:
                del current_arrays
                gc.collect()

    def get_arrays_by_indices(
        self, db_indices: List[int]
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Get specific arrays by their database indices.
        Lightning fast if arrays are stored in database!
        """
        if not db_indices:
            return []

        indices_str = ",".join(map(str, db_indices))

        # Check if we have array data stored
        cursor = self.conn.execute(f"""
            SELECT id, array_data, dtype, shape
            FROM array_index 
            WHERE id IN ({indices_str}) AND array_data IS NOT NULL
        """)

        result = []
        arrays_from_db = 0

        # Collect arrays that are in the database
        db_arrays = {}
        for db_id, array_data, dtype_str, shape_str in cursor:
            if array_data:
                array = np.frombuffer(array_data, dtype=np.dtype(dtype_str))
                array = array.reshape(eval(shape_str))
                db_arrays[db_id] = array
                arrays_from_db += 1

        # Check if we got all arrays from database
        missing_indices = [idx for idx in db_indices if idx not in db_arrays]

        if missing_indices:
            print(
                f"Loading {len(missing_indices)} arrays from pickle files (not in database)"
            )
            # Fallback to pickle loading for missing arrays
            missing_str = ",".join(map(str, missing_indices))
            cursor = self.conn.execute(f"""
                SELECT id, file_path, array_idx
                FROM array_index 
                WHERE id IN ({missing_str})
            """)

            # Group by file
            file_groups = defaultdict(list)
            for db_id, file_path, array_idx in cursor:
                file_groups[file_path].append((db_id, array_idx))

            # Load from pickle files
            for file_path, requests in file_groups.items():
                with open(file_path, "rb") as f:
                    arrays = pickle.load(f)

                for db_id, array_idx in requests:
                    if array_idx < len(arrays):
                        array = arrays[array_idx]
                        if isinstance(array, np.ndarray):
                            db_arrays[db_id] = array

                del arrays
                gc.collect()

        # Return in the same order as requested
        for db_id in db_indices:
            if db_id in db_arrays:
                result.append((db_id, db_arrays[db_id]))

        if arrays_from_db > 0:
            print(f"✅ Loaded {arrays_from_db} arrays instantly from database")

        return result

    def get_shape_statistics(self, target_shape: tuple) -> Dict:
        """Get statistics about arrays with specified shape"""
        shape_str = str(target_shape)

        cursor = self.conn.execute(
            """
            SELECT COUNT(*) as total_count,
                   COUNT(DISTINCT array_hash) as unique_count,
                   COUNT(DISTINCT file_path) as files_with_shape,
                   COUNT(CASE WHEN array_data IS NOT NULL THEN 1 END) as stored_in_db
            FROM array_index 
            WHERE shape = ?
        """,
            (shape_str,),
        )

        total, unique, files, stored = cursor.fetchone()

        return {
            "shape": target_shape,
            "total_arrays": total,
            "unique_arrays": unique,
            "duplicate_arrays": total - unique,
            "files_containing_shape": files,
            "duplicate_ratio": (total - unique) / total if total > 0 else 0,
            "stored_in_database": stored,
            "storage_ratio": stored / total if total > 0 else 0,
        }

    def get_indices_by_shape(self, target_shape: tuple) -> List[int]:
        """Get all database indices for arrays with the specified shape"""
        shape_str = str(target_shape)

        cursor = self.conn.execute(
            """
            SELECT id
            FROM array_index 
            WHERE shape = ?
            ORDER BY file_chunk_num, array_idx
        """,
            (shape_str,),
        )

        return [row[0] for row in cursor.fetchall()]

    def get_all_shapes(self) -> List[Tuple[str, int]]:
        """Get all unique shapes and their counts"""
        cursor = self.conn.execute("""
            SELECT shape, COUNT(*) as count
            FROM array_index 
            GROUP BY shape 
            ORDER BY count DESC
        """)

        return [(eval(shape), count) for shape, count in cursor.fetchall()]

    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics"""
        # Overall stats
        cursor = self.conn.execute("SELECT COUNT(*) FROM array_index")
        total_arrays = cursor.fetchone()[0]

        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM array_index WHERE array_data IS NOT NULL"
        )
        stored_arrays = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(DISTINCT file_path) FROM array_index")
        total_files = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(DISTINCT shape) FROM array_index")
        unique_shapes = cursor.fetchone()[0]

        # Database size
        db_size_gb = Path(self.index_db_path).stat().st_size / (1024**3)

        return {
            "total_arrays": total_arrays,
            "arrays_stored_in_db": stored_arrays,
            "storage_percentage": (stored_arrays / total_arrays * 100)
            if total_arrays > 0
            else 0,
            "total_files": total_files,
            "unique_shapes": unique_shapes,
            "database_size_gb": db_size_gb,
        }

    def close(self):
        """Close database connection"""
        self.conn.close()


# Usage examples
def main():
    # Initialize analyzer
    analyzer = ShapeGroupedArrayAnalyzer(
        "/ogre/pdd_hashes", index_db_path="/ogre/array_index.db"
    )

    # Build index with array storage (do this once)
    print("Building index with array storage...")
    analyzer.build_index(force_rebuild=False, store_arrays=True)

    # Show database stats
    stats = analyzer.get_database_stats()
    print("\n=== DATABASE STATS ===")
    print(f"Total arrays: {stats['total_arrays']:,}")
    print(
        f"Arrays in database: {stats['arrays_stored_in_db']:,} ({stats['storage_percentage']:.1f}%)"
    )
    print(f"Database size: {stats['database_size_gb']:.1f} GB")

    # Test lightning-fast access
    target_shape = (100, 50)
    indices = analyzer.get_indices_by_shape(target_shape)

    if indices:
        print("\n=== TESTING FAST ACCESS ===")
        import time

        # Test getting arrays by indices (should be instant!)
        start = time.time()
        sample_indices = indices[:100] if len(indices) > 100 else indices
        arrays = analyzer.get_arrays_by_indices(sample_indices)
        access_time = time.time() - start

        print(f"Loaded {len(arrays)} arrays in {access_time:.3f}s")
        print(f"Speed: {len(arrays) / access_time:.0f} arrays/second")

        if access_time < 0.1:
            print("🚀 Lightning fast! Arrays are stored in database.")
        else:
            print("⚠️  Still loading from pickle files.")

    analyzer.close()


if __name__ == "__main__":
    main()
