#!/usr/bin/env python3
import argparse
import time
from multiprocessing import Manager, Pool, cpu_count, shared_memory

import numpy as np
from tqdm import tqdm


def check_batch_subsets(args):
    batch_start, batch_end, shm_name, shape, dtype, element_counts, progress_dict = args

    shm = shared_memory.SharedMemory(name=shm_name)
    unique_sets = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    n = len(unique_sets)
    local_non_maximal = []
    batch_size = batch_end - batch_start

    for i in range(batch_start, batch_end):
        current = unique_sets[i]
        current_count = element_counts[i]

        if i + 1 < n:
            count_mask = element_counts[i + 1 :] <= current_count
            potential_indices = np.where(count_mask)[0] + i + 1

            if len(potential_indices) > 0:
                candidates = unique_sets[potential_indices]
                no_extra = ~np.any(candidates & ~current, axis=1)
                subset_indices = potential_indices[no_extra]

                if len(subset_indices) > 0:
                    local_non_maximal.extend(subset_indices.tolist())

        if (i - batch_start + 1) % max(1, batch_size // 20) == 0:
            progress_dict["completed"] = progress_dict.get("completed", 0) + 1

    shm.close()
    return local_non_maximal


def find_maximal_parallel(comp_vecs, n_workers=None):
    if n_workers is None:
        n_workers = min(cpu_count(), 32)

    start_total = time.time()

    print("Finding unique sets...")
    comp_bool = (comp_vecs > 0).astype(np.uint8)
    unique_sets = np.unique(comp_bool, axis=0)
    n = len(unique_sets)
    print(f"Found {n:,} unique sets from {len(comp_vecs):,} compositions")

    element_counts = np.sum(unique_sets, axis=1)
    sorted_indices = np.argsort(-element_counts)
    unique_sets = unique_sets[sorted_indices]
    element_counts = element_counts[sorted_indices]

    print(f"Finding maximal sets ({n_workers} workers)...")

    shm = shared_memory.SharedMemory(create=True, size=unique_sets.nbytes)
    shared_array = np.ndarray(
        unique_sets.shape, dtype=unique_sets.dtype, buffer=shm.buf
    )
    shared_array[:] = unique_sets[:]

    manager = Manager()
    progress_dict = manager.dict()
    progress_dict["completed"] = 0

    batch_size = max(1, n // (n_workers * 10))
    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batches.append(
            (
                start,
                end,
                shm.name,
                unique_sets.shape,
                unique_sets.dtype,
                element_counts,
                progress_dict,
            )
        )

    total_items = len(batches) * 20

    with Pool(n_workers) as pool:
        with tqdm(
            total=total_items,
            desc="Processing",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            async_result = pool.map_async(check_batch_subsets, batches)

            while not async_result.ready():
                time.sleep(0.5)
                current = progress_dict.get("completed", 0)
                pbar.n = current
                pbar.refresh()

            results = async_result.get()
            pbar.n = total_items
            pbar.refresh()

    shm.close()
    shm.unlink()

    is_maximal = np.ones(n, dtype=bool)
    for non_maximal_indices in results:
        if non_maximal_indices:
            is_maximal[non_maximal_indices] = False

    maximal_sets = unique_sets[is_maximal]

    elapsed = time.time() - start_total
    print(f"\nFound {np.sum(is_maximal):,} maximal sets ({elapsed:.1f}s)")
    print(f"Eliminated {n - np.sum(is_maximal):,} non-maximal sets")

    return maximal_sets


def main():
    parser = argparse.ArgumentParser(description="Find maximal sets")
    parser.add_argument("--input", type=str, default="all_compositions.npy")
    parser.add_argument("--output", type=str, default="maximal_chemical_sets.npy")
    parser.add_argument("--n_workers", type=int, default=None)

    args = parser.parse_args()

    comp_vecs = np.load(args.input)
    print(f"Loaded {comp_vecs.shape}")

    maximal_sets = find_maximal_parallel(comp_vecs, args.n_workers)

    np.save(args.output, maximal_sets)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
