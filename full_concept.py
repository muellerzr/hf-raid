import os
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Dict
import safetensors
from safetensors.torch import load_file
import torch
import numpy as np
from tqdm import tqdm

def get_weight_files(dir1: str, dir2: str) -> Tuple[List[str], List[str]]:
    """Get weight files ensuring sequential ordering from each RAID."""
    # First RAID has shards 1-8
    files1 = sorted([
        str(p) for p in Path(dir1).glob("*.safetensors")
    ])
    
    # Second RAID has shards 9-16 plus config files
    files2 = sorted([
        str(p) for p in Path(dir2).glob("*.safetensors")
    ])
    
    if len(files1) != 8 or len(files2) != 8:
        raise ValueError(f"Expected 8 shards on each RAID. Found {len(files1)} on RAID0 and {len(files2)} on RAID1")
    
    # Get shard numbers for information
    shard_nums1 = [int(Path(f).name.split('-')[1]) for f in files1]
    shard_nums2 = [int(Path(f).name.split('-')[1]) for f in files2]
    
    print(f"Found {len(files1)} shards in {dir1} (shards 1-8)")
    print(f"Found {len(files2)} shards in {dir2} (shards 9-16)")
    
    # List other files in RAID1 for information
    other_files = [
        p.name for p in Path(dir2).glob("*") 
        if p.is_file() and not p.name.startswith("model-") and not p.name.endswith(".safetensors")
    ]
    if other_files:
        print("\nAdditional files in RAID1:")
        for f in sorted(other_files):
            print(f"  {f}")
    
    return files1, files2

def load_raid_shards(args: Tuple[List[str], str, int]) -> Dict:
    """Load all shards from one RAID drive."""
    shard_files, raid_name, worker_id = args
    
    results = []
    total_size = 0
    total_time = 0
    
    # Process all shards from this RAID sequentially
    for i, shard_file in enumerate(shard_files):
        start_time = time.time()
        tensors = load_file(shard_file)
        
        # Force load into memory
        shard_size = 0
        for v in tensors.values():
            if isinstance(v, torch.Tensor):
                _ = v.mean().item()
                shard_size += v.nelement() * v.element_size()
        
        load_time = time.time() - start_time
        size_gb = shard_size / (1024 ** 3)
        
        total_size += size_gb
        total_time += load_time
        
        results.append({
            'file': shard_file,
            'size': size_gb,
            'time': load_time,
            'shard': int(Path(shard_file).name.split('-')[1])
        })
    
    return {
        'raid_name': raid_name,
        'worker_id': worker_id,
        'shards': results,
        'total_size': total_size,
        'total_time': total_time
    }

def measure_split_transfer_speed(dir1: str, dir2: str):
    """Measure transfer speed of loading split weights from two RAID drives."""
    print(f"Measuring transfer speed between {dir1} and {dir2}")
    
    try:
        # Get weight files from both directories
        files1, files2 = get_weight_files(dir1, dir2)
        
        # Use 2 workers - one for each RAID
        num_workers = 2
        print(f"\nUsing {num_workers} workers for parallel loading")
        print("Worker 0 will process all shards from RAID0")
        print("Worker 1 will process all shards from RAID1")
        
        # Prepare arguments for workers
        worker_args = [
            (files1, "RAID0", 0),
            (files2, "RAID1", 1)
        ]
        
        with mp.Pool(processes=num_workers) as pool:
            print("\nLoading shards in parallel...")
            results = pool.map(load_raid_shards, worker_args)
            
            # Process results
            for raid_result in results:
                raid_name = raid_result['raid_name']
                print(f"\n{raid_name} Results:")
                print(f"Total Size: {raid_result['total_size']:.2f} GB")
                print(f"Total Time: {raid_result['total_time']:.2f} seconds")
                print(f"Average Speed: {raid_result['total_size']/raid_result['total_time']:.2f} GB/s")
                
                print("\nIndividual Shard Details:")
                for shard in raid_result['shards']:
                    print(f"Shard {shard['shard']}: {Path(shard['file']).name}")
                    print(f"  Size: {shard['size']:.2f} GB")
                    print(f"  Speed: {shard['size']/shard['time']:.2f} GB/s")
        
        # Calculate overall statistics
        total_size = sum(r['total_size'] for r in results)
        max_time = max(r['total_time'] for r in results)
        
        print("\nOverall Statistics:")
        for r in results:
            print(f"{r['raid_name']}: {r['total_size']:.2f} GB at {r['total_size']/r['total_time']:.2f} GB/s")
        print(f"Combined Data Loaded: {total_size:.2f} GB")
        print(f"Total Time: {max_time:.2f} seconds")
        print(f"Effective Speed: {total_size/max_time:.2f} GB/s")
            
    except Exception as e:
        print(f"Error during measurement: {str(e)}")

if __name__ == "__main__":
    SOURCE_DIR1 = "/mnt/raid0n0/Qwen3-30B-A3B"
    SOURCE_DIR2 = "/mnt/raid0n1/Qwen3-30B-A3B"
    
    try:
        measure_split_transfer_speed(SOURCE_DIR1, SOURCE_DIR2)
    except KeyboardInterrupt:
        print("\nMeasurement interrupted by user")
    except Exception as e:
        print(f"Error during measurement: {str(e)}")
