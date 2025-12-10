#!/usr/bin/env python3
"""
Run EACPS pipeline for all tasks in project-label.json in chunks, distributed across multiple GPUs.

Usage:
    python src/run_all_tasks.py --json_file project-label.json --chunk_size 5 --gpus 0 1 2 3
"""
import argparse
import json
import subprocess
import sys
import time
import os
import threading
from pathlib import Path
from typing import List

def run_chunk_on_gpu(chunk: List[str], chunk_idx: int, total_chunks: int, gpu_id: int):
    """Run a chunk of task IDs using inpaint_eacps/run.py on a specific GPU"""
    print(f"\n[GPU {gpu_id}] Starting Chunk {chunk_idx + 1}/{total_chunks} (Tasks: {len(chunk)})")
    
    cmd = [
        sys.executable,
        "inpaint_eacps/run.py",
        "--task_id"
    ] + chunk
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        # Run the command
        # We capture output to avoid interleaving print statements too much, 
        # or we can just let it flow if we don't mind the mess.
        # For simplicity, let it flow but prefix with GPU ID if possible (hard with subprocess directly)
        subprocess.run(cmd, check=True, env=env)
        print(f"[GPU {gpu_id}] Chunk {chunk_idx + 1} Completed Successfully")
    except subprocess.CalledProcessError as e:
        print(f"[GPU {gpu_id}] Error in Chunk {chunk_idx + 1}: {e}")
    except KeyboardInterrupt:
        print(f"\n[GPU {gpu_id}] Interrupted.")
        return

def gpu_worker(gpu_id: int, tasks: List[str], chunk_size: int, worker_id: int):
    """Worker thread that processes a subset of tasks on a specific GPU"""
    
    # Split this worker's tasks into chunks
    chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]
    total_chunks = len(chunks)
    
    print(f"Worker {worker_id} (GPU {gpu_id}) assigned {len(tasks)} tasks ({total_chunks} chunks).")
    
    for i, chunk in enumerate(chunks):
        run_chunk_on_gpu(chunk, i, total_chunks, gpu_id)
        # Optional sleep between chunks
        time.sleep(2)

def main():
    parser = argparse.ArgumentParser(description="Run all tasks from JSON file distributed across GPUs")
    parser.add_argument("--json_file", type=str, default="project-label.json", help="Path to JSON file with tasks")
    parser.add_argument("--chunk_size", type=int, default=5, help="Number of tasks to run in one batch per GPU")
    parser.add_argument("--start_from", type=int, default=0, help="Index to start from (global index)")
    parser.add_argument("--task_ids", nargs="+", help="List of specific task IDs to run (ignores json file if provided)")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0], help="List of GPU IDs to use (e.g. 0 1 2 3)")
    
    args = parser.parse_args()
    
    # 1. Get all Task IDs
    all_ids = []
    if args.task_ids:
        all_ids = args.task_ids
        print(f"Running {len(all_ids)} specified tasks from command line.")
    else:
        json_path = Path(args.json_file)
        if not json_path.exists():
            print(f"Error: {json_path} not found.")
            sys.exit(1)
            
        with open(json_path) as f:
            data = json.load(f)
            
        if isinstance(data, list):
            for item in data:
                if "id" in item:
                    all_ids.append(str(item["id"]))
                elif "task_id" in item:
                    all_ids.append(str(item["task_id"]))
        
        if not all_ids:
            print("No task IDs found in JSON file.")
            sys.exit(1)
            
        print(f"Found {len(all_ids)} tasks in {json_path}.")
    
    # 2. Filter by start_from
    if args.start_from > 0:
        if args.start_from >= len(all_ids):
            print(f"Start index {args.start_from} is out of range (total {len(all_ids)}).")
            sys.exit(1)
        all_ids = all_ids[args.start_from:]
        print(f"Resuming from index {args.start_from}. Remaining tasks: {len(all_ids)}")

    # 3. Distribute tasks across GPUs
    num_gpus = len(args.gpus)
    tasks_per_gpu = [[] for _ in range(num_gpus)]
    
    # Round-robin distribution ensures even load even if tasks are sorted by difficulty
    for i, task_id in enumerate(all_ids):
        tasks_per_gpu[i % num_gpus].append(task_id)
        
    # 4. Start Workers
    threads = []
    print(f"Starting execution on {num_gpus} GPUs...")
    
    for i, gpu_id in enumerate(args.gpus):
        gpu_tasks = tasks_per_gpu[i]
        if not gpu_tasks:
            continue
            
        t = threading.Thread(
            target=gpu_worker,
            args=(gpu_id, gpu_tasks, args.chunk_size, i)
        )
        t.start()
        threads.append(t)
        
    # Wait for all threads
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nMain process interrupted. Waiting for threads to stop...")
        # In a real scenario we'd need a stop event, but for now we just exit
        sys.exit(1)
        
    print("All tasks completed.")

if __name__ == "__main__":
    main()
