#!/usr/bin/env python3
"""
Multi-GPU batch processing for dual output EACPS.
Distributes tasks across available GPUs for parallel execution.

Usage:
    # Auto-detect GPUs and distribute 7 tasks
    python3 inpaint_eacps/run_multigpu_batch.py \
      --task_ids 71650389 71678832 71680285 71634650 71630498 71656881 71673477 \
      --gemini_api_key $GEMINI_API_KEY
    
    # Specify GPUs manually
    python3 inpaint_eacps/run_multigpu_batch.py \
      --task_ids 71650389 71678832 71680285 71634650 71630498 71656881 71673477 \
      --gemini_api_key $GEMINI_API_KEY \
      --gpus 0 1 2 3
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess
import logging
from typing import List
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_available_gpus() -> List[int]:
    """Detect available GPUs."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        gpus = []
        for line in result.stdout.strip().split('\n'):
            idx, mem = line.split(',')
            idx = int(idx.strip())
            mem = int(mem.strip())
            if mem > 15000:  # At least 15GB free
                gpus.append(idx)
        return gpus
    except Exception as e:
        logger.warning(f"Could not detect GPUs: {e}")
        return [0]


def split_tasks(task_ids: List[str], num_gpus: int) -> List[List[str]]:
    """Split tasks evenly across GPUs."""
    chunks = [[] for _ in range(num_gpus)]
    for i, task_id in enumerate(task_ids):
        chunks[i % num_gpus].append(task_id)
    return chunks


def run_gpu_worker(
    gpu_id: int,
    task_ids: List[str],
    gemini_api_key: str,
    output_dir: str,
    from_file: str,
    fast: bool,
    no_moondream: bool,
) -> subprocess.Popen:
    """Launch worker process on specific GPU."""
    
    task_ids_str = ' '.join(task_ids)
    
    cmd = [
        'python3',
        'inpaint_eacps/run_batch_dual.py',
        '--task_ids', *task_ids,
        '--gemini_api_key', gemini_api_key,
        '--output_dir', f"{output_dir}/gpu{gpu_id}",
        '--from_file', from_file,
        '--device', f'cuda:{gpu_id}',
    ]
    
    if fast:
        cmd.append('--fast')
    if no_moondream:
        cmd.append('--no_moondream')
    
    env = os.environ.copy()
    
    log_file = Path(output_dir) / f"gpu{gpu_id}_log.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"GPU {gpu_id}: Starting {len(task_ids)} tasks -> {log_file}")
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    
    return process


def monitor_progress(output_dir: Path, num_tasks: int, processes: List):
    """Monitor overall progress across all GPUs."""
    from tqdm import tqdm
    import glob
    
    with tqdm(total=num_tasks * 2, desc="Total outputs", unit="img") as pbar:
        last_count = 0
        no_progress_count = 0
        while True:
            # Count completed outputs
            pattern = str(output_dir / "gpu*/task_*/result_*.png")
            completed = len(glob.glob(pattern))
            
            if completed > last_count:
                pbar.update(completed - last_count)
                last_count = completed
                no_progress_count = 0
            else:
                no_progress_count += 1
            
            # Check if all processes are dead
            all_dead = all(p.poll() is not None for _, p in processes)
            if all_dead and completed < num_tasks * 2:
                logger.error("\nAll workers died! Check logs for errors.")
                break
            
            if completed >= num_tasks * 2:
                break
            
            # Timeout after 5 minutes of no progress
            if no_progress_count > 60:
                logger.warning("\nNo progress for 5 minutes. Check logs.")
                break
            
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU batch EACPS")
    
    parser.add_argument("--task_ids", type=str, nargs='+', required=True)
    parser.add_argument("--from_file", type=str, default="project-label.json")
    parser.add_argument("--gemini_api_key", type=str, default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--output_dir", type=str, default="outputs/multigpu_batch")
    parser.add_argument("--gpus", type=int, nargs='+', default=None, help="GPU IDs to use")
    
    parser.add_argument("--no_moondream", action="store_true")
    parser.add_argument("--fast", action="store_true")
    
    args = parser.parse_args()
    
    if not args.gemini_api_key:
        logger.error("Gemini API key required!")
        sys.exit(1)
    
    # Detect or use specified GPUs
    if args.gpus:
        gpus = args.gpus
    else:
        gpus = get_available_gpus()
    
    if not gpus:
        logger.error("No GPUs available!")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("MULTI-GPU BATCH CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Tasks: {len(args.task_ids)}")
    logger.info(f"GPUs: {gpus}")
    logger.info(f"Tasks per GPU: {[len(chunk) for chunk in split_tasks(args.task_ids, len(gpus))]}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Fast mode: {args.fast}")
    logger.info(f"Moondream: {'DISABLED' if args.no_moondream else 'ENABLED'}")
    logger.info("=" * 80)
    
    # Split tasks across GPUs
    task_chunks = split_tasks(args.task_ids, len(gpus))
    
    # Launch workers
    processes = []
    for gpu_id, tasks in zip(gpus, task_chunks):
        if not tasks:
            continue
        
        process = run_gpu_worker(
            gpu_id=gpu_id,
            task_ids=tasks,
            gemini_api_key=args.gemini_api_key,
            output_dir=args.output_dir,
            from_file=args.from_file,
            fast=args.fast,
            no_moondream=args.no_moondream,
        )
        processes.append((gpu_id, process))
    
    logger.info(f"\nLaunched {len(processes)} workers")
    logger.info("Monitoring progress...\n")
    
    # Monitor in background
    start_time = datetime.now()
    
    try:
        # Monitor progress
        output_dir = Path(args.output_dir)
        monitor_progress(output_dir, len(args.task_ids), processes)
        
        # Wait for all processes
        logger.info("\nWaiting for all workers to complete...")
        for gpu_id, process in processes:
            returncode = process.wait()
            if returncode != 0:
                logger.error(f"GPU {gpu_id}: Failed with code {returncode}. Check {output_dir}/gpu{gpu_id}_log.txt")
            else:
                logger.info(f"GPU {gpu_id}: Completed")
        
    except KeyboardInterrupt:
        logger.warning("\nInterrupted! Terminating workers...")
        for gpu_id, process in processes:
            process.terminate()
            process.wait()
        sys.exit(1)
    
    elapsed = datetime.now() - start_time
    
    # Aggregate results
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATING RESULTS")
    logger.info("=" * 80)
    
    all_results = []
    total_api_calls = 0
    
    for gpu_id in gpus:
        summary_file = Path(args.output_dir) / f"gpu{gpu_id}" / "batch_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                data = json.load(f)
                all_results.extend(data.get('results', []))
                total_api_calls += data.get('api_calls', 0)
    
    # Save consolidated summary
    consolidated = {
        "timestamp": datetime.now().isoformat(),
        "total_tasks": len(args.task_ids),
        "gpus_used": gpus,
        "tasks_completed": len(all_results),
        "total_api_calls": total_api_calls,
        "estimated_cost_usd": total_api_calls * 0.00375,
        "elapsed_seconds": elapsed.total_seconds(),
        "fast_mode": args.fast,
        "results": all_results,
    }
    
    summary_path = Path(args.output_dir) / "consolidated_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(consolidated, f, indent=2)
    
    logger.info(f"Tasks completed: {len(all_results)}/{len(args.task_ids)}")
    logger.info(f"Total Gemini calls: {total_api_calls}")
    logger.info(f"Actual cost: ${total_api_calls * 0.00375:.2f}")
    logger.info(f"Total time: {elapsed}")
    logger.info(f"Summary: {summary_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
