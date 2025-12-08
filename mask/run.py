#!/usr/bin/env python3
"""
CLI entry point for the mask pipeline.
Fetches tasks from Label Studio and processes them with EACPS.

Usage:
    python3 mask/run.py \
        --project_id 123 \
        --api_key "your_api_key" \
        --url "http://localhost:8080" \
        --output_dir outputs/mask_results \
        --devices cuda:0,cuda:1,cuda:2,cuda:3 \
        --k_global 4 \
        --m_global 2 \
        --k_local 2
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Character face replacement pipeline with EACPS inference scaling"
    )
    
    # Label Studio settings
    parser.add_argument(
        "--project_id", type=int, required=True,
        help="Label Studio project ID"
    )
    parser.add_argument(
        "--api_key", type=str, 
        default=os.getenv("LABEL_STUDIO_API_KEY", ""),
        help="Label Studio API key (or set LABEL_STUDIO_API_KEY env var)"
    )
    parser.add_argument(
        "--url", type=str,
        default=os.getenv("LABEL_STUDIO_URL", "http://localhost:8080"),
        help="Label Studio URL (or set LABEL_STUDIO_URL env var)"
    )
    
    # Output settings
    parser.add_argument(
        "--output_dir", type=str, default="outputs/mask_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--cache_dir", type=str, default="data/cache",
        help="Cache directory for downloaded images"
    )
    
    # GPU settings
    parser.add_argument(
        "--devices", type=str, default="cuda:0",
        help="Comma-separated GPU devices (e.g., cuda:0,cuda:1,cuda:2,cuda:3)"
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Process sequentially on first device (for debugging)"
    )
    
    # EACPS settings
    parser.add_argument(
        "--k_global", type=int, default=4,
        help="Number of candidates in global exploration"
    )
    parser.add_argument(
        "--m_global", type=int, default=2,
        help="Number of top candidates for local refinement"
    )
    parser.add_argument(
        "--k_local", type=int, default=2,
        help="Number of local refinements per candidate"
    )
    
    # Model settings
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen-Image-Edit",
        help="Model ID"
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--cfg", type=float, default=5.0,
        help="CFG scale"
    )
    
    # Task filtering
    parser.add_argument(
        "--task_ids", type=str, nargs="+",
        help="Process only these task IDs"
    )
    parser.add_argument(
        "--limit", type=int,
        help="Limit number of tasks to process"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate API key
    if not args.api_key:
        print("Error: API key required. Set --api_key or LABEL_STUDIO_API_KEY env var")
        sys.exit(1)
    
    # Import after arg parsing (for faster --help)
    from config import PipelineConfig
    from labelstudio import LabelStudioClient, LoadedTask
    from worker import GPUWorkerPool, process_tasks_sequential
    
    # Create config
    config = PipelineConfig.from_args(
        project_id=args.project_id,
        api_key=args.api_key,
        url=args.url,
        output_dir=args.output_dir,
        devices=args.devices,
        k_global=args.k_global,
        m_global=args.m_global,
        k_local=args.k_local,
        steps=args.steps,
        cfg=args.cfg,
    )
    config.model.model_id = args.model
    config.cache_dir = args.cache_dir
    
    # Create output and cache dirs
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("MASK PIPELINE: Character Face Replacement with EACPS")
    print("=" * 70)
    print(f"Label Studio: {config.label_studio.url}")
    print(f"Project ID: {config.label_studio.project_id}")
    print(f"Devices: {config.gpu.devices}")
    print(f"EACPS: k_global={config.eacps.k_global}, m_global={config.eacps.m_global}, k_local={config.eacps.k_local}")
    print(f"Output: {output_dir}")
    print()
    
    # Fetch tasks from Label Studio
    print("Fetching tasks from Label Studio...")
    client = LabelStudioClient(config.label_studio.url, config.label_studio.api_key)
    
    try:
        tasks = client.fetch_project_tasks(config.label_studio.project_id)
    except Exception as e:
        print(f"Error fetching tasks: {e}")
        sys.exit(1)
    
    print(f"Found {len(tasks)} tasks")
    
    # Filter tasks if requested
    if args.task_ids:
        task_id_set = set(args.task_ids)
        tasks = [t for t in tasks if t.id in task_id_set]
        print(f"Filtered to {len(tasks)} tasks by ID")
    
    if args.limit:
        tasks = tasks[:args.limit]
        print(f"Limited to {len(tasks)} tasks")
    
    if not tasks:
        print("No tasks to process")
        sys.exit(0)
    
    # Download images and create LoadedTasks
    print("\nDownloading images...")
    loaded_tasks: List[LoadedTask] = []
    
    for i, task in enumerate(tasks):
        print(f"  [{i+1}/{len(tasks)}] Task {task.id}: {task.character_name}")
        loaded = LoadedTask.from_task(task, cache_dir)
        if loaded is not None:
            loaded_tasks.append(loaded)
        else:
            print(f"    Skipped (failed to load images)")
    
    print(f"\nLoaded {len(loaded_tasks)} tasks")
    
    if not loaded_tasks:
        print("No valid tasks to process")
        sys.exit(0)
    
    # Process tasks
    print("\nProcessing tasks...")
    
    if args.sequential or len(config.gpu.devices) == 1:
        # Sequential processing
        device = config.gpu.devices[0]
        results = process_tasks_sequential(loaded_tasks, config, device)
    else:
        # Multi-GPU parallel processing
        with GPUWorkerPool(config.gpu.devices, config) as pool:
            results = pool.process_batch(loaded_tasks)
    
    # Save summary
    summary = {
        "project_id": config.label_studio.project_id,
        "total_tasks": len(loaded_tasks),
        "successful": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "config": {
            "devices": config.gpu.devices,
            "k_global": config.eacps.k_global,
            "m_global": config.eacps.m_global,
            "k_local": config.eacps.k_local,
            "steps": config.model.num_inference_steps,
            "cfg": config.model.true_cfg_scale,
        },
        "results": [
            {
                "task_id": r.task_id,
                "success": r.success,
                "error": r.error,
                **(r.result or {}),
            }
            for r in results
        ],
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total tasks: {summary['total_tasks']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    
    # List failed tasks
    failed = [r for r in results if not r.success]
    if failed:
        print("\nFailed tasks:")
        for r in failed:
            print(f"  {r.task_id}: {r.error}")


if __name__ == "__main__":
    main()
