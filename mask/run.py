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
        --devices cuda:0,cuda:2,cuda:3 \
        --k_global 4 \
        --m_global 2 \
        --k_local 2
"""
import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
def setup_logging(output_dir: Path, debug: bool = False):
    """Setup logging to both file and console."""
    log_level = logging.DEBUG if debug else logging.INFO
    log_file = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)


def check_gpu_availability(devices: List[str]) -> List[str]:
    """Check which GPUs are available and have enough memory."""
    import torch
    
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return ["cpu"]
    
    available = []
    min_memory_gb = 20  # Minimum free memory required
    
    for device in devices:
        if device == "cpu":
            available.append(device)
            continue
        
        try:
            device_id = int(device.split(":")[1]) if ":" in device else 0
            if device_id >= torch.cuda.device_count():
                print(f"  {device}: Not found (only {torch.cuda.device_count()} GPUs)")
                continue
            
            props = torch.cuda.get_device_properties(device_id)
            total_mem = props.total_memory / (1024**3)
            
            # Get free memory
            torch.cuda.set_device(device_id)
            free_mem = (torch.cuda.get_device_properties(device_id).total_memory - 
                       torch.cuda.memory_allocated(device_id)) / (1024**3)
            
            if free_mem >= min_memory_gb:
                available.append(device)
                print(f"  {device}: OK ({free_mem:.1f}GB free / {total_mem:.1f}GB total)")
            else:
                print(f"  {device}: SKIP (only {free_mem:.1f}GB free, need {min_memory_gb}GB)")
        except Exception as e:
            print(f"  {device}: ERROR ({e})")
    
    return available


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
        "--devices", type=str, default="cuda:0,cuda:2,cuda:3",
        help="Comma-separated GPU devices (e.g., cuda:0,cuda:2,cuda:3)"
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Process sequentially on first device (for debugging)"
    )
    parser.add_argument(
        "--auto_select_gpus", action="store_true",
        help="Automatically select GPUs with enough free memory"
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
    
    # Debug settings
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Fetch tasks but don't process (for testing)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate API key
    if not args.api_key:
        print("Error: API key required. Set --api_key or LABEL_STUDIO_API_KEY env var")
        sys.exit(1)
    
    # Create output dir early for logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir, args.debug)
    
    # Import after arg parsing (for faster --help)
    from config import PipelineConfig
    from labelstudio import LabelStudioClient, LoadedTask
    from worker import GPUWorkerPool, process_tasks_sequential
    
    logger.info("=" * 70)
    logger.info("MASK PIPELINE: Character Face Replacement with EACPS")
    logger.info("=" * 70)
    
    # Parse and validate GPUs
    requested_devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    
    logger.info("Checking GPU availability...")
    if args.auto_select_gpus:
        available_devices = check_gpu_availability(requested_devices)
    else:
        available_devices = check_gpu_availability(requested_devices)
        # Only use the ones that passed the check
    
    if not available_devices:
        logger.error("No GPUs available!")
        sys.exit(1)
    
    logger.info(f"Using devices: {available_devices}")
    
    # Create config
    config = PipelineConfig.from_args(
        project_id=args.project_id,
        api_key=args.api_key,
        url=args.url,
        output_dir=args.output_dir,
        devices=",".join(available_devices),
        k_global=args.k_global,
        m_global=args.m_global,
        k_local=args.k_local,
        steps=args.steps,
        cfg=args.cfg,
    )
    config.model.model_id = args.model
    config.cache_dir = args.cache_dir
    
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Label Studio: {config.label_studio.url}")
    logger.info(f"Project ID: {config.label_studio.project_id}")
    logger.info(f"Devices: {config.gpu.devices}")
    logger.info(f"EACPS: k_global={config.eacps.k_global}, m_global={config.eacps.m_global}, k_local={config.eacps.k_local}")
    logger.info(f"Output: {output_dir}")
    
    # Fetch tasks from Label Studio
    logger.info("Fetching tasks from Label Studio...")
    client = LabelStudioClient(config.label_studio.url, config.label_studio.api_key)
    
    try:
        tasks = client.fetch_project_tasks(config.label_studio.project_id)
    except Exception as e:
        logger.error(f"Error fetching tasks: {e}")
        sys.exit(1)
    
    logger.info(f"Found {len(tasks)} tasks")
    
    # Filter tasks if requested
    if args.task_ids:
        task_id_set = set(args.task_ids)
        tasks = [t for t in tasks if t.id in task_id_set]
        logger.info(f"Filtered to {len(tasks)} tasks by ID")
    
    if args.limit:
        tasks = tasks[:args.limit]
        logger.info(f"Limited to {len(tasks)} tasks")
    
    if not tasks:
        logger.warning("No tasks to process")
        sys.exit(0)
    
    # Dry run - just show what would be processed
    if args.dry_run:
        logger.info("DRY RUN - Tasks that would be processed:")
        for i, task in enumerate(tasks):
            logger.info(f"  [{i+1}] ID={task.id}, character={task.character_name}")
        sys.exit(0)
    
    # Download images and create LoadedTasks
    logger.info("Downloading images...")
    loaded_tasks: List[LoadedTask] = []
    
    for i, task in enumerate(tasks):
        logger.info(f"  [{i+1}/{len(tasks)}] Task {task.id}: {task.character_name}")
        loaded = LoadedTask.from_task(task, cache_dir)
        if loaded is not None:
            loaded_tasks.append(loaded)
            logger.debug(f"    init: {task.init_image_url[:50]}...")
            logger.debug(f"    mask: {task.mask_image_url[:50]}...")
            logger.debug(f"    char: {task.character_image_url[:50]}...")
        else:
            logger.warning(f"    Skipped (failed to load images)")
    
    logger.info(f"Loaded {len(loaded_tasks)} tasks successfully")
    
    if not loaded_tasks:
        logger.error("No valid tasks to process")
        sys.exit(0)
    
    # Process tasks
    logger.info("Processing tasks...")
    start_time = datetime.now()
    
    if args.sequential or len(config.gpu.devices) == 1:
        # Sequential processing
        device = config.gpu.devices[0]
        logger.info(f"Sequential processing on {device}")
        results = process_tasks_sequential(loaded_tasks, config, device)
    else:
        # Multi-GPU parallel processing
        logger.info(f"Parallel processing on {len(config.gpu.devices)} GPUs")
        with GPUWorkerPool(config.gpu.devices, config) as pool:
            results = pool.process_batch(loaded_tasks)
    
    elapsed = datetime.now() - start_time
    
    # Save summary
    summary = {
        "project_id": config.label_studio.project_id,
        "total_tasks": len(loaded_tasks),
        "successful": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "elapsed_seconds": elapsed.total_seconds(),
        "devices_used": config.gpu.devices,
        "config": {
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
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total tasks: {summary['total_tasks']}")
    logger.info(f"Successful: {summary['successful']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Elapsed: {elapsed}")
    logger.info(f"Avg per task: {elapsed.total_seconds() / max(1, len(loaded_tasks)):.1f}s")
    logger.info(f"Results saved to: {output_dir}")
    
    # List failed tasks
    failed = [r for r in results if not r.success]
    if failed:
        logger.warning("Failed tasks:")
        for r in failed:
            logger.warning(f"  {r.task_id}: {r.error}")


if __name__ == "__main__":
    main()
