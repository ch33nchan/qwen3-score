#!/usr/bin/env python3
"""
Simple face swap pipeline using InsightFace.
No EACPS, no diffusion model - just direct face swap.

Usage:
    python3 mask/run_faceswap.py --from_file project-label.json --task_ids 71651078
    python3 mask/run_faceswap.py --project_id 14514 --task_ids 71651078 --api_key '...' --url '...'
"""
import argparse
import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Face swap using InsightFace")
    
    # Data source
    parser.add_argument("--project_id", type=int, nargs="+", help="Label Studio project ID(s)")
    parser.add_argument("--api_key", type=str, default=os.getenv("LABEL_STUDIO_API_KEY", ""))
    parser.add_argument("--url", type=str, default="https://label.dashtoon.ai")
    parser.add_argument("--from_file", type=str, help="Read tasks from local JSON file")
    
    # Task filtering
    parser.add_argument("--task_ids", type=str, nargs="+", help="Process only these task IDs")
    parser.add_argument("--limit", type=int, help="Limit number of tasks")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/faceswap_results")
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    
    # Debug
    parser.add_argument("--dry_run", action="store_true", help="Show tasks without processing")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate
    if not args.project_id and not args.from_file:
        print("Error: Either --project_id or --from_file required")
        sys.exit(1)
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Import here to avoid slow startup for --help
    sys.path.insert(0, str(Path(__file__).parent))
    from labelstudio import Task, LabelStudioClient, LoadedTask
    from faceswap import process_faceswap_task
    
    # Fetch tasks
    tasks = []
    
    if args.from_file:
        logger.info(f"Reading tasks from {args.from_file}")
        with open(args.from_file) as f:
            data = json.load(f)
        for item in data:
            task = Task.from_dict(item)
            if task.is_valid():
                tasks.append(task)
        logger.info(f"Loaded {len(tasks)} tasks from file")
    else:
        client = LabelStudioClient(args.url, args.api_key)
        for pid in args.project_id:
            logger.info(f"Fetching project {pid}...")
            project_tasks = client.fetch_project_tasks(pid, deduplicate=False)
            tasks.extend(project_tasks)
            logger.info(f"  Found {len(project_tasks)} tasks")
    
    # Filter by task IDs
    if args.task_ids:
        task_id_set = set(args.task_ids)
        tasks = [t for t in tasks if t.id in task_id_set]
        logger.info(f"Filtered to {len(tasks)} tasks by ID")
    
    # Limit
    if args.limit:
        tasks = tasks[:args.limit]
    
    if not tasks:
        logger.warning("No tasks to process")
        sys.exit(0)
    
    # Dry run
    if args.dry_run:
        logger.info("DRY RUN - Tasks:")
        for i, t in enumerate(tasks):
            logger.info(f"  [{i+1}] ID={t.id}, character={t.character_name}")
        sys.exit(0)
    
    # Process tasks
    logger.info(f"Processing {len(tasks)} tasks...")
    results = []
    
    for i, task in enumerate(tasks):
        logger.info(f"[{i+1}/{len(tasks)}] Task {task.id} ({task.character_name})")
        
        # Load images
        loaded = LoadedTask.from_task(task, cache_dir)
        if loaded is None:
            logger.error(f"  Failed to load images")
            results.append({"task_id": task.id, "success": False, "error": "Image load failed"})
            continue
        
        # Process
        try:
            result = process_faceswap_task(
                init_image=loaded.init_image,
                mask_image=loaded.mask_image,
                character_image=loaded.character_image,
                output_dir=output_dir,
                task_id=task.id,
            )
            results.append(result)
            
            if result["success"]:
                logger.info(f"  Success")
            else:
                logger.error(f"  Failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"  Error: {e}")
            results.append({"task_id": task.id, "success": False, "error": str(e)})
    
    # Summary
    success = sum(1 for r in results if r.get("success"))
    logger.info(f"Done: {success}/{len(results)} successful")
    
    # Save summary
    summary = {
        "total": len(results),
        "success": success,
        "failed": len(results) - success,
        "results": [{k: v for k, v in r.items() if k != "result_image"} for r in results]
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
