#!/usr/bin/env python3
"""
Push EACPS results to Label Studio project.

Usage:
    python src/push_results.py \
        --input_dir outputs/inpaint_eacps \
        --project_id 14559 \
        --api_key YOUR_KEY \
        --api_url https://label.dashtoon.ai
"""
import argparse
import json
import os
import base64
import requests
from pathlib import Path
from typing import Dict, List, Any


def image_to_base64(image_path: Path) -> str:
    """Convert image to base64 data URL."""
    if not image_path.exists():
        return ""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{data}"


def push_results(
    input_dir: Path,
    project_id: int,
    api_key: str,
    api_url: str,
):
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }
    
    # 1. Fetch existing tasks to map external task_id -> internal Label Studio ID
    print("Fetching existing tasks to check for updates...")
    existing_tasks_map = {} # external_id -> internal_id
    
    # Pagination loop to get all tasks
    page = 1
    while True:
        try:
            resp = requests.get(
                f"{api_url.rstrip('/')}/api/projects/{project_id}/tasks",
                headers=headers,
                params={"page": page, "page_size": 100},
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            
            # Handle different response formats (list vs dict with 'results')
            if isinstance(data, list):
                current_tasks = data
                total_count = len(data) # Likely no pagination if list returned
            else:
                current_tasks = data.get("results", [])
                total_count = data.get("count", 0)
            
            if not current_tasks:
                break
                
            for t in current_tasks:
                # Label Studio stores our custom data in 'data' field
                if "data" in t and "task_id" in t["data"]:
                    existing_tasks_map[str(t["data"]["task_id"])] = t["id"]
            
            if isinstance(data, list) or not data.get("next"):
                break
            page += 1
            
        except Exception as e:
            print(f"Warning: Could not fetch existing tasks: {e}")
            break
            
    print(f"Found {len(existing_tasks_map)} existing tasks in project.")

    tasks_to_create = []
    tasks_to_update = [] # List of (internal_id, data_dict)

    # Find all task directories
    task_dirs = sorted(list(input_dir.glob("task_*")))
    print(f"Found {len(task_dirs)} task directories in {input_dir}")
    
    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue
            
        # Read metrics
        metrics_file = task_dir / "metrics.json"
        metrics = {}
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
        
        task_id = metrics.get("task_id", task_dir.name.replace("task_", ""))
        
        # Images
        result_img = task_dir / "result.png"
        if not result_img.exists():
            # Fallback to eacps_best.png if result.png doesn't exist
            result_img = task_dir / "eacps_best.png"
        
        if not result_img.exists():
            print(f"Skipping {task_dir.name}: No result image found")
            continue
            
        # Convert to base64
        image_b64 = image_to_base64(result_img)
        
        # Additional images for context
        init_img = task_dir / "init.png"
        char_img = task_dir / "character.png"
        
        if init_img.exists():
            init_b64 = image_to_base64(init_img)
            print(f"  Found init.png ({len(init_b64)} bytes)")
        else:
            init_b64 = ""
            print(f"  WARNING: init.png not found in {task_dir}")

        if char_img.exists():
            char_b64 = image_to_base64(char_img)
            print(f"  Found character.png ({len(char_b64)} bytes)")
        else:
            char_b64 = ""
            print(f"  WARNING: character.png not found in {task_dir}")
        
        # Construct task data
        # We assume the project has an <Image value="$image"/> tag
        # We also include other metadata
        task_data = {
            "data": {
                "image": image_b64,
                "init_image": init_b64,
                "character_image": char_b64,
                "task_id": task_id,
                "character_name": metrics.get("character_name", ""),
                "best_potential": metrics.get("best_potential", 0),
                "best_seed": metrics.get("best_seed", 0),
                "scores": metrics.get("best_scores", {}),
                "filename": result_img.name,
            }
        }
        
        # DEBUG: Print keys being sent
        print(f"  Task Data Keys: {list(task_data['data'].keys())}")
        if not init_b64:
            print("  ERROR: init_image is empty!")
        if not char_b64:
            print("  ERROR: character_image is empty!")
            
        # Check if we should update or create
        if str(task_id) in existing_tasks_map:
            internal_id = existing_tasks_map[str(task_id)]
            tasks_to_update.append((internal_id, task_data["data"]))
            print(f"  -> Will UPDATE existing task {internal_id}")
        else:
            tasks_to_create.append(task_data)
            print(f"  -> Will CREATE new task")
    
    # 1. Handle Updates
    if tasks_to_update:
        print(f"\nUpdating {len(tasks_to_update)} existing tasks...")
        for internal_id, data in tasks_to_update:
            try:
                # PATCH /api/tasks/{id}
                update_url = f"{api_url.rstrip('/')}/api/tasks/{internal_id}"
                resp = requests.patch(
                    update_url,
                    headers=headers,
                    json={"data": data},
                    timeout=60
                )
                resp.raise_for_status()
                print(f"  Updated task {internal_id}")
            except Exception as e:
                print(f"  Error updating task {internal_id}: {e}")

    # 2. Handle Creations
    if not tasks_to_create:
        print("\nNo new tasks to create.")
        return

    print(f"\nCreating {len(tasks_to_create)} new tasks...")
    
    # Upload in batches
    batch_size = 1
    import_url = f"{api_url.rstrip('/')}/api/projects/{project_id}/import"
    
    for i in range(0, len(tasks_to_create), batch_size):
        batch = tasks_to_create[i:i+batch_size]
        print(f"Uploading batch {i//batch_size + 1} ({len(batch)} tasks)...")
        
        try:
            resp = requests.post(
                import_url,
                headers=headers,
                json=batch,
                timeout=120 # Large timeout for base64 images
            )
            resp.raise_for_status()
            print(f"  Success: {resp.json()}")
        except Exception as e:
            print(f"  Error uploading batch: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"  Response: {e.response.text}")

    print("Upload complete")


def main():
    parser = argparse.ArgumentParser(description="Push results to Label Studio")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with task results")
    parser.add_argument("--project_id", type=int, required=True, help="Label Studio Project ID")
    parser.add_argument("--api_key", type=str, required=True, help="Label Studio API Key")
    parser.add_argument("--api_url", type=str, default="https://label.dashtoon.ai", help="Label Studio URL")
    
    args = parser.parse_args()
    
    push_results(
        Path(args.input_dir),
        args.project_id,
        args.api_key,
        args.api_url
    )


if __name__ == "__main__":
    main()
