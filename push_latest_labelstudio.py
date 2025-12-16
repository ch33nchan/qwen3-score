#!/usr/bin/env python3
"""
Push latest inpaint_eacps results to Label Studio as new tasks with embedded images.

Creates tasks under the target project where each task's `data.image` is a data URI
of the local result.png. This avoids needing external storage.
"""
import argparse
import base64
import json
import sys
from pathlib import Path
from typing import List

import requests


def encode_image_to_data_uri(image_path: Path) -> str:
    # Infer mime from suffix
    suffix = image_path.suffix.lower().lstrip(".")
    mime = "png"
    if suffix in {"jpg", "jpeg"}:
        mime = "jpeg"
    elif suffix == "webp":
        mime = "webp"
    img_bytes = image_path.read_bytes()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def collect_tasks(results_root: Path, limit: int) -> List[dict]:
    task_dirs = sorted(
        [p for p in results_root.glob("task_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    selected = task_dirs[:limit]
    tasks = []

    for d in selected:
        task_id = d.name.replace("task_", "")
        result_path = d / "result.png"

        if not result_path.exists():
            print(f"[skip] {d} missing result.png", file=sys.stderr)
            continue

        data_uri = encode_image_to_data_uri(result_path)

        tasks.append({
            "data": {
                "image": data_uri,
                "result": data_uri
            }
        })

    return tasks


def get_existing_tasks(api_url: str, api_key: str, project_id: str) -> List[dict]:
    """Get all existing tasks from Label Studio project."""
    url = f"{api_url.rstrip('/')}/api/projects/{project_id}/tasks"
    headers = {"Authorization": f"Token {api_key}"}
    
    all_tasks = []
    page = 1
    page_size = 100
    
    while True:
        params = {"page": page, "page_size": page_size}
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        # Handle both list response and dict with pagination
        if isinstance(data, list):
            all_tasks.extend(data)
            break
        elif isinstance(data, dict):
            all_tasks.extend(data.get("results", []))
            if not data.get("has_next", False):
                break
        else:
            break
        
        page += 1
    
    return all_tasks


def delete_annotated_tasks(api_url: str, api_key: str, project_id: str) -> int:
    """Delete all annotated tasks, keep only unannotated ones."""
    headers = {"Authorization": f"Token {api_key}"}
    
    # Get all tasks
    existing = get_existing_tasks(api_url, api_key, project_id)
    
    annotated_ids = []
    for task in existing:
        # Check if task has annotations
        if task.get("annotations") and len(task.get("annotations", [])) > 0:
            annotated_ids.append(task["id"])
    
    if not annotated_ids:
        print("No annotated tasks to delete.")
        return 0
    
    # Delete annotated tasks
    deleted = 0
    for task_id in annotated_ids:
        delete_url = f"{api_url.rstrip('/')}/api/tasks/{task_id}"
        try:
            resp = requests.delete(delete_url, headers=headers, timeout=30)
            resp.raise_for_status()
            deleted += 1
        except requests.HTTPError as e:
            print(f"Failed to delete task {task_id}: {e}", file=sys.stderr)
    
    print(f"Deleted {deleted} annotated task(s).")
    return deleted


def push_to_labelstudio(api_url: str, api_key: str, project_id: str, tasks: List[dict]) -> None:
    if not tasks:
        print("No tasks to push.")
        return

    url = f"{api_url.rstrip('/')}/api/projects/{project_id}/import"
    headers = {"Authorization": f"Token {api_key}"}

    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')
        temp_file = f.name
    
    try:
        with open(temp_file, 'rb') as f:
            files = {'file': ('tasks.jsonl', f, 'application/jsonl')}
            resp = requests.post(url, headers=headers, files=files, timeout=120)
            resp.raise_for_status()
            result = resp.json()
            print(f"Pushed {len(tasks)} task(s) to project {project_id}.")
            print(f"Response: {result}")
    except requests.HTTPError as e:
        print(f"Failed to push tasks: {resp.status_code}", file=sys.stderr)
        print(f"Full server response: {resp.text}", file=sys.stderr)
        try:
            error_data = resp.json()
            print(f"Parsed error: {json.dumps(error_data, indent=2)}", file=sys.stderr)
        except Exception:
            pass
        raise e
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def main():
    parser = argparse.ArgumentParser(description="Push latest inpaint_eacps results to Label Studio")
    parser.add_argument("--api_key", required=True, help="Label Studio API key")
    parser.add_argument("--api_url", required=True, help="Label Studio base URL (e.g., https://label.dashtoon.ai)")
    parser.add_argument("--project_id", required=True, help="Label Studio project ID")
    parser.add_argument("--results_root", default="inpaint_eacps", help="Path to results root")
    parser.add_argument("--n", type=int, default=5, help="Number of latest tasks to push")
    parser.add_argument("--delete_annotated", action="store_true", help="Delete annotated tasks before pushing")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.exists():
        print(f"Results root not found: {results_root}", file=sys.stderr)
        sys.exit(1)

    # Delete annotated tasks if requested
    if args.delete_annotated:
        print("Deleting annotated tasks...")
        delete_annotated_tasks(args.api_url, args.api_key, args.project_id)

    tasks = collect_tasks(results_root, args.n)
    push_to_labelstudio(args.api_url, args.api_key, args.project_id, tasks)


if __name__ == "__main__":
    main()
