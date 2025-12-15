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
    img_bytes = image_path.read_bytes()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


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
        tasks.append(
            {
                "data": {
                    "id": task_id,
                    "image": data_uri,
                    "source": "inpaint_eacps",
                    "task_dir": str(d),
                }
            }
        )

    return tasks


def push_to_labelstudio(api_url: str, api_key: str, project_id: str, tasks: List[dict]) -> None:
    if not tasks:
        print("No tasks to push.")
        return

    url = f"{api_url.rstrip('/')}/api/projects/{project_id}/import"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }

    try:
        # Use json= to let requests set headers/encoding
        resp = requests.post(url, headers=headers, json=tasks, timeout=120)
        resp.raise_for_status()
        print(f"Pushed {len(tasks)} task(s) to project {project_id}.")
        print(f"Response: {resp.text}")
    except requests.HTTPError as e:
        print(f"Failed to push tasks: {resp.status_code}", file=sys.stderr)
        try:
            print(f"Server response: {resp.text}", file=sys.stderr)
        except Exception:
            pass
        print("Payload preview (first task):", file=sys.stderr)
        if tasks:
            print(json.dumps(tasks[0], indent=2)[:2000], file=sys.stderr)
        raise e


def main():
    parser = argparse.ArgumentParser(description="Push latest inpaint_eacps results to Label Studio")
    parser.add_argument("--api_key", required=True, help="Label Studio API key")
    parser.add_argument("--api_url", required=True, help="Label Studio base URL (e.g., https://label.dashtoon.ai)")
    parser.add_argument("--project_id", required=True, help="Label Studio project ID")
    parser.add_argument("--results_root", default="outputs/inpaint_eacps", help="Path to results root")
    parser.add_argument("--n", type=int, default=5, help="Number of latest tasks to push")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.exists():
        print(f"Results root not found: {results_root}", file=sys.stderr)
        sys.exit(1)

    tasks = collect_tasks(results_root, args.n)
    push_to_labelstudio(args.api_url, args.api_key, args.project_id, tasks)


if __name__ == "__main__":
    main()
