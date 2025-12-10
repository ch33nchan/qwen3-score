#!/usr/bin/env python3
"""
Delete tasks from Label Studio project.

Usage:
    python src/delete_tasks.py \
        --project_id 14559 \
        --api_key YOUR_KEY \
        --api_url https://label.dashtoon.ai \
        --count 3
"""
import argparse
import requests
import sys

def delete_first_n_tasks(
    project_id: int,
    api_key: str,
    api_url: str,
    count: int
):
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }
    
    # 1. Fetch tasks
    tasks_url = f"{api_url.rstrip('/')}/api/projects/{project_id}/tasks"
    print(f"Fetching tasks from {tasks_url}...")
    
    try:
        resp = requests.get(
            tasks_url,
            headers=headers,
            params={"page_size": count},
            timeout=30
        )
        resp.raise_for_status()
        tasks = resp.json()
        
        if not tasks:
            print("No tasks found in project.")
            return

        # If response is paginated, it might be in 'results' key, or list directly
        if isinstance(tasks, dict) and "results" in tasks:
            tasks_to_delete = tasks["results"]
        elif isinstance(tasks, list):
            tasks_to_delete = tasks
        else:
            print(f"Unexpected response format: {type(tasks)}")
            return
            
        # Limit to requested count
        tasks_to_delete = tasks_to_delete[:count]
        
        if not tasks_to_delete:
            print("No tasks to delete.")
            return
            
        print(f"Found {len(tasks_to_delete)} tasks to delete.")
        
        # 2. Delete tasks
        for task in tasks_to_delete:
            task_id = task["id"]
            delete_url = f"{api_url.rstrip('/')}/api/tasks/{task_id}"
            print(f"Deleting task {task_id}...")
            
            del_resp = requests.delete(delete_url, headers=headers)
            if del_resp.status_code == 204:
                print(f"  Deleted task {task_id}")
            else:
                print(f"  Failed to delete task {task_id}: {del_resp.status_code} {del_resp.text}")
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Delete tasks from Label Studio")
    parser.add_argument("--project_id", type=int, required=True, help="Label Studio Project ID")
    parser.add_argument("--api_key", type=str, required=True, help="Label Studio API Key")
    parser.add_argument("--api_url", type=str, default="https://label.dashtoon.ai", help="Label Studio URL")
    parser.add_argument("--count", type=int, default=3, help="Number of tasks to delete from the start")
    
    args = parser.parse_args()
    
    delete_first_n_tasks(
        args.project_id,
        args.api_key,
        args.api_url,
        args.count
    )

if __name__ == "__main__":
    main()
