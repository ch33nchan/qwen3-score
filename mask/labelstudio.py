"""
Label Studio API client for fetching project tasks.
"""
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
from PIL import Image
from io import BytesIO


@dataclass
class Task:
    """Represents a single task from Label Studio."""
    id: str
    character_id: str
    character_name: str
    character_image_url: str
    init_image_url: str
    mask_image_url: str
    prompt: str
    image_url: Optional[str] = None  # Additional reference image if present
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Parse task from Label Studio JSON format."""
        # Handle both nested {"data": {...}} and flat formats
        task_data = data.get("data", data)
        return cls(
            id=str(task_data.get("id", "")),
            character_id=str(task_data.get("character_id", "")),
            character_name=str(task_data.get("character_name", "")),
            character_image_url=str(task_data.get("character_image_url", "")),
            init_image_url=str(task_data.get("init_image_url", "")),
            mask_image_url=str(task_data.get("mask_image_url", "")),
            prompt=str(task_data.get("prompt", "")),
            image_url=task_data.get("image_url"),
        )
    
    def is_valid(self) -> bool:
        """Check if task has all required fields."""
        return bool(
            self.id and 
            self.character_image_url and 
            self.init_image_url and 
            self.mask_image_url
        )


class LabelStudioClient:
    """Client for Label Studio API."""
    
    def __init__(self, url: str, api_key: str):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        })
    
    def fetch_project_tasks(self, project_id: int) -> List[Task]:
        """Fetch all tasks from a Label Studio project."""
        endpoint = f"{self.url}/api/projects/{project_id}/tasks"
        tasks = []
        page = 1
        
        while True:
            response = self.session.get(
                endpoint,
                params={"page": page, "page_size": 100},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            
            # Handle paginated response
            if isinstance(data, dict):
                items = data.get("tasks", data.get("results", []))
                total = data.get("total", len(items))
            else:
                items = data
                total = len(items)
            
            for item in items:
                task = Task.from_dict(item)
                if task.is_valid():
                    tasks.append(task)
            
            if len(tasks) >= total or not items:
                break
            page += 1
        
        return tasks
    
    def fetch_task_by_id(self, task_id: int) -> Optional[Task]:
        """Fetch a single task by ID."""
        endpoint = f"{self.url}/api/tasks/{task_id}"
        response = self.session.get(endpoint, timeout=30)
        
        if response.status_code == 404:
            return None
        
        response.raise_for_status()
        task = Task.from_dict(response.json())
        return task if task.is_valid() else None


def download_image(url: str, cache_dir: Path, timeout: int = 30) -> Optional[Path]:
    """Download image from URL and cache locally."""
    if not url or not url.strip():
        return None
    
    # Create cache filename from URL
    filename = url.split("/")[-1].split("?")[0]
    if not filename or len(filename) > 100:
        import hashlib
        filename = hashlib.md5(url.encode()).hexdigest() + ".png"
    
    cache_path = cache_dir / filename
    if cache_path.exists():
        return cache_path
    
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        # Validate it's an image
        img = Image.open(BytesIO(response.content))
        img.verify()
        
        # Save to cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(response.content)
        
        return cache_path
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None


def load_image_from_url(url: str, cache_dir: Path) -> Optional[Image.Image]:
    """Download and load image from URL."""
    path = download_image(url, cache_dir)
    if path is None:
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Failed to load image {path}: {e}")
        return None


@dataclass
class LoadedTask:
    """Task with downloaded images ready for processing."""
    task: Task
    init_image: Image.Image
    mask_image: Image.Image
    character_image: Image.Image
    
    @classmethod
    def from_task(cls, task: Task, cache_dir: Path) -> Optional["LoadedTask"]:
        """Download all images and create LoadedTask."""
        init_image = load_image_from_url(task.init_image_url, cache_dir)
        if init_image is None:
            print(f"Task {task.id}: Failed to load init_image")
            return None
        
        mask_path = download_image(task.mask_image_url, cache_dir)
        if mask_path is None:
            print(f"Task {task.id}: Failed to load mask_image")
            return None
        mask_image = Image.open(mask_path).convert("L")
        
        char_path = download_image(task.character_image_url, cache_dir)
        if char_path is None:
            print(f"Task {task.id}: Failed to load character_image")
            return None
        character_image = Image.open(char_path).convert("RGBA")
        
        return cls(
            task=task,
            init_image=init_image,
            mask_image=mask_image,
            character_image=character_image,
        )
