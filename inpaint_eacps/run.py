#!/usr/bin/env python3
"""
Run inpainting with EACPS on Label Studio tasks.

Usage:
    python3 inpaint_eacps/run.py --task_id 71651078 --gemini_api_key YOUR_KEY
    python3 inpaint_eacps/run.py --from_file project-label.json --task_id 71651078
"""
import argparse
import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import requests
from io import BytesIO
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def download_image(url: str, cache_dir: Path) -> Image.Image:
    """Download image from URL with caching."""
    import hashlib
    
    url_hash = hashlib.md5(url.encode()).hexdigest()
    ext = url.split(".")[-1].split("?")[0]
    if ext not in ["png", "jpg", "jpeg", "webp"]:
        ext = "png"
    
    cache_path = cache_dir / f"{url_hash}.{ext}"
    
    if cache_path.exists():
        return Image.open(cache_path)
    
    logger.info(f"  Downloading {url[:50]}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    img = Image.open(BytesIO(response.content))
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    img.save(cache_path)
    
    return img


def load_task_from_file(filepath: str, task_id: str) -> dict:
    """Load a specific task from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    
    for item in data:
        item_data = item.get("data", item)
        if str(item_data.get("id")) == str(task_id):
            return item_data
    
    raise ValueError(f"Task {task_id} not found in {filepath}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inpainting with EACPS")
    
    # Task selection
    parser.add_argument("--task_id", type=str, required=True, help="Task ID to process")
    parser.add_argument("--from_file", type=str, default="project-label.json", help="JSON file with tasks")
    
    # API keys
    parser.add_argument("--gemini_api_key", type=str, default=os.getenv("GEMINI_API_KEY", ""),
                        help="Gemini API key")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/inpaint_eacps")
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda:0")
    
    # EACPS settings
    parser.add_argument("--k_global", type=int, default=4)
    parser.add_argument("--m_global", type=int, default=2)
    parser.add_argument("--k_local", type=int, default=2)
    
    # Model settings
    parser.add_argument("--use_moondream", action="store_true", default=True,
                        help="Use Moondream for realism scoring (default: True)")
    parser.add_argument("--no_moondream", dest="use_moondream", action="store_false",
                        help="Disable Moondream (faster but less accurate)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load task
    logger.info(f"Loading task {args.task_id} from {args.from_file}")
    task_data = load_task_from_file(args.from_file, args.task_id)
    
    logger.info(f"Task: {task_data['character_name']}")
    logger.info(f"  init: {task_data['init_image_url'][:50]}...")
    logger.info(f"  mask: {task_data['mask_image_url'][:50]}...")
    logger.info(f"  char: {task_data['character_image_url'][:50]}...")
    
    # Download images
    logger.info("Downloading images...")
    init_image = download_image(task_data["init_image_url"], cache_dir).convert("RGB")
    mask_image = download_image(task_data["mask_image_url"], cache_dir).convert("L")
    character_image = download_image(task_data["character_image_url"], cache_dir).convert("RGBA")
    
    logger.info(f"  init: {init_image.size}")
    logger.info(f"  mask: {mask_image.size}")
    logger.info(f"  char: {character_image.size}")
    
    # Import pipeline (after arg parsing for faster --help)
    sys.path.insert(0, str(Path(__file__).parent))
    from config import PipelineConfig, EACPSConfig, ModelConfig
    from pipeline import process_task
    
    # Create config
    config = PipelineConfig(
        eacps=EACPSConfig(
            k_global=args.k_global,
            m_global=args.m_global,
            k_local=args.k_local,
        ),
        model=ModelConfig(
            gemini_api_key=args.gemini_api_key,
        ),
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        device=args.device,
    )
    
    if not config.model.gemini_api_key:
        logger.warning("No Gemini API key provided. Scoring will use defaults.")
    
    # Process
    logger.info("=" * 60)
    logger.info("Starting EACPS inpainting")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    result = process_task(
        init_image=init_image,
        mask_image=mask_image,
        character_image=character_image,
        character_name=task_data["character_name"],
        task_id=task_data["id"],
        config=config,
        output_dir=output_dir,
        use_moondream=args.use_moondream,
    )
    
    elapsed = datetime.now() - start_time
    
    # Summary
    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)
    logger.info(f"Task: {result['task_id']}")
    logger.info(f"Character: {result['character_name']}")
    logger.info(f"Best seed: {result['best_seed']} ({result['best_phase']})")
    logger.info(f"Best potential: {result['best_potential']:.2f}")
    logger.info(f"Scores: {result['best_scores']}")
    logger.info(f"Elapsed: {elapsed}")
    logger.info(f"Output: {result.get('output_dir', output_dir)}")
    
    # Auto-push to git
    try:
        import subprocess
        repo_root = Path(__file__).parent.parent
        
        logger.info("Pushing results to git...")
        subprocess.run(
            ["git", "add", str(output_dir / f"task_{result['task_id']}")],
            cwd=repo_root,
            check=False,
        )
        subprocess.run(
            ["git", "commit", "-m", f"Inpaint EACPS result: task {result['task_id']} ({result['character_name']})"],
            cwd=repo_root,
            check=False,
        )
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=repo_root,
            check=False,
        )
        logger.info("Git push completed")
    except Exception as e:
        logger.warning(f"Git push failed (non-critical): {e}")


if __name__ == "__main__":
    main()
