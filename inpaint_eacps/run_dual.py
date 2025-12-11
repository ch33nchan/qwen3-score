#!/usr/bin/env python3
"""
Run inpainting with EACPS - DUAL OUTPUT MODE
Generates two versions:
1. Init hair version: Preserves hairstyle from init image
2. Character hair version: Uses hairstyle from character image

Usage:
    python3 inpaint_eacps/run_dual.py --task_id 71651078 --gemini_api_key YOUR_KEY
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inpaint_eacps.config import PipelineConfig, EACPSConfig, ModelConfig
from inpaint_eacps.pipeline import (
    process_task,
    QwenEditPipeline,
    MultiModelScorer,
    run_eacps_inpaint,
    swap_face_insightface,
    composite_character_face_on_mask,
)

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


def process_task_dual_output(
    init_image: Image.Image,
    mask_image: Image.Image,
    character_image: Image.Image,
    character_name: str,
    task_id: str,
    config: PipelineConfig,
    output_dir: Path,
    qwen_pipe: QwenEditPipeline,
    scorer: MultiModelScorer,
) -> dict:
    """
    Process task with dual output mode.
    """
    logger.info("=" * 80)
    logger.info(f"DUAL OUTPUT MODE: Task {task_id} ({character_name})")
    logger.info("=" * 80)
    
    task_dir = output_dir / f"task_{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Save inputs
    init_image.save(task_dir / "init.png")
    mask_image.save(task_dir / "mask.png")
    character_image.convert("RGB").save(task_dir / "character.png")
    
    # ============================================================
    # VERSION 1: PRESERVE INIT HAIR
    # ============================================================
    logger.info("")
    logger.info("🎨 VERSION 1: Preserving init image hairstyle")
    logger.info("-" * 80)
    
    best_init_hair, candidates_init = run_eacps_inpaint(
        init_image=init_image,
        mask_image=mask_image,
        character_image=character_image,
        character_name=character_name,
        qwen_pipe=qwen_pipe,
        scorer=scorer,
        config=config,
        verbose=True,
        preserve_init_hair=True,
    )
    
    best_cand_init = candidates_init[0]
    best_init_hair.save(task_dir / "result_init_hair.png")
    
    logger.info(f"  ✓ Init hair version complete")
    logger.info(f"    Best seed: {best_cand_init.seed}")
    logger.info(f"    Potential: {best_cand_init.potential:.2f}")
    logger.info(f"    Scores: {best_cand_init.scores}")
    
    # Save top 5 candidates for init hair
    for i, cand in enumerate(candidates_init[:5]):
        cand.image.save(task_dir / f"init_hair_candidate_{i+1}_seed{cand.seed}.png")
    
    # ============================================================
    # VERSION 2: USE CHARACTER HAIR
    # ============================================================
    logger.info("")
    logger.info("🎭 VERSION 2: Using character image hairstyle")
    logger.info("-" * 80)
    
    best_char_hair, candidates_char = run_eacps_inpaint(
        init_image=init_image,
        mask_image=mask_image,
        character_image=character_image,
        character_name=character_name,
        qwen_pipe=qwen_pipe,
        scorer=scorer,
        config=config,
        verbose=True,
        preserve_init_hair=False,
    )
    
    best_cand_char = candidates_char[0]
    best_char_hair.save(task_dir / "result_char_hair.png")
    
    logger.info(f"  ✓ Character hair version complete")
    logger.info(f"    Best seed: {best_cand_char.seed}")
    logger.info(f"    Potential: {best_cand_char.potential:.2f}")
    logger.info(f"    Scores: {best_cand_char.scores}")
    
    # Save top 5 candidates for char hair
    for i, cand in enumerate(candidates_char[:5]):
        cand.image.save(task_dir / f"char_hair_candidate_{i+1}_seed{cand.seed}.png")
    
    # ============================================================
    # SAVE COMPARISON AND METADATA
    # ============================================================
    
    # Save face swap base for reference
    swapped = swap_face_insightface(init_image, character_image, mask_image)
    if swapped:
        swapped.save(task_dir / "faceswap_base.png")
    
    # Save metrics
    result = {
        "task_id": task_id,
        "character_name": character_name,
        "dual_output": True,
        "init_hair_version": {
            "seed": best_cand_init.seed,
            "phase": best_cand_init.phase,
            "potential": best_cand_init.potential,
            "scores": best_cand_init.scores,
        },
        "char_hair_version": {
            "seed": best_cand_char.seed,
            "phase": best_cand_char.phase,
            "potential": best_cand_char.potential,
            "scores": best_cand_char.scores,
        },
        "total_candidates_per_version": len(candidates_init),
    }
    
    with open(task_dir / "metrics_dual.json", "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ DUAL OUTPUT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {task_dir}")
    logger.info(f"  - result_init_hair.png (preserves init hairstyle)")
    logger.info(f"  - result_char_hair.png (uses character hairstyle)")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Inpainting with EACPS - Dual Output Mode")
    
    parser.add_argument("--task_id", type=str, required=True, help="Task ID to process")
    parser.add_argument("--from_file", type=str, default="project-label.json", help="JSON file with tasks")
    parser.add_argument("--gemini_api_key", type=str, default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--output_dir", type=str, default="outputs/inpaint_eacps_dual")
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    # EACPS settings
    parser.add_argument("--k_global", type=int, default=8, help="Global exploration candidates")
    parser.add_argument("--m_global", type=int, default=3, help="Top candidates for refinement")
    parser.add_argument("--k_local", type=int, default=4, help="Local refinement per candidate")
    
    args = parser.parse_args()
    
    if not args.gemini_api_key:
        logger.error("Gemini API key required. Set GEMINI_API_KEY or use --gemini_api_key")
        sys.exit(1)
    
    # Setup paths
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load task
    logger.info(f"Loading task {args.task_id} from {args.from_file}")
    task_data = load_task_from_file(args.from_file, args.task_id)
    
    # Download images
    logger.info("Downloading images...")
    init_image = download_image(task_data["init_image"], cache_dir)
    mask_image = download_image(task_data["mask"], cache_dir)
    character_image = download_image(task_data["character_image"], cache_dir)
    
    logger.info(f"  Init: {init_image.size}")
    logger.info(f"  Mask: {mask_image.size}")
    logger.info(f"  Character: {character_image.size}")
    
    # Setup config
    eacps_config = EACPSConfig(
        k_global=args.k_global,
        m_global=args.m_global,
        k_local=args.k_local,
    )
    
    model_config = ModelConfig(
        gemini_api_key=args.gemini_api_key,
    )
    
    config = PipelineConfig(
        eacps=eacps_config,
        model=model_config,
        device=args.device,
    )
    
    # Initialize models once
    logger.info("Initializing models...")
    qwen_pipe = QwenEditPipeline(config.model.qwen_model_id, config.device)
    scorer = MultiModelScorer(
        gemini_api_key=config.model.gemini_api_key,
        gemini_model=config.model.gemini_model,
        moondream_model_id=config.model.moondream_model_id,
        device=config.device,
        use_gemini=True,
        use_moondream=True,
    )
    
    # Process with dual output
    start_time = datetime.now()
    
    result = process_task_dual_output(
        init_image=init_image,
        mask_image=mask_image,
        character_image=character_image,
        character_name=task_data["name"],
        task_id=task_data["id"],
        config=config,
        output_dir=output_dir,
        qwen_pipe=qwen_pipe,
        scorer=scorer,
    )
    
    elapsed = datetime.now() - start_time
    logger.info(f"Total elapsed time: {elapsed}")


if __name__ == "__main__":
    main()
