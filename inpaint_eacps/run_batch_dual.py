#!/usr/bin/env python3
"""
Batch process multiple tasks with dual output mode.
Optimized for API usage with optional Moondream.

Usage:
    # With both Gemini and Moondream (best quality)
    python3 inpaint_eacps/run_batch_dual.py --task_ids 71650389 71678832 --gemini_api_key KEY
    
    # Gemini only (faster, less VRAM)
    python3 inpaint_eacps/run_batch_dual.py --task_ids 71650389 71678832 --gemini_api_key KEY --no_moondream
    
    # Reduced candidates to save API calls (K_global=4, K_local=2 = 12 candidates vs 20)
    python3 inpaint_eacps/run_batch_dual.py --task_ids 71650389 71678832 --gemini_api_key KEY --fast
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from inpaint_eacps.config import PipelineConfig, EACPSConfig, ModelConfig
from inpaint_eacps.pipeline import (
    QwenEditPipeline,
    MultiModelScorer,
    run_eacps_inpaint,
    swap_face_insightface,
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
    
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    img = Image.open(BytesIO(response.content))
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    img.save(cache_path)
    
    return img


def load_task_from_file(filepath: str, task_id: str) -> dict:
    """Load a specific task from Label Studio JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    
    for item in data:
        item_data = item.get("data", {})
        if str(item_data.get("id")) == str(task_id):
            return item_data
    
    raise ValueError(f"Task {task_id} not found in {filepath}")


def process_task_dual(
    task_id: str,
    task_data: dict,
    cache_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    qwen_pipe: QwenEditPipeline,
    scorer: MultiModelScorer,
    api_call_counter: dict,
    pbar: tqdm = None,
) -> dict:
    """Process single task with dual output and track API calls."""
    
    logger.info("=" * 80)
    logger.info(f"Task {task_id}: {task_data.get('character_name', 'Unknown')}")
    logger.info("=" * 80)
    
    # Download images (Label Studio format)
    init_image = download_image(task_data["init_image_url"], cache_dir)
    mask_image = download_image(task_data["mask_image_url"], cache_dir)
    character_image = download_image(task_data["character_image_url"], cache_dir)
    
    task_dir = output_dir / f"task_{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Save inputs
    init_image.save(task_dir / "init.png")
    mask_image.save(task_dir / "mask.png")
    character_image.convert("RGB").save(task_dir / "character.png")
    
    # Track API calls before
    calls_before = api_call_counter['total']
    
    # ============================================================
    # VERSION 1: PRESERVE INIT HAIR
    # ============================================================
    if pbar:
        pbar.set_description(f"Task {task_id} - Init hair")
    logger.info("🎨 VERSION 1: Init hair preservation")
    
    best_init_hair, candidates_init = run_eacps_inpaint(
        init_image=init_image,
        mask_image=mask_image,
        character_image=character_image,
        character_name=task_data.get("character_name", "character"),
        qwen_pipe=qwen_pipe,
        scorer=scorer,
        config=config,
        verbose=True,
        preserve_init_hair=True,
    )
    
    best_cand_init = candidates_init[0]
    best_init_hair.save(task_dir / "result_init_hair.png")
    
    # Save top 3 candidates
    for i, cand in enumerate(candidates_init[:3]):
        cand.image.save(task_dir / f"init_hair_top{i+1}_seed{cand.seed}.png")
    
    logger.info(f"  ✓ Seed={best_cand_init.seed}, Potential={best_cand_init.potential:.2f}")
    
    # ============================================================
    # VERSION 2: USE CHARACTER HAIR
    # ============================================================
    if pbar:
        pbar.set_description(f"Task {task_id} - Char hair")
    logger.info("🎭 VERSION 2: Character hair")
    
    best_char_hair, candidates_char = run_eacps_inpaint(
        init_image=init_image,
        mask_image=mask_image,
        character_image=character_image,
        character_name=task_data.get("character_name", "character"),
        qwen_pipe=qwen_pipe,
        scorer=scorer,
        config=config,
        verbose=True,
        preserve_init_hair=False,
    )
    
    best_cand_char = candidates_char[0]
    best_char_hair.save(task_dir / "result_char_hair.png")
    
    # Save top 3 candidates
    for i, cand in enumerate(candidates_char[:3]):
        cand.image.save(task_dir / f"char_hair_top{i+1}_seed{cand.seed}.png")
    
    logger.info(f"  ✓ Seed={best_cand_char.seed}, Potential={best_cand_char.potential:.2f}")
    
    # Track API calls after
    calls_after = api_call_counter['total']
    calls_this_task = calls_after - calls_before
    api_call_counter['total'] = calls_after
    
    # Save face swap base
    swapped = swap_face_insightface(init_image, character_image, mask_image)
    if swapped:
        swapped.save(task_dir / "faceswap_base.png")
    
    # Save metrics
    result = {
        "task_id": task_id,
        "character_name": task_data.get("character_name", "Unknown"),
        "character_id": task_data.get("character_id"),
        "init_hair": {
            "seed": best_cand_init.seed,
            "phase": best_cand_init.phase,
            "potential": best_cand_init.potential,
            "scores": best_cand_init.scores,
        },
        "char_hair": {
            "seed": best_cand_char.seed,
            "phase": best_cand_char.phase,
            "potential": best_cand_char.potential,
            "scores": best_cand_char.scores,
        },
        "candidates_per_version": len(candidates_init),
        "api_calls_this_task": calls_this_task,
    }
    
    with open(task_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"  ✓ API calls this task: {calls_this_task}")
    logger.info(f"  ✓ Output: {task_dir}")
    
    if pbar:
        pbar.update(1)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Batch dual-output EACPS")
    
    parser.add_argument("--task_ids", type=str, nargs='+', required=True, help="Task IDs to process")
    parser.add_argument("--from_file", type=str, default="project-label.json")
    parser.add_argument("--gemini_api_key", type=str, default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--output_dir", type=str, default="outputs/dual_hair_batch")
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    # Optimization flags
    parser.add_argument("--no_moondream", action="store_true", help="Disable Moondream (save VRAM)")
    parser.add_argument("--fast", action="store_true", help="Use fewer candidates (K=4, M=2, L=2)")
    
    # Manual EACPS config
    parser.add_argument("--k_global", type=int, default=None)
    parser.add_argument("--m_global", type=int, default=None)
    parser.add_argument("--k_local", type=int, default=None)
    
    args = parser.parse_args()
    
    if not args.gemini_api_key:
        logger.error("Gemini API key required!")
        sys.exit(1)
    
    # Setup
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # EACPS config
    if args.fast:
        k_global, m_global, k_local = 4, 2, 2  # 12 candidates (50% fewer API calls)
    else:
        k_global = args.k_global or 8
        m_global = args.m_global or 3
        k_local = args.k_local or 4  # 20 candidates (default)
    
    eacps_config = EACPSConfig(
        k_global=k_global,
        m_global=m_global,
        k_local=k_local,
    )
    
    model_config = ModelConfig(gemini_api_key=args.gemini_api_key)
    config = PipelineConfig(
        eacps=eacps_config,
        model=model_config,
        device=args.device,
    )
    
    # Calculate expected API calls
    candidates_per_run = k_global + m_global * k_local
    runs_per_task = 2  # dual output
    calls_per_task = candidates_per_run * runs_per_task
    total_expected_calls = calls_per_task * len(args.task_ids)
    
    logger.info("=" * 80)
    logger.info("BATCH CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Tasks: {len(args.task_ids)}")
    logger.info(f"EACPS: K_global={k_global}, M={m_global}, K_local={k_local}")
    logger.info(f"Candidates per run: {candidates_per_run}")
    logger.info(f"Runs per task: {runs_per_task} (dual output)")
    logger.info(f"Gemini calls per task: {calls_per_task}")
    logger.info(f"TOTAL expected Gemini calls: {total_expected_calls}")
    logger.info(f"Moondream: {'DISABLED' if args.no_moondream else 'ENABLED'}")
    
    estimated_cost = total_expected_calls * 0.00375
    logger.info(f"Estimated Gemini cost: ${estimated_cost:.2f}")
    logger.info("=" * 80)
    
    # Initialize models once
    logger.info("Initializing models...")
    qwen_pipe = QwenEditPipeline(config.model.qwen_model_id, config.device)
    scorer = MultiModelScorer(
        gemini_api_key=config.model.gemini_api_key,
        gemini_model=config.model.gemini_model,
        moondream_model_id=config.model.moondream_model_id,
        device=config.device,
        use_gemini=True,
        use_moondream=not args.no_moondream,
    )
    
    # Process all tasks
    api_call_counter = {'total': 0}
    results = []
    start_time = datetime.now()
    
    with tqdm(total=len(args.task_ids), desc="Batch Progress", unit="task", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for i, task_id in enumerate(args.task_ids, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"TASK {i}/{len(args.task_ids)}: {task_id}")
            logger.info(f"{'='*80}")
            
            try:
                task_data = load_task_from_file(args.from_file, task_id)
                
                result = process_task_dual(
                    task_id=task_id,
                    task_data=task_data,
                    cache_dir=cache_dir,
                    output_dir=output_dir,
                    config=config,
                    qwen_pipe=qwen_pipe,
                    scorer=scorer,
                    api_call_counter=api_call_counter,
                    pbar=pbar,
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed task {task_id}: {e}")
                import traceback
                traceback.print_exc()
                pbar.update(1)
                continue
    
    # Final summary
    elapsed = datetime.now() - start_time
    
    logger.info("\n" + "=" * 80)
    logger.info("BATCH COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Processed: {len(results)}/{len(args.task_ids)} tasks")
    logger.info(f"Total Gemini API calls: {api_call_counter['total']}")
    logger.info(f"Actual cost: ${api_call_counter['total'] * 0.00375:.2f}")
    logger.info(f"Total time: {elapsed}")
    logger.info(f"Output directory: {output_dir}")
    
    # Save batch summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "tasks_processed": len(results),
        "total_tasks": len(args.task_ids),
        "api_calls": api_call_counter['total'],
        "estimated_cost_usd": api_call_counter['total'] * 0.00375,
        "elapsed_seconds": elapsed.total_seconds(),
        "eacps_config": {
            "k_global": k_global,
            "m_global": m_global,
            "k_local": k_local,
        },
        "moondream_enabled": not args.no_moondream,
        "results": results,
    }
    
    with open(output_dir / "batch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved: {output_dir}/batch_summary.json")


if __name__ == "__main__":
    main()
