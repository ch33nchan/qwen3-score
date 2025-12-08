#!/usr/bin/env python3
"""
Label Studio Character Replacement with Inference Scaling

Takes tasks from Label Studio with:
- init_image: background/scene image
- character_image: character to insert
- mask_image: where to place the character
- prompt: instruction for the edit

Uses Qwen-Image-Edit with TT-FLUX/EACPS inference scaling.

Usage:
    # Test with single task ID
    python src/labelstudio_inpaint.py \
        --api_key YOUR_API_KEY \
        --project_id 14514 \
        --task_id 12345 \
        --output_dir experiments/labelstudio_test \
        --device cuda:0

    # Process multiple tasks
    python src/labelstudio_inpaint.py \
        --api_key YOUR_API_KEY \
        --project_id 14514 \
        --task_ids 123 456 789 \
        --output_dir experiments/labelstudio_batch \
        --device cuda:0
"""
import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import time
import requests
from io import BytesIO

os.environ["DIFFUSERS_USE_FLASH_ATTENTION"] = "0"
os.environ["XFORMERS_DISABLED"] = "1"

import torch
from PIL import Image
from tqdm import tqdm

from ttflux.verifiers import LAIONAestheticVerifier


LABEL_STUDIO_URL = "https://label.dashtoon.ai"


@dataclass
class LabelStudioTask:
    id: int
    character_id: str
    character_name: str
    character_image_url: str
    image_url: str
    init_image_url: str
    mask_image_url: str
    prompt: str


@dataclass 
class InpaintResult:
    task_id: int
    character_id: str
    character_name: str
    ttflux_best_score: float
    ttflux_best_seed: int
    eacps_best_score: float
    eacps_best_seed: int
    winner: str
    score_diff: float
    time_seconds: float


def fetch_image(url: str, api_key: str = None) -> Image.Image:
    headers = {}
    if api_key and "label.dashtoon.ai" in url:
        headers["Authorization"] = f"Token {api_key}"
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def fetch_task_by_id(api_key: str, task_id: int) -> Optional[Dict]:
    """Fetch task by Label Studio task ID (direct API call)."""
    url = f"{LABEL_STUDIO_URL}/api/tasks/{task_id}"
    headers = {"Authorization": f"Token {api_key}"}
    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def fetch_task_by_data_id(api_key: str, project_id: int, data_id: str) -> Optional[Dict]:
    """Fetch task by data.id field (requires searching through project tasks)."""
    page = 1
    page_size = 100
    while True:
        url = f"{LABEL_STUDIO_URL}/api/projects/{project_id}/tasks"
        headers = {"Authorization": f"Token {api_key}"}
        params = {"page_size": page_size, "page": page}
        response = requests.get(url, headers=headers, params=params, timeout=60)
        response.raise_for_status()
        tasks = response.json()
        if not tasks:
            break
        for task in tasks:
            if task.get("data", {}).get("id") == data_id:
                return task
        if len(tasks) < page_size:
            break
        page += 1
    return None


def fetch_task(api_key: str, project_id: int, task_id: int) -> LabelStudioTask:
    """Fetch task - tries task ID first, then searches by data.id."""
    task_data = fetch_task_by_id(api_key, task_id)
    
    if task_data is None:
        print(f"Task ID {task_id} not found directly, searching by data.id...")
        task_data = fetch_task_by_data_id(api_key, project_id, str(task_id))
        if task_data is None:
            raise ValueError(f"Task with ID or data.id={task_id} not found in project {project_id}")
    
    data = task_data.get("data", task_data)
    actual_task_id = task_data.get("id", task_id)
    
    return LabelStudioTask(
        id=actual_task_id,
        character_id=data.get("character_id", ""),
        character_name=data.get("character_name", ""),
        character_image_url=data.get("character_image_url", ""),
        image_url=data.get("image_url", ""),
        init_image_url=data.get("init_image_url", ""),
        mask_image_url=data.get("mask_image_url", ""),
        prompt=data.get("prompt", ""),
    )


def fetch_project_tasks(api_key: str, project_id: int, limit: int = 100) -> List[Dict]:
    url = f"{LABEL_STUDIO_URL}/api/projects/{project_id}/tasks"
    headers = {"Authorization": f"Token {api_key}"}
    params = {"page_size": limit}
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def load_qwen_pipeline(device: str):
    from diffusers import QwenImageEditPipeline
    
    print(f"Loading Qwen-Image-Edit pipeline on {device}...")
    
    # Check for multi-GPU (comma-separated devices)
    if "," in device:
        print(f"Multi-GPU mode detected: using device_map='balanced'")
        pipe = QwenImageEditPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit",
            torch_dtype=torch.bfloat16,
            device_map="balanced",
        )
    else:
        pipe = QwenImageEditPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit",
            torch_dtype=torch.bfloat16,
        )
        pipe.to(device)
    
    pipe.set_progress_bar_config(disable=True)
    return pipe


def composite_character_on_mask(
    init_image: Image.Image,
    character_image: Image.Image, 
    mask_image: Image.Image,
) -> Image.Image:
    """
    Composite character onto init image using mask.
    The mask indicates where the character should be placed.
    """
    init_image = init_image.convert("RGBA")
    character_image = character_image.convert("RGBA")
    mask_image = mask_image.convert("L")
    
    if character_image.size != init_image.size:
        character_image = character_image.resize(init_image.size, Image.Resampling.LANCZOS)
    if mask_image.size != init_image.size:
        mask_image = mask_image.resize(init_image.size, Image.Resampling.LANCZOS)
    
    result = Image.composite(character_image, init_image, mask_image)
    return result.convert("RGB")


def generate_inpaint(
    pipe,
    init_image: Image.Image,
    mask_image: Image.Image,
    character_image: Image.Image,
    prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    device: str,
    use_compositing: bool = True,
    negative_prompt: str = "",
) -> Image.Image:
    """
    Generate inpaint with optional character reference pre-compositing.
    
    Args:
        use_compositing: If True, pre-composite character onto masked region before editing.
                        This helps preserve character appearance for replacement tasks.
        negative_prompt: Custom negative prompt to avoid unwanted features.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Optionally pre-composite character onto masked region as starting point
    if use_compositing and character_image is not None:
        input_image = composite_character_on_mask(init_image, character_image, mask_image)
    else:
        input_image = init_image
    
    result = pipe(
        image=input_image,
        prompt=prompt,
        generator=generator,
        num_inference_steps=steps,
        true_cfg_scale=cfg,
        negative_prompt=negative_prompt,
    )
    return result.images[0]


def run_ttflux_inpaint(
    pipe,
    init_image: Image.Image,
    mask_image: Image.Image,
    character_image: Image.Image,
    prompt: str,
    verifier: LAIONAestheticVerifier,
    num_samples: int,
    steps: int,
    cfg: float,
    device: str,
    output_dir: Path,
    use_compositing: bool = True,
    negative_prompt: str = "",
) -> Tuple[Dict, float]:
    print(f"\nTT-FLUX: Generating {num_samples} random candidates")
    results = []
    
    start_time = time.time()
    for i in tqdm(range(num_samples), desc="TT-FLUX"):
        seed = torch.randint(0, 2**31 - 1, (1,)).item()
        edited = generate_inpaint(pipe, init_image, mask_image, character_image, prompt, seed, steps, cfg, device, use_compositing, negative_prompt)
        
        inputs = verifier.prepare_inputs(images=[edited], prompts=prompt)
        scores = verifier.score(inputs)
        score = scores[0]["laion_aesthetic_score"]
        
        edited.save(output_dir / f"ttflux_{i:02d}_seed{seed}_score{score:.3f}.png")
        results.append({"seed": seed, "image": edited, "score": score})
    
    elapsed = time.time() - start_time
    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]
    best["image"].save(output_dir / "ttflux_best.png")
    
    print(f"TT-FLUX Best: seed={best['seed']}, score={best['score']:.4f}, time={elapsed:.1f}s")
    return {"best": best, "all": results}, elapsed


def run_eacps_inpaint(
    pipe,
    init_image: Image.Image,
    mask_image: Image.Image,
    character_image: Image.Image,
    prompt: str,
    verifier: LAIONAestheticVerifier,
    k_global: int,
    m_global: int,
    k_local: int,
    steps: int,
    cfg: float,
    device: str,
    output_dir: Path,
    use_compositing: bool = True,
    negative_prompt: str = "",
) -> Tuple[Dict, float]:
    print(f"\nEACPS Stage 1: Global exploration ({k_global} candidates)")
    global_results = []
    
    start_time = time.time()
    for i in tqdm(range(k_global), desc="EACPS Global"):
        seed = torch.randint(0, 2**31 - 1, (1,)).item()
        edited = generate_inpaint(pipe, init_image, mask_image, character_image, prompt, seed, steps, cfg, device, use_compositing, negative_prompt)
        
        inputs = verifier.prepare_inputs(images=[edited], prompts=prompt)
        scores = verifier.score(inputs)
        score = scores[0]["laion_aesthetic_score"]
        
        edited.save(output_dir / f"eacps_global_{i:02d}_seed{seed}_score{score:.3f}.png")
        global_results.append({"seed": seed, "image": edited, "score": score})
    
    global_results.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = global_results[:m_global]
    
    print(f"\nEACPS Stage 2: Refining top {m_global} candidates ({k_local} each)")
    all_results = list(global_results)
    
    for rank, candidate in enumerate(top_candidates):
        base_seed = candidate["seed"]
        print(f"  Refining candidate {rank+1}: seed={base_seed}, score={candidate['score']:.4f}")
        
        for j in tqdm(range(k_local), desc=f"Refine #{rank+1}", leave=False):
            refined_seed = base_seed + j + 1
            edited = generate_inpaint(pipe, init_image, mask_image, character_image, prompt, refined_seed, steps, cfg, device, use_compositing, negative_prompt)
            
            inputs = verifier.prepare_inputs(images=[edited], prompts=prompt)
            scores = verifier.score(inputs)
            score = scores[0]["laion_aesthetic_score"]
            
            edited.save(output_dir / f"eacps_refine_{rank}_{j}_seed{refined_seed}_score{score:.3f}.png")
            all_results.append({"seed": refined_seed, "image": edited, "score": score})
    
    elapsed = time.time() - start_time
    all_results.sort(key=lambda x: x["score"], reverse=True)
    best = all_results[0]
    best["image"].save(output_dir / "eacps_best.png")
    
    print(f"EACPS Best: seed={best['seed']}, score={best['score']:.4f}, time={elapsed:.1f}s")
    return {"best": best, "all": all_results}, elapsed


def process_task(
    task: LabelStudioTask,
    pipe,
    verifier: LAIONAestheticVerifier,
    api_key: str,
    output_dir: Path,
    device: str,
    num_samples: int = 8,
    k_global: int = 8,
    m_global: int = 2,
    k_local: int = 4,
    steps: int = 50,
    cfg: float = 4.0,
    use_compositing: bool = True,
    custom_prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,
) -> InpaintResult:
    task_dir = output_dir / f"task_{task.id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print(f"Processing Task {task.id}")
    print("=" * 70)
    print(f"Character: {task.character_name} ({task.character_id})")
    print(f"Prompt: {task.prompt}")
    
    print("\nFetching images...")
    init_image = fetch_image(task.init_image_url, api_key)
    character_image = fetch_image(task.character_image_url, api_key)
    mask_image = fetch_image(task.mask_image_url, api_key)
    
    init_image.save(task_dir / "init_image.png")
    character_image.save(task_dir / "character_image.png")
    mask_image.save(task_dir / "mask_image.png")
    
    # Save composited reference for debugging (if using compositing)
    if use_compositing:
        composited_ref = composite_character_on_mask(init_image, character_image, mask_image)
        composited_ref.save(task_dir / "composited_reference.png")
    
    # Construct edit prompt based on strategy
    if custom_prompt:
        edit_prompt = custom_prompt
    elif use_compositing:
        # When compositing, focus on blending/refinement since character is already placed
        edit_prompt = f"Refine and blend the character naturally into the scene. {task.prompt}"
    else:
        # When not compositing, need more explicit instruction
        edit_prompt = f"Replace the masked region with {task.character_name}. {task.prompt}"
    
    # Set negative prompt
    if negative_prompt is None:
        negative_prompt = "distorted, deformed, wrong anatomy, poor quality, blurry"
    
    with open(task_dir / "task_info.json", "w") as f:
        json.dump(asdict(task), f, indent=2)
    
    start_time = time.time()
    
    ttflux_dir = task_dir / "ttflux"
    ttflux_dir.mkdir(exist_ok=True)
    ttflux_result, ttflux_time = run_ttflux_inpaint(
        pipe=pipe,
        init_image=init_image,
        mask_image=mask_image,
        character_image=character_image,
        prompt=edit_prompt,
        verifier=verifier,
        num_samples=num_samples,
        steps=steps,
        cfg=cfg,
        device=device,
        output_dir=ttflux_dir,
        use_compositing=use_compositing,
        negative_prompt=negative_prompt,
    )
    
    eacps_dir = task_dir / "eacps"
    eacps_dir.mkdir(exist_ok=True)
    eacps_result, eacps_time = run_eacps_inpaint(
        pipe=pipe,
        init_image=init_image,
        mask_image=mask_image,
        character_image=character_image,
        prompt=edit_prompt,
        verifier=verifier,
        k_global=k_global,
        m_global=m_global,
        k_local=k_local,
        steps=steps,
        cfg=cfg,
        device=device,
        output_dir=eacps_dir,
        use_compositing=use_compositing,
        negative_prompt=negative_prompt,
    )
    
    total_time = time.time() - start_time
    
    ttflux_best = ttflux_result["best"]
    eacps_best = eacps_result["best"]
    score_diff = eacps_best["score"] - ttflux_best["score"]
    
    if abs(score_diff) < 0.01:
        winner = "TIE"
    elif score_diff > 0:
        winner = "EACPS"
    else:
        winner = "TTFLUX"
    
    result = InpaintResult(
        task_id=task.id,
        character_id=task.character_id,
        character_name=task.character_name,
        ttflux_best_score=ttflux_best["score"],
        ttflux_best_seed=ttflux_best["seed"],
        eacps_best_score=eacps_best["score"],
        eacps_best_seed=eacps_best["seed"],
        winner=winner,
        score_diff=score_diff,
        time_seconds=total_time,
    )
    
    with open(task_dir / "result.json", "w") as f:
        json.dump(asdict(result), f, indent=2)
    
    print("\n" + "-" * 70)
    print(f"Task {task.id} Results:")
    print(f"  TT-FLUX: score={ttflux_best['score']:.4f}, seed={ttflux_best['seed']}")
    print(f"  EACPS:   score={eacps_best['score']:.4f}, seed={eacps_best['seed']}")
    print(f"  Winner:  {winner} (diff: {score_diff:+.4f})")
    print(f"  Time:    {total_time:.1f}s")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Label Studio Character Inpainting with Inference Scaling")
    parser.add_argument("--api_key", type=str, required=True, help="Label Studio API key")
    parser.add_argument("--project_id", type=int, default=14514, help="Label Studio project ID")
    parser.add_argument("--task_id", type=int, default=None, help="Single task ID to process")
    parser.add_argument("--task_ids", type=int, nargs="+", default=None, help="Multiple task IDs")
    parser.add_argument("--output_dir", type=str, default="experiments/labelstudio_inpaint")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=8, help="TT-FLUX samples")
    parser.add_argument("--k_global", type=int, default=8, help="EACPS global candidates")
    parser.add_argument("--m_global", type=int, default=2, help="EACPS top to refine")
    parser.add_argument("--k_local", type=int, default=4, help="EACPS refinements each")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=4.0, help="true_cfg_scale")
    parser.add_argument("--use_compositing", type=bool, default=True, help="Pre-composite character before editing")
    parser.add_argument("--custom_prompt", type=str, default=None, help="Override prompt construction")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Custom negative prompt")
    parser.add_argument("--list_tasks", action="store_true", help="List available tasks and exit")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.list_tasks:
        print(f"Fetching tasks from project {args.project_id}...")
        tasks = fetch_project_tasks(args.api_key, args.project_id)
        print(f"\nFound {len(tasks)} tasks:")
        for t in tasks[:20]:
            data = t.get("data", {})
            print(f"  ID: {t['id']}, Character: {data.get('character_name', 'N/A')}")
        if len(tasks) > 20:
            print(f"  ... and {len(tasks) - 20} more")
        return
    
    task_ids = []
    if args.task_id:
        task_ids = [args.task_id]
    elif args.task_ids:
        task_ids = args.task_ids
    else:
        print("Error: Specify --task_id or --task_ids, or use --list_tasks")
        return
    
    print("=" * 70)
    print("Label Studio Character Inpainting")
    print("=" * 70)
    print(f"Project: {args.project_id}")
    print(f"Tasks: {task_ids}")
    print(f"Device: {args.device}")
    print(f"TT-FLUX samples: {args.num_samples}")
    print(f"EACPS: k_global={args.k_global}, m_global={args.m_global}, k_local={args.k_local}")
    
    pipe = load_qwen_pipeline(args.device)
    verifier = LAIONAestheticVerifier(device=args.device)
    
    results = []
    for task_id in task_ids:
        task = fetch_task(args.api_key, args.project_id, task_id)
        result = process_task(
            task=task,
            pipe=pipe,
            verifier=verifier,
            api_key=args.api_key,
            output_dir=output_dir,
            device=args.device,
            num_samples=args.num_samples,
            k_global=args.k_global,
            m_global=args.m_global,
            k_local=args.k_local,
            steps=args.steps,
            cfg=args.cfg,
            use_compositing=args.use_compositing,
            custom_prompt=args.custom_prompt,
            negative_prompt=args.negative_prompt,
        )
        results.append(result)
    
    with open(output_dir / "all_results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print("\n" + "=" * 70)
    print("ALL RESULTS SUMMARY")
    print("=" * 70)
    ttflux_wins = sum(1 for r in results if r.winner == "TTFLUX")
    eacps_wins = sum(1 for r in results if r.winner == "EACPS")
    ties = sum(1 for r in results if r.winner == "TIE")
    print(f"TT-FLUX wins: {ttflux_wins}")
    print(f"EACPS wins:   {eacps_wins}")
    print(f"Ties:         {ties}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
