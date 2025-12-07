#!/usr/bin/env python3
"""
Character Inpainting with Inference Scaling

Takes task data with:
- init_image_url: background/scene image  
- character_image_url: reference character image
- mask_image_url: where to place the character
- prompt: instruction

Uses Qwen-Image-Edit with TT-FLUX/EACPS inference scaling.

Usage:
    python src/character_inpaint.py \
        --task_json '{"id": "71671165", "character_image_url": "...", ...}' \
        --output_dir experiments/character_test \
        --device cuda:0
"""
import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple
import time
import requests
from io import BytesIO

os.environ["DIFFUSERS_USE_FLASH_ATTENTION"] = "0"
os.environ["XFORMERS_DISABLED"] = "1"

import torch
from PIL import Image
from tqdm import tqdm

from ttflux.verifiers import LAIONAestheticVerifier


@dataclass
class TaskData:
    id: str
    character_id: str
    character_name: str
    character_image_url: str
    image_url: str
    init_image_url: str
    mask_image_url: str
    prompt: str


@dataclass 
class InpaintResult:
    task_id: str
    character_id: str
    character_name: str
    ttflux_best_score: float
    ttflux_best_seed: int
    ttflux_samples: int
    eacps_best_score: float
    eacps_best_seed: int
    eacps_samples: int
    winner: str
    score_diff: float
    total_time_seconds: float


def fetch_image(url: str) -> Image.Image:
    print(f"  Fetching: {url[:80]}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def fetch_mask(url: str) -> Image.Image:
    print(f"  Fetching mask: {url[:80]}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("L")


def load_qwen_pipeline(device: str):
    from diffusers import QwenImageEditPipeline
    
    print(f"Loading Qwen-Image-Edit on {device}...")
    pipe = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        torch_dtype=torch.bfloat16,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_inpaint(
    pipe,
    image: Image.Image,
    prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    device: str,
) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        image=image,
        prompt=prompt,
        generator=generator,
        num_inference_steps=steps,
        true_cfg_scale=cfg,
        negative_prompt=" ",
    )
    return result.images[0]


def run_ttflux(
    pipe,
    image: Image.Image,
    prompt: str,
    verifier: LAIONAestheticVerifier,
    num_samples: int,
    steps: int,
    cfg: float,
    device: str,
    output_dir: Path,
) -> Tuple[dict, float]:
    print(f"\nTT-FLUX: {num_samples} random candidates")
    results = []
    
    start = time.time()
    for i in tqdm(range(num_samples), desc="TT-FLUX"):
        seed = torch.randint(0, 2**31 - 1, (1,)).item()
        edited = generate_inpaint(pipe, image, prompt, seed, steps, cfg, device)
        
        inputs = verifier.prepare_inputs(images=[edited], prompts=prompt)
        scores = verifier.score(inputs)
        score = scores[0]["laion_aesthetic_score"]
        
        edited.save(output_dir / f"ttflux_{i:02d}_seed{seed}_score{score:.3f}.png")
        results.append({"seed": seed, "image": edited, "score": score})
    
    elapsed = time.time() - start
    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]
    best["image"].save(output_dir / "ttflux_best.png")
    
    print(f"TT-FLUX Best: seed={best['seed']}, score={best['score']:.4f}")
    return {"best": best, "all": results}, elapsed


def run_eacps(
    pipe,
    image: Image.Image,
    prompt: str,
    verifier: LAIONAestheticVerifier,
    k_global: int,
    m_global: int,
    k_local: int,
    steps: int,
    cfg: float,
    device: str,
    output_dir: Path,
) -> Tuple[dict, float]:
    print(f"\nEACPS Stage 1: {k_global} global candidates")
    global_results = []
    
    start = time.time()
    for i in tqdm(range(k_global), desc="EACPS Global"):
        seed = torch.randint(0, 2**31 - 1, (1,)).item()
        edited = generate_inpaint(pipe, image, prompt, seed, steps, cfg, device)
        
        inputs = verifier.prepare_inputs(images=[edited], prompts=prompt)
        scores = verifier.score(inputs)
        score = scores[0]["laion_aesthetic_score"]
        
        edited.save(output_dir / f"eacps_global_{i:02d}_seed{seed}_score{score:.3f}.png")
        global_results.append({"seed": seed, "image": edited, "score": score})
    
    global_results.sort(key=lambda x: x["score"], reverse=True)
    top = global_results[:m_global]
    
    print(f"\nEACPS Stage 2: Refining top {m_global} ({k_local} each)")
    all_results = list(global_results)
    
    for rank, cand in enumerate(top):
        base_seed = cand["seed"]
        for j in tqdm(range(k_local), desc=f"Refine #{rank+1}", leave=False):
            refined_seed = base_seed + j + 1
            edited = generate_inpaint(pipe, image, prompt, refined_seed, steps, cfg, device)
            
            inputs = verifier.prepare_inputs(images=[edited], prompts=prompt)
            scores = verifier.score(inputs)
            score = scores[0]["laion_aesthetic_score"]
            
            edited.save(output_dir / f"eacps_refine_{rank}_{j}_seed{refined_seed}_score{score:.3f}.png")
            all_results.append({"seed": refined_seed, "image": edited, "score": score})
    
    elapsed = time.time() - start
    all_results.sort(key=lambda x: x["score"], reverse=True)
    best = all_results[0]
    best["image"].save(output_dir / "eacps_best.png")
    
    print(f"EACPS Best: seed={best['seed']}, score={best['score']:.4f}")
    return {"best": best, "all": all_results}, elapsed


def main():
    parser = argparse.ArgumentParser(description="Character Inpainting with Inference Scaling")
    parser.add_argument("--task_json", type=str, default=None, help="Task data as JSON string")
    parser.add_argument("--task_file", type=str, default=None, help="Task data from JSON file")
    parser.add_argument("--output_dir", type=str, default="experiments/character_inpaint")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=8, help="TT-FLUX samples")
    parser.add_argument("--k_global", type=int, default=8, help="EACPS global")
    parser.add_argument("--m_global", type=int, default=2, help="EACPS top to refine")
    parser.add_argument("--k_local", type=int, default=4, help="EACPS refinements")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=4.0, help="true_cfg_scale")
    args = parser.parse_args()
    
    if args.task_json:
        task_data = json.loads(args.task_json)
    elif args.task_file:
        with open(args.task_file) as f:
            task_data = json.load(f)
    else:
        task_data = {
            "id": "71671165",
            "character_id": "CHA54a8fd5c3ff74200",
            "character_name": "Hrivaan rameshvar",
            "character_image_url": "https://dev-content.dashtoon.ai/user-uploaded-files/303b8aa374804d4ca8ecf18acabb82d0.jpeg",
            "image_url": "https://content.dashtoon.ai/stability-images/e7eccdeb-7f66-4aec-a5e5-2c8145a4406f.png",
            "init_image_url": "https://content.dashtoon.ai/user-uploaded-images/e70b10a2-4266-49a2-b2b3-35f582572098.webp",
            "mask_image_url": "https://content.dashtoon.ai/user-uploaded-images/806b2e94-c53f-4b6d-bffa-161d2c63c41a.png",
            "prompt": "check if there is character consistency in the image pairs"
        }
    
    if "data" in task_data:
        task_data = task_data["data"]
    
    task = TaskData(**task_data)
    
    output_dir = Path(args.output_dir) / f"task_{task.id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Character Inpainting with Inference Scaling")
    print("=" * 70)
    print(f"Task ID: {task.id}")
    print(f"Character: {task.character_name}")
    print(f"Device: {args.device}")
    print(f"TT-FLUX: {args.num_samples} samples")
    print(f"EACPS: {args.k_global} global + {args.m_global}x{args.k_local} refine")
    print()
    
    print("Fetching images...")
    init_image = fetch_image(task.init_image_url)
    character_image = fetch_image(task.character_image_url)
    mask_image = fetch_mask(task.mask_image_url)
    
    if task.image_url:
        reference_image = fetch_image(task.image_url)
        reference_image.save(output_dir / "reference_result.png")
    
    init_image.save(output_dir / "init_image.png")
    character_image.save(output_dir / "character_image.png")
    mask_image.save(output_dir / "mask_image.png")
    
    with open(output_dir / "task_data.json", "w") as f:
        json.dump(asdict(task), f, indent=2)
    
    edit_prompt = f"Replace the masked area with the character {task.character_name}. Maintain character consistency, same face, same appearance. High quality, detailed."
    
    print(f"\nEdit prompt: {edit_prompt}")
    
    pipe = load_qwen_pipeline(args.device)
    verifier = LAIONAestheticVerifier(device=args.device)
    
    total_start = time.time()
    
    ttflux_dir = output_dir / "ttflux"
    ttflux_dir.mkdir(exist_ok=True)
    ttflux_result, ttflux_time = run_ttflux(
        pipe=pipe,
        image=init_image,
        prompt=edit_prompt,
        verifier=verifier,
        num_samples=args.num_samples,
        steps=args.steps,
        cfg=args.cfg,
        device=args.device,
        output_dir=ttflux_dir,
    )
    
    eacps_dir = output_dir / "eacps"
    eacps_dir.mkdir(exist_ok=True)
    eacps_result, eacps_time = run_eacps(
        pipe=pipe,
        image=init_image,
        prompt=edit_prompt,
        verifier=verifier,
        k_global=args.k_global,
        m_global=args.m_global,
        k_local=args.k_local,
        steps=args.steps,
        cfg=args.cfg,
        device=args.device,
        output_dir=eacps_dir,
    )
    
    total_time = time.time() - total_start
    
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
        ttflux_samples=args.num_samples,
        eacps_best_score=eacps_best["score"],
        eacps_best_seed=eacps_best["seed"],
        eacps_samples=len(eacps_result["all"]),
        winner=winner,
        score_diff=score_diff,
        total_time_seconds=total_time,
    )
    
    with open(output_dir / "result.json", "w") as f:
        json.dump(asdict(result), f, indent=2)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Task: {task.id} ({task.character_name})")
    print(f"\nTT-FLUX ({args.num_samples} samples, {ttflux_time:.1f}s):")
    print(f"  Best Score: {ttflux_best['score']:.4f}")
    print(f"  Best Seed:  {ttflux_best['seed']}")
    print(f"\nEACPS ({len(eacps_result['all'])} samples, {eacps_time:.1f}s):")
    print(f"  Best Score: {eacps_best['score']:.4f}")
    print(f"  Best Seed:  {eacps_best['seed']}")
    print(f"\nWINNER: {winner} (diff: {score_diff:+.4f})")
    print(f"Total time: {total_time:.1f}s")
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
