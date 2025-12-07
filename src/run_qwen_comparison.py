#!/usr/bin/env python3
"""
EACPS vs TT-FLUX Comparison for Qwen-Image-Edit

Compares TT-FLUX random search against EACPS adaptive search
using Qwen-Image-Edit for image editing tasks.

Optimized for multi-GPU (2-3 H100s).

Usage:
    python src/run_qwen_comparison.py \
        --image data/bear.png \
        --prompt "Add a colorful art board and paintbrush in the bear's hands" \
        --output_dir experiments/results/qwen_comparison \
        --device cuda:0
"""
import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import time

os.environ["DIFFUSERS_USE_FLASH_ATTENTION"] = "0"
os.environ["XFORMERS_DISABLED"] = "1"

import torch
from PIL import Image
from tqdm import tqdm

from ttflux.verifiers import LAIONAestheticVerifier


@dataclass
class ComparisonResult:
    prompt: str
    image_path: str
    ttflux_best_score: float
    ttflux_best_seed: int
    ttflux_total_samples: int
    ttflux_time_seconds: float
    eacps_best_score: float
    eacps_best_seed: int
    eacps_total_samples: int
    eacps_time_seconds: float
    winner: str
    score_diff: float


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_qwen_pipeline(device: str):
    from diffusers import QwenImageEditPipeline
    
    print(f"Loading Qwen-Image-Edit pipeline on {device}...")
    pipe = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        torch_dtype=torch.bfloat16,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_edit(
    pipe, 
    image: Image.Image, 
    prompt: str, 
    seed: int, 
    steps: int, 
    cfg: float, 
    device: str
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


def run_ttflux_search(
    pipe,
    image: Image.Image,
    prompt: str,
    verifier: LAIONAestheticVerifier,
    num_samples: int,
    steps: int,
    cfg: float,
    device: str,
    output_dir: Path,
) -> Tuple[Dict, float]:
    print(f"\nTT-FLUX: Generating {num_samples} random candidates")
    results = []
    
    start_time = time.time()
    for i in tqdm(range(num_samples), desc="TT-FLUX"):
        seed = torch.randint(0, 2**31 - 1, (1,)).item()
        edited = generate_edit(pipe, image, prompt, seed, steps, cfg, device)
        
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


def run_eacps_search(
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
) -> Tuple[Dict, float]:
    print(f"\nEACPS Stage 1: Global exploration ({k_global} candidates)")
    global_results = []
    
    start_time = time.time()
    for i in tqdm(range(k_global), desc="EACPS Global"):
        seed = torch.randint(0, 2**31 - 1, (1,)).item()
        edited = generate_edit(pipe, image, prompt, seed, steps, cfg, device)
        
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
            edited = generate_edit(pipe, image, prompt, refined_seed, steps, cfg, device)
            
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


def main():
    parser = argparse.ArgumentParser(description="EACPS vs TT-FLUX for Qwen-Image-Edit")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--prompt", type=str, required=True, help="Edit prompt")
    parser.add_argument("--output_dir", type=str, default="experiments/results/qwen_comparison")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=16, help="TT-FLUX total samples")
    parser.add_argument("--k_global", type=int, default=8, help="EACPS global candidates")
    parser.add_argument("--m_global", type=int, default=2, help="EACPS top candidates to refine")
    parser.add_argument("--k_local", type=int, default=4, help="EACPS refinements per candidate")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=4.0, help="true_cfg_scale for Qwen-Image-Edit")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    original = load_image(args.image)
    original.save(output_dir / "original.png")
    
    print("=" * 70)
    print("EACPS vs TT-FLUX Comparison (Qwen-Image-Edit)")
    print("=" * 70)
    print(f"Image: {args.image}")
    print(f"Prompt: {args.prompt}")
    print(f"Device: {args.device}")
    print(f"TT-FLUX samples: {args.num_samples}")
    print(f"EACPS: k_global={args.k_global}, m_global={args.m_global}, k_local={args.k_local}")
    print(f"Total EACPS samples: {args.k_global + args.m_global * args.k_local}")
    print()
    
    pipe = load_qwen_pipeline(args.device)
    verifier = LAIONAestheticVerifier(device=args.device)
    
    ttflux_dir = output_dir / "ttflux"
    ttflux_dir.mkdir(exist_ok=True)
    ttflux_result, ttflux_time = run_ttflux_search(
        pipe=pipe,
        image=original,
        prompt=args.prompt,
        verifier=verifier,
        num_samples=args.num_samples,
        steps=args.steps,
        cfg=args.cfg,
        device=args.device,
        output_dir=ttflux_dir,
    )
    
    eacps_dir = output_dir / "eacps"
    eacps_dir.mkdir(exist_ok=True)
    eacps_result, eacps_time = run_eacps_search(
        pipe=pipe,
        image=original,
        prompt=args.prompt,
        verifier=verifier,
        k_global=args.k_global,
        m_global=args.m_global,
        k_local=args.k_local,
        steps=args.steps,
        cfg=args.cfg,
        device=args.device,
        output_dir=eacps_dir,
    )
    
    ttflux_best = ttflux_result["best"]
    eacps_best = eacps_result["best"]
    
    score_diff = eacps_best["score"] - ttflux_best["score"]
    if abs(score_diff) < 0.01:
        winner = "TIE"
    elif score_diff > 0:
        winner = "EACPS"
    else:
        winner = "TTFLUX"
    
    comparison = ComparisonResult(
        prompt=args.prompt,
        image_path=args.image,
        ttflux_best_score=ttflux_best["score"],
        ttflux_best_seed=ttflux_best["seed"],
        ttflux_total_samples=args.num_samples,
        ttflux_time_seconds=ttflux_time,
        eacps_best_score=eacps_best["score"],
        eacps_best_seed=eacps_best["seed"],
        eacps_total_samples=len(eacps_result["all"]),
        eacps_time_seconds=eacps_time,
        winner=winner,
        score_diff=score_diff,
    )
    
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(asdict(comparison), f, indent=2)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nTT-FLUX ({args.num_samples} samples, {ttflux_time:.1f}s):")
    print(f"  Best Score: {ttflux_best['score']:.4f}")
    print(f"  Best Seed:  {ttflux_best['seed']}")
    
    print(f"\nEACPS ({len(eacps_result['all'])} samples, {eacps_time:.1f}s):")
    print(f"  Best Score: {eacps_best['score']:.4f}")
    print(f"  Best Seed:  {eacps_best['seed']}")
    
    print(f"\nWINNER: {winner}")
    print(f"Score Difference: {score_diff:+.4f}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
