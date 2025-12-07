#!/usr/bin/env python3
"""
EACPS vs TT-FLUX Comparison Script

Compares EACPS (Efficient Adaptive Candidate-based Prompt Search) against
the actual TT-FLUX implementation for image generation/editing tasks.

Usage:
    python src/run_comparison.py \
        --prompt "a tiny astronaut hatching from an egg on the moon" \
        --output_dir experiments/results/comparison \
        --device cuda:0 \
        --search_rounds 4

For image editing with Qwen:
    python src/run_comparison.py \
        --image data/bear.png \
        --prompt "Add a colorful art board and paintbrush in the bear's hands" \
        --output_dir experiments/results/comparison \
        --device cuda:0 \
        --mode edit
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

os.environ["DIFFUSERS_USE_FLASH_ATTENTION"] = "0"
os.environ["XFORMERS_DISABLED"] = "1"

import torch
from PIL import Image
from tqdm import tqdm

from ttflux import TTFluxPipeline, LAIONAestheticVerifier
from ttflux.pipeline import TTFluxConfig, SearchResult


@dataclass
class ComparisonResult:
    prompt: str
    mode: str
    ttflux_best_score: float
    ttflux_best_seed: int
    ttflux_total_samples: int
    eacps_best_score: float
    eacps_best_seed: int
    eacps_total_samples: int
    winner: str
    score_diff: float


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def run_ttflux(
    pipe,
    prompt: str,
    config: TTFluxConfig,
    device: str,
    verbose: bool = True,
) -> SearchResult:
    verifier = LAIONAestheticVerifier(device=device)
    ttflux = TTFluxPipeline(pipe=pipe, config=config, verifier=verifier, device=device)
    return ttflux(prompt, verbose=verbose)


def run_eacps_generation(
    pipe,
    prompt: str,
    k_global: int,
    m_global: int,
    k_local: int,
    steps: int,
    cfg: float,
    device: str,
    verifier,
) -> Dict:
    from ttflux.pipeline import get_noises
    
    config = TTFluxConfig(
        height=1024,
        width=1024,
        num_inference_steps=steps,
        guidance_scale=cfg,
    )
    
    print(f"EACPS Stage 1: Global exploration with {k_global} candidates")
    noises = get_noises(
        num_samples=k_global,
        height=config.height,
        width=config.width,
        device=device,
        dtype=torch.bfloat16,
    )
    
    global_results = []
    for seed, latent in tqdm(noises.items(), desc="Global candidates"):
        result = pipe(
            prompt=prompt,
            latents=latent,
            height=config.height,
            width=config.width,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
        )
        image = result.images[0]
        inputs = verifier.prepare_inputs(images=[image], prompts=prompt)
        scores = verifier.score(inputs)
        score = scores[0]["laion_aesthetic_score"]
        global_results.append({
            "seed": seed,
            "image": image,
            "score": score,
            "latent": latent,
        })
    
    global_results.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = global_results[:m_global]
    print(f"EACPS Stage 2: Refining top {m_global} candidates with {k_local} variations each")
    
    all_results = list(global_results)
    for candidate in tqdm(top_candidates, desc="Refinement"):
        base_latent = candidate["latent"]
        for _ in range(k_local):
            noise = torch.randn_like(base_latent) * 0.1
            refined_latent = base_latent + noise
            result = pipe(
                prompt=prompt,
                latents=refined_latent,
                height=config.height,
                width=config.width,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
            )
            image = result.images[0]
            inputs = verifier.prepare_inputs(images=[image], prompts=prompt)
            scores = verifier.score(inputs)
            score = scores[0]["laion_aesthetic_score"]
            all_results.append({
                "seed": candidate["seed"],
                "image": image,
                "score": score,
                "latent": refined_latent,
            })
    
    all_results.sort(key=lambda x: x["score"], reverse=True)
    best = all_results[0]
    
    return {
        "best_image": best["image"],
        "best_score": best["score"],
        "best_seed": best["seed"],
        "total_samples": len(all_results),
        "all_results": all_results,
    }


def run_comparison_generation(
    prompt: str,
    output_dir: Path,
    device: str,
    search_rounds: int = 4,
    k_global: int = 8,
    m_global: int = 2,
    k_local: int = 4,
    steps: int = 50,
    cfg: float = 3.5,
    model_id: str = "black-forest-labs/FLUX.1-dev",
) -> ComparisonResult:
    from diffusers import FluxPipeline
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading FLUX pipeline: {model_id}")
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    verifier = LAIONAestheticVerifier(device=device)
    
    print("\n" + "=" * 60)
    print("Running TT-FLUX (Random Search)")
    print("=" * 60)
    
    ttflux_config = TTFluxConfig(
        search_rounds=search_rounds,
        search_method="random",
        verifier_name="laion_aesthetic",
        choice_of_metric="laion_aesthetic_score",
        batch_size=1,
        height=1024,
        width=1024,
        num_inference_steps=steps,
        guidance_scale=cfg,
    )
    
    ttflux_result = run_ttflux(pipe, prompt, ttflux_config, device)
    ttflux_result.best_image.save(output_dir / "ttflux_best.png")
    
    ttflux_total = sum(2**i for i in range(1, search_rounds + 1))
    
    print("\n" + "=" * 60)
    print("Running EACPS (Adaptive Candidate Search)")
    print("=" * 60)
    
    eacps_result = run_eacps_generation(
        pipe=pipe,
        prompt=prompt,
        k_global=k_global,
        m_global=m_global,
        k_local=k_local,
        steps=steps,
        cfg=cfg,
        device=device,
        verifier=verifier,
    )
    eacps_result["best_image"].save(output_dir / "eacps_best.png")
    
    winner = "TTFLUX" if ttflux_result.best_score > eacps_result["best_score"] else "EACPS"
    if abs(ttflux_result.best_score - eacps_result["best_score"]) < 0.01:
        winner = "TIE"
    
    comparison = ComparisonResult(
        prompt=prompt,
        mode="generation",
        ttflux_best_score=ttflux_result.best_score,
        ttflux_best_seed=ttflux_result.best_seed,
        ttflux_total_samples=ttflux_total,
        eacps_best_score=eacps_result["best_score"],
        eacps_best_seed=eacps_result["best_seed"],
        eacps_total_samples=eacps_result["total_samples"],
        winner=winner,
        score_diff=eacps_result["best_score"] - ttflux_result.best_score,
    )
    
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(asdict(comparison), f, indent=2)
    
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print(f"\nTT-FLUX:")
    print(f"  Best Score: {ttflux_result.best_score:.4f}")
    print(f"  Best Seed: {ttflux_result.best_seed}")
    print(f"  Total Samples: {ttflux_total}")
    print(f"\nEACPS:")
    print(f"  Best Score: {eacps_result['best_score']:.4f}")
    print(f"  Best Seed: {eacps_result['best_seed']}")
    print(f"  Total Samples: {eacps_result['total_samples']}")
    print(f"\nWinner: {winner} (diff: {comparison.score_diff:+.4f})")
    
    return comparison


def run_comparison_edit(
    image_path: str,
    prompt: str,
    output_dir: Path,
    device: str,
    num_samples: int = 8,
    k_global: int = 8,
    m_global: int = 2,
    k_local: int = 4,
    steps: int = 50,
    cfg: float = 5.0,
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
) -> ComparisonResult:
    from diffusers import QwenImageEditPipeline
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    original_image = load_image(image_path)
    original_image.save(output_dir / "original.png")
    
    print(f"Loading Qwen Edit pipeline: {model_id}")
    pipe = QwenImageEditPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(device)
    
    verifier = LAIONAestheticVerifier(device=device)
    
    print("\n" + "=" * 60)
    print("Running TT-FLUX Style (Random Search for Editing)")
    print("=" * 60)
    
    ttflux_results = []
    for i in tqdm(range(num_samples), desc="TT-FLUX candidates"):
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(seed)
        result = pipe(
            image=original_image,
            prompt=prompt,
            generator=generator,
            num_inference_steps=steps,
            true_cfg_scale=cfg,
        )
        image = result.images[0]
        inputs = verifier.prepare_inputs(images=[image], prompts=prompt)
        scores = verifier.score(inputs)
        score = scores[0]["laion_aesthetic_score"]
        ttflux_results.append({"seed": seed, "image": image, "score": score})
    
    ttflux_results.sort(key=lambda x: x["score"], reverse=True)
    ttflux_best = ttflux_results[0]
    ttflux_best["image"].save(output_dir / "ttflux_best.png")
    
    print("\n" + "=" * 60)
    print("Running EACPS (Adaptive Candidate Search for Editing)")
    print("=" * 60)
    
    print(f"Stage 1: Global exploration with {k_global} candidates")
    global_results = []
    for i in tqdm(range(k_global), desc="Global candidates"):
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(seed)
        result = pipe(
            image=original_image,
            prompt=prompt,
            generator=generator,
            num_inference_steps=steps,
            true_cfg_scale=cfg,
        )
        image = result.images[0]
        inputs = verifier.prepare_inputs(images=[image], prompts=prompt)
        scores = verifier.score(inputs)
        score = scores[0]["laion_aesthetic_score"]
        global_results.append({"seed": seed, "image": image, "score": score})
    
    global_results.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = global_results[:m_global]
    
    print(f"Stage 2: Refining top {m_global} with {k_local} variations each")
    all_eacps_results = list(global_results)
    for candidate in tqdm(top_candidates, desc="Refinement"):
        base_seed = candidate["seed"]
        for j in range(k_local):
            refined_seed = base_seed + j + 1
            generator = torch.Generator(device=device).manual_seed(refined_seed)
            result = pipe(
                image=original_image,
                prompt=prompt,
                generator=generator,
                num_inference_steps=steps,
                true_cfg_scale=cfg,
            )
            image = result.images[0]
            inputs = verifier.prepare_inputs(images=[image], prompts=prompt)
            scores = verifier.score(inputs)
            score = scores[0]["laion_aesthetic_score"]
            all_eacps_results.append({"seed": refined_seed, "image": image, "score": score})
    
    all_eacps_results.sort(key=lambda x: x["score"], reverse=True)
    eacps_best = all_eacps_results[0]
    eacps_best["image"].save(output_dir / "eacps_best.png")
    
    winner = "TTFLUX" if ttflux_best["score"] > eacps_best["score"] else "EACPS"
    if abs(ttflux_best["score"] - eacps_best["score"]) < 0.01:
        winner = "TIE"
    
    comparison = ComparisonResult(
        prompt=prompt,
        mode="edit",
        ttflux_best_score=ttflux_best["score"],
        ttflux_best_seed=ttflux_best["seed"],
        ttflux_total_samples=num_samples,
        eacps_best_score=eacps_best["score"],
        eacps_best_seed=eacps_best["seed"],
        eacps_total_samples=len(all_eacps_results),
        winner=winner,
        score_diff=eacps_best["score"] - ttflux_best["score"],
    )
    
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(asdict(comparison), f, indent=2)
    
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print(f"\nTT-FLUX:")
    print(f"  Best Score: {ttflux_best['score']:.4f}")
    print(f"  Best Seed: {ttflux_best['seed']}")
    print(f"  Total Samples: {num_samples}")
    print(f"\nEACPS:")
    print(f"  Best Score: {eacps_best['score']:.4f}")
    print(f"  Best Seed: {eacps_best['seed']}")
    print(f"  Total Samples: {len(all_eacps_results)}")
    print(f"\nWinner: {winner} (diff: {comparison.score_diff:+.4f})")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="EACPS vs TT-FLUX Comparison")
    parser.add_argument("--prompt", type=str, required=True, help="Generation/edit prompt")
    parser.add_argument("--image", type=str, default=None, help="Input image for editing mode")
    parser.add_argument("--output_dir", type=str, default="experiments/results/comparison")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", type=str, choices=["generate", "edit"], default=None,
                        help="Mode: generate (FLUX) or edit (Qwen). Auto-detected if --image provided")
    parser.add_argument("--search_rounds", type=int, default=4, help="TT-FLUX search rounds")
    parser.add_argument("--num_samples", type=int, default=8, help="TT-FLUX samples for edit mode")
    parser.add_argument("--k_global", type=int, default=8, help="EACPS global candidates")
    parser.add_argument("--m_global", type=int, default=2, help="EACPS top candidates to refine")
    parser.add_argument("--k_local", type=int, default=4, help="EACPS refinements per candidate")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=3.5, help="Guidance scale (3.5 for FLUX, 5.0 for Qwen)")
    parser.add_argument("--model_id", type=str, default=None, help="Model ID override")
    args = parser.parse_args()
    
    mode = args.mode
    if mode is None:
        mode = "edit" if args.image else "generate"
    
    output_dir = Path(args.output_dir)
    
    if mode == "generate":
        cfg = args.cfg if args.cfg != 3.5 or args.model_id else 3.5
        model_id = args.model_id or "black-forest-labs/FLUX.1-dev"
        run_comparison_generation(
            prompt=args.prompt,
            output_dir=output_dir,
            device=args.device,
            search_rounds=args.search_rounds,
            k_global=args.k_global,
            m_global=args.m_global,
            k_local=args.k_local,
            steps=args.steps,
            cfg=cfg,
            model_id=model_id,
        )
    else:
        if not args.image:
            raise ValueError("--image required for edit mode")
        cfg = args.cfg if args.cfg != 3.5 else 5.0
        model_id = args.model_id or "Qwen/Qwen2.5-VL-7B-Instruct"
        run_comparison_edit(
            image_path=args.image,
            prompt=args.prompt,
            output_dir=output_dir,
            device=args.device,
            num_samples=args.num_samples,
            k_global=args.k_global,
            m_global=args.m_global,
            k_local=args.k_local,
            steps=args.steps,
            cfg=cfg,
            model_id=model_id,
        )


if __name__ == "__main__":
    main()
