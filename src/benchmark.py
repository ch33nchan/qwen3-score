#!/usr/bin/env python3
"""
Benchmark script for comparing EACPS against TT-FLUX paper results.
Computes the same metrics as the TT-FLUX paper:
- CLIPScore (0-1 normalized)
- Aesthetic Score (1-10)
- ImageReward (human preference)

Usage:
    python benchmark.py --dataset data/drawbench.jsonl --output_dir results/benchmark
"""
import os
import sys

os.environ["DIFFUSERS_USE_FLASH_ATTENTION"] = "0"
os.environ["XFORMERS_DISABLED"] = "1"

try:
    import flash_attn.flash_attn_interface as fai
    if not hasattr(fai, '_wrapped_flash_attn_backward'):
        fai._wrapped_flash_attn_backward = lambda *args, **kwargs: None
    if not hasattr(fai, '_wrapped_flash_attn_forward'):
        fai._wrapped_flash_attn_forward = lambda *args, **kwargs: None
except (ImportError, AttributeError, ValueError):
    import types
    from importlib.machinery import ModuleSpec
    dummy_interface = types.ModuleType('flash_attn.flash_attn_interface')
    dummy_interface._wrapped_flash_attn_backward = lambda *args, **kwargs: None
    dummy_interface._wrapped_flash_attn_forward = lambda *args, **kwargs: None
    dummy_interface.__spec__ = ModuleSpec('flash_attn.flash_attn_interface', None)
    dummy_flash_attn = types.ModuleType('flash_attn')
    dummy_flash_attn.flash_attn_interface = dummy_interface
    dummy_flash_attn.__spec__ = ModuleSpec('flash_attn', None)
    sys.modules['flash_attn'] = dummy_flash_attn
    sys.modules['flash_attn.flash_attn_interface'] = dummy_interface

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np


@dataclass
class BenchmarkResult:
    prompt_id: str
    prompt: str
    method: str
    clip_score: float
    aesthetic_score: float
    imagereward: Optional[float]
    num_candidates: int


@dataclass
class AggregatedMetrics:
    method: str
    n_samples: int
    clip_score_mean: float
    clip_score_std: float
    aesthetic_mean: float
    aesthetic_std: float
    imagereward_mean: Optional[float]
    imagereward_std: Optional[float]


def load_prompts(dataset_path: str) -> List[Dict]:
    prompts = []
    with open(dataset_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def run_random_search(
    prompt: str,
    pipe,
    clip_model,
    clip_processor,
    aesthetic_model,
    imagereward_model,
    num_samples: int = 8,
    device: str = "cuda:0",
) -> Dict:
    """
    TT-FLUX style random search.
    Generate N candidates, pick best by CLIPScore.
    """
    from scorers import compute_clip_score, compute_aesthetic_score, compute_imagereward

    candidates = []
    for i in range(num_samples):
        generator = torch.Generator(device=device if device.startswith("cuda") else "cpu").manual_seed(i)
        with torch.inference_mode():
            out = pipe(
                prompt=prompt,
                generator=generator,
                num_inference_steps=20,
                guidance_scale=3.5,
            )
            candidates.append(out.images[0])

    scores = []
    for img in candidates:
        clip = compute_clip_score(img, prompt, clip_model, clip_processor)
        aesthetic = compute_aesthetic_score(img, clip_model, clip_processor, aesthetic_model)
        ir = compute_imagereward(img, prompt, imagereward_model)
        scores.append({
            "image": img,
            "clip_score": clip,
            "aesthetic_score": aesthetic,
            "imagereward": ir,
        })

    best = max(scores, key=lambda x: x["clip_score"])
    return {
        "best_image": best["image"],
        "clip_score": best["clip_score"],
        "aesthetic_score": best["aesthetic_score"],
        "imagereward": best["imagereward"],
        "all_scores": scores,
    }


def run_eacps_search(
    prompt: str,
    pipe,
    scorer,
    clip_model,
    clip_processor,
    aesthetic_model,
    imagereward_model,
    k_global: int = 8,
    m_global: int = 2,
    k_local: int = 2,
    device: str = "cuda:0",
) -> Dict:
    """
    EACPS multi-stage search.
    """
    from scorers import compute_clip_score, compute_aesthetic_score, compute_imagereward

    global_candidates = []
    for i in range(k_global):
        generator = torch.Generator(device=device if device.startswith("cuda") else "cpu").manual_seed(i)
        with torch.inference_mode():
            out = pipe(
                prompt=prompt,
                generator=generator,
                num_inference_steps=20,
                guidance_scale=3.5,
            )
            global_candidates.append({"seed": i, "image": out.images[0]})

    global_images = [c["image"] for c in global_candidates]
    global_scores = scorer.score_batch(
        images=global_images,
        prompts=[prompt] * len(global_images),
    )

    for cand, sc in zip(global_candidates, global_scores):
        pf = sc.get("PF", 0.0)
        aesthetic = sc.get("aesthetic", 5.0)
        cand["potential"] = pf + aesthetic * 0.1

    global_candidates.sort(key=lambda c: c["potential"], reverse=True)
    top_for_local = global_candidates[:m_global]

    local_candidates = []
    for parent in top_for_local:
        parent_seed = parent["seed"]
        for j in range(k_local):
            local_seed = (parent_seed + 1) * 100000 + j
            generator = torch.Generator(device=device if device.startswith("cuda") else "cpu").manual_seed(local_seed)
            with torch.inference_mode():
                out = pipe(
                    prompt=prompt,
                    generator=generator,
                    num_inference_steps=20,
                    guidance_scale=3.5,
                )
                local_candidates.append({"seed": local_seed, "image": out.images[0], "parent_seed": parent_seed})

    if local_candidates:
        local_images = [c["image"] for c in local_candidates]
        local_scores = scorer.score_batch(
            images=local_images,
            prompts=[prompt] * len(local_images),
        )
        for cand, sc in zip(local_candidates, local_scores):
            pf = sc.get("PF", 0.0)
            aesthetic = sc.get("aesthetic", 5.0)
            cand["potential"] = pf + aesthetic * 0.1

    all_candidates = global_candidates + local_candidates
    best_cand = max(all_candidates, key=lambda c: c["potential"])
    best_image = best_cand["image"]

    clip = compute_clip_score(best_image, prompt, clip_model, clip_processor)
    aesthetic = compute_aesthetic_score(best_image, clip_model, clip_processor, aesthetic_model)
    ir = compute_imagereward(best_image, prompt, imagereward_model)

    return {
        "best_image": best_image,
        "clip_score": clip,
        "aesthetic_score": aesthetic,
        "imagereward": ir,
        "num_candidates": len(all_candidates),
    }


def aggregate_results(results: List[BenchmarkResult], method: str) -> AggregatedMetrics:
    method_results = [r for r in results if r.method == method]
    n = len(method_results)
    if n == 0:
        return None

    clip_scores = [r.clip_score for r in method_results]
    aesthetic_scores = [r.aesthetic_score for r in method_results]
    imagereward_scores = [r.imagereward for r in method_results if r.imagereward is not None]

    return AggregatedMetrics(
        method=method,
        n_samples=n,
        clip_score_mean=float(np.mean(clip_scores)),
        clip_score_std=float(np.std(clip_scores)),
        aesthetic_mean=float(np.mean(aesthetic_scores)),
        aesthetic_std=float(np.std(aesthetic_scores)),
        imagereward_mean=float(np.mean(imagereward_scores)) if imagereward_scores else None,
        imagereward_std=float(np.std(imagereward_scores)) if imagereward_scores else None,
    )


def print_comparison_table(ttflux_agg: AggregatedMetrics, eacps_agg: AggregatedMetrics):
    """Print comparison table matching TT-FLUX paper format."""
    print()
    print("=" * 90)
    print("BENCHMARK RESULTS - Comparison with TT-FLUX Paper")
    print("=" * 90)
    print()
    print("TT-FLUX Paper Reference (FLUX.1-dev on DrawBench):")
    print("  Baseline (no search):  Aesthetic=5.79, CLIPScore=0.71, ImageReward=0.97")
    print("  Random Search (N=8):   Aesthetic≈5.85, CLIPScore≈0.72, ImageReward≈1.05")
    print()
    print("-" * 90)
    print(f"{'Method':<30} | {'CLIPScore':<15} | {'Aesthetic':<15} | {'ImageReward':<15}")
    print("-" * 90)

    if ttflux_agg:
        ir_str = f"{ttflux_agg.imagereward_mean:.4f}" if ttflux_agg.imagereward_mean else "N/A"
        print(f"{'Random Search (Ours)':<30} | {ttflux_agg.clip_score_mean:.4f} ± {ttflux_agg.clip_score_std:.3f}  | {ttflux_agg.aesthetic_mean:.2f} ± {ttflux_agg.aesthetic_std:.2f}    | {ir_str}")

    if eacps_agg:
        ir_str = f"{eacps_agg.imagereward_mean:.4f}" if eacps_agg.imagereward_mean else "N/A"
        print(f"{'EACPS (Ours)':<30} | {eacps_agg.clip_score_mean:.4f} ± {eacps_agg.clip_score_std:.3f}  | {eacps_agg.aesthetic_mean:.2f} ± {eacps_agg.aesthetic_std:.2f}    | {ir_str}")

    print("-" * 90)

    if ttflux_agg and eacps_agg:
        clip_delta = eacps_agg.clip_score_mean - ttflux_agg.clip_score_mean
        aes_delta = eacps_agg.aesthetic_mean - ttflux_agg.aesthetic_mean
        print(f"{'Delta (EACPS - Random)':<30} | {clip_delta:+.4f}          | {aes_delta:+.2f}           |")
        print()
        clip_winner = "EACPS" if clip_delta > 0 else "Random Search"
        aes_winner = "EACPS" if aes_delta > 0 else "Random Search"
        print(f"Winner - CLIPScore: {clip_winner}")
        print(f"Winner - Aesthetic: {aes_winner}")

    print("=" * 90)


def run():
    parser = argparse.ArgumentParser(description="Benchmark EACPS vs TT-FLUX")
    parser.add_argument("--dataset", type=str, default="data/drawbench.jsonl", help="Dataset path")
    parser.add_argument("--output_dir", type=str, default="results/benchmark", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--num_samples", type=int, default=8, help="Candidates for random search")
    parser.add_argument("--k_global", type=int, default=8, help="EACPS k_global")
    parser.add_argument("--m_global", type=int, default=2, help="EACPS m_global")
    parser.add_argument("--k_local", type=int, default=2, help="EACPS k_local")
    parser.add_argument("--max_prompts", type=int, default=None, help="Max prompts to evaluate")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev", help="Model to use")
    parser.add_argument("--skip_random", action="store_true", help="Skip random search baseline")
    parser.add_argument("--skip_eacps", action="store_true", help="Skip EACPS")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BENCHMARK: TT-FLUX Metrics Evaluation")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print()

    if "cuda" in args.device and not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    prompts = load_prompts(args.dataset)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"Loaded {len(prompts)} prompts")

    print("Loading models...")

    if "FLUX" in args.model.upper():
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
        )
        pipe.to(args.device)
    else:
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
        )
        pipe.to(args.device)

    pipe.set_progress_bar_config(disable=True)

    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(args.device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    from scorers import AestheticMLP, EditScorer
    aesthetic_model = None
    try:
        import urllib.request
        cache_dir = os.path.expanduser("~/.cache/aesthetic")
        os.makedirs(cache_dir, exist_ok=True)
        weights_path = os.path.join(cache_dir, "sac+logos+ava1-l14-linearMSE.pth")
        if not os.path.exists(weights_path):
            url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
            print("Downloading aesthetic predictor weights...")
            urllib.request.urlretrieve(url, weights_path)
        aesthetic_model = AestheticMLP(input_size=768)
        state_dict = torch.load(weights_path, map_location="cpu")
        aesthetic_model.load_state_dict(state_dict)
        aesthetic_model.to(args.device)
        aesthetic_model.eval()
        print("Aesthetic predictor loaded.")
    except Exception as e:
        print(f"Warning: Could not load aesthetic predictor: {e}")

    imagereward_model = None
    try:
        import ImageReward as RM
        imagereward_model = RM.load("ImageReward-v1.0", device=args.device)
        print("ImageReward loaded.")
    except ImportError:
        print("Warning: ImageReward not installed. Run: pip install image-reward")
    except Exception as e:
        print(f"Warning: Could not load ImageReward: {e}")

    scorer = EditScorer(
        device=args.device,
        use_lpips=False,
        use_aesthetic=True,
        use_imagereward=False,
    )

    print("Models loaded.")
    print()

    results: List[BenchmarkResult] = []

    for item in tqdm(prompts, desc="Evaluating"):
        prompt_id = item["id"]
        prompt = item["prompt"]

        if not args.skip_random:
            try:
                random_result = run_random_search(
                    prompt=prompt,
                    pipe=pipe,
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    aesthetic_model=aesthetic_model,
                    imagereward_model=imagereward_model,
                    num_samples=args.num_samples,
                    device=args.device,
                )
                results.append(BenchmarkResult(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    method="random_search",
                    clip_score=random_result["clip_score"],
                    aesthetic_score=random_result["aesthetic_score"],
                    imagereward=random_result["imagereward"],
                    num_candidates=args.num_samples,
                ))
                random_result["best_image"].save(output_dir / f"{prompt_id}_random.png")
            except Exception as e:
                print(f"Error on {prompt_id} (random): {e}")

        if not args.skip_eacps:
            try:
                eacps_result = run_eacps_search(
                    prompt=prompt,
                    pipe=pipe,
                    scorer=scorer,
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    aesthetic_model=aesthetic_model,
                    imagereward_model=imagereward_model,
                    k_global=args.k_global,
                    m_global=args.m_global,
                    k_local=args.k_local,
                    device=args.device,
                )
                results.append(BenchmarkResult(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    method="eacps",
                    clip_score=eacps_result["clip_score"],
                    aesthetic_score=eacps_result["aesthetic_score"],
                    imagereward=eacps_result["imagereward"],
                    num_candidates=eacps_result["num_candidates"],
                ))
                eacps_result["best_image"].save(output_dir / f"{prompt_id}_eacps.png")
            except Exception as e:
                print(f"Error on {prompt_id} (eacps): {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_path = output_dir / "benchmark_results.json"
    with results_path.open("w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    ttflux_agg = aggregate_results(results, "random_search")
    eacps_agg = aggregate_results(results, "eacps")

    print_comparison_table(ttflux_agg, eacps_agg)

    agg_path = output_dir / "aggregated_metrics.json"
    with agg_path.open("w") as f:
        agg_data = {}
        if ttflux_agg:
            agg_data["random_search"] = asdict(ttflux_agg)
        if eacps_agg:
            agg_data["eacps"] = asdict(eacps_agg)
        json.dump(agg_data, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    run()

