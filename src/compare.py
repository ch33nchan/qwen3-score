#!/usr/bin/env python3
"""
Head-to-head comparison: TT-FLUX vs EACPS (Qwen-Image-Edit)
Compares using the same metrics as TT-FLUX paper:
- CLIPScore (text-image alignment, 0-1 normalized)
- Aesthetic Score (LAION aesthetic predictor, 1-10)
- ImageReward (human preference aligned)
- LPIPS (perceptual distance)
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
from typing import Dict, List, Tuple, Optional
import torch
from PIL import Image
from tqdm import tqdm
import urllib.request


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_aesthetic_model(device: str):
    """Load LAION aesthetic predictor (same as TT-FLUX paper)."""
    from scorers import AestheticMLP
    
    cache_dir = os.path.expanduser("~/.cache/aesthetic")
    os.makedirs(cache_dir, exist_ok=True)
    weights_path = os.path.join(cache_dir, "sac+logos+ava1-l14-linearMSE.pth")
    
    if not os.path.exists(weights_path):
        url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
        print("Downloading aesthetic predictor weights...")
        urllib.request.urlretrieve(url, weights_path)
    
    model = AestheticMLP(input_size=768)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_imagereward_model(device: str):
    """Load ImageReward model (human preference aligned)."""
    try:
        import ImageReward as RM
        return RM.load("ImageReward-v1.0", device=device)
    except ImportError:
        print("ImageReward not installed. Run: pip install image-reward")
        return None
    except Exception as e:
        print(f"Could not load ImageReward: {e}")
        return None


def compute_metrics(
    image: Image.Image,
    prompt: str,
    original: Optional[Image.Image],
    clip_model,
    clip_processor,
    aesthetic_model,
    imagereward_model,
    lpips_model,
    device: str,
) -> Dict[str, float]:
    """Compute all TT-FLUX paper metrics for an image."""
    from scorers import compute_clip_score, compute_aesthetic_score, compute_imagereward
    import torchvision.transforms.functional as F
    
    metrics = {}
    
    clip = compute_clip_score(image, prompt, clip_model, clip_processor)
    metrics["clip_score"] = clip
    
    aesthetic = compute_aesthetic_score(image, clip_model, clip_processor, aesthetic_model)
    metrics["aesthetic"] = aesthetic
    
    ir = compute_imagereward(image, prompt, imagereward_model)
    metrics["imagereward"] = ir if ir is not None else 0.0
    
    if original is not None and lpips_model is not None:
        target_size = [256, 256]
        
        def to_tensor(img):
            im_r = F.resize(img, target_size)
            t = F.to_tensor(im_r)
            t = t * 2.0 - 1.0
            return t.unsqueeze(0).to(device)
        
        t_edit = to_tensor(image)
        t_orig = to_tensor(original)
        
        with torch.no_grad():
            lpips_val = lpips_model(t_edit, t_orig).item()
        metrics["lpips"] = lpips_val
    
    return metrics


def format_edit_prompt(prompt: str) -> str:
    """
    Format prompt for Qwen-Image-Edit.
    The model works best with detailed, instruction-style prompts.
    Adds realism keywords to reduce AI artifacts.
    """
    prompt = prompt.strip()
    # Add strong realism emphasis
    realism_suffix = ", RAW photo, DSLR, 8k uhd, natural skin pores, natural skin texture, natural hair strands, detailed skin imperfections, unedited, no retouching, no airbrushing"
    prompt = prompt + realism_suffix
    if not prompt.endswith('.'):
        prompt = prompt + '.'
    return prompt


def run_ttflux_style(
    image: Image.Image,
    prompt: str,
    pipe,
    num_samples: int,
    device: str,
    num_inference_steps: int = 50,
    true_cfg_scale: float = 4.0,
) -> List[Image.Image]:
    """TT-FLUX style: Random search - generate N candidates."""
    formatted_prompt = format_edit_prompt(prompt)
    candidates = []
    print(f"  Generating {num_samples} candidates...")
    for i in tqdm(range(num_samples), desc="  TT-FLUX candidates"):
        seed = i * 17 + 100  # Different seed range from EACPS
        generator = torch.Generator(device=device if device.startswith("cuda") else "cpu").manual_seed(seed)
        width, height = image.size
        inputs = {
            "image": image,
            "prompt": formatted_prompt,
            "generator": generator,
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": "soft skin, airbrushed, smooth skin, brushed hair, AI generated, artificial, oversaturated, cartoon, painting, illustration, digital art, unrealistic, fake, synthetic, plastic skin, perfect skin, flawless skin, hairline blur, smooth texture, overly smooth, retouched, filtered, beautified, porcelain skin, wax figure, mannequin, doll-like, photoshopped, beauty filter, skin smoothing, noise reduction, over-processed",
            "num_inference_steps": num_inference_steps,
            "height": height,
            "width": width,
        }
        with torch.inference_mode():
            out = pipe(**inputs)
            candidates.append(out.images[0])
    return candidates


def run_eacps_style(
    image: Image.Image,
    prompt: str,
    pipe,
    scorer,
    k_global: int,
    m_global: int,
    k_local: int,
    device: str,
    num_inference_steps: int = 50,
    true_cfg_scale: float = 4.0,
) -> Tuple[Image.Image, List[Image.Image]]:
    """EACPS style: Multi-stage search with scoring."""
    from eacps import generate_edit, compute_potential
    
    formatted_prompt = format_edit_prompt(prompt)
    
    # Global stage - use different seed range from TT-FLUX
    print(f"  Global stage: generating {k_global} candidates...")
    global_candidates = []
    for i in tqdm(range(k_global), desc="  EACPS global"):
        seed = i * 31 + 5000  # Different seed range
        img = generate_edit(
            pipe=pipe,
            image=image,
            prompt=formatted_prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            device=device,
        )
        global_candidates.append({"seed": seed, "image": img, "phase": "global"})
    
    print("  Scoring global candidates...")
    global_images = [c["image"] for c in global_candidates]
    global_scores = scorer.score_batch(
        images=global_images,
        originals=[image] * len(global_images),
        prompts=[formatted_prompt] * len(global_images),
    )
    
    for cand, scores in zip(global_candidates, global_scores):
        scores_norm = {
            "clip_prompt_following": scores.get("PF", 0.0),
            "clip_consistency": scores.get("CONS", 0.0),
            "lpips": scores.get("LPIPS", 0.0),
        }
        cand["scores"] = scores_norm
        cand["potential"] = compute_potential(
            scores_norm,
            alpha_cons=10.0,
            beta_lpips=3.0,
            gamma_realism=0.0,
        )
    
    global_candidates.sort(key=lambda c: c["potential"], reverse=True)
    top_for_local = global_candidates[:m_global]
    
    print(f"  Selected top {m_global} candidates for local refinement")
    for i, c in enumerate(top_for_local):
        print(f"    #{i+1}: seed={c['seed']}, potential={c['potential']:.4f}")
    
    # Local stage
    print(f"  Local stage: generating {m_global * k_local} refinements...")
    local_candidates = []
    for parent in tqdm(top_for_local, desc="  EACPS local"):
        parent_seed = parent["seed"]
        for j in range(k_local):
            local_seed = parent_seed * 100 + j + 1
            img = generate_edit(
                pipe=pipe,
                image=image,
                prompt=formatted_prompt,
                seed=local_seed,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                device=device,
            )
            local_candidates.append({"seed": local_seed, "image": img, "phase": "local", "parent_seed": parent_seed})
    
    if local_candidates:
        print("  Scoring local candidates...")
        local_images = [c["image"] for c in local_candidates]
        local_scores = scorer.score_batch(
            images=local_images,
            originals=[image] * len(local_images),
            prompts=[formatted_prompt] * len(local_images),
        )
        for cand, scores in zip(local_candidates, local_scores):
            scores_norm = {
                "clip_prompt_following": scores.get("PF", 0.0),
                "clip_consistency": scores.get("CONS", 0.0),
                "lpips": scores.get("LPIPS", 0.0),
            }
            cand["scores"] = scores_norm
            cand["potential"] = compute_potential(
                scores_norm,
                alpha_cons=10.0,
                beta_lpips=3.0,
                gamma_realism=0.0,
            )
    
    all_candidates = global_candidates + local_candidates
    best_cand = max(all_candidates, key=lambda c: c["potential"])
    
    return best_cand["image"], [c["image"] for c in all_candidates]


def run():
    parser = argparse.ArgumentParser(description="Head-to-head: TT-FLUX vs EACPS")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--prompt", type=str, required=True, help="Edit prompt")
    parser.add_argument("--output_dir", type=str, default="head2head_comparison", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of candidates for TT-FLUX style")
    parser.add_argument("--k_global", type=int, default=8, help="EACPS k_global")
    parser.add_argument("--m_global", type=int, default=2, help="EACPS m_global")
    parser.add_argument("--k_local", type=int, default=2, help="EACPS k_local")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps (50 recommended)")
    parser.add_argument("--cfg", type=float, default=4.0, help="CFG scale (4.0-7.0 recommended)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 90)
    print("HEAD-TO-HEAD COMPARISON: TT-FLUX vs EACPS")
    print("=" * 90)
    print(f"Image: {args.image}")
    print(f"Prompt: {args.prompt}")
    print()
    
    if "cuda" in args.device:
        if not torch.cuda.is_available():
            print("ERROR: CUDA is not available")
            sys.exit(1)
        print(f"CUDA available: {torch.cuda.device_count()} device(s)")
        print(f"Using device: {args.device}")
    
    print("\nLoading models...", flush=True)
    sys.stdout.flush()
    
    from eacps import build_pipeline
    from scorers import EditScorer
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("  [1/6] Loading Qwen-Image-Edit pipeline...", flush=True)
    sys.stdout.flush()
    pipe = build_pipeline("Qwen/Qwen-Image-Edit", args.device)
    pipe.set_progress_bar_config(disable=True)
    print("  [1/6] Pipeline loaded.", flush=True)
    sys.stdout.flush()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("  [2/6] Loading EditScorer...", flush=True)
    sys.stdout.flush()
    scorer = EditScorer(device=args.device, use_lpips=True)
    print("  [2/6] EditScorer loaded.", flush=True)
    sys.stdout.flush()
    
    print("  [3/6] Loading CLIP model...", flush=True)
    sys.stdout.flush()
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(args.device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    print("  [3/6] CLIP model loaded.", flush=True)
    sys.stdout.flush()
    
    print("  [4/6] Loading aesthetic predictor (LAION)...", flush=True)
    sys.stdout.flush()
    aesthetic_model = load_aesthetic_model(args.device)
    print("  [4/6] Aesthetic predictor loaded.", flush=True)
    sys.stdout.flush()
    
    print("  [5/6] Loading ImageReward...", flush=True)
    sys.stdout.flush()
    imagereward_model = load_imagereward_model(args.device)
    if imagereward_model is None:
        print("  [5/6] (ImageReward unavailable, skipping this metric)", flush=True)
    else:
        print("  [5/6] ImageReward loaded.", flush=True)
    sys.stdout.flush()
    
    print("  [6/6] Loading LPIPS...", flush=True)
    sys.stdout.flush()
    import lpips
    lpips_model = lpips.LPIPS(net="vgg").to(args.device)
    lpips_model.eval()
    print("  [6/6] LPIPS loaded.", flush=True)
    sys.stdout.flush()
    
    print("All models loaded.\n", flush=True)
    sys.stdout.flush()
    
    input_image = load_image(args.image)
    input_image.save(output_dir / "input.png")
    
    print(f"Running TT-FLUX style (Random Search) with {args.steps} steps, cfg={args.cfg}...")
    ttflux_candidates = run_ttflux_style(
        input_image, args.prompt, pipe, args.num_samples, args.device,
        num_inference_steps=args.steps, true_cfg_scale=args.cfg
    )
    
    print("Scoring TT-FLUX candidates...")
    ttflux_all_metrics = []
    for i, img in enumerate(tqdm(ttflux_candidates, desc="  Scoring TT-FLUX")):
        m = compute_metrics(
            img, args.prompt, input_image,
            clip_model, clip_processor, aesthetic_model, imagereward_model, lpips_model,
            args.device
        )
        m["candidate_idx"] = i
        ttflux_all_metrics.append(m)
        img.save(output_dir / f"ttflux_candidate_{i}.png")
    
    ttflux_best_idx = max(range(len(ttflux_all_metrics)), key=lambda i: ttflux_all_metrics[i]["clip_score"])
    ttflux_best = ttflux_candidates[ttflux_best_idx]
    ttflux_best_metrics = ttflux_all_metrics[ttflux_best_idx]
    ttflux_best.save(output_dir / "ttflux_best.png")
    
    print(f"Running EACPS style (Multi-stage Search) with {args.steps} steps, cfg={args.cfg}...")
    eacps_best, eacps_all = run_eacps_style(
        input_image, args.prompt, pipe, scorer,
        k_global=args.k_global,
        m_global=args.m_global,
        k_local=args.k_local,
        device=args.device,
        num_inference_steps=args.steps,
        true_cfg_scale=args.cfg,
    )
    
    print("Scoring EACPS result...")
    eacps_metrics = compute_metrics(
        eacps_best, args.prompt, input_image,
        clip_model, clip_processor, aesthetic_model, imagereward_model, lpips_model,
        args.device
    )
    eacps_best.save(output_dir / "eacps_best.png")
    
    for i, img in enumerate(eacps_all):
        img.save(output_dir / f"eacps_candidate_{i}.png")
    
    print()
    print("=" * 90)
    print("RESULTS COMPARISON")
    print("=" * 90)
    print()
    print("TT-FLUX Paper Reference (FLUX.1-dev, DrawBench, no search):")
    print("  CLIPScore: 0.71  |  Aesthetic: 5.79  |  ImageReward: 0.97")
    print()
    print("-" * 90)
    print(f"{'Metric':<25} | {'TT-FLUX (Best of N)':<20} | {'EACPS':<20} | {'Winner':<12}")
    print("-" * 90)
    
    metrics_info = [
        ("CLIPScore ↑", "clip_score", True),
        ("Aesthetic Score ↑", "aesthetic", True),
        ("LPIPS ↓", "lpips", False),
    ]
    
    if imagereward_model is not None:
        metrics_info.insert(2, ("ImageReward ↑", "imagereward", True))
    
    comparison = {}
    for metric_name, metric_key, higher_better in metrics_info:
        tt_val = ttflux_best_metrics.get(metric_key, 0.0)
        eacps_val = eacps_metrics.get(metric_key, 0.0)
        
        if higher_better:
            winner = "EACPS" if eacps_val > tt_val else "TT-FLUX"
            delta = eacps_val - tt_val
        else:
            winner = "EACPS" if eacps_val < tt_val else "TT-FLUX"
            delta = tt_val - eacps_val
        
        print(f"{metric_name:<25} | {tt_val:>18.4f}   | {eacps_val:>18.4f}   | {winner:<12} (Δ={delta:+.4f})")
        comparison[metric_key] = {
            "ttflux": tt_val,
            "eacps": eacps_val,
            "delta": delta,
            "winner": winner
        }
    
    print("-" * 90)
    print()
    
    wins = sum(1 for v in comparison.values() if v["winner"] == "EACPS")
    print(f"EACPS wins {wins}/{len(comparison)} metrics")
    print("=" * 90)
    
    results = {
        "input_image": str(args.image),
        "prompt": args.prompt,
        "ttflux_paper_reference": {
            "model": "FLUX.1-dev",
            "benchmark": "DrawBench",
            "clip_score": 0.71,
            "aesthetic": 5.79,
            "imagereward": 0.97,
        },
        "ttflux": {
            "method": "Random Search (Best of N by CLIPScore)",
            "model": "Qwen-Image-Edit",
            "best_candidate_idx": ttflux_best_idx,
            "metrics": ttflux_best_metrics,
            "all_candidates": ttflux_all_metrics,
        },
        "eacps": {
            "method": "Multi-stage Search (EACPS)",
            "model": "Qwen-Image-Edit",
            "metrics": eacps_metrics,
        },
        "comparison": comparison,
        "config": {
            "num_samples": args.num_samples,
            "k_global": args.k_global,
            "m_global": args.m_global,
            "k_local": args.k_local,
        }
    }
    
    results_path = output_dir / "comparison_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"Images saved to {output_dir}/")
    print(f"\nBest outputs:")
    print(f"  TT-FLUX: {output_dir}/ttflux_best.png")
    print(f"  EACPS:   {output_dir}/eacps_best.png")


if __name__ == "__main__":
    run()
