import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

# Disable flash attention before importing diffusers
os.environ["DIFFUSERS_USE_FLASH_ATTENTION"] = "0"
os.environ["XFORMERS_DISABLED"] = "1"

# Monkey-patch flash_attn to prevent import errors
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

import torch
from PIL import Image
from tqdm import tqdm

from diffusers import QwenImageEditPipeline
from scorers import EditScorer


@dataclass
class Sample:
    id: str
    image_path: Path
    prompt: str


def load_dataset(dataset_jsonl: Path) -> List[Sample]:
    dataset_dir = dataset_jsonl.parent
    samples: List[Sample] = []
    with dataset_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            image_rel = Path(obj["image_path"])
            if image_rel.is_absolute():
                image_path = image_rel
            else:
                image_path = (dataset_dir / image_rel).resolve()
            samples.append(
                Sample(
                    id=str(obj["id"]),
                    image_path=image_path,
                    prompt=str(obj["prompt"]),
                )
            )
    return samples


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def build_pipeline(model_id: str, device: str) -> QwenImageEditPipeline:
    # Check if CUDA is actually available
    if "cuda" in device and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device '{device}' requested but CUDA is not available. "
            f"Check: nvidia-smi, CUDA installation, and terminal session."
        )
    
    use_cuda = "cuda" in device and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    
    print(f"    Loading model from {model_id}...", flush=True)
    import sys
    sys.stdout.flush()
    # Check for multi-GPU (e.g., "cuda:0,1" or "cuda:0,cuda:1")
    if "," in device:
        print(f"    Multi-GPU mode: using device_map='balanced' across GPUs", flush=True)
        sys.stdout.flush()
        pipe = QwenImageEditPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="balanced",
        )
    else:
        print(f"    Single-GPU mode: loading on {device}", flush=True)
        sys.stdout.flush()
        pipe = QwenImageEditPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
    print(f"    Model loaded from cache/download.", flush=True)
    sys.stdout.flush()
    if use_cuda:
        # Clear cache before moving to device
        torch.cuda.empty_cache()
        print(f"    Moving pipeline to device: {device}", flush=True)
        sys.stdout.flush()
        pipe.to(device)
        torch.cuda.empty_cache()
        print(f"    Pipeline on device: {device}, CUDA device count: {torch.cuda.device_count()}", flush=True)
        sys.stdout.flush()
    return pipe


def build_scorer(device: str) -> EditScorer:
    # For multi-GPU, put scorer on first GPU
    scorer_device = device.split(",")[0] if "," in device else device
    print(f"Building scorer on device {scorer_device}")
    return EditScorer(device=scorer_device)


def generate_edit(
    pipe: QwenImageEditPipeline,
    image: Image.Image,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    true_cfg_scale: float,
    device: str,
    negative_prompt: str = "soft skin, airbrushed, smooth skin, brushed hair, AI generated, artificial, oversaturated, cartoon, painting, illustration, digital art, unrealistic, fake, synthetic, plastic skin, perfect skin, flawless skin, hairline blur, smooth texture, overly smooth, retouched, filtered, beautified, porcelain skin, wax figure, mannequin, doll-like, photoshopped, beauty filter, skin smoothing, noise reduction, over-processed",
) -> Image.Image:
    # For multi-GPU (device_map), use first GPU or cpu for generator
    if "," in device:
        gen_device = "cuda:0"
    elif device.startswith("cuda"):
        gen_device = device
    else:
        gen_device = "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed)
    width, height = image.size
    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": generator,
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "height": height,
        "width": width,
    }
    with torch.inference_mode():
        out = pipe(**inputs)
    return out.images[0]


def canonicalize_scores(raw: Dict[str, Any]) -> Dict[str, float]:
    """
    Map whatever the scorer returns into canonical keys:
    - clip_prompt_following
    - clip_consistency
    - lpips
    - realism
    Handles missing keys gracefully.
    """
    out: Dict[str, float] = {}
    numeric: Dict[str, float] = {}

    for k, v in raw.items():
        try:
            numeric[k] = float(v)
        except Exception:
            continue

    # Prompt following
    if "clip_prompt_following" in numeric:
        out["clip_prompt_following"] = numeric["clip_prompt_following"]
    else:
        pf_key = None
        for k, v in numeric.items():
            kl = k.lower()
            if kl.startswith("lpips"):
                continue
            if "realism" in kl:
                continue
            if "consistency" in kl:
                continue
            if v > 1.0:
                pf_key = k
                break
        if pf_key is None:
            for k, v in numeric.items():
                kl = k.lower()
                if kl.startswith("lpips") or "realism" in kl:
                    continue
                pf_key = k
                break
        if pf_key is not None:
            out["clip_prompt_following"] = numeric[pf_key]

    # Consistency
    if "clip_consistency" in numeric:
        out["clip_consistency"] = numeric["clip_consistency"]
    else:
        cons_key = None
        for k, v in numeric.items():
            kl = k.lower()
            if kl.startswith("lpips"):
                continue
            if "realism" in kl:
                continue
            if "prompt" in kl:
                continue
            if v <= 1.5:
                cons_key = k
                break
        if cons_key is not None:
            out["clip_consistency"] = numeric[cons_key]

    # LPIPS
    for k, v in numeric.items():
        if k.lower().startswith("lpips"):
            out["lpips"] = v
            break

    # Realism
    for k, v in numeric.items():
        if "realism" in k.lower():
            out["realism"] = v
            break

    return out


def compute_potential(
    scores: Dict[str, float],
    alpha_cons: float,
    beta_lpips: float,
    gamma_realism: float,
) -> float:
    pf = float(scores.get("clip_prompt_following", 0.0))
    cons = float(scores.get("clip_consistency", 0.0))
    lpips_val = float(scores.get("lpips", 0.0))
    realism = float(scores.get("realism", 0.0))

    potential = pf + alpha_cons * cons + gamma_realism * realism
    if beta_lpips != 0.0:
        potential -= beta_lpips * lpips_val
    return potential


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model_id", type=str, default="Qwen/Qwen-Image-Edit")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)

    parser.add_argument("--k_global", type=int, default=4)
    parser.add_argument("--m_global", type=int, default=2)
    parser.add_argument("--k_local", type=int, default=2)

    parser.add_argument("--alpha_cons", type=float, default=10.0)
    parser.add_argument("--beta_lpips", type=float, default=3.0)
    parser.add_argument("--gamma_realism", type=float, default=0.0)

    # Early stopping and output controls
    parser.add_argument("--max_candidates", type=int, default=32,
                        help="Maximum candidates per record before stopping")
    parser.add_argument("--patience", type=int, default=2,
                        help="Stop if no improvement after this many rounds")
    parser.add_argument("--save_all_candidates", action="store_true",
                        help="Save all candidates (not just winner) for analysis")

    # Resume support
    parser.add_argument("--start_index", type=int, default=0,
                        help="Skip records before this index (for resuming crashed runs)")
    parser.add_argument("--skip_on_oom", action="store_true",
                        help="Skip records that cause OOM instead of crashing")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "dataset_results_eacps.jsonl"

    samples = load_dataset(dataset_path)
    print(f"Loaded {len(samples)} records from {dataset_path}")
    print(f"Config: k_global={args.k_global}, m_global={args.m_global}, k_local={args.k_local}")
    print(f"        num_samples={args.num_samples}, max_candidates={args.max_candidates}, patience={args.patience}")
    print(f"        alpha_cons={args.alpha_cons}, beta_lpips={args.beta_lpips}, gamma_realism={args.gamma_realism}")
    print(f"        save_all_candidates={args.save_all_candidates}")
    if args.start_index > 0:
        print(f"        RESUMING from index {args.start_index}")

    pipe = build_pipeline(args.model_id, args.device)
    scorer = build_scorer(args.device)

    records_out: List[Dict[str, Any]] = []
    write_mode = "a" if args.start_index > 0 else "w"

    for sample_idx, sample in enumerate(samples):
        if sample_idx < args.start_index:
            continue

        torch.cuda.empty_cache()
        print(f"\n[{sample_idx+1}/{len(samples)}] Processing id={sample.id}")

        try:
            orig_image = load_image(sample.image_path)
            prompt = sample.prompt
            total_candidates = 0
            best_potential_so_far = float("-inf")
            rounds_without_improvement = 0

            # ---------- Global stage ----------
            print(f"  Stage 1 (Global): generating {args.num_samples} candidates...")
            global_candidates: List[Dict[str, Any]] = []
            for i in tqdm(range(args.num_samples), desc="  Global", leave=False):
                if total_candidates >= args.max_candidates:
                    print(f"  Hit max_candidates={args.max_candidates}, stopping global stage.")
                    break
                seed = args.seed_offset + sample_idx * args.num_samples + i
                img = generate_edit(
                    pipe=pipe,
                    image=orig_image,
                    prompt=prompt,
                    seed=seed,
                    num_inference_steps=args.num_inference_steps,
                    true_cfg_scale=args.true_cfg_scale,
                    device=args.device,
                )
                global_candidates.append({
                    "seed": seed,
                    "image": img,
                    "phase": "stage1",
                })
                total_candidates += 1

            # Score global candidates
            if global_candidates:
                global_images = [c["image"] for c in global_candidates]
                global_scores_raw = scorer.score_batch(
                    images=global_images,
                    originals=[orig_image] * len(global_images),
                    prompts=[prompt] * len(global_images),
                )
                for cand, raw_scores in zip(global_candidates, global_scores_raw):
                    scores = canonicalize_scores(raw_scores)
                    cand["scores"] = scores
                    cand["potential"] = compute_potential(
                        scores,
                        alpha_cons=args.alpha_cons,
                        beta_lpips=args.beta_lpips,
                        gamma_realism=args.gamma_realism,
                    )
                global_candidates.sort(key=lambda c: c["potential"], reverse=True)
                current_best = global_candidates[0]["potential"]
                if current_best > best_potential_so_far + 0.01:
                    best_potential_so_far = current_best
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                print(f"  Global best potential: {current_best:.2f} (PF={global_candidates[0]['scores'].get('clip_prompt_following', 0):.2f})")

            k_global = min(args.k_global, len(global_candidates))
            m_global = min(args.m_global, k_global)
            top_for_local = global_candidates[:m_global]

            # ---------- Local stage ----------
            local_candidates: List[Dict[str, Any]] = []
            if rounds_without_improvement >= args.patience:
                print(f"  Skipping local stage: no improvement for {args.patience} rounds")
            elif total_candidates >= args.max_candidates:
                print(f"  Skipping local stage: hit max_candidates={args.max_candidates}")
            else:
                print(f"  Stage 2 (Local): refining top {m_global} candidates with {args.k_local} variants each...")
                for parent in top_for_local:
                    if total_candidates >= args.max_candidates:
                        break
                    parent_seed = int(parent["seed"])
                    for j in range(args.k_local):
                        if total_candidates >= args.max_candidates:
                            break
                        local_seed = (parent_seed + 1) * 100000 + j
                        img = generate_edit(
                            pipe=pipe,
                            image=orig_image,
                            prompt=prompt,
                            seed=local_seed,
                            num_inference_steps=args.num_inference_steps,
                            true_cfg_scale=args.true_cfg_scale,
                            device=args.device,
                        )
                        local_candidates.append({
                            "seed": local_seed,
                            "parent_seed": parent_seed,
                            "image": img,
                            "phase": "stage2",
                        })
                        total_candidates += 1

            if local_candidates:
                local_images = [c["image"] for c in local_candidates]
                local_scores_raw = scorer.score_batch(
                    images=local_images,
                    originals=[orig_image] * len(local_images),
                    prompts=[prompt] * len(local_images),
                )
                for cand, raw_scores in zip(local_candidates, local_scores_raw):
                    scores = canonicalize_scores(raw_scores)
                    cand["scores"] = scores
                    cand["potential"] = compute_potential(
                        scores,
                        alpha_cons=args.alpha_cons,
                        beta_lpips=args.beta_lpips,
                        gamma_realism=args.gamma_realism,
                    )
                local_best = max(local_candidates, key=lambda c: c["potential"])
                print(f"  Local best potential: {local_best['potential']:.2f} (PF={local_best['scores'].get('clip_prompt_following', 0):.2f})")

            # ---------- Selection ----------
            all_candidates = global_candidates + local_candidates
            best_cand = max(all_candidates, key=lambda c: c["potential"])
            print(f"  Final: {total_candidates} candidates, best from {best_cand['phase']}, potential={best_cand['potential']:.2f}")

            record = {
                "id": sample.id,
                "prompt": prompt,
                "seed": int(best_cand["seed"]),
                "phase": best_cand["phase"],
                "clip_prompt_following": best_cand["scores"].get("clip_prompt_following", 0.0),
                "clip_consistency": best_cand["scores"].get("clip_consistency", 0.0),
                "lpips": best_cand["scores"].get("lpips", 0.0),
                "potential": best_cand["potential"],
                "total_candidates": total_candidates,
            }
            records_out.append(record)

            # Write incrementally
            with results_path.open(write_mode, encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            write_mode = "a"

            if args.save_all_candidates:
                all_cands_path = output_dir / "all_candidates.jsonl"
                cands_mode = "a" if sample_idx > args.start_index else ("a" if args.start_index > 0 else "w")
                with all_cands_path.open(cands_mode, encoding="utf-8") as f:
                    for cand in all_candidates:
                        cand_record = {
                            "phase": cand["phase"],
                            "id": sample.id,
                            "prompt": prompt,
                            "image_path": str(sample.image_path),
                            "seed": int(cand["seed"]),
                            "model_id": args.model_id,
                            "num_inference_steps": args.num_inference_steps,
                            "true_cfg_scale": args.true_cfg_scale,
                            "clip_prompt_following": cand["scores"].get("clip_prompt_following", 0.0),
                            "clip_consistency": cand["scores"].get("clip_consistency", 0.0),
                            "lpips": cand["scores"].get("lpips", 0.0),
                            "potential": cand["potential"],
                        }
                        if "parent_seed" in cand:
                            cand_record["parent_seed"] = cand["parent_seed"]
                        f.write(json.dumps(cand_record) + "\n")

        except torch.cuda.OutOfMemoryError as e:
            print(f"  ERROR: CUDA OOM on {sample.id}: {e}")
            torch.cuda.empty_cache()
            if args.skip_on_oom:
                print(f"  Skipping {sample.id} and continuing...")
                continue
            else:
                print("  Use --skip_on_oom to continue past OOM errors")
                raise

        except Exception as e:
            print(f"  ERROR processing {sample.id}: {e}")
            raise

    print(f"\nCompleted! Processed {len(records_out)} records.")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
