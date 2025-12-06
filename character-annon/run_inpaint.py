#!/usr/bin/env python3
"""
Character Inpainting Pipeline
Supports Label Studio exports: JSON, JSON-MIN, CSV, TSV, YOLOv8

Takes init_image + mask_image + character_image and inpaints the character
into the masked region using TT-FLUX or EACPS approaches.
"""
import os
import sys
import json
import csv
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image, ImageFilter, ImageOps
from io import BytesIO
import torch
import numpy as np

os.environ["DIFFUSERS_USE_FLASH_ATTENTION"] = "0"
os.environ["XFORMERS_DISABLED"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eacps import build_pipeline, generate_edit
from scorers import EditScorer


def download_image(url: str, cache_dir: Path) -> Optional[Path]:
    """Download image from URL and cache locally."""
    if not url or url.strip() == "":
        return None
    
    filename = url.split("/")[-1].split("?")[0]
    if not filename:
        filename = f"img_{hash(url)}.png"
    
    cache_path = cache_dir / filename
    if cache_path.exists():
        return cache_path
    
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        img.save(cache_path)
        return cache_path
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None


def load_local_image(path: str) -> Optional[Path]:
    """Load image from local path."""
    p = Path(path)
    if p.exists():
        return p
    return None


def get_image_path(url_or_path: str, cache_dir: Path) -> Optional[Path]:
    """Get image path from URL or local path."""
    if not url_or_path or url_or_path.strip() == "":
        return None
    
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return download_image(url_or_path, cache_dir)
    else:
        return load_local_image(url_or_path)


def parse_json(filepath: Path) -> List[Dict[str, Any]]:
    """Parse Label Studio JSON export."""
    with open(filepath, "r") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        items = []
        for item in data:
            if "data" in item:
                items.append(item["data"])
            else:
                items.append(item)
        return items
    elif isinstance(data, dict) and "data" in data:
        return [data["data"]]
    else:
        return [data]


def parse_json_min(filepath: Path) -> List[Dict[str, Any]]:
    """Parse Label Studio JSON-MIN export."""
    return parse_json(filepath)


def parse_csv(filepath: Path) -> List[Dict[str, Any]]:
    """Parse CSV export."""
    items = []
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(dict(row))
    return items


def parse_tsv(filepath: Path) -> List[Dict[str, Any]]:
    """Parse TSV export."""
    items = []
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            items.append(dict(row))
    return items


def parse_yolov8(dirpath: Path) -> List[Dict[str, Any]]:
    """Parse YOLOv8 OBB export structure."""
    notes_path = dirpath / "notes.json"
    if not notes_path.exists():
        raise FileNotFoundError(f"notes.json not found in {dirpath}")
    
    with open(notes_path, "r") as f:
        notes = json.load(f)
    
    items = []
    if isinstance(notes, list):
        items = notes
    elif isinstance(notes, dict):
        if "data" in notes:
            items = notes["data"] if isinstance(notes["data"], list) else [notes["data"]]
        else:
            items = [notes]
    
    images_dir = dirpath / "images"
    for item in items:
        if "image_url" not in item and "id" in item:
            img_candidates = list(images_dir.glob(f"*{item['id']}*"))
            if img_candidates:
                item["image_url"] = str(img_candidates[0])
    
    return items


def load_data(filepath: str) -> List[Dict[str, Any]]:
    """Load data from Label Studio export file."""
    path = Path(filepath)
    
    if path.is_dir():
        return parse_yolov8(path)
    
    suffix = path.suffix.lower()
    name = path.name.lower()
    
    if suffix == ".json":
        if "min" in name:
            return parse_json_min(path)
        return parse_json(path)
    elif suffix == ".csv":
        return parse_csv(path)
    elif suffix == ".tsv":
        return parse_tsv(path)
    else:
        try:
            return parse_json(path)
        except:
            return parse_csv(path)


def filter_items(items: List[Dict], ids: Optional[List[int]] = None, 
                 start: Optional[int] = None, end: Optional[int] = None) -> List[Dict]:
    """Filter items by IDs or range. Checks both top-level id and data.id fields."""
    if ids is not None:
        # Convert to both int and str for flexible matching
        id_set_int = set(ids)
        id_set_str = set(str(x) for x in ids)
        filtered = []
        for i, item in enumerate(items):
            # Check top-level id
            item_id = item.get("id")
            # Check data.id (Label Studio format)
            data_id = item.get("data", {}).get("id")
            # Match against any of: top-level id, data.id, or index (as int or str)
            match = (
                item_id in id_set_int or 
                str(item_id) in id_set_str or
                data_id in id_set_int or 
                str(data_id) in id_set_str or
                i in id_set_int
            )
            if match:
                filtered.append(item)
        return filtered
    
    if start is not None or end is not None:
        start = start or 0
        end = end or len(items)
        return items[start:end]
    
    return items


def feather_mask(mask: Image.Image, feather_radius: int = 5, threshold: int = 32) -> Image.Image:
    """Normalize mask and apply feathering for smooth blending."""
    # Normalize mask contrast and threshold to focus on the region of interest
    mask_norm = ImageOps.autocontrast(mask)
    mask_binary = mask_norm.point(lambda p: 255 if p > threshold else 0)
    if feather_radius > 0:
        return mask_binary.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    return mask_binary


def blend_with_mask(output_img: Image.Image, original_img: Image.Image, 
                    mask_img: Image.Image, feather: bool = True) -> Image.Image:
    """
    Blend output with original using mask with smooth feathering.
    Keep output ONLY in masked area, keep original OUTSIDE mask.
    Uses feathered mask for smooth transitions.
    """
    # Resize output to match original if needed
    if output_img.size != original_img.size:
        output_img = output_img.resize(original_img.size, Image.Resampling.LANCZOS)
    
    # Resize mask to match
    mask_resized = mask_img.convert("L").resize(original_img.size, Image.Resampling.LANCZOS)
    
    # Prepare mask (normalize + optional feathering)
    if feather:
        mask_processed = feather_mask(mask_resized, feather_radius=6, threshold=30)
    else:
        mask_processed = mask_resized
    
    output_rgb = output_img.convert("RGB")
    original_rgb = original_img.convert("RGB")
    
    # Blend so that only masked area comes from output
    result = Image.composite(output_rgb, original_rgb, mask_processed)
    return result


def crop_to_content(image: Image.Image) -> Image.Image:
    """Crop image to its non-transparent/non-white content."""
    img_array = np.array(image.convert("RGBA"))
    
    # Check alpha channel if exists, otherwise check for non-white pixels
    if img_array.shape[2] == 4:
        # Use alpha channel - find non-transparent pixels
        alpha = img_array[:, :, 3]
        non_empty = alpha > 10
    else:
        # Find non-white pixels (assuming white background)
        rgb = img_array[:, :, :3]
        non_empty = np.any(rgb < 250, axis=2)
    
    rows = np.any(non_empty, axis=1)
    cols = np.any(non_empty, axis=0)
    
    if not rows.any() or not cols.any():
        return image
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Add small padding
    padding = 2
    y_min = max(0, y_min - padding)
    y_max = min(image.height - 1, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(image.width - 1, x_max + padding)
    
    return image.crop((x_min, y_min, x_max + 1, y_max + 1))


def match_color_grading(source: Image.Image, reference: Image.Image) -> Image.Image:
    """
    Match the color grading of source image to reference image.
    Uses simple histogram matching in LAB color space for natural results.
    """
    src_array = np.array(source.convert("RGB")).astype(np.float32)
    ref_array = np.array(reference.convert("RGB")).astype(np.float32)
    
    # Calculate mean and std for each channel
    src_mean = src_array.mean(axis=(0, 1))
    src_std = src_array.std(axis=(0, 1)) + 1e-6
    ref_mean = ref_array.mean(axis=(0, 1))
    ref_std = ref_array.std(axis=(0, 1)) + 1e-6
    
    # Apply color transfer: normalize source, then scale to reference stats
    result = (src_array - src_mean) * (ref_std / src_std) + ref_mean
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result)


def composite_character(init_image: Image.Image, character_image: Image.Image, 
                        mask_image: Optional[Image.Image]) -> Image.Image:
    """
    Composite the character image into the mask region of init_image.
    
    Strategy for proper face/character alignment:
    1. Crop character to content
    2. Scale character to COVER the mask (aspect fill) - no distortion
    3. Center the scaled character over the mask
    4. Apply the mask to cut out only the masked region
    5. Color match and blend seamlessly
    """
    if character_image is None:
        return init_image
    
    init_rgb = init_image.convert("RGB")
    init_rgba = init_image.convert("RGBA")
    char_rgba = character_image.convert("RGBA")
    
    # Crop character to its actual content (remove transparent borders)
    char_cropped = crop_to_content(char_rgba)
    
    if mask_image is None:
        char_resized = char_cropped.resize(init_rgb.size, Image.Resampling.LANCZOS)
        composite = Image.alpha_composite(init_rgba, char_resized)
        return composite.convert("RGB")
    
    # Resize mask to match init image size
    mask_resized = mask_image.convert("L").resize(init_rgb.size, Image.Resampling.LANCZOS)
    
    # Find bounding box of the mask (white region)
    mask_array = np.array(mask_resized)
    rows = np.any(mask_array > 10, axis=1)
    cols = np.any(mask_array > 10, axis=0)
    
    if not rows.any() or not cols.any():
        return init_image.convert("RGB")
    
    # Get bounding box of mask
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1
    mask_center_x = (x_min + x_max) // 2
    mask_center_y = (y_min + y_max) // 2
    
    # Match color grading of character to init image BEFORE resizing
    char_rgb = char_cropped.convert("RGB")
    char_color_matched = match_color_grading(char_rgb, init_rgb)
    
    # Preserve alpha channel
    if char_cropped.mode == "RGBA":
        char_color_matched = char_color_matched.convert("RGBA")
        char_color_matched.putalpha(char_cropped.split()[3])
    
    # Scale character to COVER the mask region (aspect fill - no distortion)
    char_w, char_h = char_color_matched.size
    scale_w = bbox_width / char_w
    scale_h = bbox_height / char_h
    scale = max(scale_w, scale_h)  # Use max to ensure full coverage
    
    new_w = int(char_w * scale)
    new_h = int(char_h * scale)
    char_scaled = char_color_matched.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Center the scaled character over the mask center
    paste_x = mask_center_x - new_w // 2
    paste_y = mask_center_y - new_h // 2
    
    # Create a layer for the character
    char_layer = Image.new("RGBA", init_rgb.size, (0, 0, 0, 0))
    char_layer.paste(char_scaled, (paste_x, paste_y))
    
    # Create feathered mask for smooth edge blending
    mask_feathered = feather_mask(mask_resized, feather_radius=10, threshold=20)
    
    # Composite: character appears ONLY in masked region
    composite = Image.composite(char_layer, init_rgba, mask_feathered)
    
    return composite.convert("RGB")


def run_ttflux_inpaint(pipe, init_image: Image.Image, mask_image: Image.Image,
                       character_image: Image.Image, prompt: str, 
                       num_samples: int, device: str, steps: int, cfg: float,
                       negative_prompt: str, scorer: EditScorer) -> tuple:
    """Run TT-FLUX style random search for character editing."""
    candidates = []
    scores = []
    
    input_image = composite_character(init_image, character_image, mask_image)
    
    for i in range(num_samples):
        seed = i * 17 + 100
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        raw_result = pipe(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        ).images[0]
        
        # Blend: keep ONLY masked area from output, rest from original
        result = blend_with_mask(raw_result, init_image, mask_image)
        
        clip_pf = float(scorer._clip_pf([result], [prompt])[0])
        candidates.append({"image": result, "seed": seed, "clip_score": clip_pf})
        scores.append(clip_pf)
        print(f"  TT-FLUX candidate {i+1}/{num_samples}: CLIP={clip_pf:.4f}")
    
    best_idx = scores.index(max(scores))
    return candidates[best_idx]["image"], candidates


def compute_mask_preservation(input_img: Image.Image, output_img: Image.Image, 
                               mask_img: Image.Image) -> float:
    """
    Compute how well the output preserves the content in the masked region.
    Higher score = better preservation of the character in mask area.
    """
    # Resize output to match input size if different
    if output_img.size != input_img.size:
        output_img = output_img.resize(input_img.size, Image.Resampling.LANCZOS)
    
    # Convert to arrays
    input_arr = np.array(input_img.convert("RGB")).astype(np.float32)
    output_arr = np.array(output_img.convert("RGB")).astype(np.float32)
    mask_arr = np.array(mask_img.convert("L").resize(input_img.size, Image.Resampling.LANCZOS))
    
    # Normalize mask to 0-1
    mask_norm = mask_arr.astype(np.float32) / 255.0
    mask_3d = np.stack([mask_norm] * 3, axis=-1)
    
    # Compute difference only in masked region
    diff = np.abs(input_arr - output_arr) * mask_3d
    
    # Normalize to 0-1 range (255 max diff)
    mse_masked = np.sum(diff) / (np.sum(mask_norm) * 3 * 255 + 1e-8)
    
    # Return preservation score (1 - error), higher is better
    preservation = 1.0 - min(1.0, mse_masked)
    return preservation


def run_eacps_inpaint(pipe, init_image: Image.Image, mask_image: Image.Image,
                      character_image: Image.Image, prompt: str,
                      k_global: int, m_global: int, k_local: int,
                      device: str, steps: int, cfg: float,
                      negative_prompt: str, scorer: EditScorer) -> tuple:
    """Run EACPS multi-stage search for character editing with mask-only output."""
    
    input_image = composite_character(init_image, character_image, mask_image)
    
    global_candidates = []
    print(f"  EACPS Stage 1: Global exploration ({k_global} candidates)")
    
    for i in range(k_global):
        seed = i * 31 + 5000
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        raw_result = pipe(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        ).images[0]
        
        # Blend: keep ONLY masked area from output, rest from original
        result = blend_with_mask(raw_result, init_image, mask_image)
        
        pf = float(scorer._clip_pf([result], [prompt])[0])
        cons = float(scorer._clip_cons([result], [input_image])[0])
        lpips_val = float(scorer._lpips_dist([result], [input_image])[0])
        
        # Potential score for ranking
        potential = pf + 10.0 * cons - 3.0 * lpips_val
        
        global_candidates.append({
            "image": result, "seed": seed, "pf": pf, 
            "cons": cons, "lpips": lpips_val, "potential": potential
        })
        print(f"    Global {i+1}/{k_global}: PF={pf:.4f}, CONS={cons:.4f}, LPIPS={lpips_val:.4f}, POT={potential:.4f}")
    
    global_candidates.sort(key=lambda x: x["potential"], reverse=True)
    top_candidates = global_candidates[:m_global]
    
    print(f"  EACPS Stage 2: Local refinement ({k_local} per top-{m_global})")
    all_candidates = list(global_candidates)
    
    for j, parent in enumerate(top_candidates):
        for k in range(k_local):
            child_seed = parent["seed"] * 10 + k + 1
            generator = torch.Generator(device="cpu").manual_seed(child_seed)
            
            raw_result = pipe(
                image=input_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
            ).images[0]
            
            # Blend: keep ONLY masked area from output, rest from original
            result = blend_with_mask(raw_result, init_image, mask_image)
            
            pf = float(scorer._clip_pf([result], [prompt])[0])
            cons = float(scorer._clip_cons([result], [input_image])[0])
            lpips_val = float(scorer._lpips_dist([result], [input_image])[0])
            potential = pf + 10.0 * cons - 3.0 * lpips_val
            
            all_candidates.append({
                "image": result, "seed": child_seed, "pf": pf,
                "cons": cons, "lpips": lpips_val, "potential": potential
            })
            print(f"    Local {j+1}.{k+1}: PF={pf:.4f}, CONS={cons:.4f}, LPIPS={lpips_val:.4f}, POT={potential:.4f}")
    
    all_candidates.sort(key=lambda x: x["potential"], reverse=True)
    return all_candidates[0]["image"], all_candidates


def process_item(item: Dict, pipe, cache_dir: Path, output_dir: Path,
                 method: str, device: str, num_samples: int,
                 k_global: int, m_global: int, k_local: int,
                 steps: int, cfg: float) -> Dict:
    """Process a single item."""
    item_id = item.get("id", "unknown")
    char_name = item.get("character_name", "character")
    
    print(f"\nProcessing item {item_id} ({char_name})")
    
    init_url = item.get("init_image_url", item.get("image_url", ""))
    mask_url = item.get("mask_image_url", "")
    char_url = item.get("character_image_url", "")
    prompt = item.get("prompt", f"Add {char_name} to the image")
    
    init_path = get_image_path(init_url, cache_dir)
    mask_path = get_image_path(mask_url, cache_dir)
    char_path = get_image_path(char_url, cache_dir)
    
    if not init_path:
        print(f"  Skipping: no init image")
        return {"id": item_id, "status": "skipped", "reason": "no init image"}
    
    init_image = Image.open(init_path).convert("RGB")
    mask_image = Image.open(mask_path).convert("L") if mask_path else None
    char_image = Image.open(char_path).convert("RGBA") if char_path else None
    
    edit_prompt = f"Blend {char_name} naturally into the scene, maintain lighting consistency, photorealistic, seamless integration"
    if "consistency" in prompt.lower():
        edit_prompt = f"Improve character consistency for {char_name}, natural blending, same lighting, photorealistic"
    
    negative_prompt = (
        "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
        "bad proportions, extra limbs, cloned face, disfigured, "
        "out of frame, watermark, signature, text, seams, visible edges, "
        "inconsistent lighting, artificial, unnatural"
    )
    
    item_output_dir = output_dir / f"item_{item_id}"
    item_output_dir.mkdir(parents=True, exist_ok=True)
    
    init_image.save(item_output_dir / "init.png")
    if mask_image:
        mask_image.save(item_output_dir / "mask.png")
    if char_image:
        char_image.convert("RGB").save(item_output_dir / "character.png")
    
    composite = composite_character(init_image, char_image, mask_image)
    composite.save(item_output_dir / "composite.png")
    print(f"  Saved composite image")
    
    # Create shared scorer with LPIPS enabled
    scorer = EditScorer(device=device, use_lpips=True)
    
    results = {"id": item_id, "character_name": char_name, "prompt": edit_prompt, "original_prompt": prompt}
    
    if method in ["ttflux", "both"]:
        print(f"  Running TT-FLUX ({num_samples} samples)")
        ttflux_best, ttflux_candidates = run_ttflux_inpaint(
            pipe, init_image, mask_image, char_image, edit_prompt,
            num_samples, device, steps, cfg, negative_prompt, scorer
        )
        ttflux_best.save(item_output_dir / "ttflux_best.png")
        results["ttflux"] = {
            "best_clip": max(c["clip_score"] for c in ttflux_candidates),
            "output": str(item_output_dir / "ttflux_best.png")
        }
    
    if method in ["eacps", "both"]:
        print(f"  Running EACPS (k_global={k_global}, m_global={m_global}, k_local={k_local})")
        eacps_best, eacps_candidates = run_eacps_inpaint(
            pipe, init_image, mask_image, char_image, edit_prompt,
            k_global, m_global, k_local, device, steps, cfg, negative_prompt, scorer
        )
        eacps_best.save(item_output_dir / "eacps_best.png")
        results["eacps"] = {
            "best_potential": max(c["potential"] for c in eacps_candidates),
            "output": str(item_output_dir / "eacps_best.png")
        }
    
    with open(item_output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Character Inpainting Pipeline")
    parser.add_argument("--input", required=True, help="Path to Label Studio export (JSON/CSV/TSV/YOLOv8 dir)")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--cache_dir", default="data/cache", help="Image cache directory")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--method", choices=["ttflux", "eacps", "both"], default="both", help="Method to use")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Process all items")
    group.add_argument("--ids", type=int, nargs="+", help="Process specific item IDs")
    group.add_argument("--range", type=int, nargs=2, metavar=("START", "END"), help="Process range [START, END)")
    group.add_argument("--first", type=int, help="Process first N items")
    
    parser.add_argument("--num_samples", type=int, default=4, help="TT-FLUX: number of samples")
    parser.add_argument("--k_global", type=int, default=4, help="EACPS: global candidates")
    parser.add_argument("--m_global", type=int, default=2, help="EACPS: top candidates to refine")
    parser.add_argument("--k_local", type=int, default=2, help="EACPS: local refinements per candidate")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=5.0, help="Guidance scale")
    parser.add_argument("--model", default="Qwen/Qwen-Image-Edit", help="Model to use")
    
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {args.input}")
    items = load_data(args.input)
    print(f"Loaded {len(items)} items")
    
    if args.ids:
        items = filter_items(items, ids=args.ids)
    elif args.range:
        items = filter_items(items, start=args.range[0], end=args.range[1])
    elif args.first:
        items = filter_items(items, end=args.first)
    elif not args.all and len(items) > 5:
        print(f"Warning: {len(items)} items found. Use --all to process all, or --ids/--range/--first to select.")
        items = items[:2]
        print(f"Processing first 2 items by default")
    
    print(f"Will process {len(items)} items")
    
    print(f"\nLoading model {args.model} on {args.device}")
    pipe = build_pipeline(args.model, args.device)
    
    all_results = []
    for item in items:
        result = process_item(
            item, pipe, cache_dir, output_dir,
            args.method, args.device, args.num_samples,
            args.k_global, args.m_global, args.k_local,
            args.steps, args.cfg
        )
        all_results.append(result)
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDone! Results saved to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

