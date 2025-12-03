#!/usr/bin/env python3
"""
Batch comparison: Run TT-FLUX vs EACPS on multiple prompts.
Supports multi-GPU processing.

Usage:
    python batch_compare.py --prompts data/eval_prompts.jsonl --output_dir results/batch_eval --devices cuda:0,cuda:2
"""
import os
import sys

os.environ["DIFFUSERS_USE_FLASH_ATTENTION"] = "0"
os.environ["XFORMERS_DISABLED"] = "1"

try:
    import flash_attn.flash_attn_interface as fai
    if not hasattr(fai, '_wrapped_flash_attn_backward'):
        fai._wrapped_flash_attn_backward = lambda *args, **kwargs: None
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
import subprocess
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def load_prompts(prompts_file: str) -> List[Dict]:
    prompts = []
    with open(prompts_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def run_single_comparison(
    prompt_data: Dict,
    data_dir: Path,
    output_dir: Path,
    device: str,
    num_samples: int,
    k_global: int,
    m_global: int,
    k_local: int,
    steps: int,
    cfg: float,
) -> Dict:
    """Run a single comparison and return results."""
    prompt_id = prompt_data["id"]
    image_path = data_dir / prompt_data["image"]
    prompt = prompt_data["prompt"]
    task_output_dir = output_dir / prompt_id
    
    cmd = [
        sys.executable, "compare.py",
        "--image", str(image_path),
        "--prompt", prompt,
        "--output_dir", str(task_output_dir),
        "--device", device,
        "--num_samples", str(num_samples),
        "--k_global", str(k_global),
        "--m_global", str(m_global),
        "--k_local", str(k_local),
        "--steps", str(steps),
        "--cfg", str(cfg),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            text=True,
            cwd=str(Path(__file__).parent),
        )
        
        if result.returncode != 0:
            return {
                "id": prompt_id,
                "status": "error",
                "error": "Process failed with non-zero exit code",
            }
        
        results_file = task_output_dir / "comparison_results.json"
        if results_file.exists():
            with open(results_file, "r") as f:
                comparison = json.load(f)
            return {
                "id": prompt_id,
                "status": "success",
                "output_dir": str(task_output_dir),
                "comparison": comparison.get("comparison", {}),
            }
        else:
            return {
                "id": prompt_id,
                "status": "error",
                "error": "Results file not found",
            }
    except Exception as e:
        return {
            "id": prompt_id,
            "status": "error",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Batch comparison: TT-FLUX vs EACPS")
    parser.add_argument("--prompts", type=str, required=True, help="Prompts JSONL file")
    parser.add_argument("--output_dir", type=str, default="results/batch_eval", help="Output directory")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory with images")
    parser.add_argument("--devices", type=str, default="cuda:0", help="Comma-separated GPU devices")
    parser.add_argument("--num_samples", type=int, default=4, help="TT-FLUX candidates")
    parser.add_argument("--k_global", type=int, default=4, help="EACPS k_global")
    parser.add_argument("--m_global", type=int, default=2, help="EACPS m_global")
    parser.add_argument("--k_local", type=int, default=2, help="EACPS k_local")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers (default: num devices)")
    args = parser.parse_args()
    
    prompts = load_prompts(args.prompts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    
    devices = [d.strip() for d in args.devices.split(",")]
    num_workers = args.workers or len(devices)
    
    print("=" * 70)
    print("BATCH COMPARISON: TT-FLUX vs EACPS")
    print("=" * 70)
    print(f"Prompts: {len(prompts)}")
    print(f"Devices: {devices}")
    print(f"Workers: {num_workers}")
    print(f"Output: {output_dir}")
    print()
    
    results = []
    
    # Process sequentially with device rotation for stability
    for i, prompt_data in enumerate(tqdm(prompts, desc="Processing")):
        device = devices[i % len(devices)]
        print(f"\n[{i+1}/{len(prompts)}] {prompt_data['id']} on {device}")
        print(f"  Image: {prompt_data['image']}")
        print(f"  Prompt: {prompt_data['prompt'][:50]}...")
        
        result = run_single_comparison(
            prompt_data=prompt_data,
            data_dir=data_dir,
            output_dir=output_dir,
            device=device,
            num_samples=args.num_samples,
            k_global=args.k_global,
            m_global=args.m_global,
            k_local=args.k_local,
            steps=args.steps,
            cfg=args.cfg,
        )
        results.append(result)
        
        if result["status"] == "success":
            comp = result.get("comparison", {})
            wins = sum(1 for v in comp.values() if v.get("winner") == "EACPS")
            print(f"  Result: EACPS wins {wins}/{len(comp)} metrics")
        else:
            print(f"  Error: {result.get('error', 'Unknown')[:100]}")
    
    # Save batch results
    batch_results_file = output_dir / "batch_results.json"
    with open(batch_results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        # Aggregate wins
        eacps_wins = {"clip_score": 0, "aesthetic": 0, "lpips": 0}
        ttflux_wins = {"clip_score": 0, "aesthetic": 0, "lpips": 0}
        
        for r in successful:
            comp = r.get("comparison", {})
            for metric in ["clip_score", "aesthetic", "lpips"]:
                if metric in comp:
                    if comp[metric].get("winner") == "EACPS":
                        eacps_wins[metric] += 1
                    else:
                        ttflux_wins[metric] += 1
        
        print("\nAggregate wins across all prompts:")
        print(f"  CLIPScore:  EACPS {eacps_wins['clip_score']} / TT-FLUX {ttflux_wins['clip_score']}")
        print(f"  Aesthetic:  EACPS {eacps_wins['aesthetic']} / TT-FLUX {ttflux_wins['aesthetic']}")
        print(f"  LPIPS:      EACPS {eacps_wins['lpips']} / TT-FLUX {ttflux_wins['lpips']}")
        
        total_eacps = sum(eacps_wins.values())
        total_ttflux = sum(ttflux_wins.values())
        print(f"\nOverall: EACPS {total_eacps} / TT-FLUX {total_ttflux}")
    
    print(f"\nResults saved to: {batch_results_file}")
    print(f"Individual results in: {output_dir}/")
    
    # Generate Label Studio tasks
    print("\nGenerating Label Studio tasks...")
    task_dirs = [output_dir / r["id"] for r in successful]
    if task_dirs:
        from setup_labelstudio import create_labelstudio_tasks
        ls_output = output_dir / "labelstudio_export"
        tasks = create_labelstudio_tasks(task_dirs, ls_output, randomize_ab=False)
        print(f"Label Studio tasks: {ls_output / 'tasks.json'}")


if __name__ == "__main__":
    main()

