"""
EACPS refinement pipeline for character face replacement.
Multi-stage search with mask-aware blending.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Disable flash attention before imports
os.environ["DIFFUSERS_USE_FLASH_ATTENTION"] = "0"
os.environ["XFORMERS_DISABLED"] = "1"

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from PIL import Image
from tqdm import tqdm

from compositor import composite_face_to_mask, blend_with_mask
from config import EACPSConfig, ModelConfig


def _setup_flash_attn_mock():
    """Mock flash_attn to prevent import errors."""
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

_setup_flash_attn_mock()

from diffusers import QwenImageEditPipeline
from scorers import EditScorer


@dataclass
class Candidate:
    """A single candidate result from EACPS."""
    image: Image.Image
    seed: int
    phase: str  # "global" or "local"
    scores: Dict[str, float]
    potential: float
    parent_seed: Optional[int] = None


def build_pipeline(model_id: str, device: str) -> QwenImageEditPipeline:
    """Load Qwen-Image-Edit pipeline."""
    if "cuda" in device and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device '{device}' requested but CUDA is not available")
    
    use_cuda = "cuda" in device and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    
    print(f"Loading model {model_id} on {device}...")
    pipe = QwenImageEditPipeline.from_pretrained(model_id, torch_dtype=dtype)
    
    if use_cuda:
        torch.cuda.empty_cache()
        pipe.to(device)
        torch.cuda.empty_cache()
    
    return pipe


def build_scorer(device: str) -> EditScorer:
    """Build multi-metric scorer."""
    return EditScorer(device=device, use_lpips=True)


def generate_edit(
    pipe: QwenImageEditPipeline,
    image: Image.Image,
    prompt: str,
    seed: int,
    config: ModelConfig,
    device: str,
) -> Image.Image:
    """Generate a single edit with the given seed."""
    gen_device = device if device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed)
    
    width, height = image.size
    with torch.inference_mode():
        out = pipe(
            image=image,
            prompt=prompt,
            generator=generator,
            true_cfg_scale=config.true_cfg_scale,
            negative_prompt=config.negative_prompt,
            num_inference_steps=config.num_inference_steps,
            height=height,
            width=width,
        )
    return out.images[0]


def compute_potential(
    scores: Dict[str, float],
    config: EACPSConfig,
) -> float:
    """Compute potential score from metrics."""
    pf = scores.get("PF", 0.0)
    cons = scores.get("CONS", 0.0)
    lpips_val = scores.get("LPIPS", 0.0)
    
    potential = pf + config.alpha_cons * cons
    if config.beta_lpips != 0.0:
        potential -= config.beta_lpips * lpips_val
    
    return potential


def run_eacps_refinement(
    composited: Image.Image,
    init_image: Image.Image,
    mask: Image.Image,
    prompt: str,
    pipe: QwenImageEditPipeline,
    scorer: EditScorer,
    eacps_config: EACPSConfig,
    model_config: ModelConfig,
    device: str,
    verbose: bool = True,
) -> Tuple[Image.Image, List[Candidate]]:
    """
    Run EACPS multi-stage refinement.
    
    Args:
        composited: Pre-composited image (character in mask region)
        init_image: Original base image (for blending output)
        mask: Mask image defining edit region
        prompt: Edit prompt for refinement
        pipe: Qwen-Image-Edit pipeline
        scorer: Multi-metric scorer
        eacps_config: EACPS hyperparameters
        model_config: Model inference parameters
        device: CUDA device
        verbose: Whether to print progress
    
    Returns:
        Tuple of (best_result, all_candidates)
    """
    k_global = eacps_config.k_global
    m_global = eacps_config.m_global
    k_local = eacps_config.k_local
    
    # Stage 1: Global exploration
    if verbose:
        print(f"  Stage 1: Global exploration ({k_global} candidates)")
    
    global_candidates: List[Candidate] = []
    iterator = tqdm(range(k_global), desc="  Global", leave=False) if verbose else range(k_global)
    
    for i in iterator:
        seed = i * 31 + 5000
        raw_result = generate_edit(
            pipe=pipe,
            image=composited,
            prompt=prompt,
            seed=seed,
            config=model_config,
            device=device,
        )
        
        # Blend: keep ONLY masked area from output
        result = blend_with_mask(raw_result, init_image, mask)
        
        # Score
        scores_list = scorer.score_batch(
            images=[result],
            originals=[init_image],
            prompts=[prompt],
        )
        scores = scores_list[0]
        potential = compute_potential(scores, eacps_config)
        
        global_candidates.append(Candidate(
            image=result,
            seed=seed,
            phase="global",
            scores=scores,
            potential=potential,
        ))
        
        if verbose:
            print(f"    Global {i+1}/{k_global}: PF={scores.get('PF', 0):.2f}, "
                  f"CONS={scores.get('CONS', 0):.4f}, LPIPS={scores.get('LPIPS', 0):.4f}, "
                  f"POT={potential:.2f}")
    
    # Rank by potential
    global_candidates.sort(key=lambda c: c.potential, reverse=True)
    top_candidates = global_candidates[:m_global]
    
    if verbose:
        print(f"  Selected top {m_global} for local refinement")
    
    # Stage 2: Local refinement
    if verbose:
        print(f"  Stage 2: Local refinement ({k_local} per candidate)")
    
    local_candidates: List[Candidate] = []
    
    for j, parent in enumerate(top_candidates):
        for k in range(k_local):
            child_seed = parent.seed * 10 + k + 1
            
            raw_result = generate_edit(
                pipe=pipe,
                image=composited,
                prompt=prompt,
                seed=child_seed,
                config=model_config,
                device=device,
            )
            
            # Blend
            result = blend_with_mask(raw_result, init_image, mask)
            
            # Score
            scores_list = scorer.score_batch(
                images=[result],
                originals=[init_image],
                prompts=[prompt],
            )
            scores = scores_list[0]
            potential = compute_potential(scores, eacps_config)
            
            local_candidates.append(Candidate(
                image=result,
                seed=child_seed,
                phase="local",
                scores=scores,
                potential=potential,
                parent_seed=parent.seed,
            ))
            
            if verbose:
                print(f"    Local {j+1}.{k+1}: PF={scores.get('PF', 0):.2f}, "
                      f"CONS={scores.get('CONS', 0):.4f}, LPIPS={scores.get('LPIPS', 0):.4f}, "
                      f"POT={potential:.2f}")
    
    # Final selection
    all_candidates = global_candidates + local_candidates
    all_candidates.sort(key=lambda c: c.potential, reverse=True)
    best = all_candidates[0]
    
    if verbose:
        print(f"  Best: phase={best.phase}, seed={best.seed}, potential={best.potential:.2f}")
    
    return best.image, all_candidates


def process_single_task(
    init_image: Image.Image,
    mask_image: Image.Image,
    character_image: Image.Image,
    prompt: str,
    pipe: QwenImageEditPipeline,
    scorer: EditScorer,
    eacps_config: EACPSConfig,
    model_config: ModelConfig,
    device: str,
    output_dir: Optional[Path] = None,
    task_id: str = "unknown",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Process a single task: composite + EACPS refinement.
    
    Returns dict with results and metrics.
    """
    if verbose:
        print(f"\nProcessing task {task_id}")
    
    # Step 1: Composite character into mask region
    if verbose:
        print("  Compositing character into mask region...")
    composited = composite_face_to_mask(init_image, character_image, mask_image)
    
    # Step 2: Build refinement prompt
    edit_prompt = f"Blend character naturally into the scene, maintain lighting consistency, photorealistic, seamless integration. {prompt}"
    
    # Step 3: Run EACPS refinement
    best_result, all_candidates = run_eacps_refinement(
        composited=composited,
        init_image=init_image,
        mask=mask_image,
        prompt=edit_prompt,
        pipe=pipe,
        scorer=scorer,
        eacps_config=eacps_config,
        model_config=model_config,
        device=device,
        verbose=verbose,
    )
    
    # Collect results
    best_cand = all_candidates[0]
    result = {
        "task_id": task_id,
        "best_seed": best_cand.seed,
        "best_phase": best_cand.phase,
        "best_potential": best_cand.potential,
        "best_scores": best_cand.scores,
        "total_candidates": len(all_candidates),
    }
    
    # Save outputs if output_dir provided
    if output_dir is not None:
        task_dir = output_dir / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        
        init_image.save(task_dir / "init.png")
        mask_image.save(task_dir / "mask.png")
        character_image.convert("RGB").save(task_dir / "character.png")
        composited.save(task_dir / "composite.png")
        best_result.save(task_dir / "result.png")
        
        # Save metrics
        import json
        with open(task_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        
        result["output_dir"] = str(task_dir)
    
    result["result_image"] = best_result
    result["composite_image"] = composited
    
    return result
