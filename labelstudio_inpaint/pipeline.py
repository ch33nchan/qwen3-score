"""
EACPS refinement pipeline for character face replacement.
Uses inpainting with character identity reference - NOT compositing.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

# Disable flash attention before imports
os.environ["DIFFUSERS_USE_FLASH_ATTENTION"] = "0"
os.environ["XFORMERS_DISABLED"] = "1"

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

from config import EACPSConfig, ModelConfig

logger = logging.getLogger(__name__)


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
    phase: str
    scores: Dict[str, float]
    potential: float
    parent_seed: Optional[int] = None


def build_pipeline(model_id: str, device: str) -> QwenImageEditPipeline:
    """Load Qwen-Image-Edit pipeline."""
    if "cuda" in device and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device '{device}' requested but CUDA is not available")
    
    use_cuda = "cuda" in device and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    
    logger.info(f"Loading model {model_id} on {device}...")
    pipe = QwenImageEditPipeline.from_pretrained(model_id, torch_dtype=dtype)
    
    if use_cuda:
        torch.cuda.empty_cache()
        pipe.to(device)
        torch.cuda.empty_cache()
    
    return pipe


def build_scorer(device: str) -> EditScorer:
    """Build multi-metric scorer."""
    return EditScorer(device=device, use_lpips=True)


def prepare_masked_image(
    init_image: Image.Image,
    mask_image: Image.Image,
    noise_strength: float = 0.5,
) -> Image.Image:
    """
    Prepare init image with masked region filled with noise/blur.
    This helps the model understand where to generate.
    """
    init_rgb = init_image.convert("RGB")
    mask_resized = mask_image.convert("L").resize(init_rgb.size, Image.Resampling.LANCZOS)
    
    # Create blurred version of init for masked region
    init_blurred = init_rgb.filter(ImageFilter.GaussianBlur(radius=20))
    
    # Create noise
    init_array = np.array(init_rgb)
    noise = np.random.randint(0, 256, init_array.shape, dtype=np.uint8)
    
    # Mix blur and noise
    mixed = (init_array.astype(float) * (1 - noise_strength) + 
             noise.astype(float) * noise_strength).astype(np.uint8)
    mixed_img = Image.fromarray(mixed)
    
    # Apply only in masked region
    mask_array = np.array(mask_resized) / 255.0
    mask_3d = np.stack([mask_array] * 3, axis=-1)
    
    result_array = (np.array(mixed_img) * mask_3d + 
                    np.array(init_rgb) * (1 - mask_3d)).astype(np.uint8)
    
    return Image.fromarray(result_array)


def blend_with_mask(
    output_img: Image.Image,
    original_img: Image.Image,
    mask_img: Image.Image,
    feather_radius: int = 10,
) -> Image.Image:
    """Blend output with original - keep output ONLY in masked area."""
    if output_img.size != original_img.size:
        output_img = output_img.resize(original_img.size, Image.Resampling.LANCZOS)
    
    mask_resized = mask_img.convert("L").resize(original_img.size, Image.Resampling.LANCZOS)
    
    # Feather mask edges
    mask_feathered = mask_resized.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    
    output_rgb = output_img.convert("RGB")
    original_rgb = original_img.convert("RGB")
    
    return Image.composite(output_rgb, original_rgb, mask_feathered)


def describe_character(character_image: Image.Image) -> str:
    """
    Generate a text description of the character's facial features.
    This is used as the prompt for face generation.
    """
    # For now, use a generic description
    # In production, you could use BLIP/LLaVA to generate this
    return (
        "person with natural facial features, realistic skin texture, "
        "natural lighting, photorealistic face"
    )


def create_side_by_side(
    init_image: Image.Image,
    character_image: Image.Image,
) -> Image.Image:
    """
    Create side-by-side reference image: [character_face | init_image]
    This gives the model both the target identity and the scene context.
    """
    # Resize character to match init height
    char_h = init_image.height
    char_w = int(character_image.width * (char_h / character_image.height))
    char_resized = character_image.convert("RGB").resize(
        (char_w, char_h), Image.Resampling.LANCZOS
    )
    
    # Create side-by-side
    total_w = char_w + init_image.width
    combined = Image.new("RGB", (total_w, char_h))
    combined.paste(char_resized, (0, 0))
    combined.paste(init_image.convert("RGB"), (char_w, 0))
    
    return combined


def generate_inpaint(
    pipe: QwenImageEditPipeline,
    init_image: Image.Image,
    character_image: Image.Image,
    mask_image: Image.Image,
    prompt: str,
    seed: int,
    config: ModelConfig,
    device: str,
) -> Image.Image:
    """
    Generate face using inpainting with character reference.
    
    Strategy: Create a reference image showing the character, 
    then ask the model to edit the init image's masked region
    to match the character's face.
    """
    gen_device = device if device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed)
    
    # Create side-by-side reference
    reference = create_side_by_side(init_image, character_image)
    
    # Full prompt for face replacement
    full_prompt = (
        f"Edit the right image: replace the face in the masked region with "
        f"the face from the left reference image. Match the identity, facial features, "
        f"skin tone from the left person. Keep the pose, lighting, and composition "
        f"from the right image. Blend naturally, photorealistic result. {prompt}"
    )
    
    width, height = reference.size
    
    with torch.inference_mode():
        out = pipe(
            image=reference,
            prompt=full_prompt,
            generator=generator,
            true_cfg_scale=config.true_cfg_scale,
            negative_prompt=config.negative_prompt,
            num_inference_steps=config.num_inference_steps,
            height=height,
            width=width,
        )
    
    result = out.images[0]
    
    # Extract only the right half (the edited init image)
    char_w = int(character_image.width * (init_image.height / character_image.height))
    result_cropped = result.crop((char_w, 0, result.width, result.height))
    
    # Resize to match init exactly
    if result_cropped.size != init_image.size:
        result_cropped = result_cropped.resize(init_image.size, Image.Resampling.LANCZOS)
    
    return result_cropped


def generate_direct_edit(
    pipe: QwenImageEditPipeline,
    init_image: Image.Image,
    character_image: Image.Image,
    mask_image: Image.Image,
    prompt: str,
    seed: int,
    config: ModelConfig,
    device: str,
) -> Image.Image:
    """
    Alternative: Direct edit on init image with character description.
    """
    gen_device = device if device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed)
    
    # Prepare init with masked region hinted
    masked_init = prepare_masked_image(init_image, mask_image, noise_strength=0.3)
    
    # Build prompt describing the character
    char_desc = describe_character(character_image)
    full_prompt = (
        f"Replace the face/head in the center of this image with a new face: "
        f"{char_desc}. Match the lighting and style of the surrounding image. "
        f"Seamless blend, photorealistic. {prompt}"
    )
    
    width, height = init_image.size
    
    with torch.inference_mode():
        out = pipe(
            image=masked_init,
            prompt=full_prompt,
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
    init_image: Image.Image,
    character_image: Image.Image,
    mask_image: Image.Image,
    prompt: str,
    pipe: QwenImageEditPipeline,
    scorer: EditScorer,
    eacps_config: EACPSConfig,
    model_config: ModelConfig,
    device: str,
    verbose: bool = True,
) -> Tuple[Image.Image, List[Candidate]]:
    """
    Run EACPS multi-stage refinement with inpainting approach.
    """
    k_global = eacps_config.k_global
    m_global = eacps_config.m_global
    k_local = eacps_config.k_local
    
    # Stage 1: Global exploration
    if verbose:
        print(f"  Stage 1: Global exploration ({k_global} candidates)")
    
    global_candidates: List[Candidate] = []
    
    for i in range(k_global):
        seed = i * 31 + 5000
        
        # Generate using inpainting approach
        raw_result = generate_inpaint(
            pipe=pipe,
            init_image=init_image,
            character_image=character_image,
            mask_image=mask_image,
            prompt=prompt,
            seed=seed,
            config=model_config,
            device=device,
        )
        
        # Blend: enforce mask boundary
        result = blend_with_mask(raw_result, init_image, mask_image)
        
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
            
            raw_result = generate_inpaint(
                pipe=pipe,
                init_image=init_image,
                character_image=character_image,
                mask_image=mask_image,
                prompt=prompt,
                seed=child_seed,
                config=model_config,
                device=device,
            )
            
            result = blend_with_mask(raw_result, init_image, mask_image)
            
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
    Process a single task using inpainting with character identity reference.
    """
    if verbose:
        print(f"\nProcessing task {task_id}")
    
    # Build edit prompt
    edit_prompt = f"Replace face with character identity, natural blending. {prompt}"
    
    # Run EACPS refinement with inpainting
    if verbose:
        print("  Running EACPS with inpainting approach...")
    
    best_result, all_candidates = run_eacps_refinement(
        init_image=init_image,
        character_image=character_image,
        mask_image=mask_image,
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
    
    # Save outputs
    if output_dir is not None:
        task_dir = output_dir / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        
        init_image.save(task_dir / "init.png")
        mask_image.save(task_dir / "mask.png")
        character_image.convert("RGB").save(task_dir / "character.png")
        
        # Save the side-by-side reference used
        reference = create_side_by_side(init_image, character_image)
        reference.save(task_dir / "reference.png")
        
        best_result.save(task_dir / "result.png")
        
        import json
        with open(task_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        
        result["output_dir"] = str(task_dir)
    
    result["result_image"] = best_result
    
    return result
