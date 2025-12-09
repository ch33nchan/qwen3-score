"""
Inpainting pipeline with EACPS inference scaling.
Uses Qwen-Edit for generation and Gemini/Moondream for scoring.
"""
import os
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image, ImageFilter
import numpy as np

from config import EACPSConfig, ModelConfig, PipelineConfig
from scorers import MultiModelScorer

logger = logging.getLogger(__name__)

# Disable flash attention
os.environ["DIFFUSERS_USE_FLASH_ATTENTION"] = "0"


@dataclass
class Candidate:
    """A single candidate from EACPS."""
    image: Image.Image
    seed: int
    phase: str
    scores: Dict[str, float]
    potential: float
    parent_seed: Optional[int] = None


def _setup_flash_attn_mock():
    """Mock flash_attn to prevent import errors."""
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


_setup_flash_attn_mock()


def detect_face_region(image: Image.Image) -> tuple:
    """
    Detect face region in character image.
    Returns (x, y, width, height) or None if no face found.
    """
    try:
        import cv2
        import numpy as np
        
        # Convert PIL to OpenCV
        img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Use OpenCV's Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        if len(faces) > 0:
            # Take the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Expand region slightly for context (20% padding)
            padding = int(0.2 * max(w, h))
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.width - x, w + 2 * padding)
            h = min(image.height - y, h + 2 * padding)
            
            return (x, y, w, h)
    except:
        pass
    
    return None


def composite_character_face_on_mask(
    init_image: Image.Image,
    character_image: Image.Image,
    mask_image: Image.Image,
) -> Image.Image:
    """
    Composite character face onto init image using mask.
    Uses face detection to extract the exact face region from character image.
    """
    init_image = init_image.convert("RGBA")
    character_image = character_image.convert("RGBA")
    mask_image = mask_image.convert("L")
    
    if mask_image.size != init_image.size:
        mask_image = mask_image.resize(init_image.size, Image.Resampling.LANCZOS)
    
    # Find mask bounding box
    mask_array = np.array(mask_image)
    coords = np.column_stack(np.where(mask_array > 128))
    if len(coords) == 0:
        return init_image.convert("RGB")
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    mask_width = x_max - x_min
    mask_height = y_max - y_min
    
    # Try to detect face in character image
    face_region = detect_face_region(character_image)
    
    if face_region:
        # Use detected face region
        fx, fy, fw, fh = face_region
        character_face = character_image.crop((fx, fy, fx + fw, fy + fh))
        logger.info(f"Face detected at ({fx}, {fy}, {fw}, {fh})")
    else:
        # Fallback: use center-weighted crop favoring upper portion
        char_width, char_height = character_image.size
        # Take center 60% width, top 50% height
        crop_width = int(char_width * 0.6)
        crop_height = int(char_height * 0.5)
        left = (char_width - crop_width) // 2
        character_face = character_image.crop((left, 0, left + crop_width, crop_height))
        logger.warning("No face detected, using fallback crop")
    
    # Resize to match mask dimensions
    character_face = character_face.resize((mask_width, mask_height), Image.Resampling.LANCZOS)
    
    # Paste character face at mask location
    result = init_image.copy()
    result.paste(character_face, (x_min, y_min), mask_image.crop((x_min, y_min, x_max, y_max)))
    
    return result.convert("RGB")


class QwenEditPipeline:
    """Wrapper for Qwen-Image-Edit pipeline."""
    
    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device
        self._pipe = None
    
    def _load(self):
        if self._pipe is None:
            from diffusers import QwenImageEditPipeline
            
            logger.info(f"Loading {self.model_id} on {self.device}...")
            dtype = torch.bfloat16 if "cuda" in self.device else torch.float32
            
            self._pipe = QwenImageEditPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
            )
            
            if "cuda" in self.device:
                torch.cuda.empty_cache()
                self._pipe.to(self.device)
                torch.cuda.empty_cache()
            
            logger.info("Model loaded")
        return self._pipe
    
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        seed: int,
        num_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: str = "",
    ) -> Image.Image:
        """Generate a single edit."""
        pipe = self._load()
        
        gen_device = self.device if "cuda" in self.device else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(seed)
        
        width, height = image.size
        
        with torch.inference_mode():
            result = pipe(
                image=image,
                prompt=prompt,
                generator=generator,
                true_cfg_scale=guidance_scale,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=num_steps,
                height=height,
                width=width,
            )
        
        return result.images[0]


def blend_with_mask(
    output: Image.Image,
    original: Image.Image,
    mask: Image.Image,
    feather_radius: int = 10,
) -> Image.Image:
    """Blend output with original using mask."""
    if output.size != original.size:
        output = output.resize(original.size, Image.Resampling.LANCZOS)
    
    mask_resized = mask.convert("L").resize(original.size, Image.Resampling.LANCZOS)
    
    if feather_radius > 0:
        mask_feathered = mask_resized.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    else:
        mask_feathered = mask_resized
    
    return Image.composite(output.convert("RGB"), original.convert("RGB"), mask_feathered)


def create_inpaint_prompt(character_name: str) -> str:
    """Create prompt for face inpainting refinement (character already composited)."""
    return (
        f"Refine and blend the face naturally into the scene. "
        f"Maintain the exact facial features and appearance. "
        f"Match lighting perfectly - shadows, highlights, and color temperature must be consistent. "
        f"The result must look like an unedited, real photograph. "
        f"Natural skin texture, realistic hair, proper shadows. "
        f"Seamless integration with no visible seams or boundaries. "
        f"RAW photo, DSLR quality, photorealistic."
    )


def run_eacps_inpaint(
    init_image: Image.Image,
    mask_image: Image.Image,
    character_image: Image.Image,
    character_name: str,
    qwen_pipe: QwenEditPipeline,
    scorer: MultiModelScorer,
    config: PipelineConfig,
    verbose: bool = True,
) -> Tuple[Image.Image, List[Candidate]]:
    """
    Run EACPS inference scaling for inpainting.
    
    Returns (best_result, all_candidates)
    """
    eacps = config.eacps
    model = config.model
    
    prompt = create_inpaint_prompt(character_name)
    
    # Stage 1: Global exploration
    if verbose:
        print(f"  Stage 1: Global exploration ({eacps.k_global} candidates)")
    
    global_candidates: List[Candidate] = []
    
    # Pre-composite character face onto masked region
    composited_input = composite_character_face_on_mask(init_image, character_image, mask_image)
    
    for i in range(eacps.k_global):
        seed = i * 31 + 5000
        
        if verbose:
            print(f"    Generating candidate {i+1}/{eacps.k_global} (seed={seed})...")
        
        # Generate from composited image (character face already placed)
        raw_result = qwen_pipe.generate(
            image=composited_input,
            prompt=prompt,
            seed=seed,
            num_steps=model.num_inference_steps,
            guidance_scale=model.guidance_scale,
            negative_prompt=model.negative_prompt,
        )
        
        # Blend with mask
        result = blend_with_mask(raw_result, init_image, mask_image)
        
        # Score with multi-model scorer
        scores = scorer.score(result, character_image, init_image)
        potential = scorer.compute_potential(scores, eacps)
        
        global_candidates.append(Candidate(
            image=result,
            seed=seed,
            phase="global",
            scores=scores,
            potential=potential,
        ))
        
        if verbose:
            print(f"      Scores: {scores}")
            print(f"      Potential: {potential:.2f}")
    
    # Rank by potential
    global_candidates.sort(key=lambda c: c.potential, reverse=True)
    top_candidates = global_candidates[:eacps.m_global]
    
    if verbose:
        print(f"  Selected top {eacps.m_global} for local refinement")
    
    # Stage 2: Local refinement
    if verbose:
        print(f"  Stage 2: Local refinement ({eacps.k_local} per candidate)")
    
    local_candidates: List[Candidate] = []
    
    for j, parent in enumerate(top_candidates):
        for k in range(eacps.k_local):
            child_seed = parent.seed * 10 + k + 1
            
            if verbose:
                print(f"    Refining {j+1}.{k+1} (seed={child_seed})...")
            
            # Use composited input (character face already placed)
            raw_result = qwen_pipe.generate(
                image=composited_input,
                prompt=prompt,
                seed=child_seed,
                num_steps=model.num_inference_steps,
                guidance_scale=model.guidance_scale,
                negative_prompt=model.negative_prompt,
            )
            
            result = blend_with_mask(raw_result, init_image, mask_image)
            
            scores = scorer.score(result, character_image, init_image)
            potential = scorer.compute_potential(scores, eacps)
            
            local_candidates.append(Candidate(
                image=result,
                seed=child_seed,
                phase="local",
                scores=scores,
                potential=potential,
                parent_seed=parent.seed,
            ))
            
            if verbose:
                print(f"      Potential: {potential:.2f}")
    
    # Final selection
    all_candidates = global_candidates + local_candidates
    all_candidates.sort(key=lambda c: c.potential, reverse=True)
    best = all_candidates[0]
    
    if verbose:
        print(f"  Best: phase={best.phase}, seed={best.seed}, potential={best.potential:.2f}")
    
    return best.image, all_candidates


def process_task(
    init_image: Image.Image,
    mask_image: Image.Image,
    character_image: Image.Image,
    character_name: str,
    task_id: str,
    config: PipelineConfig,
    output_dir: Optional[Path] = None,
    use_moondream: bool = True,
) -> Dict[str, Any]:
    """
    Process a single inpainting task.
    """
    logger.info(f"Processing task {task_id} ({character_name})")
    
    # Initialize pipeline
    qwen_pipe = QwenEditPipeline(config.model.qwen_model_id, config.device)
    
    # Initialize scorer with Moondream enabled for better realism scoring
    scorer = MultiModelScorer(
        gemini_api_key=config.model.gemini_api_key,
        gemini_model=config.model.gemini_model,
        moondream_model_id=config.model.moondream_model_id,
        device=config.device,
        use_gemini=bool(config.model.gemini_api_key),
        use_moondream=use_moondream,  # Enabled for better realism and identity scoring
    )
    
    # Run EACPS
    best_result, all_candidates = run_eacps_inpaint(
        init_image=init_image,
        mask_image=mask_image,
        character_image=character_image,
        character_name=character_name,
        qwen_pipe=qwen_pipe,
        scorer=scorer,
        config=config,
        verbose=True,
    )
    
    # Prepare result
    best_cand = all_candidates[0]
    result = {
        "task_id": task_id,
        "character_name": character_name,
        "best_seed": best_cand.seed,
        "best_phase": best_cand.phase,
        "best_potential": best_cand.potential,
        "best_scores": best_cand.scores,
        "total_candidates": len(all_candidates),
    }
    
    # Save outputs
    if output_dir:
        task_dir = output_dir / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        
        init_image.save(task_dir / "init.png")
        mask_image.save(task_dir / "mask.png")
        character_image.convert("RGB").save(task_dir / "character.png")
        best_result.save(task_dir / "result.png")
        
        # Save composited reference for debugging
        composited_ref = composite_character_face_on_mask(init_image, character_image, mask_image)
        composited_ref.save(task_dir / "composited_reference.png")
        
        # Save all candidates
        for i, cand in enumerate(all_candidates[:5]):  # Top 5
            cand.image.save(task_dir / f"candidate_{i+1}_seed{cand.seed}.png")
        
        import json
        with open(task_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        
        result["output_dir"] = str(task_dir)
    
    result["result_image"] = best_result
    
    return result
