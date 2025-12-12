"""
Inpainting pipeline with EACPS inference scaling.
Uses InsightFace for face swap and Qwen-Edit for refinement.
Gemini/Moondream for scoring.
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
import cv2

from config import EACPSConfig, ModelConfig, PipelineConfig
from scorers import MultiModelScorer

logger = logging.getLogger(__name__)

# Disable flash attention
os.environ["DIFFUSERS_USE_FLASH_ATTENTION"] = "0"

# InsightFace global cache
_FACE_ANALYZER = None
_FACE_SWAPPER = None


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


def _init_insightface():
    """Initialize InsightFace models."""
    global _FACE_ANALYZER, _FACE_SWAPPER
    
    if _FACE_ANALYZER is not None:
        return _FACE_ANALYZER, _FACE_SWAPPER
    
    try:
        import insightface
        from insightface.app import FaceAnalysis
        
        logger.info("Loading InsightFace models...")
        _FACE_ANALYZER = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        _FACE_ANALYZER.prepare(ctx_id=0, det_size=(640, 640))
        
        model_path = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")
        
        if not os.path.exists(model_path):
            logger.error(f"Face swapper model not found at {model_path}")
            logger.error("Download manually:")
            logger.error("  1. Go to https://huggingface.co/ezioruan/inswapper_128.onnx/tree/main")
            logger.error("  2. Download inswapper_128.onnx")
            logger.error(f"  3. Place at {model_path}")
            _FACE_SWAPPER = None
        else:
            file_size = os.path.getsize(model_path)
            if file_size < 1_000_000:
                logger.error(f"Face swapper model at {model_path} is too small ({file_size} bytes) - likely corrupt")
                logger.error("Download manually:")
                logger.error("  1. Go to https://huggingface.co/ezioruan/inswapper_128.onnx/tree/main")
                logger.error("  2. Download inswapper_128.onnx (~500MB)")
                logger.error(f"  3. Place at {model_path}")
                _FACE_SWAPPER = None
            else:
                _FACE_SWAPPER = insightface.model_zoo.get_model(model_path)
        
        logger.info("InsightFace models loaded")
        return _FACE_ANALYZER, _FACE_SWAPPER
        
    except ImportError:
        logger.error("InsightFace not installed. Run: pip install insightface onnxruntime-gpu")
        return None, None


def swap_face_insightface(
    init_image: Image.Image,
    character_image: Image.Image,
    mask_image: Optional[Image.Image] = None,
) -> Optional[Image.Image]:
    """
    Swap face using InsightFace - preserves exact identity from character.
    """
    analyzer, swapper = _init_insightface()
    
    if analyzer is None or swapper is None:
        logger.error("InsightFace not available")
        return None
    
    # Convert to BGR for InsightFace
    init_array = np.array(init_image.convert("RGB"))
    init_bgr = cv2.cvtColor(init_array, cv2.COLOR_RGB2BGR)
    
    char_array = np.array(character_image.convert("RGB"))
    char_bgr = cv2.cvtColor(char_array, cv2.COLOR_RGB2BGR)
    
    # Detect faces
    init_faces = analyzer.get(init_bgr)
    char_faces = analyzer.get(char_bgr)
    
    if not init_faces:
        logger.error("No face detected in init image")
        return None
    
    if not char_faces:
        logger.error("No face detected in character image")
        return None
    
    # Get source face (from character) and target face (in init image)
    source_face = char_faces[0]
    
    # If mask provided, find face in mask region
    if mask_image is not None:
        mask_array = np.array(mask_image.convert("L").resize(init_image.size, Image.Resampling.LANCZOS))
        mask_points = np.where(mask_array > 128)
        
        if len(mask_points[0]) > 0:
            mask_center_y = mask_points[0].mean()
            mask_center_x = mask_points[1].mean()
            
            # Find face closest to mask center
            best_face = None
            best_dist = float('inf')
            
            for face in init_faces:
                bbox = face.bbox
                face_center_x = (bbox[0] + bbox[2]) / 2
                face_center_y = (bbox[1] + bbox[3]) / 2
                
                dist = np.sqrt((face_center_x - mask_center_x)**2 + (face_center_y - mask_center_y)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_face = face
            
            target_face = best_face or init_faces[0]
        else:
            target_face = init_faces[0]
    else:
        target_face = init_faces[0]
    
    logger.info(f"Face swap: source bbox={source_face.bbox.astype(int).tolist()}, target bbox={target_face.bbox.astype(int).tolist()}")
    
    # Perform swap
    result_bgr = swapper.get(init_bgr, target_face, source_face, paste_back=True)
    
    # Convert back to RGB
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


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


def create_facial_feature_mask(
    face_bbox: np.ndarray,
    image_size: Tuple[int, int],
    include_hair: bool = False,
) -> Image.Image:
    """
    Create precise mask covering only facial features (eyes, nose, mouth, skin).
    Excludes hair region if include_hair=False.
    
    Args:
        face_bbox: [x1, y1, x2, y2] bounding box from face detection
        image_size: (width, height) of target image
        include_hair: If False, exclude top 30% of face bbox (hair region)
    """
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    
    x1, y1, x2, y2 = face_bbox.astype(int)
    
    if not include_hair:
        # Exclude hair: only use lower 70% of face bbox
        face_height = y2 - y1
        y1 = y1 + int(face_height * 0.30)  # Start from 30% down
    
    # Fill facial feature region (excluding hair if specified)
    mask[y1:y2, x1:x2] = 255
    
    return Image.fromarray(mask)


def blend_with_mask(
    output: Image.Image,
    original: Image.Image,
    mask: Image.Image,
    feather_radius: int = 25,
    preserve_texture: bool = True,
) -> Image.Image:
    """
    Blend output with original using mask with feature-preserving feathering.
    
    Args:
        preserve_texture: If True, use minimal blur to preserve skin texture and hair detail
    """
    if output.size != original.size:
        output = output.resize(original.size, Image.Resampling.LANCZOS)
    
    mask_resized = mask.convert("L").resize(original.size, Image.Resampling.LANCZOS)
    
    # Convert to numpy for advanced processing
    mask_array = np.array(mask_resized).astype(float) / 255.0
    
    # Erode mask by 3 pixels to avoid edge artifacts (reduced from 5)
    kernel = np.ones((3, 3), np.uint8)
    mask_binary = (mask_array > 0.5).astype(np.uint8) * 255
    mask_eroded = cv2.erode(mask_binary, kernel, iterations=1)
    
    # Apply feathering only at edges, preserve interior
    if feather_radius > 0:
        # Create distance transform to find mask interior
        dist_transform = cv2.distanceTransform(mask_eroded, cv2.DIST_L2, 5)
        
        # Feather only within 'feather_radius' pixels of edge
        feather_zone = (dist_transform > 0) & (dist_transform < feather_radius)
        
        # Apply Gaussian blur only in feather zone
        mask_float = mask_eroded.astype(float)
        if preserve_texture:
            # Minimal blur for texture preservation
            blurred = cv2.GaussianBlur(mask_float, (0, 0), sigmaX=feather_radius // 3)
        else:
            blurred = cv2.GaussianBlur(mask_float, (0, 0), sigmaX=feather_radius // 2)
        
        # Blend: use blurred in feather zone, original elsewhere
        mask_final = np.where(feather_zone, blurred, mask_float)
        mask_final = np.clip(mask_final / 255.0, 0, 1)
    else:
        mask_final = mask_eroded.astype(float) / 255.0
    
    mask_pil = Image.fromarray((mask_final * 255).astype(np.uint8))
    
    return Image.composite(output.convert("RGB"), original.convert("RGB"), mask_pil)


def create_inpaint_prompt(character_name: str, preserve_init_hair: bool = False) -> str:
    """Create prompt for post-faceswap refinement."""
    if preserve_init_hair:
        # Mode 1: Preserve init image hairstyle
        return (
            f"seamlessly blend the face into the photograph, "
            f"match skin tone and lighting exactly to the scene, "
            f"preserve the original hairstyle and hair texture from the photograph, "
            f"keep exact hair color and style from background, "
            f"natural photorealistic face with real skin pores and texture, "
            f"no soft focus, no airbrushing, no smooth skin"
        )
    else:
        # Mode 2: Use character hairstyle
        return (
            f"seamlessly blend the face and hair into the photograph, "
            f"match lighting and colors to the scene, "
            f"keep character's hairstyle and hair texture, "
            f"natural photorealistic face with real skin pores and texture, "
            f"no soft focus, no airbrushing, no smooth skin, "
            f"preserve facial features exactly"
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
    dual_output: bool = True,
    preserve_init_hair: bool = False,
    override_prompt: Optional[str] = None,
) -> Tuple[Image.Image, List[Candidate], Optional[Image.Image]]:
    """
    Run EACPS inference scaling for inpainting.
    
    Args:
        dual_output: If True, generate both init-hair and char-hair versions
        preserve_init_hair: Which mode to use if dual_output=False
    
    Returns (best_result, all_candidates, optional_second_best_result)
    """
    eacps = config.eacps
    model = config.model
    
    # Allow caller to provide a full custom inpaint prompt.
    if override_prompt:
        prompt = override_prompt
    else:
        prompt = create_inpaint_prompt(character_name, preserve_init_hair=preserve_init_hair)
    
    # Stage 0: Face swap using InsightFace (preserves exact identity)
    if verbose:
        print(f"  Stage 0: InsightFace face swap")
    
    swapped_base = swap_face_insightface(init_image, character_image, mask_image)
    
    if swapped_base is None:
        logger.error("Face swap failed, falling back to compositing")
        swapped_base = composite_character_face_on_mask(init_image, character_image, mask_image)
    else:
        if verbose:
            print(f"    Face swap successful")
    
    # Stage 1: Global exploration - refine the face-swapped result
    if verbose:
        print(f"  Stage 1: Global exploration ({eacps.k_global} candidates)")
    
    global_candidates: List[Candidate] = []
    
    # Pre-composite character face onto masked region
    composited_input = composite_character_face_on_mask(init_image, character_image, mask_image)
    
    # CRITICAL: Add the raw face swap as a candidate (no diffusion)
    # This ensures we always have the "pure" InsightFace result
    swapped_blended = blend_with_mask(swapped_base, init_image, mask_image)
    scores_raw = scorer.score(swapped_blended, character_image, init_image)
    potential_raw = scorer.compute_potential(scores_raw, eacps)
    global_candidates.append(Candidate(
        image=swapped_blended,
        seed=0,
        phase="raw_swap",
        scores=scores_raw,
        potential=potential_raw,
    ))
    if verbose:
        print(f"    Raw face swap: Potential={potential_raw:.2f}")
    
    for i in range(eacps.k_global):
        seed = i * 31 + 5000
        
        if verbose:
            print(f"    Generating candidate {i+1}/{eacps.k_global} (seed={seed})...")
        
        # Generate from face-swapped image (exact identity preserved)
        raw_result = qwen_pipe.generate(
            image=swapped_base,
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
            
            # Use face-swapped base for refinement
            raw_result = qwen_pipe.generate(
                image=swapped_base,
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
    qwen_pipe: Optional['QwenEditPipeline'] = None,
    scorer: Optional[MultiModelScorer] = None,
) -> Dict[str, Any]:
    """
    Process a single inpainting task.
    """
    logger.info(f"Processing task {task_id} ({character_name})")
    
    # Initialize pipeline
    if qwen_pipe is None:
        qwen_pipe = QwenEditPipeline(config.model.qwen_model_id, config.device)
    
    # Initialize scorer with Moondream enabled for better realism scoring
    if scorer is None:
        scorer = MultiModelScorer(
            gemini_api_key=config.model.gemini_api_key,
            gemini_model=config.model.gemini_model,
            moondream_model_id=config.model.moondream_model_id,
            device=config.device,
            use_gemini=bool(config.model.gemini_api_key),
            use_moondream=use_moondream,  # Enabled for better realism and identity scoring
        )
    
    # Run EACPS
    # Allow caller to pass an override_prompt via config.model.override_prompt (optional)
    override_prompt = None
    if hasattr(config.model, 'override_prompt') and config.model.override_prompt:
        override_prompt = config.model.override_prompt

    best_result, all_candidates = run_eacps_inpaint(
        init_image=init_image,
        mask_image=mask_image,
        character_image=character_image,
        character_name=character_name,
        qwen_pipe=qwen_pipe,
        scorer=scorer,
        config=config,
        verbose=True,
        override_prompt=override_prompt,
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
        
        # Save face-swapped base
        swapped = swap_face_insightface(init_image, character_image, mask_image)
        if swapped:
            swapped.save(task_dir / "faceswap_base.png")
        
        # Save all candidates
        for i, cand in enumerate(all_candidates[:5]):  # Top 5
            cand.image.save(task_dir / f"candidate_{i+1}_seed{cand.seed}.png")
        
        import json
        with open(task_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        
        result["output_dir"] = str(task_dir)
    
    result["result_image"] = best_result
    
    return result
