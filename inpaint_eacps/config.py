"""
Configuration for inpainting with EACPS.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EACPSConfig:
    """EACPS hyperparameters."""
    k_global: int = 8  # More exploration for better candidates
    m_global: int = 3  # Top candidates for refinement
    k_local: int = 4   # More local refinements per top candidate
    
    # Potential function weights
    # REALISM is king - we want photorealistic outputs above all else
    alpha_consistency: float = 1.0  # Scene consistency
    beta_quality: float = 4.0  # HEAVILY weight realism - this is the key metric
    gamma_identity: float = 1.5  # Identity preservation (already handled by face swap)


@dataclass
class ModelConfig:
    """Model configuration."""
    # Qwen Edit
    qwen_model_id: str = "Qwen/Qwen-Image-Edit"
    num_inference_steps: int = 8  # Ultra-minimal steps to preserve original quality
    guidance_scale: float = 1.2  # Extremely low CFG - only subtle refinement
    negative_prompt: str = (
        "soft skin, smooth skin, airbrushed skin, blurred skin, porcelain skin, plastic skin, "
        "soft focus, gaussian blur, beauty filter, skin smoothing, touched up photo, "
        "oversaturated, overexposed, high contrast, boosted colors, vivid colors, "
        "caricature, cartoon, big head, oversized face, disproportionate face, "
        "exaggerated features, stretched face, compressed face, distorted proportions, "
        "fake hair, wig-like hair, painted hair, synthetic hair texture, "
        "visible seams, hard edges, color banding, posterization, "
        "deepfake artifacts, face morph, blending artifacts, "
        "added facial hair, mustache, beard, stubble that wasn't there, "
        "illustration, painting, 3d render, cgi, anime, manga, "
        "watermark, text, logo, signature, frame, "
        "changed background, modified background, altered background, "
        "changed body position, modified pose, altered pose, different angle, "
        "changed clothing, modified clothing, altered clothing, "
        "changed lighting, modified lighting, different lighting, "
        "changed composition, modified composition, cropped image, "
        "any changes outside the masked region, any modifications to unmasked areas"
    )
    
    # Gemini
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = "gemini-2.5-pro"  # Best quality vision model
    
    # Moondream V3
    moondream_model_id: str = "vikhyatk/moondream2"  # moondream2 is the latest version


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    eacps: EACPSConfig = field(default_factory=EACPSConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    output_dir: str = "outputs/inpaint_eacps"
    cache_dir: str = "data/cache"
    device: str = "cuda:0"
