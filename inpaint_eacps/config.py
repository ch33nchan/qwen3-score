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
    num_inference_steps: int = 15  # MUCH lower - minimal editing
    guidance_scale: float = 2.5  # Very low - barely touch the face swap
    negative_prompt: str = (
        "blurry, low quality, distorted, artifacts, double face, triple face, "
        "unnatural, fake, AI generated, deepfake artifacts, "
        "cartoon, painting, illustration, anime, 3d render, cgi, "
        "oversaturated, airbrushed, plastic skin, smooth skin, unrealistic skin, "
        "visible seams, color mismatch, lighting mismatch, "
        "caricature, exaggerated features, doll-like, mannequin, "
        "overprocessed, oversharpened, hdr artifacts, "
        "added facial hair, mustache, beard, stubble, extra hair, "
        "watermark, text, logo, signature"
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
