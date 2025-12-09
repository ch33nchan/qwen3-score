"""
Configuration for inpainting with EACPS.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EACPSConfig:
    """EACPS hyperparameters."""
    k_global: int = 6  # Increased for better exploration
    m_global: int = 3  # More top candidates for refinement
    k_local: int = 3   # More local refinements
    
    # Potential function weights
    alpha_consistency: float = 1.5  # Increased weight
    beta_quality: float = 2.5  # Much higher weight for realism
    gamma_identity: float = 2.0  # Higher weight for identity preservation


@dataclass
class ModelConfig:
    """Model configuration."""
    # Qwen Edit
    qwen_model_id: str = "Qwen/Qwen-Image-Edit"
    num_inference_steps: int = 50  # Balanced for quality/speed
    guidance_scale: float = 6.0  # Balanced guidance
    negative_prompt: str = (
        "blurry, low quality, distorted, artifacts, double face, "
        "unnatural, fake, AI generated, cartoon, painting, illustration, "
        "oversaturated, airbrushed, smooth skin, unrealistic"
    )
    
    # Gemini
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = "gemini-1.5-flash"
    
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
