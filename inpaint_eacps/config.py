"""
Configuration for inpainting with EACPS.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EACPSConfig:
    """EACPS hyperparameters."""
    k_global: int = 4  # Global exploration candidates
    m_global: int = 2  # Top candidates for local refinement
    k_local: int = 2   # Local refinement per candidate
    
    # Potential function weights
    alpha_consistency: float = 1.0
    beta_quality: float = 1.0
    gamma_identity: float = 2.0  # Higher weight for identity preservation


@dataclass
class ModelConfig:
    """Model configuration."""
    # Qwen Edit
    qwen_model_id: str = "Qwen/Qwen-Image-Edit"
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    
    # Gemini
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = "gemini-1.5-flash"
    
    # Moondream
    moondream_model_id: str = "vikhyatk/moondream2"


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    eacps: EACPSConfig = field(default_factory=EACPSConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    output_dir: str = "outputs/inpaint_eacps"
    cache_dir: str = "data/cache"
    device: str = "cuda:0"
