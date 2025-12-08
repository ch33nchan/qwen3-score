"""
Configuration for the mask pipeline.
Supports environment variables for sensitive values.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LabelStudioConfig:
    """Label Studio API configuration."""
    url: str = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
    api_key: str = os.getenv("LABEL_STUDIO_API_KEY", "")
    project_id: Optional[int] = None


@dataclass
class GPUConfig:
    """GPU configuration for multi-device processing."""
    devices: List[str] = field(default_factory=lambda: ["cuda:0"])
    batch_size_per_gpu: int = 1  # Memory-bound, process one at a time per GPU
    
    @classmethod
    def from_string(cls, devices_str: str) -> "GPUConfig":
        """Parse device string like 'cuda:0,cuda:1,cuda:2,cuda:3'."""
        devices = [d.strip() for d in devices_str.split(",") if d.strip()]
        return cls(devices=devices)
    
    @property
    def num_workers(self) -> int:
        return len(self.devices)


@dataclass
class EACPSConfig:
    """EACPS algorithm hyperparameters."""
    k_global: int = 4       # Number of candidates in global exploration
    m_global: int = 2       # Top candidates selected for refinement
    k_local: int = 2        # Local variations per selected candidate
    alpha_cons: float = 10.0  # Weight for consistency term
    beta_lpips: float = 3.0   # Weight for LPIPS penalty
    gamma_realism: float = 0.0  # Weight for realism/aesthetic score


@dataclass
class ModelConfig:
    """Diffusion model configuration."""
    model_id: str = "Qwen/Qwen-Image-Edit"
    num_inference_steps: int = 50
    true_cfg_scale: float = 5.0
    negative_prompt: str = (
        "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
        "bad proportions, extra limbs, cloned face, disfigured, "
        "out of frame, watermark, signature, text, seams, visible edges, "
        "inconsistent lighting, artificial, unnatural"
    )


@dataclass
class PipelineConfig:
    """Combined pipeline configuration."""
    label_studio: LabelStudioConfig = field(default_factory=LabelStudioConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    eacps: EACPSConfig = field(default_factory=EACPSConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output_dir: str = "outputs/mask_results"
    cache_dir: str = "data/cache"
    
    @classmethod
    def from_args(
        cls,
        project_id: int,
        api_key: str,
        url: str,
        output_dir: str,
        devices: str,
        k_global: int = 4,
        m_global: int = 2,
        k_local: int = 2,
        steps: int = 50,
        cfg: float = 5.0,
    ) -> "PipelineConfig":
        """Create config from CLI arguments."""
        return cls(
            label_studio=LabelStudioConfig(
                url=url,
                api_key=api_key,
                project_id=project_id,
            ),
            gpu=GPUConfig.from_string(devices),
            eacps=EACPSConfig(
                k_global=k_global,
                m_global=m_global,
                k_local=k_local,
            ),
            model=ModelConfig(
                num_inference_steps=steps,
                true_cfg_scale=cfg,
            ),
            output_dir=output_dir,
        )
