from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Union, List, Dict
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VERIFIER_PROMPT_PATH = os.path.join(SCRIPT_DIR, "verifier_prompt.txt")


class BaseVerifier(ABC):
    SUPPORTED_METRIC_CHOICES = None

    def __init__(self, seed: int = 1994, prompt_path: str = None):
        self.seed = seed
        prompt_path = prompt_path or DEFAULT_VERIFIER_PROMPT_PATH
        if os.path.exists(prompt_path):
            with open(prompt_path, "r") as f:
                self.verifier_prompt = f.read()
        else:
            self.verifier_prompt = ""

    @abstractmethod
    def prepare_inputs(self, images, prompts, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def score(self, inputs, **kwargs) -> List[Dict[str, float]]:
        raise NotImplementedError


class AestheticMLP(nn.Module):
    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LAIONAestheticVerifier(BaseVerifier):
    SUPPORTED_METRIC_CHOICES = ["laion_aesthetic_score"]

    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        # For multi-GPU strings like "cuda:0,1", use first GPU for verifier
        if "," in device:
            self.device = device.split(",")[0]
        else:
            self.device = device if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.aesthetic_model = self._load_aesthetic_model()

    def _load_aesthetic_model(self) -> AestheticMLP:
        import urllib.request
        cache_dir = os.path.expanduser("~/.cache/aesthetic")
        os.makedirs(cache_dir, exist_ok=True)
        weights_path = os.path.join(cache_dir, "sac+logos+ava1-l14-linearMSE.pth")
        if not os.path.exists(weights_path):
            url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
            urllib.request.urlretrieve(url, weights_path)
        model = AestheticMLP(input_size=768)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def prepare_inputs(
        self, images: Union[List[Image.Image], Image.Image], prompts=None, **kwargs
    ) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]
        inputs = self.clip_processor(images=images, return_tensors="pt", padding=True)
        return inputs["pixel_values"].to(self.device)

    @torch.no_grad()
    @torch.inference_mode()
    def score(self, inputs: torch.Tensor, **kwargs) -> List[Dict[str, float]]:
        image_features = self.clip_model.get_image_features(pixel_values=inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        scores = self.aesthetic_model(image_features)
        return [{"laion_aesthetic_score": s.item()} for s in scores]


class CLIPScoreVerifier(BaseVerifier):
    SUPPORTED_METRIC_CHOICES = ["clip_score"]

    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        # For multi-GPU strings like "cuda:0,1", use first GPU for verifier
        if "," in device:
            self.device = device.split(",")[0]
        else:
            self.device = device if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def prepare_inputs(
        self, images: Union[List[Image.Image], Image.Image], prompts: Union[List[str], str], **kwargs
    ) -> Dict:
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        inputs = self.clip_processor(
            text=prompts, images=images, return_tensors="pt", padding=True, truncation=True
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    @torch.no_grad()
    @torch.inference_mode()
    def score(self, inputs: Dict, **kwargs) -> List[Dict[str, float]]:
        outputs = self.clip_model(**inputs)
        logits = outputs.logits_per_image.diag()
        normalized = logits / 100.0
        return [{"clip_score": s.item()} for s in normalized]


def get_verifier(name: str, **kwargs) -> BaseVerifier:
    verifiers = {
        "laion_aesthetic": LAIONAestheticVerifier,
        "clip_score": CLIPScoreVerifier,
    }
    if name not in verifiers:
        raise ValueError(f"Unknown verifier: {name}. Available: {list(verifiers.keys())}")
    return verifiers[name](**kwargs)
