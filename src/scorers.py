from __future__ import annotations

from typing import List, Dict, Optional
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torchvision.transforms.functional as F


class AestheticMLP(nn.Module):
    """
    LAION Aesthetic Predictor MLP head.
    Trained on AVA dataset aesthetic ratings.
    Input: 768-dim CLIP ViT-L/14 features
    Output: scalar aesthetic score (1-10 scale)
    """
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


class EditScorer:
    """
    Multi-metric scorer for image editing evaluation.
    
    Metrics:
    - PF: CLIP text-image similarity (prompt following)
    - CONS: CLIP image-image cosine similarity (preservation vs original)
    - LPIPS: perceptual distance (lower = more similar)
    - Aesthetic: LAION aesthetic score (1-10)
    - ImageReward: human preference score
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        use_lpips: bool = False,
        use_aesthetic: bool = False,
        use_imagereward: bool = False,
    ) -> None:
        if device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.use_lpips = use_lpips
        self.lpips = None
        if self.use_lpips:
            import lpips
            self.lpips = lpips.LPIPS(net="vgg").to(self.device)
            self.lpips.eval()

        self.use_aesthetic = use_aesthetic
        self.aesthetic_model = None
        if self.use_aesthetic:
            self._init_aesthetic()

        self.use_imagereward = use_imagereward
        self.imagereward_model = None
        if self.use_imagereward:
            self._init_imagereward()

    def _init_aesthetic(self) -> None:
        """Initialize LAION aesthetic predictor."""
        try:
            import urllib.request
            import os
            
            cache_dir = os.path.expanduser("~/.cache/aesthetic")
            os.makedirs(cache_dir, exist_ok=True)
            weights_path = os.path.join(cache_dir, "sac+logos+ava1-l14-linearMSE.pth")
            
            if not os.path.exists(weights_path):
                url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
                print(f"Downloading aesthetic predictor weights...")
                urllib.request.urlretrieve(url, weights_path)
            
            self.aesthetic_model = AestheticMLP(input_size=768)
            state_dict = torch.load(weights_path, map_location="cpu")
            self.aesthetic_model.load_state_dict(state_dict)
            self.aesthetic_model.to(self.device)
            self.aesthetic_model.eval()
        except Exception as e:
            print(f"Failed to load aesthetic predictor: {e}")
            self.aesthetic_model = None

    def _init_imagereward(self) -> None:
        """Initialize ImageReward model."""
        try:
            import ImageReward as RM
            self.imagereward_model = RM.load("ImageReward-v1.0", device=self.device)
        except ImportError:
            print("ImageReward not installed. Run: pip install image-reward")
            self.imagereward_model = None
        except Exception as e:
            print(f"Failed to load ImageReward: {e}")
            self.imagereward_model = None

    @torch.no_grad()
    def _clip_pf(
        self,
        images: List[Image.Image],
        prompts: List[str],
    ) -> torch.Tensor:
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        return logits_per_image[:, 0].detach().cpu()

    @torch.no_grad()
    def _clip_cons(
        self,
        images: List[Image.Image],
        originals: List[Image.Image],
    ) -> torch.Tensor:
        imgs_proc = self.processor(images=images, return_tensors="pt")
        pix_edit = imgs_proc["pixel_values"].to(self.device)
        emb_edit = self.model.get_image_features(pixel_values=pix_edit)
        emb_edit = emb_edit / emb_edit.norm(dim=-1, keepdim=True)

        orig_proc = self.processor(images=originals, return_tensors="pt")
        pix_orig = orig_proc["pixel_values"].to(self.device)
        emb_orig = self.model.get_image_features(pixel_values=pix_orig)
        emb_orig = emb_orig / emb_orig.norm(dim=-1, keepdim=True)

        cons = (emb_edit * emb_orig).sum(dim=-1)
        return cons.detach().cpu()

    @torch.no_grad()
    def _lpips_dist(
        self,
        images: List[Image.Image],
        originals: List[Image.Image],
    ) -> torch.Tensor:
        if self.lpips is None:
            return torch.zeros(len(images))

        target_size = [256, 256]

        def pil_batch_to_tensor(imgs: List[Image.Image]) -> torch.Tensor:
            ts = []
            for im in imgs:
                im_r = F.resize(im, target_size)
                t = F.to_tensor(im_r)
                t = t * 2.0 - 1.0
                ts.append(t)
            return torch.stack(ts, dim=0)

        t_edit = pil_batch_to_tensor(images).to(self.device)
        t_orig = pil_batch_to_tensor(originals).to(self.device)

        d = self.lpips(t_edit, t_orig).view(-1)
        return d.detach().cpu()

    @torch.no_grad()
    def _aesthetic_score(
        self,
        images: List[Image.Image],
    ) -> torch.Tensor:
        if self.aesthetic_model is None:
            return torch.full((len(images),), 5.0)

        imgs_proc = self.processor(images=images, return_tensors="pt")
        pix = imgs_proc["pixel_values"].to(self.device)
        features = self.model.get_image_features(pixel_values=pix)
        features = features / features.norm(dim=-1, keepdim=True)
        scores = self.aesthetic_model(features).squeeze(-1)
        return scores.detach().cpu()

    def _imagereward_score(
        self,
        images: List[Image.Image],
        prompts: List[str],
    ) -> torch.Tensor:
        if self.imagereward_model is None:
            return torch.zeros(len(images))

        scores = []
        for img, prompt in zip(images, prompts):
            try:
                score = self.imagereward_model.score(prompt, img)
                scores.append(float(score))
            except Exception:
                scores.append(0.0)
        return torch.tensor(scores)

    @torch.no_grad()
    def score_batch(
        self,
        images: List[Image.Image],
        prompts: List[str],
        originals: Optional[List[Image.Image]] = None,
    ) -> List[Dict[str, float]]:
        if len(images) != len(prompts):
            raise ValueError("images and prompts must have same length")

        pf_scores = self._clip_pf(images, prompts)

        cons_scores: Optional[torch.Tensor] = None
        lpips_scores: Optional[torch.Tensor] = None

        if originals is not None:
            if len(originals) != len(images):
                raise ValueError("originals must be None or same length as images")
            cons_scores = self._clip_cons(images, originals)
            if self.use_lpips:
                lpips_scores = self._lpips_dist(images, originals)

        aesthetic_scores: Optional[torch.Tensor] = None
        if self.use_aesthetic:
            aesthetic_scores = self._aesthetic_score(images)

        imagereward_scores: Optional[torch.Tensor] = None
        if self.use_imagereward:
            imagereward_scores = self._imagereward_score(images, prompts)

        results: List[Dict[str, float]] = []
        for i in range(len(images)):
            rec: Dict[str, float] = {"PF": float(pf_scores[i].item())}
            if cons_scores is not None:
                rec["CONS"] = float(cons_scores[i].item())
            if lpips_scores is not None:
                rec["LPIPS"] = float(lpips_scores[i].item())
            if aesthetic_scores is not None:
                rec["aesthetic"] = float(aesthetic_scores[i].item())
            if imagereward_scores is not None:
                rec["imagereward"] = float(imagereward_scores[i].item())
            results.append(rec)

        return results


def compute_clip_score(
    image: Image.Image,
    prompt: str,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
) -> float:
    """Compute CLIPScore (text-image similarity) normalized to 0-1 scale."""
    inputs = clip_processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    device = next(clip_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        score = logits_per_image[0, 0].item()
    return float(score) / 100.0


def compute_aesthetic_score(
    image: Image.Image,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    aesthetic_model: Optional[AestheticMLP] = None,
) -> float:
    """Compute LAION aesthetic score (1-10 scale)."""
    device = next(clip_model.parameters()).device
    
    if aesthetic_model is None:
        inputs = clip_processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            score = float(features.norm().item() * 1.5 + 4.5)
        return max(1.0, min(10.0, score))
    
    inputs = clip_processor(images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        score = aesthetic_model(features).squeeze().item()
    return float(max(1.0, min(10.0, score)))


def compute_imagereward(
    image: Image.Image,
    prompt: str,
    imagereward_model=None,
) -> Optional[float]:
    """Compute ImageReward score (human preference aligned)."""
    if imagereward_model is None:
        try:
            import ImageReward as RM
            imagereward_model = RM.load("ImageReward-v1.0")
        except ImportError:
            return None
        except Exception:
            return None
    
    try:
        score = imagereward_model.score(prompt, image)
        return float(score)
    except Exception:
        return None
