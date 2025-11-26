import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List

import torch
from PIL import Image
from diffusers import QwenImageEditPipeline
from transformers import CLIPProcessor, CLIPModel


@dataclass
class QwenEditConfig:
    model_id: str = "Qwen/Qwen-Image-Edit"
    device: str = "cuda"
    true_cfg_scale: float = 4.0
    num_inference_steps: int = 50
    negative_prompt: str = ""
    dtype: str = "bfloat16"  # used only on CUDA


class QwenEditModel:
    def __init__(self, config: QwenEditConfig) -> None:
        self.config = config

        # Device selection
        if config.device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(config.device)
            use_bf16 = True
        else:
            self.device = torch.device("cpu")
            use_bf16 = False

        # Load pipeline
        self.pipeline = QwenImageEditPipeline.from_pretrained(config.model_id)

        if use_bf16:
            self.pipeline.to(torch.bfloat16)
        else:
            self.pipeline.to(torch.float32)

        self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=None)

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        seed: int,
        num_inference_steps: int | None = None,
        true_cfg_scale: float | None = None,
    ) -> Image.Image:
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_steps
        if true_cfg_scale is None:
            true_cfg_scale = self.config.true_cfg_scale

        generator = torch.Generator(device=self.device.type).manual_seed(seed)

        inputs: Dict[str, Any] = {
            "image": image,
            "prompt": prompt,
            "generator": generator,
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": self.config.negative_prompt or " ",
            "num_inference_steps": num_inference_steps,
        }

        with torch.inference_mode():
            output = self.pipeline(**inputs)
            return output.images[0]


class CLIPScorer:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
    ) -> None:
        if device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def text_image_score(self, prompt: str, image: Image.Image) -> float:
        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
        return float(logits_per_image[0, 0].cpu().item())

    def image_image_score(self, image_a: Image.Image, image_b: Image.Image) -> float:
        inputs = self.processor(images=[image_a, image_b], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            image_embs = self.model.get_image_features(pixel_values=pixel_values)
        image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
        sim = (image_embs[0] * image_embs[1]).sum()
        return float(sim.cpu().item())


def slugify(text: str) -> str:
    keep = []
    for ch in text.strip():
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    slug = "".join(keep)
    return slug[:80] if slug else "prompt"


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen-Image-Edit baseline + simple CLIP evaluation",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image to edit",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Editing instruction",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save edited images and metadata",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of edited images to sample (Best-of-N style)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed; seeds will be seed + i",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen-Image-Edit",
        help="HF model id for Qwen image editing model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string, e.g. 'cuda', 'cuda:0', or 'cpu'",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Denoising steps for Qwen-Image-Edit",
    )
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=4.0,
        help="true_cfg_scale parameter used by Qwen-Image-Edit",
    )
    parser.add_argument(
        "--no_clip",
        action="store_true",
        help="If set, skip CLIP-based metrics",
    )

    args = parser.parse_args()

    image_path = Path(args.image_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_image = Image.open(image_path).convert("RGB")

    config = QwenEditConfig(
        model_id=args.model_id,
        device=args.device,
        true_cfg_scale=args.true_cfg_scale,
        num_inference_steps=args.num_inference_steps,
    )
    qwen_model = QwenEditModel(config)

    clip_scorer = None
    if not args.no_clip:
        clip_scorer = CLIPScorer(device=args.device)

    prompt_slug = slugify(args.prompt)
    edit_subdir = output_dir / prompt_slug
    edit_subdir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "config": {
            "qwen": asdict(config),
            "clip_model": "openai/clip-vit-large-patch14"
            if not args.no_clip
            else None,
        },
        "input": {
            "image_path": str(image_path.resolve()),
            "prompt": args.prompt,
        },
        "samples": [],
    }

    for i in range(args.num_samples):
        seed = int(args.seed) + i
        edited = qwen_model.edit(
            image=input_image,
            prompt=args.prompt,
            seed=seed,
            num_inference_steps=args.num_inference_steps,
            true_cfg_scale=args.true_cfg_scale,
        )

        out_name = f"{prompt_slug}_seed{seed}.png"
        out_path = edit_subdir / out_name
        edited.save(out_path)

        sample_record: Dict[str, Any] = {
            "seed": seed,
            "output_path": str(out_path.resolve()),
        }

        if clip_scorer is not None:
            pf_score = clip_scorer.text_image_score(args.prompt, edited)
            cons_score = clip_scorer.image_image_score(input_image, edited)
            sample_record["clip_prompt_following"] = pf_score
            sample_record["clip_consistency"] = cons_score

        results["samples"].append(sample_record)

    metadata_path = output_dir / f"{prompt_slug}_results.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results['samples'])} edited images to {edit_subdir}")
    print(f"Metadata written to {metadata_path}")
    if not args.no_clip:
        print("Each sample has CLIP-based prompt-following and consistency scores.")


if __name__ == "__main__":
    run()
