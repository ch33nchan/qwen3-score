from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .verifiers import BaseVerifier, get_verifier


MAX_SEED = np.iinfo(np.int32).max


@dataclass
class TTFluxConfig:
    search_rounds: int = 4
    search_method: str = "random"
    verifier_name: str = "laion_aesthetic"
    choice_of_metric: str = "laion_aesthetic_score"
    batch_size: int = 1
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 3.5
    max_sequence_length: int = 512
    threshold: float = 0.95
    num_neighbors: int = 4


@dataclass
class SearchResult:
    prompt: str
    search_round: int
    num_noises: int
    best_seed: int
    best_score: float
    best_image: Image.Image
    best_noise: torch.Tensor
    all_images: List[Tuple[int, Image.Image, float]] = field(default_factory=list)


def prepare_latents_for_flux(
    batch_size: int,
    height: int,
    width: int,
    generator: torch.Generator,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_channels_latents = 16
    vae_scale_factor = 8
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor
    shape = (batch_size, num_channels_latents, latent_height, latent_width)
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    packed_latents = latents.view(batch_size, num_channels_latents, -1).transpose(1, 2)
    return packed_latents


def get_noises(
    num_samples: int,
    height: int,
    width: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[int, torch.Tensor]:
    noises = {}
    for _ in range(num_samples):
        seed = np.random.randint(0, MAX_SEED)
        generator = torch.Generator(device=device).manual_seed(seed)
        latent = prepare_latents_for_flux(
            batch_size=1,
            height=height,
            width=width,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        noises[seed] = latent
    return noises


def generate_neighbors(
    x: torch.Tensor, threshold: float = 0.95, num_neighbors: int = 4
) -> torch.Tensor:
    batch_size = x.shape[0]
    neighbors = []
    for _ in range(num_neighbors):
        noise = torch.randn_like(x)
        neighbor = threshold * x + np.sqrt(1 - threshold**2) * noise
        neighbors.append(neighbor)
    return torch.stack(neighbors, dim=1).view(batch_size * num_neighbors, *x.shape[1:])


class TTFluxPipeline:
    def __init__(
        self,
        pipe,
        config: TTFluxConfig = None,
        verifier: BaseVerifier = None,
        device: str = "cuda",
    ):
        self.pipe = pipe
        self.config = config or TTFluxConfig()
        self.device = device if torch.cuda.is_available() else "cpu"
        if verifier is None:
            self.verifier = get_verifier(self.config.verifier_name, device=self.device)
        else:
            self.verifier = verifier

    def _generate_batch(
        self,
        prompt: str,
        latents: torch.Tensor,
        **kwargs,
    ) -> List[Image.Image]:
        batch_size = latents.shape[0]
        prompts = [prompt] * batch_size
        call_args = {
            "prompt": prompts,
            "latents": latents,
            "height": self.config.height,
            "width": self.config.width,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "max_sequence_length": self.config.max_sequence_length,
        }
        call_args.update(kwargs)
        result = self.pipe(**call_args)
        return result.images

    def _score_images(
        self, images: List[Image.Image], prompt: str
    ) -> List[Dict[str, float]]:
        inputs = self.verifier.prepare_inputs(images=images, prompts=prompt)
        return self.verifier.score(inputs)

    def sample_round(
        self,
        prompt: str,
        noises: Dict[int, torch.Tensor],
        search_round: int,
    ) -> SearchResult:
        noise_items = list(noises.items())
        all_images = []
        all_scores = []
        all_seeds = []
        all_noises = []

        for i in range(0, len(noise_items), self.config.batch_size):
            batch = noise_items[i : i + self.config.batch_size]
            seeds, latents = zip(*batch)
            latents_tensor = torch.cat(list(latents), dim=0)
            images = self._generate_batch(prompt, latents_tensor)
            scores = self._score_images(images, prompt)
            for seed, img, score, lat in zip(seeds, images, scores, latents):
                metric_score = score[self.config.choice_of_metric]
                all_images.append(img)
                all_scores.append(metric_score)
                all_seeds.append(seed)
                all_noises.append(lat)

        best_idx = np.argmax(all_scores)
        return SearchResult(
            prompt=prompt,
            search_round=search_round,
            num_noises=len(noises),
            best_seed=all_seeds[best_idx],
            best_score=all_scores[best_idx],
            best_image=all_images[best_idx],
            best_noise=all_noises[best_idx],
            all_images=list(zip(all_seeds, all_images, all_scores)),
        )

    def run_random_search(
        self,
        prompt: str,
        search_rounds: int = None,
        verbose: bool = True,
    ) -> SearchResult:
        search_rounds = search_rounds or self.config.search_rounds
        best_result = None

        for round_idx in range(1, search_rounds + 1):
            num_noises = 2**round_idx
            if verbose:
                print(f"Round {round_idx}: generating {num_noises} candidates")
            noises = get_noises(
                num_samples=num_noises,
                height=self.config.height,
                width=self.config.width,
                device=self.device,
                dtype=torch.bfloat16,
            )
            result = self.sample_round(prompt, noises, round_idx)
            if best_result is None or result.best_score > best_result.best_score:
                best_result = result
            if verbose:
                print(f"  Best seed={result.best_seed}, score={result.best_score:.4f}")

        return best_result

    def run_zero_order_search(
        self,
        prompt: str,
        search_rounds: int = None,
        verbose: bool = True,
    ) -> SearchResult:
        search_rounds = search_rounds or self.config.search_rounds
        best_result = None
        current_noise = None

        for round_idx in range(1, search_rounds + 1):
            if current_noise is None:
                noises = get_noises(
                    num_samples=1,
                    height=self.config.height,
                    width=self.config.width,
                    device=self.device,
                    dtype=torch.bfloat16,
                )
                base_seed, base_noise = next(iter(noises.items()))
            else:
                base_seed, base_noise = current_noise

            neighbors = generate_neighbors(
                base_noise,
                threshold=self.config.threshold,
                num_neighbors=self.config.num_neighbors,
            )
            all_latents = torch.cat([base_noise, neighbors], dim=0)
            noises = {base_seed + i: lat.unsqueeze(0) for i, lat in enumerate(all_latents)}

            if verbose:
                print(f"Round {round_idx}: base + {self.config.num_neighbors} neighbors")

            result = self.sample_round(prompt, noises, round_idx)

            if best_result is None or result.best_score > best_result.best_score:
                best_result = result
                current_noise = (result.best_seed, result.best_noise)

            if verbose:
                print(f"  Best seed={result.best_seed}, score={result.best_score:.4f}")

        return best_result

    def __call__(
        self,
        prompt: str,
        search_rounds: int = None,
        search_method: str = None,
        verbose: bool = True,
    ) -> SearchResult:
        method = search_method or self.config.search_method
        if method == "random":
            return self.run_random_search(prompt, search_rounds, verbose)
        elif method == "zero-order":
            return self.run_zero_order_search(prompt, search_rounds, verbose)
        else:
            raise ValueError(f"Unknown search method: {method}")
