#!/usr/bin/env python3
"""
Run a single inpainting task with a custom inpaint prompt.
Saves outputs to `outputs/inpaint_eacps/task_<id>`.

Usage:
  python3 inpaint_eacps/run_single_with_prompt.py --task_id 71650389 \
      --gemini_api_key YOUR_KEY --prompt-file myprompt.txt --device cuda:1

You can also pass the prompt inline with --prompt 'long prompt here'.
"""
import argparse
import json
import os
from pathlib import Path
import requests
from io import BytesIO
from PIL import Image

from inpaint_eacps.config import PipelineConfig, ModelConfig, EACPSConfig
from inpaint_eacps.pipeline import process_task


def download_image(url: str, cache_dir: Path) -> Image.Image:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = cache_dir / Path(url.split('/')[-1].split('?')[0])
    if fname.exists():
        return Image.open(fname)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert('RGB')
    img.save(fname)
    return img


def load_labelstudio_task(file: Path, task_id: str) -> dict:
    with open(file) as f:
        data = json.load(f)
    for item in data:
        d = item.get('data', {})
        if str(d.get('id')) == str(task_id):
            return d
    raise ValueError(f"Task {task_id} not found in {file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', required=True)
    parser.add_argument('--from_file', default='project-label.json')
    parser.add_argument('--gemini_api_key', required=True)
    parser.add_argument('--prompt_file', help='Path to text file containing the full prompt')
    parser.add_argument('--prompt', help='Prompt string (use instead of --prompt_file)')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--output_dir', default='outputs/inpaint_eacps')
    args = parser.parse_args()

    if not args.prompt and not args.prompt_file:
        parser.error('Provide --prompt or --prompt_file')

    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt_text = f.read()
    else:
        prompt_text = args.prompt

    task = load_labelstudio_task(Path(args.from_file), args.task_id)
    init_url = task.get('init_image_url') or task.get('init_image')
    mask_url = task.get('mask_image_url') or task.get('mask')
    char_url = task.get('character_image_url') or task.get('character_image')

    if not (init_url and mask_url and char_url):
        raise ValueError('Task missing one of init/mask/character image URLs')

    cache_dir = Path('data/cache')
    init_image = download_image(init_url, cache_dir)
    mask_image = download_image(mask_url, cache_dir)
    character_image = download_image(char_url, cache_dir)

    # Build config and inject prompt into model config as override_prompt
    eacps = EACPSConfig()
    model = ModelConfig()
    model.gemini_api_key = args.gemini_api_key
    # attach override_prompt attribute used by pipeline.process_task
    setattr(model, 'override_prompt', prompt_text)

    cfg = PipelineConfig(eacps=eacps, model=model)
    cfg.device = args.device
    cfg.output_dir = args.output_dir

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = process_task(
        init_image=init_image,
        mask_image=mask_image,
        character_image=character_image,
        character_name=task.get('character_name', 'character'),
        task_id=str(args.task_id),
        config=cfg,
        output_dir=out_dir,
        use_moondream=True,
    )

    print('\nDone. Result saved to:', result.get('output_dir'))


if __name__ == '__main__':
    main()
