#!/usr/bin/env python3
"""
inpaint_eacps CLI

Single canonical CLI to run one Label-Studio task with a full custom inpaint prompt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from io import BytesIO
import requests
from PIL import Image

from .config import PipelineConfig, ModelConfig, EACPSConfig
from .pipeline import process_task


def download_image(url: str, cache_dir: Path) -> Image.Image:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = cache_dir / Path(url.split('/')[-1].split('?')[0])
    if fname.exists():
        return Image.open(fname).convert('RGB')
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert('RGB')
    img.save(fname)
    return img


def load_labelstudio_task(file: Path, task_id: str) -> dict:
    with open(file, 'r') as f:
        data = json.load(f)
    for item in data:
        d = item.get('data', {})
        if str(d.get('id')) == str(task_id):
            return d
    raise ValueError(f"Task {task_id} not found in {file}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='inpaint_eacps.cli')
    p.add_argument('--task_id', required=True)
    p.add_argument('--from_file', default='project-label.json')
    p.add_argument('--prompt', required=True, help='Full inpaint prompt (quoted)')
    p.add_argument('--gemini_api_key', required=True)
    p.add_argument('--moondream_model', default='vikhyatk/moondream2')
    p.add_argument('--use_moondream', action='store_true', help='Enable Moondream scoring')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output_dir', default='outputs/inpaint_eacps')
    p.add_argument('--cache_dir', default='data/cache')
    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    task = load_labelstudio_task(Path(args.from_file), args.task_id)
    init_url = task.get('init_image_url') or task.get('init_image')
    mask_url = task.get('mask_image_url') or task.get('mask_image') or task.get('mask')
    char_url = task.get('character_image_url') or task.get('character_image')

    if not (init_url and mask_url and char_url):
        raise SystemExit('Task missing one of init/mask/character image URLs')

    cache_dir = Path(args.cache_dir)
    init_image = download_image(init_url, cache_dir)
    mask_image = download_image(mask_url, cache_dir)
    character_image = download_image(char_url, cache_dir)

    # Build config and inject prompt as override
    eacps = EACPSConfig()
    model = ModelConfig()
    model.gemini_api_key = args.gemini_api_key
    model.moondream_model_id = args.moondream_model
    setattr(model, 'override_prompt', args.prompt)

    cfg = PipelineConfig(eacps=eacps, model=model)
    cfg.device = args.device
    cfg.output_dir = args.output_dir
    cfg.cache_dir = args.cache_dir

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting task {args.task_id} on device {cfg.device} -> output {out_dir}")
    result = process_task(
        init_image=init_image,
        mask_image=mask_image,
        character_image=character_image,
        character_name=task.get('character_name', 'character'),
        task_id=str(args.task_id),
        config=cfg,
        output_dir=out_dir,
        use_moondream=args.use_moondream,
    )

    print('Done. Saved:', result.get('output_dir'))
    print('Metrics:', json.dumps({k: result.get(k) for k in ['best_seed','best_phase','best_potential']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
