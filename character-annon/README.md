# Character Annotation Pipeline

Inpainting pipeline for Label Studio character annotation projects.

## Supported Input Formats

- **JSON** - Label Studio Common Format
- **JSON-MIN** - Minimal JSON format
- **CSV** - Comma-separated values
- **TSV** - Tab-separated values  
- **YOLOv8 OBB** - Directory with `classes.txt`, `images/`, `labels/`, `notes.json`

## Expected Data Schema

```json
{
    "id": "unique_id",
    "character_id": "char_001",
    "character_name": "John",
    "character_image_url": "https://...",
    "image_url": "https://...",
    "init_image_url": "https://...",
    "mask_image_url": "https://...",
    "prompt": "Add character to scene",
    "Issues": ""
}
```

## Usage

### Process All Items

```bash
python run_inpaint.py \
    --input project-label.json \
    --output_dir outputs \
    --device cuda:0 \
    --method both \
    --all
```

### Process Specific IDs

```bash
python run_inpaint.py \
    --input project-label.json \
    --output_dir outputs \
    --ids 1 5 10 15
```

### Process Range

```bash
python run_inpaint.py \
    --input project-label.json \
    --output_dir outputs \
    --range 0 10
```

### Process First N Items

```bash
python run_inpaint.py \
    --input project-label.json \
    --output_dir outputs \
    --first 5
```

### Method Selection

- `--method ttflux` - TT-FLUX random search only
- `--method eacps` - EACPS multi-stage search only
- `--method both` - Run both methods (default)

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | required | Path to Label Studio export file/directory |
| `--output_dir` | outputs | Output directory |
| `--device` | cuda:0 | GPU device |
| `--method` | both | ttflux, eacps, or both |
| `--num_samples` | 4 | TT-FLUX: number of candidates |
| `--k_global` | 4 | EACPS: global exploration candidates |
| `--m_global` | 2 | EACPS: top candidates to refine |
| `--k_local` | 2 | EACPS: refinements per candidate |
| `--steps` | 50 | Diffusion inference steps |
| `--cfg` | 5.0 | Guidance scale |

## Output Structure

```
outputs/
  item_001/
    init.png
    mask.png
    character.png
    ttflux_best.png
    eacps_best.png
    results.json
  item_002/
    ...
  summary.json
```

