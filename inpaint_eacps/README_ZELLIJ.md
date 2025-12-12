# Running Inpaint EACPS with Zellij

## Quick Start

### Single Task
```bash
./inpaint_eacps/start_zellij.sh --task_id 71680285 --gemini_api_key YOUR_KEY
```

### Multiple Tasks
```bash
./inpaint_eacps/start_zellij.sh \
  --task_id 71680285 71651078 71634264 \
  --gemini_api_key YOUR_KEY \
  --device cuda:0
```

## Full Options

```bash
./inpaint_eacps/start_zellij.sh \
  --task_id ID1 ID2 ID3 ... \
  --from_file project-label.json \
  --output_dir inpaint_eacps \
  --device cuda:0 \
  --gemini_api_key YOUR_KEY \
  --k_global 8 \
  --m_global 3 \
  --k_local 4
```

## Arguments

- `--task_id`: One or more task IDs (required)
- `--from_file`: JSON file with tasks (default: `project-label.json`)
- `--output_dir`: Output directory (default: `inpaint_eacps`)
- `--device`: CUDA device (default: `cuda:0`)
- `--gemini_api_key`: Gemini API key (or set `GEMINI_API_KEY` env var)
- `--k_global`: Global exploration candidates (default: 8)
- `--m_global`: Top candidates for refinement (default: 3)
- `--k_local`: Local refinements per candidate (default: 4)

## Output Structure

All outputs are stored in `inpaint_eacps/task_{ID}/`:

```
inpaint_eacps/
├── task_71680285/
│   ├── init.png                    # Original image
│   ├── mask.png                    # Mask image
│   ├── character.png               # Character reference
│   ├── result.png                  # Best result
│   ├── faceswap_base.png           # InsightFace swap (before refinement)
│   ├── composited_reference.png    # Composite reference
│   ├── candidate_1_seed5000.png   # Top candidates
│   ├── candidate_2_seed5031.png
│   ├── ...
│   └── metrics.json                # Scores and metrics
├── task_71651078/
│   └── ...
└── ...
```

## Progress Bars

The pipeline shows detailed progress:

1. **Overall Progress**: Shows task completion across all tasks
2. **Global Exploration**: Shows candidate generation (k_global candidates)
3. **Local Refinement**: Shows refinement progress (m_global × k_local candidates)

Example output:
```
Overall Progress: 2/5 tasks [00:15<00:45] current: 71680285 character: Sandeep Khanna potential: 28.45
  Global exploration: 8/8 [00:12<00:00] potential: 28.45
  Local refinement: 12/12 [00:08<00:00] potential: 29.12
```

## Zellij Session Management

### Attach to Session
```bash
zellij attach inpaint_eacps
```

### List Sessions
```bash
zellij list-sessions
```

### Kill Session
```bash
zellij kill-session inpaint_eacps
```

### Create New Session (if exists)
The script will prompt you to either attach or recreate if the session already exists.

## Environment Variables

Set these before running:

```bash
export GEMINI_API_KEY="your_key_here"
```

Then you can omit `--gemini_api_key` from the command.

## Example: Processing Multiple Tasks

```bash
# Process 3 tasks on cuda:0
./inpaint_eacps/start_zellij.sh \
  --task_id 71680285 71651078 71634264 \
  --gemini_api_key "$GEMINI_API_KEY" \
  --device cuda:0 \
  --output_dir inpaint_eacps

# In another terminal, monitor progress
zellij attach inpaint_eacps
```

## Troubleshooting

### Session Already Exists
The script will prompt you to attach or recreate. Choose option 2 to kill and recreate.

### Output Directory
By default, outputs go to `inpaint_eacps/` (not `outputs/inpaint_eacps/`). 
Use `--output_dir` to change this.

### Progress Not Showing
Make sure you're running in a terminal that supports tqdm progress bars. 
If using SSH, ensure your terminal is properly configured.
