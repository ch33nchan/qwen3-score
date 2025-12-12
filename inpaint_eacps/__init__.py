"""inpaint_eacps package

Entry points:
- `python -m inpaint_eacps` runs the CLI.

This package provides the EACPS inpainting pipeline and a single canonical
CLI at `inpaint_eacps.cli` for running one Label-Studio task with a full
custom prompt. Avoid creating ad-hoc helper scripts; use the CLI instead.
"""

__all__ = ["cli"]
# Inpainting with EACPS inference scaling
# Uses Qwen-Edit, Gemini, and Moondream for multi-model scoring
