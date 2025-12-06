#!/usr/bin/env python3
"""
Setup Label Studio for human evaluation of image edits.
Creates tasks for side-by-side comparison: Option A = TT-FLUX, Option B = EACPS

Usage:
    # Generate tasks from batch results
    python setup_labelstudio.py --results_dir results/batch_eval --output_dir labelstudio_export
    
    # Upload to Label Studio API
    python setup_labelstudio.py --results_dir results/batch_eval --upload --api_url https://your-labelstudio.com --api_key YOUR_KEY
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import shutil
import base64


LABELING_CONFIG = """
<View>
  <Header value="Image Edit Comparison"/>
  <Text name="prompt" value="Edit prompt: $prompt"/>
  
  <View style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0;">
    <View style="text-align: center;">
      <Header value="Original Image"/>
      <Image name="original" value="$original"/>
    </View>
    <View style="text-align: center;">
      <Header value="Option A (TT-FLUX)"/>
      <Image name="option_a" value="$option_a"/>
    </View>
    <View style="text-align: center;">
      <Header value="Option B (EACPS)"/>
      <Image name="option_b" value="$option_b"/>
    </View>
  </View>
  
  <Header value="1. Which edit better follows the prompt?"/>
  <Choices name="prompt_following" toName="original" choice="single" required="true">
    <Choice value="A much better"/>
    <Choice value="A slightly better"/>
    <Choice value="Equal"/>
    <Choice value="B slightly better"/>
    <Choice value="B much better"/>
  </Choices>
  
  <Header value="2. Which edit has better visual quality?"/>
  <Choices name="visual_quality" toName="original" choice="single" required="true">
    <Choice value="A much better"/>
    <Choice value="A slightly better"/>
    <Choice value="Equal"/>
    <Choice value="B slightly better"/>
    <Choice value="B much better"/>
  </Choices>
  
  <Header value="3. Which edit better preserves the original (where it should)?"/>
  <Choices name="preservation" toName="original" choice="single" required="true">
    <Choice value="A much better"/>
    <Choice value="A slightly better"/>
    <Choice value="Equal"/>
    <Choice value="B slightly better"/>
    <Choice value="B much better"/>
  </Choices>
  
  <Header value="4. Overall, which edit is better?"/>
  <Choices name="overall" toName="original" choice="single" required="true">
    <Choice value="A is better"/>
    <Choice value="B is better"/>
    <Choice value="Both equally good"/>
    <Choice value="Both equally bad"/>
  </Choices>
  
  <Header value="5. Comments (optional)"/>
  <TextArea name="comments" toName="original" rows="2" placeholder="Any additional observations..."/>
</View>
"""


def load_comparison_results(results_dir: Path) -> Dict:
    results_file = results_dir / "comparison_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    with open(results_file, "r") as f:
        return json.load(f)


def image_to_base64(image_path: Path) -> str:
    """Convert image to base64 data URL."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{data}"


def create_labelstudio_tasks(
    results_dirs: List[Path],
    output_dir: Path,
    randomize_ab: bool = False,
    use_base64: bool = False,
) -> List[Dict]:
    """
    Create Label Studio tasks from comparison results.
    Option A = TT-FLUX, Option B = EACPS (fixed mapping for fair comparison)
    """
    tasks = []
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, results_dir in enumerate(results_dirs):
        try:
            results = load_comparison_results(results_dir)
        except FileNotFoundError:
            print(f"  Skipping {results_dir}: no results file")
            continue
        
        original_src = results_dir / "input.png"
        ttflux_src = results_dir / "ttflux_best.png"
        eacps_src = results_dir / "eacps_best.png"
        
        if not all(p.exists() for p in [original_src, ttflux_src, eacps_src]):
            print(f"  Skipping {results_dir}: missing images")
            continue
        
        task_id = f"task_{idx:03d}"
        
        if use_base64:
            original_url = image_to_base64(original_src)
            option_a_url = image_to_base64(ttflux_src)
            option_b_url = image_to_base64(eacps_src)
        else:
            # Copy images
            original_dst = images_dir / f"{task_id}_original.png"
            ttflux_dst = images_dir / f"{task_id}_ttflux.png"
            eacps_dst = images_dir / f"{task_id}_eacps.png"
            
            shutil.copy(original_src, original_dst)
            shutil.copy(ttflux_src, ttflux_dst)
            shutil.copy(eacps_src, eacps_dst)
            
            original_url = f"/data/local-files/?d=images/{task_id}_original.png"
            option_a_url = f"/data/local-files/?d=images/{task_id}_ttflux.png"
            option_b_url = f"/data/local-files/?d=images/{task_id}_eacps.png"
        
        # Fixed mapping: A = TT-FLUX, B = EACPS
        task = {
            "id": idx,
            "data": {
                "task_id": task_id,
                "prompt": results["prompt"],
                "original": original_url,
                "option_a": option_a_url,
                "option_b": option_b_url,
                "source_dir": str(results_dir),
                "mapping": {
                    "A": "ttflux",
                    "B": "eacps"
                },
                "auto_metrics": {
                    "ttflux": results.get("ttflux", {}).get("metrics", {}),
                    "eacps": results.get("eacps", {}).get("metrics", {}),
                }
            }
        }
        tasks.append(task)
    
    return tasks


def upload_to_labelstudio(
    api_url: str,
    api_key: str,
    project_name: str,
    tasks: List[Dict],
    labeling_config: str,
) -> Dict:
    """Upload tasks to Label Studio via API."""
    import requests
    
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }
    
    # Check if project exists
    projects_url = f"{api_url.rstrip('/')}/api/projects/"
    resp = requests.get(projects_url, headers=headers)
    resp.raise_for_status()
    
    existing = [p for p in resp.json().get("results", []) if p["title"] == project_name]
    
    if existing:
        project_id = existing[0]["id"]
        print(f"Using existing project: {project_name} (ID: {project_id})")
    else:
        # Create new project
        create_resp = requests.post(
            projects_url,
            headers=headers,
            json={
                "title": project_name,
                "label_config": labeling_config,
            }
        )
        create_resp.raise_for_status()
        project_id = create_resp.json()["id"]
        print(f"Created new project: {project_name} (ID: {project_id})")
    
    # Import tasks
    import_url = f"{api_url.rstrip('/')}/api/projects/{project_id}/import"
    import_resp = requests.post(
        import_url,
        headers=headers,
        json=tasks,
    )
    import_resp.raise_for_status()
    
    result = import_resp.json()
    print(f"Imported {result.get('task_count', len(tasks))} tasks")
    
    return {
        "project_id": project_id,
        "project_url": f"{api_url.rstrip('/')}/projects/{project_id}/",
        "tasks_imported": result.get("task_count", len(tasks)),
    }


def main():
    parser = argparse.ArgumentParser(description="Setup Label Studio for evaluation")
    parser.add_argument("--results_dir", type=str, nargs="+", required=True,
                        help="Directory(ies) containing comparison results")
    parser.add_argument("--output_dir", type=str, default="labelstudio_export",
                        help="Output directory for Label Studio files")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to Label Studio API")
    parser.add_argument("--api_url", type=str, default=os.environ.get("LABEL_STUDIO_URL", ""),
                        help="Label Studio API URL (or set LABEL_STUDIO_URL env)")
    parser.add_argument("--api_key", type=str, default=os.environ.get("LABEL_STUDIO_API_KEY", ""),
                        help="Label Studio API key (or set LABEL_STUDIO_API_KEY env)")
    parser.add_argument("--project_name", type=str, default="EACPS vs TT-FLUX Evaluation",
                        help="Label Studio project name")
    parser.add_argument("--base64", action="store_true",
                        help="Embed images as base64 (useful for API upload)")
    args = parser.parse_args()
    
    # Handle both single directory and batch results
    results_dirs = []
    for rd in args.results_dir:
        rd_path = Path(rd)
        if (rd_path / "comparison_results.json").exists():
            results_dirs.append(rd_path)
        else:
            # Check for subdirectories with results
            for subdir in rd_path.iterdir():
                if subdir.is_dir() and (subdir / "comparison_results.json").exists():
                    results_dirs.append(subdir)
    
    if not results_dirs:
        print("No results directories found with comparison_results.json")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating Label Studio tasks from {len(results_dirs)} result directories...")
    print("Mapping: Option A = TT-FLUX, Option B = EACPS")
    
    use_base64 = args.base64 or args.upload
    tasks = create_labelstudio_tasks(
        results_dirs,
        output_dir,
        randomize_ab=False,  # Fixed: A=TT-FLUX, B=EACPS
        use_base64=use_base64,
    )
    
    if not tasks:
        print("No valid tasks created")
        return
    
    # Save tasks locally
    tasks_file = output_dir / "tasks.json"
    with open(tasks_file, "w") as f:
        json.dump(tasks, f, indent=2)
    
    # Save labeling config
    config_file = output_dir / "labeling_config.xml"
    with open(config_file, "w") as f:
        f.write(LABELING_CONFIG)
    
    print(f"\nCreated {len(tasks)} tasks")
    print(f"Tasks file: {tasks_file}")
    print(f"Labeling config: {config_file}")
    
    if args.upload:
        if not args.api_url or not args.api_key:
            print("\nError: --api_url and --api_key required for upload")
            print("Or set LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY environment variables")
            return
        
        print(f"\nUploading to Label Studio: {args.api_url}")
        result = upload_to_labelstudio(
            api_url=args.api_url,
            api_key=args.api_key,
            project_name=args.project_name,
            tasks=tasks,
            labeling_config=LABELING_CONFIG,
        )
        print(f"Project URL: {result['project_url']}")
    else:
        print("\n" + "=" * 60)
        print("To upload to Label Studio:")
        print("=" * 60)
        print(f"python setup_labelstudio.py --results_dir {' '.join(args.results_dir)} \\")
        print(f"    --upload --api_url YOUR_LABELSTUDIO_URL --api_key YOUR_API_KEY")
        print()
        print("Or import manually via web UI")
        print("=" * 60)


if __name__ == "__main__":
    main()
