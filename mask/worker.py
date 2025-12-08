"""
Multi-GPU worker pool for parallel task processing.
Uses torch.multiprocessing for CUDA-safe parallelism.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from queue import Empty
import traceback

import torch
import torch.multiprocessing as mp

from config import PipelineConfig, EACPSConfig, ModelConfig
from labelstudio import LoadedTask


@dataclass
class TaskResult:
    """Result from processing a single task."""
    task_id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _worker_init(device: str, config: PipelineConfig):
    """Initialize worker process with model on specific GPU."""
    # Set CUDA device
    if device.startswith("cuda"):
        device_id = int(device.split(":")[1]) if ":" in device else 0
        torch.cuda.set_device(device_id)
    
    # Import here to avoid loading model in main process
    from pipeline import build_pipeline, build_scorer
    
    # Build models
    pipe = build_pipeline(config.model.model_id, device)
    scorer = build_scorer(device)
    
    return pipe, scorer


def _worker_process_task(
    task: LoadedTask,
    pipe,
    scorer,
    config: PipelineConfig,
    device: str,
    output_dir: Path,
) -> TaskResult:
    """Process a single task in worker."""
    from pipeline import process_single_task
    
    try:
        result = process_single_task(
            init_image=task.init_image,
            mask_image=task.mask_image,
            character_image=task.character_image,
            prompt=task.task.prompt,
            pipe=pipe,
            scorer=scorer,
            eacps_config=config.eacps,
            model_config=config.model,
            device=device,
            output_dir=output_dir,
            task_id=task.task.id,
            verbose=True,
        )
        
        # Remove PIL images from result (not picklable)
        result.pop("result_image", None)
        result.pop("composite_image", None)
        
        return TaskResult(
            task_id=task.task.id,
            success=True,
            result=result,
        )
    except Exception as e:
        traceback.print_exc()
        return TaskResult(
            task_id=task.task.id,
            success=False,
            error=str(e),
        )


def _worker_loop(
    device: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    config: PipelineConfig,
    output_dir: Path,
):
    """Worker process main loop."""
    print(f"[Worker {device}] Initializing...")
    
    try:
        pipe, scorer = _worker_init(device, config)
        print(f"[Worker {device}] Ready")
    except Exception as e:
        print(f"[Worker {device}] Init failed: {e}")
        traceback.print_exc()
        return
    
    while True:
        try:
            # Get task from queue (blocking with timeout)
            task = task_queue.get(timeout=1.0)
            
            if task is None:
                # Poison pill - exit
                print(f"[Worker {device}] Shutting down")
                break
            
            print(f"[Worker {device}] Processing task {task.task.id}")
            result = _worker_process_task(task, pipe, scorer, config, device, output_dir)
            result_queue.put(result)
            
            # Clear CUDA cache between tasks
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
                
        except Empty:
            continue
        except Exception as e:
            print(f"[Worker {device}] Error: {e}")
            traceback.print_exc()


class GPUWorkerPool:
    """
    Pool of GPU workers for parallel task processing.
    
    Usage:
        pool = GPUWorkerPool(devices=["cuda:0", "cuda:1"], config=config)
        results = pool.process_batch(tasks)
        pool.shutdown()
    """
    
    def __init__(self, devices: List[str], config: PipelineConfig):
        self.devices = devices
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use spawn method for CUDA compatibility
        self.ctx = mp.get_context("spawn")
        
        self.task_queue: mp.Queue = self.ctx.Queue()
        self.result_queue: mp.Queue = self.ctx.Queue()
        self.workers: List[mp.Process] = []
        
        self._started = False
    
    def start(self):
        """Start worker processes."""
        if self._started:
            return
        
        print(f"Starting {len(self.devices)} GPU workers...")
        
        for device in self.devices:
            p = self.ctx.Process(
                target=_worker_loop,
                args=(device, self.task_queue, self.result_queue, self.config, self.output_dir),
            )
            p.start()
            self.workers.append(p)
        
        self._started = True
        print(f"All workers started")
    
    def process_batch(self, tasks: List[LoadedTask]) -> List[TaskResult]:
        """Process a batch of tasks across all workers."""
        if not self._started:
            self.start()
        
        # Submit all tasks
        for task in tasks:
            self.task_queue.put(task)
        
        # Collect results
        results: List[TaskResult] = []
        for _ in range(len(tasks)):
            try:
                result = self.result_queue.get(timeout=600)  # 10 min timeout per task
                results.append(result)
            except Empty:
                print("Warning: Timeout waiting for result")
                break
        
        return results
    
    def shutdown(self):
        """Shutdown all workers."""
        if not self._started:
            return
        
        print("Shutting down workers...")
        
        # Send poison pills
        for _ in self.workers:
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for p in self.workers:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        
        self.workers.clear()
        self._started = False
        print("All workers stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


def process_tasks_sequential(
    tasks: List[LoadedTask],
    config: PipelineConfig,
    device: str = "cuda:0",
) -> List[TaskResult]:
    """
    Process tasks sequentially on a single GPU.
    Useful for debugging or when only one GPU is available.
    """
    from pipeline import build_pipeline, build_scorer, process_single_task
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model on {device}...")
    pipe = build_pipeline(config.model.model_id, device)
    scorer = build_scorer(device)
    print("Model loaded")
    
    results: List[TaskResult] = []
    
    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] Processing task {task.task.id}")
        
        try:
            result = process_single_task(
                init_image=task.init_image,
                mask_image=task.mask_image,
                character_image=task.character_image,
                prompt=task.task.prompt,
                pipe=pipe,
                scorer=scorer,
                eacps_config=config.eacps,
                model_config=config.model,
                device=device,
                output_dir=output_dir,
                task_id=task.task.id,
                verbose=True,
            )
            
            result.pop("result_image", None)
            result.pop("composite_image", None)
            
            results.append(TaskResult(
                task_id=task.task.id,
                success=True,
                result=result,
            ))
        except Exception as e:
            traceback.print_exc()
            results.append(TaskResult(
                task_id=task.task.id,
                success=False,
                error=str(e),
            ))
        
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    
    return results
