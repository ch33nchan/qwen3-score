"""
Multi-GPU worker pool for parallel task processing.
Uses torch.multiprocessing for CUDA-safe parallelism.
"""
import os
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
from queue import Empty
import traceback
import time

import torch
import torch.multiprocessing as mp

from config import PipelineConfig
from labelstudio import LoadedTask

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result from processing a single task."""
    task_id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


def _worker_init(device: str, config: PipelineConfig):
    """Initialize worker process with model on specific GPU."""
    # Set CUDA device
    if device.startswith("cuda"):
        device_id = int(device.split(":")[1]) if ":" in device else 0
        torch.cuda.set_device(device_id)
        # Clear cache
        torch.cuda.empty_cache()
    
    # Import here to avoid loading model in main process
    from pipeline import build_pipeline, build_scorer
    
    # Build models
    print(f"[Worker {device}] Loading pipeline...")
    pipe = build_pipeline(config.model.model_id, device)
    print(f"[Worker {device}] Loading scorer...")
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
    
    start_time = time.time()
    
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
        
        elapsed = time.time() - start_time
        
        return TaskResult(
            task_id=task.task.id,
            success=True,
            result=result,
            elapsed_seconds=elapsed,
        )
    except Exception as e:
        elapsed = time.time() - start_time
        traceback.print_exc()
        return TaskResult(
            task_id=task.task.id,
            success=False,
            error=str(e),
            elapsed_seconds=elapsed,
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
        # Send error result and exit
        result_queue.put(TaskResult(
            task_id="__init__",
            success=False,
            error=f"Worker init failed: {e}",
        ))
        return
    
    tasks_processed = 0
    
    while True:
        try:
            # Get task from queue (blocking with timeout)
            task = task_queue.get(timeout=2.0)
            
            if task is None:
                # Poison pill - exit
                print(f"[Worker {device}] Shutting down (processed {tasks_processed} tasks)")
                break
            
            print(f"[Worker {device}] Processing task {task.task.id} ({task.task.character_name})")
            result = _worker_process_task(task, pipe, scorer, config, device, output_dir)
            result_queue.put(result)
            tasks_processed += 1
            
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
        pool = GPUWorkerPool(devices=["cuda:0", "cuda:2", "cuda:3"], config=config)
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
        
        logger.info(f"Starting {len(self.devices)} GPU workers: {self.devices}")
        
        for device in self.devices:
            p = self.ctx.Process(
                target=_worker_loop,
                args=(device, self.task_queue, self.result_queue, self.config, self.output_dir),
                name=f"worker-{device}",
            )
            p.start()
            self.workers.append(p)
            logger.info(f"  Started worker on {device} (PID: {p.pid})")
        
        self._started = True
        
        # Wait a bit for workers to initialize
        time.sleep(2)
        logger.info("All workers started")
    
    def process_batch(self, tasks: List[LoadedTask], timeout_per_task: int = 600) -> List[TaskResult]:
        """Process a batch of tasks across all workers."""
        if not self._started:
            self.start()
        
        # Submit all tasks
        logger.info(f"Submitting {len(tasks)} tasks to queue...")
        for task in tasks:
            self.task_queue.put(task)
        
        # Collect results
        results: List[TaskResult] = []
        total_timeout = timeout_per_task * len(tasks)
        start_time = time.time()
        
        while len(results) < len(tasks):
            elapsed = time.time() - start_time
            if elapsed > total_timeout:
                logger.error(f"Timeout waiting for results (got {len(results)}/{len(tasks)})")
                break
            
            try:
                result = self.result_queue.get(timeout=30)
                if result.task_id == "__init__":
                    # Worker init failure
                    logger.error(f"Worker init failed: {result.error}")
                    continue
                results.append(result)
                status = "OK" if result.success else f"FAIL: {result.error}"
                logger.info(f"  [{len(results)}/{len(tasks)}] Task {result.task_id}: {status} ({result.elapsed_seconds:.1f}s)")
            except Empty:
                # Check if workers are still alive
                alive = sum(1 for w in self.workers if w.is_alive())
                if alive == 0:
                    logger.error("All workers died!")
                    break
        
        return results
    
    def shutdown(self):
        """Shutdown all workers."""
        if not self._started:
            return
        
        logger.info("Shutting down workers...")
        
        # Send poison pills
        for _ in self.workers:
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for p in self.workers:
            p.join(timeout=30)
            if p.is_alive():
                logger.warning(f"Worker {p.name} did not exit, terminating...")
                p.terminate()
        
        self.workers.clear()
        self._started = False
        logger.info("All workers stopped")
    
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
    
    logger.info(f"Loading model on {device}...")
    pipe = build_pipeline(config.model.model_id, device)
    scorer = build_scorer(device)
    logger.info("Model loaded")
    
    results: List[TaskResult] = []
    
    for i, task in enumerate(tasks):
        logger.info(f"[{i+1}/{len(tasks)}] Processing task {task.task.id} ({task.task.character_name})")
        
        start_time = time.time()
        
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
            
            elapsed = time.time() - start_time
            
            results.append(TaskResult(
                task_id=task.task.id,
                success=True,
                result=result,
                elapsed_seconds=elapsed,
            ))
            logger.info(f"  Completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            traceback.print_exc()
            results.append(TaskResult(
                task_id=task.task.id,
                success=False,
                error=str(e),
                elapsed_seconds=elapsed,
            ))
            logger.error(f"  Failed: {e}")
        
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    
    return results
