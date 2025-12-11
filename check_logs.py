#!/usr/bin/env python3
"""
Debug helper: Check logs from multi-GPU batch run.
Usage: python3 check_logs.py outputs/multigpu_batch
"""
import sys
from pathlib import Path
import subprocess

if len(sys.argv) < 2:
    print("Usage: python3 check_logs.py <output_dir>")
    sys.exit(1)

output_dir = Path(sys.argv[1])

if not output_dir.exists():
    print(f"Directory not found: {output_dir}")
    sys.exit(1)

log_files = sorted(output_dir.glob("gpu*_log.txt"))

if not log_files:
    print(f"No log files found in {output_dir}")
    sys.exit(1)

for log_file in log_files:
    print(f"\n{'='*80}")
    print(f"LOG: {log_file.name}")
    print('='*80)
    
    # Show last 30 lines
    try:
        result = subprocess.run(['tail', '-30', str(log_file)], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        with open(log_file) as f:
            lines = f.readlines()
            print(''.join(lines[-30:]))
