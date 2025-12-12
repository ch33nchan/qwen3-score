# Fix Git and Install Zellij

## Issue: Git Refusing to Merge Unrelated Histories

This happens when there's a forced update. Here's how to fix it:

### Option 1: Force Pull (Recommended - Overwrites Local Changes)
```bash
# Backup current state (optional)
git stash

# Force pull
git fetch origin
git reset --hard origin/main

# Verify
git status
```

### Option 2: Merge with Unrelated Histories
```bash
git pull origin main --allow-unrelated-histories
```

### Option 3: If You Have Local Changes You Want to Keep
```bash
# Stash local changes
git stash

# Force pull
git fetch origin
git reset --hard origin/main

# Reapply stashed changes (if needed)
git stash pop
```

## After Git is Fixed: Install Zellij

```bash
# Make sure you're in project root
cd /mnt/data1/srini/qwen3-score

# Install zellij locally
./inpaint_eacps/install_zellij.sh

# Verify
.venv/bin/zellij --version
```

## Then Run Tasks

```bash
./inpaint_eacps/start_zellij.sh \
  --task_id 71650389 71678832 71680285 71634650 71630498 71656881 71673477 \
  --gemini_api_key "your_key_here" \
  --device cuda:0
```
