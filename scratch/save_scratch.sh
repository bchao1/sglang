#!/usr/bin/env bash
# Auto-save scratch/ to dev/brian branch via a temporary git worktree.
# Safe to call any time — uses worktree so the main checkout is never touched.
set -euo pipefail

REPO=/home/brianchc/sglang
WDIR=$(mktemp -d /tmp/scratch-save-XXXXX)

cleanup() { git -C "$REPO" worktree remove --force "$WDIR" 2>/dev/null; rm -rf "$WDIR" 2>/dev/null || true; }
trap cleanup EXIT

cd "$REPO"

# Add a worktree for dev/brian at a temp path
git worktree add "$WDIR" dev/brian

# Sync scratch/ into the worktree (--delete removes files gone from scratch/)
rsync -a --delete --exclude='results/' scratch/ "$WDIR/scratch/"

cd "$WDIR"
git add -f scratch/

if git diff --cached --quiet; then
    echo "[save_scratch] No changes in scratch/ — nothing to commit."
    exit 0
fi

git commit -m "wip: auto-save scratch from claude session"
git push origin dev/brian
echo "[save_scratch] scratch/ saved and pushed to dev/brian."
