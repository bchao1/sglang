#!/usr/bin/env bash
# Save scratch/ to dev/brian via a temporary git worktree.
# The main checkout is NEVER touched — no branch switch, no lost files.
#
# Usage: bash scratch/save_scratch.sh [commit-message-suffix]
# Example: bash scratch/save_scratch.sh "spectral-progressive-flux: add bench script"
set -euo pipefail

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
SUFFIX=${1:-"update"}
WDIR=$(mktemp -d /tmp/scratch-save-XXXXX)

cleanup() { git -C "$REPO" worktree remove --force "$WDIR" 2>/dev/null; rm -rf "$WDIR" 2>/dev/null || true; }
trap cleanup EXIT

git -C "$REPO" worktree add "$WDIR" dev/brian -q

# Sync scratch/ into worktree; exclude results/ (local-only benchmark outputs)
rsync -a --delete --exclude='**/results/' "$REPO/scratch/" "$WDIR/scratch/"

cd "$WDIR"
git add -f scratch/

if git diff --cached --quiet; then
    echo "[save_scratch] No changes — nothing to commit."
    exit 0
fi

git commit --no-verify -m "wip(scratch): $SUFFIX"
git push origin dev/brian
echo "[save_scratch] Pushed to dev/brian. Current branch unchanged."
