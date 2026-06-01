#!/usr/bin/env bash
# Pull untracked dev files from dev/brian onto the current working tree.
# Run after cloning on a new machine or after creating a new feature branch.
set -euo pipefail

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
WDIR=$(mktemp -d /tmp/restore-dev-XXXXX)

cleanup() { git -C "$REPO" worktree remove --force "$WDIR" 2>/dev/null; rm -rf "$WDIR" 2>/dev/null || true; }
trap cleanup EXIT

git -C "$REPO" worktree add "$WDIR" dev/brian -q

# Restore untracked dev files
cp "$WDIR/CLAUDE.md" "$REPO/CLAUDE.md"
cp "$WDIR/.claude/settings.json" "$REPO/.claude/settings.json"
cp "$WDIR/.claude/settings.local.json" "$REPO/.claude/settings.local.json"

# Restore scratch/ (exclude results/ — local benchmark outputs only)
rsync -a --exclude='**/results/' "$WDIR/scratch/" "$REPO/scratch/"

# Ensure .git/info/exclude is correct so git never sees these files on any branch
EXCLUDE="$REPO/.git/info/exclude"
for pattern in "scratch/" "scratch/results/" "CLAUDE.md"; do
    grep -qxF "$pattern" "$EXCLUDE" || echo "$pattern" >> "$EXCLUDE"
done

echo "[restore_dev] Dev files restored from dev/brian onto $(git -C "$REPO" branch --show-current)."
echo "[restore_dev] .git/info/exclude verified."
