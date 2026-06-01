#!/usr/bin/env bash
# Pull untracked dev files from dev/brian onto the current working tree.
# Run this after creating a new feature branch so CLAUDE.md and .claude/settings
# are present on disk without being tracked by git.
set -euo pipefail

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
WDIR=$(mktemp -d /tmp/restore-dev-XXXXX)

cleanup() { git -C "$REPO" worktree remove --force "$WDIR" 2>/dev/null; rm -rf "$WDIR" 2>/dev/null || true; }
trap cleanup EXIT

git -C "$REPO" worktree add "$WDIR" dev/brian -q

# Restore untracked dev files (never committed on feature branches)
cp "$WDIR/CLAUDE.md" "$REPO/CLAUDE.md"
cp "$WDIR/.claude/settings.json" "$REPO/.claude/settings.json"
cp "$WDIR/.claude/settings.local.json" "$REPO/.claude/settings.local.json"

# Restore scratch/ (exclude results/ — local benchmark outputs only)
rsync -a --exclude='**/results/' "$WDIR/scratch/" "$REPO/scratch/"

echo "[restore_dev] Dev files restored from dev/brian onto $(git -C "$REPO" branch --show-current)."
