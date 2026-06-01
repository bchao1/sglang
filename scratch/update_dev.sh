#!/usr/bin/env bash
# Save CLAUDE.md + .claude/settings to dev/brian and refresh the current working tree.
# Use this whenever CLAUDE.md or settings files change.
#
# Usage: bash scratch/update_dev.sh "what changed"
set -euo pipefail

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
MSG=${1:-"update dev files"}
WDIR=$(mktemp -d /tmp/update-dev-XXXXX)

cleanup() { git -C "$REPO" worktree remove --force "$WDIR" 2>/dev/null; rm -rf "$WDIR" 2>/dev/null || true; }
trap cleanup EXIT

git -C "$REPO" worktree add "$WDIR" dev/brian -q

cp "$REPO/CLAUDE.md" "$WDIR/CLAUDE.md"
cp "$REPO/.claude/settings.json" "$WDIR/.claude/settings.json"
cp "$REPO/.claude/settings.local.json" "$WDIR/.claude/settings.local.json"

cd "$WDIR"
git add -f CLAUDE.md .claude/settings.json .claude/settings.local.json

if git diff --cached --quiet; then
    echo "[update_dev] No changes — nothing to commit."
    exit 0
fi

git commit -m "chore(dev): $MSG"
git push origin dev/brian
echo "[update_dev] Pushed to dev/brian. Current branch unchanged."
