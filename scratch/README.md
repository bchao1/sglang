# Scratch Directory

Personal test scripts and experiments for sglang contribution work.
Excluded from git on all branches via `.git/info/exclude`.
Tracked on `dev/brian` branch of `origin` for backup and history.

## Workflow

### Daily work
Put test scripts here freely — they're invisible to git on feature branches.

### Save progress to the cloud
```bash
git checkout dev/brian
git add -f scratch/
git commit -m "wip: save test scripts"
git push origin dev/brian
git checkout main   # or your feature branch
```

### Start a new feature PR
```bash
# Sync with upstream first
git fetch upstream
git checkout main
git merge upstream/main
git push origin main

# Create a clean feature branch
git checkout -b feature/my-feature
# ... write code, run tests from scratch/ as needed ...
# Only commit the real implementation files, never scratch/
git push origin feature/my-feature
# Open PR: bchao1/sglang → sgl-project/sglang, base: main
```

### Restore scratch on a new machine
```bash
git checkout dev/brian
cp -r scratch/ /tmp/scratch_backup    # copy out
git checkout main                      # switch back
cp -r /tmp/scratch_backup scratch/    # restore on disk
```

## Remotes
- `origin`   → https://github.com/bchao1/sglang.git  (your fork — push feature branches here)
- `upstream` → https://github.com/sgl-project/sglang.git  (official repo — pull updates from here)
