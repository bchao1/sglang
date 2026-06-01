# SGLang Contribution Workflow (Brian's Fork)

Local-only dev setup file. Defines non-negotiable rules Claude must follow
automatically — never wait to be asked.

---

## Dev File Contract

These files are NEVER tracked on feature branches (gitignored / git-excluded).
They live on disk and are backed up on `dev/brian` only.

| File | Purpose |
|------|---------|
| `CLAUDE.md` (this file) | Contribution workflow rules |
| `.claude/settings.json` | `bypassPermissions` mode |
| `.claude/settings.local.json` | Local permission allow-list |
| `scratch/<feature>/` | Feature test/bench scripts |

**When any of these files change** (user asks or Claude edits them):
```bash
bash scratch/update_dev.sh "<what changed>"   # saves CLAUDE.md + settings → dev/brian
bash scratch/save_scratch.sh "<feature>: <what changed>"  # if scratch/ changed too
```

---

## Creating a New Feature Branch

When the user says "new feature", "start a feature", "I want to work on X",
or anything implying new feature work, immediately ask:
1. Feature name → becomes `bchao1/<name>` and `scratch/<name>/`
2. Base: `main` or extend an existing branch? (if extend, which one?)

Then run in order:
```bash
# 1. Sync upstream (if branching from main)
git fetch upstream && git checkout main && git merge upstream/main && git push origin main

# 2. Create branch
git checkout -b bchao1/<feature-name>      # from main
# OR: git checkout bchao1/<existing> && git checkout -b bchao1/<new>

# 3. Restore untracked dev files onto the new branch (NOT tracked, just on disk)
bash scratch/restore_dev.sh

# 4. Create feature scratch subfolder
mkdir -p scratch/<feature-name>
# Create scratch/<feature-name>/description.md using spectral-progressive-flux/description.md as template
```

---

## Scratch Directory Rules

- All feature utils go in `scratch/<feature-name>/` — never at scratch root
- Every feature needs `scratch/<feature-name>/description.md`
- After adding/modifying any scratch file: `bash scratch/save_scratch.sh "<feature>: <what changed>"`
- NEVER `git checkout dev/brian` directly — always use the worktree scripts
- `scratch/` is excluded via `.git/info/exclude`; `dev/brian` is backup only

---

## PR Checklist

PR body must follow `scratch/PR_template.md`. Before opening any PR:
1. `pre-commit run --all-files` must pass
2. Unit tests added under `test/` (not scratch)
3. `scratch/<feature-name>/description.md` updated with final benchmark numbers
4. PR from `bchao1/sglang → sgl-project/sglang`, base `main`
5. Trigger CI with `/tag-and-rerun-ci` comment on the PR

---

## Coding Style (https://docs.sglang.io/docs/developer_guide/contribution_guide)

For all production code (anything going into a PR):
- `pre-commit run --all-files` passes (black, isort, ruff, trailing whitespace)
- No file > 2000 lines — split if needed
- No code duplication > 5 lines — extract to shared function
- No `.item()` or `.cpu()` on the hot path
- Cache repeated boolean checks in `__init__`, not inline
- Pure functions, no in-place argument modification
- No `pickle.loads()` / `recv_pyobj()` for network data — use msgpack/JSON

---

## Must-Read Skills Before Editing

- Speculative decoding (`srt/speculative/`, attention backends, CLI flags) → invoke `speculative-naming` skill first
- `Scheduler` / `TokenizerManager` / `ModelRunner` `__init__` → invoke `large-class-init-style` skill first
- Any `SGLANG_*` env var or `environ.py` → invoke `env-var-conventions` skill first

---

## Branch Naming

- Feature branches: `bchao1/<feature-name>` (NOT `feature/<name>`)
- Dev backup branch: `dev/brian` (never PR from this)
- `origin` = `https://github.com/bchao1/sglang.git` (push feature branches here)
- `upstream` = `https://github.com/sgl-project/sglang.git` (pull updates from here)
