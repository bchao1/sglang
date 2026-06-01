# SGLang Contribution Workflow (Brian's Fork)

This file is local-only (in .git/info/exclude). It defines non-negotiable rules
that Claude must follow automatically — never wait to be asked.

## Feature Branches

When the user says "new feature", "start a feature", "I want to work on X", or anything
implying new work, immediately ask:
1. Feature name (becomes `bchao1/<name>` and `scratch/<name>/`)
2. Base branch: `main` or extend an existing branch?

Then run:
```bash
git fetch upstream && git checkout <base> && git merge upstream/<base>  # if from main
git checkout -b bchao1/<feature-name>
mkdir -p scratch/<feature-name>
```

## Scratch Directory Rules

- Every feature's utils go in `scratch/<feature-name>/` — never at scratch root
- Every feature needs `scratch/<feature-name>/description.md` (use progressive_diffusion_integration.md format as template)
- After adding/modifying any file in scratch: `bash scratch/save_scratch.sh "<feature-name>: <what changed>"`
- NEVER use `git checkout dev/brian` to save scratch — always use `save_scratch.sh` (worktree approach, no branch switch)
- `scratch/` is excluded from all branches via `.git/info/exclude`; `dev/brian` is backup-only

## PR Rules

PR body must use `scratch/PR_template.md`. Before opening any PR:
1. `pre-commit run --all-files` must pass
2. Unit tests added in `test/` (not scratch)
3. `scratch/<feature-name>/description.md` updated with final benchmarks
4. PR opened from `bchao1/sglang → sgl-project/sglang`, base `main`
5. CI triggered with `/tag-and-rerun-ci` comment

## Coding Style (from https://docs.sglang.io/docs/developer_guide/contribution_guide)

For all production code (code that goes into a PR):
- `pre-commit run --all-files` must pass (black, isort, ruff, trailing whitespace)
- No file > 2000 lines — split if needed
- No code duplication > 5 lines — extract to shared function
- No `.item()` or `.cpu()` on the hot path
- Cache repeated boolean checks in `__init__`, not inline
- Pure functions, no in-place argument modification
- No `pickle.loads()` / `recv_pyobj()` for network data — use msgpack/JSON

## Must-Read Skills Before Editing

- Speculative decoding (`srt/speculative/`) → invoke `speculative-naming` skill
- `Scheduler`/`TokenizerManager`/`ModelRunner` `__init__` → invoke `large-class-init-style` skill
- Any `SGLANG_*` env var or `environ.py` → invoke `env-var-conventions` skill

## Branch Naming

- Feature branches: `bchao1/<feature-name>` (NOT `feature/<name>`)
- Scratch backup: `dev/brian` (never PR from this)
- `origin` = `bchao1/sglang` (push here), `upstream` = `sgl-project/sglang` (pull from here)
