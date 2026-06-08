"""
Smoke test: Cache-DiT × Progressive resolution on FLUX.1-dev.

Runs four configurations and checks that all succeed without error:
  1. fullres  + no cache-dit  (baseline, sanity)
  2. fullres  + cache-dit
  3. progressive (dct_rewind L1 δ=0.05) + no cache-dit
  4. progressive (dct_rewind L1 δ=0.05) + cache-dit  ← the newly fixed combo

Each run uses 20 steps (fast smoke) and saves output to results/cache_dit_smoke/.
Timing for each config is printed to confirm progressive speedup survives the fix.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# Pick the GPU with the most free memory.
_gpu = subprocess.check_output(
    "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
    " | awk -F', ' '{ if ($2 > max) { max=$2; idx=$1 } } END { print idx }'",
    shell=True,
    text=True,
).strip()
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
print(f"[select_gpu] CUDA_VISIBLE_DEVICES={_gpu}")

# Point at repo source so editable install picks up the patch.
sys.path.insert(0, str(Path(__file__).parents[3]))

os.environ["SGLANG_CACHE_DIT_ENABLED"] = "1"
# FA3 requires Hopper (H100+); A6000 is Ampere — force FA2.
os.environ.setdefault("SGLANG_DIFFUSION_ATTENTION_BACKEND", "flash_attn2")

from sglang.multimodal_gen import DiffGenerator  # noqa: E402

MODEL = "/miele/brian/modelscope/black-forest-labs/FLUX.1-dev"
PROMPT = "A serene mountain lake at golden hour, photorealistic, 8k"
STEPS = 20
OUT = Path(__file__).parent / "results" / "cache_dit_smoke"
OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    dict(
        label="fullres_no_cache",
        progressive_mode="fullres",
        cache_dit=False,
    ),
    dict(
        label="fullres_cache_dit",
        progressive_mode="fullres",
        cache_dit=True,
    ),
    dict(
        label="progressive_no_cache",
        progressive_mode="dct_rewind",
        cache_dit=False,
    ),
    dict(
        label="progressive_cache_dit",
        progressive_mode="dct_rewind",
        cache_dit=True,
    ),
]


def run_config(gen: DiffGenerator, cfg: dict) -> float:
    label = cfg["label"]
    if not cfg["cache_dit"]:
        os.environ["SGLANG_CACHE_DIT_ENABLED"] = "0"
    else:
        os.environ["SGLANG_CACHE_DIT_ENABLED"] = "1"

    sp_kwargs = {}
    if cfg["progressive_mode"] != "fullres":
        sp_kwargs = dict(
            progressive_mode=cfg["progressive_mode"],
            progressive_levels=1,
            progressive_delta=0.05,
        )

    print(f"\n{'='*60}")
    print(f"Config: {label}")
    print(f"  progressive_mode={cfg['progressive_mode']}, cache_dit={cfg['cache_dit']}")

    t0 = time.time()
    result = gen.generate(
        sampling_params_kwargs={
            "prompt": PROMPT,
            "num_inference_steps": STEPS,
            "height": 1024,
            "width": 1024,
            **sp_kwargs,
        }
    )
    elapsed = time.time() - t0

    # DiffGenerator swallows generation errors internally; detect failure via
    # missing images rather than relying on an exception propagating.
    images = getattr(result, "images", None)
    if not images:
        raise RuntimeError(
            f"Generation returned no images (elapsed {elapsed:.2f}s) — "
            "check logs above for the underlying error."
        )
    img = images[0]
    out_path = OUT / f"{label}.png"
    img.save(out_path)
    print(f"  Saved: {out_path}")
    print(f"  Wall time: {elapsed:.2f}s")
    return elapsed


def main():
    print(f"Model: {MODEL}")
    print(f"Steps: {STEPS}, Resolution: 1024x1024")

    gen = DiffGenerator.from_pretrained(
        model_path=MODEL,
        dit_cpu_offload=False,
    )

    timings = {}
    errors = {}
    for cfg in CONFIGS:
        label = cfg["label"]
        try:
            timings[label] = run_config(gen, cfg)
        except Exception as e:
            errors[label] = e
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for label, t in timings.items():
        print(f"  {label:<30} {t:6.2f}s")
    for label, e in errors.items():
        print(f"  {label:<30} FAILED: {e}")

    baseline = timings.get("fullres_no_cache")
    prog_cache = timings.get("progressive_cache_dit")
    if baseline and prog_cache:
        print(
            f"\n  progressive+cache_dit speedup vs fullres baseline: {baseline/prog_cache:.2f}x"
        )

    if errors:
        print(f"\nFAILED: {len(errors)} config(s) errored")
        sys.exit(1)
    else:
        print("\nAll configs passed.")


if __name__ == "__main__":
    main()
