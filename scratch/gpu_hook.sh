#!/usr/bin/env bash
# PreToolUse hook: auto-select best free GPU before CUDA experiment commands.
#
# Fires when a Bash command looks like an sglang experiment or benchmark.
# Skips commands that already set CUDA_VISIBLE_DEVICES.
# Outputs additionalContext telling Claude which GPU to use.

INPUT=$(cat)

# Extract command using Python (jq may not be on PATH in all envs)
cmd=$(echo "$INPUT" | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('command',''))" \
    2>/dev/null)

# Only fire for experiment-like commands
if ! echo "$cmd" | grep -qE \
    '(sglang (generate|serve)|bench_.*\.sh|quality.*\.sh|FLASHINFER|dit-cpu-offload|progressive-mode|diffusion)'; then
    exit 0
fi

# Already has an explicit GPU set → respect it, don't override
if echo "$cmd" | grep -q 'CUDA_VISIBLE_DEVICES'; then
    exit 0
fi

cd /home/brianchc/sglang
GPU_ID=$(bash scratch/select_gpu.sh 2>/dev/null)
FREE=$(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits \
    -i "$GPU_ID" 2>/dev/null \
    | awk -F', ' '{printf "%d/%d MiB free", $1, $2}')

python3 -c "
import json, sys
gpu = '$GPU_ID'
mem = '$FREE'
print(json.dumps({
    'hookSpecificOutput': {
        'hookEventName': 'PreToolUse',
        'additionalContext': f'GPU auto-selector: best free GPU is {gpu} ({mem}). Set CUDA_VISIBLE_DEVICES={gpu} in this command or at the top of the script.'
    }
}))
"
