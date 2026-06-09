# Final PR submission plan
Please exactly follow the described logic for the follow final PR plan.

## Feature branches involved
- bchao1/spectral-progressive-flux
- bchao1/spectral-progressive-flux2
- bchao1/spectral-progressive-wan
- bchao1/spectral-progressive-z-image
- bchao1/spectral-progressive-qwen
- dev/brian

## Explanation and background settings
- All feature branches extends bchao1/spectral-progressive-flux
- All scripts generating figures and plots, PR descriptions, etc, are in `scratch/`, and only tracked on `dev/brian` branch
- `scratch` should never be tracked on feature branches
- `.gitignore` should always match the original sglang main branch


## Steps to do
1. For all feature branches, follow the denoising router and denoising stage logic in the `bchao1/spectral-progressive-flux` branch during setup, dispatch stage and `SamplingParams` and CLI arguments.
2. Rebase `bchao1/spectral-progressive-flux` to the most recent main branch for original sglang remote. Work with me to resolve merge conflicts.
3. One by one, merge the other feature branches into the `bchao1/spectral-progressive-flux` branch. Work with me to resolve merge conflicts.
4. For unit tests of all features branches (CPU-based unit tests and E2E GPU tests), merge into single file `test_progressive.py`.
5. Run standard pre-commit, unit tests, etc and commit.
- After feature branches are done and all scripts are set, all of the following step after step 5 in the plan happens on `dev/brian` (just running visuals and benchmarking), with dev-related stuff, GPU picking, etc.
6. After all code is fixed and merged, smoke test on one prompt full resolution generation AND `dct_rewind + 0.05` version for all models and save results to scratch. 
    - For the qwen image model, might need to use DiT off load or offloading operations since there is OOM when I tested.
    - OOM'ed at final decoding stage
7. Work on PR ticket and documentation


## Final PR ticket outline
PR ticket name: `final_PR_ticket.md` in `scratch` folder. 
- Follow structure and logic of `scratch/spectral-progressive-flux/PR_description.md` in `dev/brian` branch.
- Merge all other PR tickets of feature branches (in respective `PR_description.md`) files into a singular file.
- Overall explain what files are added and changed
- Usage for all mdoels and setting params, and point to docs
- Report result for all tests of all models
- Create speedup table for all models, and singular plot for models (all curves on single plot with legend)
- Merge results that can be organized into a single table. 
- Remove torch.compile comparisons.
- For generate images, show drop down lists and images for all models -- retain all results in all `PR_description.md` files for models.
- Work with me to layout structure, remove results, add results, if unclear.


## Final Documentation outline
File path: `docs_new/docs/sglang-diffusion/progressive_resolution.mdx`. 
- Follow the structure and logic of file on `bchao1/spectral-progressive-flux` branch, but each section explain more if there is difference among models. 
- Briefly describe what these feature does
- Explain the new arguments and how to set how how it does
- Explain there is support for different models
- Explain what models are supported and how to invoke calls
- NO benchmarking on the documentation