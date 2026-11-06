# What was validated, where, and what is still pending

Numbers are quoted from the runs that produced them; "one run" is not a benchmark.

## Lifecycle on PR #303 (Cosmos QA, 2026-11-03 and 11-04)

Machine: RTX Spark N1X, Windows 11 Pro ARM64, 38 GB RAM (45.45 GiB CUDA-visible), no admin
rights, no WSL. Stack: ARM64 CPython 3.13.18 (uv), torch 2.15.0.dev20260902+cu134,
torchcodec 0.17.0a0 (built against that torch), transformers 4.57.6, peft 0.20.0,
cosmos-framework at PR #303 head 4475a3a plus the guards that are now part of the PR.

| Leg | Result |
|---|---|
| import and `train --help` | pass |
| reasoner config import and Cosmos3-Nano to VLM conversion | pass: 750 tensors, 4 shards, 16.3 GiB |
| one-step LoRA smoke | pass: 144 LoRA modules, 15,335,424 trainable parameters, 47 s first step, 33 GB DCP save |
| 20-step, 2-epoch run with validation | pass: train loss 2.17, validation loss 4.55 then 4.31, about 1.3 s per step after warm-up, 33 GB save in about two minutes; free host memory fell to 0.9 GB during the save |
| export (`export_vlm_dcp` with the base model) | pass for both checkpoints: 750 tensors, 5 shards, zero residual LoRA keys |
| Framework evaluate (binary, TorchCodec CUDA decode) | pass: 4 of 4 on the four-item smoke set |

Required overrides on that stack: `max_sequence_length 8192` (memory), sdpa on both policy
keys, async checkpointing off. All of them are now defaults of this skill's launcher or recipe.

Retest at the rebased head 7a6bca4 (main 7a4b057 plus the Windows commits; Cosmos QA, 2026-11-06), stock branch, no local
guards, no overrides, `toml` uninstalled: config import, the one-step smoke, a 20-step two-epoch run with validation
(train loss 2.17, validation 4.55 to 4.31), two 33 GB DCP saves at 132 s each, two exports with zero residual LoRA keys and
two evaluations all passed; the sdpa and synchronous-save fallbacks engaged on their own. Unit tests: 12 passed, 1 skipped,
3 failed, the failures being POSIX-only `os.statvfs` mocks. The two code findings (a second `toml` import in
`utils/workflow_status.py`, the statvfs mocks) are fixed in 69fee1f, the head this skill pins.

## Fine-tuning results on the earlier stack (ITS work, 2026-10-30 to 11-04)

Machine: RTX Spark N1X, 128 GB (101.5 GiB CUDA-visible), 1600 MHz clock cap, pacer 84/78/87 C.
Stack: cosmos-framework `02b05df` (the PR #164 base) with the vendored Windows patch, `cosmos`
attention. Data: 960 binary video QA records (synthetic traffic renders), 120-record held-out set.

| Configuration | Last-epoch train loss | Held-out accuracy / balanced | Train time | Peak C |
|---|---|---|---|---|
| base model (no training) | | 0.758 | | |
| recipe at 128x64, 3 epochs | 0.150 | 0.800 / 0.805 | 38.8 min | 75 |
| recipe v2: 8 frames at 480x288 + system prompt | 0.077 | 0.833 / 0.830 | 75.5 min (9.3 min paused) | 85 |
| recipe v2 + lr 2e-5, warm-up, 4 epochs | 0.139 | 0.783 / 0.780 | 99 min | 85 |

Memorisation test (64 records, 10 epochs, lr 1e-4, recipe v2 input): mean loss 0.478 / 0.148 /
0.004 / 0.000 at epochs 1 / 3 / 5 / 10; a matched evaluator answered 64 of 64.

## W14: why sdpa is right on main and was wrong before

On checkouts older than August 2026 the trainer dropped the attention mask and the HF `sdpa` and
`eager` paths derived false packed-sequence boundaries; training completed and reported a loss,
but the model converged to the label prior (loss 0.34 on yes/no sets, accuracy equal to the
base). The `cosmos` adapter did not have the defect, which is why the earlier Windows guidance
said "keep cosmos". Upstream fixed the mask handling in August (#198); on main sdpa receives the
mask and trains correctly, and on Windows the `cosmos` adapter's vision-tower path has no
backend, so #303 resolves it to sdpa. The memorisation test above is the one-command check for
either failure.

## Pending (as of 2026-11-05)

- This skill's commands on the stable torch `2.15.0+cu134` pairing with a matching TorchCodec:
  the QA run and the ITS runs used the 2026-09-02 nightly pairing.
- The recipe v2 numbers on main with sdpa (an equivalence run against the table above is
  scheduled on the 128 GB development unit).
- The synthetic smoke set's loss curve on both units.
- The eval cases in `eval/eval.config` require a Windows-on-Arm host; the Linux evaluator
  runners cannot execute them. Until a Windows runner class exists, a manual run with the
  report attached to the PR is the validation.
