---
name: cosmos3-windows-arm64
description: >
  Set up cosmos-framework and run the Cosmos3 Nano Reasoner lifecycle (LoRA SFT on video
  question answering, merged-safetensors export, Framework evaluation) natively on Windows 11
  on Arm, on an RTX Spark laptop or compact desktop, without WSL or containers. Use when the
  user asks "run Cosmos3 on my RTX Spark", "fine-tune Cosmos Reason on Windows on Arm",
  "LoRA SFT on a Windows laptop", "install cosmos-framework on Windows", or hits Windows-only
  failures (no NCCL or FlashAttention, pyarrow or TorchCodec wheel missing, checkpoint save out
  of memory, "Could not find a compatible Attention backend", GPU far slower than expected).
  Not for Linux hosts, for WSL, for multi-GPU, or for the Cosmos3 generator models.
---

# Cosmos3 Reasoner on Windows on Arm (RTX Spark)

One job: take the public `nvidia/Cosmos3-Nano` release, LoRA-fine-tune its Reasoner on your
video QA records on the laptop's own GPU, export a merged Hugging Face directory, and measure
it with the Framework evaluator. Input: an annotation JSON (one record per clip and question,
LLaVA conversation format) and the clips. Output: `<run>/checkpoints/epoch_N` (DCP),
`<run>/export/model.safetensors*` (merged, zero LoRA keys), and `results.json` plus a metrics
summary from the evaluator. Measured on an RTX Spark N1X: a 960-record, 3-epoch LoRA run at
8 frames of 480x288 takes about 75 minutes under the thermal limiter.

## Read this first

1. **The interpreter must be native ARM64.** `python -c "import platform; print(platform.machine())"`
   must print `ARM64`. An x64 Python under emulation installs and imports everything and cannot
   see the GPU. `scripts/preflight_windows_arm64.py` checks this before anything else.
2. **Memory is unified; nvidia-smi lies about it.** WDDM shows an 8 GiB dedicated segment.
   The CUDA budget is what the preflight prints: about 101 GiB on the 128 GB unit, about 45 GiB
   on the 64 GB unit. A DCP checkpoint save needs roughly the state-dict size (18 GB for Nano)
   in free host memory on top of the resident model, so nothing else runs during a save.
3. **Framework version: `main` with PR #303.** Until that PR merges, use the branch named in
   Variant 1 (main plus the Windows commits). Older checkouts fail on this platform in known
   ways (pyarrow at import, the attention backend, checkpoint saves); do not patch them, update.
4. **Attention: pass `sdpa` on both policy keys on every run.** On Windows the default
   `cosmos` adapter has no backend for the Qwen3-VL vision tower (FlashAttention has no Windows
   build and cuDNN rejects varlen). #303 resolves it to `sdpa` automatically; passing both keys
   makes the run self-describing. The trainer forwards the attention mask under sdpa on main,
   so sdpa trains correctly here (history: `references/validation.md` § W14).
5. **One GPU job at a time, under the thermal limiter.** `scripts/train_lora.ps1` runs the job
   inside `scripts/thermal/run_measured.ps1` with the pacer; restore default clocks after every
   job (`scripts/thermal/clock_cap.ps1 -Restore`). Thresholds and why:
   `references/thermal-operating-point.md`.
6. **The Windows overrides live on the command line.** Nine of the eleven cannot be written
   into a recipe TOML (the schema rejects them); the launcher appends them. Do not accept a
   "cleaned-up recipe" that claims to have absorbed them (`references/training-overrides.md`).
7. **Verify before reporting.** The LoRA injection line (144 modules), a falling loss, zero
   residual LoRA keys after export, and the evaluator's metrics file. Section "Verify the output".

## When to use this skill

- Use to install cosmos-framework and its dependencies on Windows on Arm and prove the GPU path
- Use to LoRA-fine-tune the Cosmos3 Nano Reasoner on video QA, export it and evaluate it, on one Spark
- Use when a Windows run fails with a platform error listed under "Things not obvious from the docs"
- Do not use for Linux or WSL (follow `../cosmos3-setup/SKILL.md` and `../cosmos3-post-training/SKILL.md`),
  for multi-GPU or multi-node, or for the generator models (`../cosmos3-generate-media/SKILL.md`)

## Path convention

Paths starting with `scripts/`, `assets/` and `references/` are relative to this skill
directory. The repository root is `../../../` from here; `python -m cosmos_framework...` works
from any directory once the package is installed editable.

## Hardware: what fits

| Unit | CUDA-visible | LoRA SFT, 8 frames at 480x288 | LoRA SFT, 8 frames at 128x64 | Notes |
|---|---|---|---|---|
| RTX Spark, 128 GB | about 101 GiB | yes; about 1.1 s per step, 25 to 28 W, 81 to 85 C under the pacer | yes; about 0.5 s per step, 17 W, 73 to 77 C | all variants below |
| RTX Spark, 64 GB | about 45 GiB | yes with `max_sequence_length = 8192` in the recipe (one line); saves take about two minutes and most of the host memory | yes | measured by QA on a 38 GB-RAM unit: 20-step run at 1.3 s per step, 33 GB DCP per save |

Disk: raw release 33 GB, processor cache 16 GB, converted model 16 GB, one DCP checkpoint 33 GB.

## Variant 1 — Environment

Reference: [`references/windows-setup.md`](references/windows-setup.md) (what each pin is for,
the dependency markers, the TorchCodec routes, the DLL search path).

Pins that are not negotiable: ARM64 CPython 3.13; torch `2.15.0+cu134` with torchvision
`0.29.0+cu134` and torchaudio `2.11.0+cu134` from NVIDIA's Windows-on-Arm index (these are the
stable wheels the PyTorch "additional platforms" page lists for RTX Spark); a TorchCodec wheel
built for that torch (`win_arm64`, CUDA 13.4). Run each block as one PowerShell call; nothing
carries over between blocks except files on disk.

```powershell
# Block 1: interpreter, torch, preflight. $ROOT is the only path you choose; $SKILL is the directory holding this SKILL.md.
$ROOT = "C:\cosmos3"; $VENV = "$ROOT\venv"; $PY = "$VENV\Scripts\python.exe"; $SKILL = "<directory of this SKILL.md>"
New-Item -ItemType Directory -Force -Path $ROOT | Out-Null
uv python install 3.13
uv venv $VENV --python 3.13 --seed --no-config
& $PY -c "import platform, sys; assert platform.machine() == 'ARM64', platform.machine(); print(platform.machine(), sys.version.split()[0])"
& $PY -m pip install --only-binary=:all: --extra-index-url https://pypi.nvidia.com/nvtorch_oot/ "torch==2.15.0+cu134" "torchvision==0.29.0+cu134" "torchaudio==2.11.0+cu134"
& $PY $SKILL\scripts\preflight_windows_arm64.py --allocation-gib 72    # 128 GB unit; use 32 on the 64 GB unit
```

Expected: the preflight prints JSON with `"status": "PASS"`, `machine: ARM64`, the torch
version, `torch_cuda: 13.4`, the device name and the total CUDA memory. The allocation gate
allocates, writes, synchronizes and frees the requested buffer, so run it on an idle GPU.

```powershell
# Block 2: the framework at main + #303, its dependency set, TorchCodec, the DLL search path.
$ROOT = "C:\cosmos3"; $VENV = "$ROOT\venv"; $PY = "$VENV\Scripts\python.exe"; $REPO = "$ROOT\cosmos-framework"; $SKILL = "<directory of this SKILL.md>"
$TORCHCODEC_WHEEL = "<path to torchcodec-*+cu134-cp313-cp313-win_arm64.whl>"   # see references/windows-setup.md § TorchCodec
& $PY -m pip install -r $SKILL\assets\requirements-windows-arm64.txt
if (-not (Test-Path $REPO)) { git clone https://github.com/NVIDIA/cosmos-framework.git $REPO }
git -C $REPO fetch origin pull/303/head:windows-compat; git -C $REPO checkout windows-compat   # after #303 merges: git -C $REPO checkout main
& $PY -m pip install -e $REPO --no-deps
& $PY -m pip install --no-deps $TORCHCODEC_WHEEL
& $PY $SKILL\scripts\configure_windows_arm64_dll_paths.py --venv $VENV
& $PY -c "import torch, torchcodec, cosmos_framework; from torchcodec.decoders import VideoDecoder; print(torch.__version__, torchcodec.__version__, torch.cuda.is_available())"
& $PY -c "import sys; sys.modules['pyarrow'] = None; import cosmos_framework.configs.base.reasoner.config; print('reasoner config imports without pyarrow')"
& $PY -m cosmos_framework.scripts.train --help | Select-Object -First 3
```

Expected: the three checks print the torch and torchcodec versions with `True`, the
pyarrow-free import line, and the trainer's usage. If `configure_windows_arm64_dll_paths.py`
reports a missing directory, add it with `--extra-dll-dir` (the FFmpeg `bin` of a self-built
TorchCodec, for example); Python 3.13 does not resolve extension DLLs from `PATH`.

## Variant 2 — Convert the base model

Reference: `references/windows-setup.md` § Model. Needs the Hugging Face gate for
`nvidia/Cosmos3-Nano` accepted and `HF_TOKEN` in the shell. The converter also fetches the
`Qwen/Qwen3-VL-8B-Instruct` processor and runs on the CPU.

```powershell
$ROOT = "C:\cosmos3"; $VENV = "$ROOT\venv"; $PY = "$VENV\Scripts\python.exe"; $MODELS = "$ROOT\models"
$env:PATH = "$VENV\Scripts;$env:PATH"
& $PY -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/Cosmos3-Nano', local_dir=r'$MODELS\Cosmos3-Nano')"
& $PY -m cosmos_framework.scripts.convert_model_to_vlm_safetensors --checkpoint-path "$MODELS\Cosmos3-Nano" -o "$MODELS\Cosmos3-Nano-VLM"
Get-ChildItem "$MODELS\Cosmos3-Nano-VLM" | Measure-Object -Property Length -Sum | ForEach-Object { "{0:N1} GiB" -f ($_.Sum / 1GB) }
```

Expected: 750 tensors in 4 shards, 16.3 GiB, with `config.json`, the safetensors index and the
tokenizer files. Every later step points `VLM_SAFETENSORS_PATH` at this directory, never at the
raw release or the Hugging Face id.

## Variant 3 — LoRA SFT

Reference: [`references/training-overrides.md`](references/training-overrides.md) (the eleven
overrides, the attention keys, the environment block, what `trainer.max_iter` does not do).

The recipe is `assets/video_sft_nano_windows.toml`: main's `video_sft_nano.toml` with the
Windows deviations marked `# WINDOWS`. As shipped it is a 20-step smoke (two epochs of ten
steps); for a real run set `trainer.steps_per_epoch` to your record count,
`trainer.num_epochs` to 3, `trainer.max_iter` to their product and `scheduler.cycle_lengths`
to the same number (the CLI override `trainer.max_iter` is accepted and ignored). On the 64 GB
unit set `max_sequence_length = 8192`.

Data: the experiment reads its records and knobs from environment variables that the launcher
sets from its parameters: `COSMOS_VIDEO_TRAIN_ANNOTATION`, `COSMOS_VIDEO_TRAIN_MEDIA`,
`COSMOS_VIDEO_VAL_ANNOTATION`, `COSMOS_VIDEO_VAL_MEDIA`, `COSMOS_VIDEO_NUM_FRAMES`,
`COSMOS_VIDEO_MAX_PIXELS` (a per-clip pixel budget, not per frame: 1179648 gives 8 frames at
480x288, the default 81920 gives 8 frames at 128x64, where a pedestrian is half a visual
token), `COSMOS_VIDEO_SYSTEM_PROMPT`. Records are LLaVA conversations: `assets/smoke/` shows
the format; `scripts/make_smoke_set.py` writes ten synthetic 33-frame clips for it.

```powershell
# Smoke on the bundled synthetic set, then the same command on your data.
$ROOT = "C:\cosmos3"; $VENV = "$ROOT\venv"; $PY = "$VENV\Scripts\python.exe"; $SKILL = "<directory of this SKILL.md>"
& $PY $SKILL\scripts\make_smoke_set.py --out "$ROOT\data\smoke"     # needs ffmpeg on PATH or FFMPEG set
powershell -ExecutionPolicy Bypass -File $SKILL\scripts\train_lora.ps1 `
  -Python $PY -Recipe $SKILL\assets\video_sft_nano_windows.toml `
  -Model "$ROOT\models\Cosmos3-Nano-VLM" -OutputRoot "$ROOT\out" -JobName smoke_20 `
  -TrainAnnotation "$ROOT\data\smoke\annotations_train.json" -TrainMedia "$ROOT\data\smoke\media" `
  -ValAnnotation "$ROOT\data\smoke\annotations_val.json" -ValMedia "$ROOT\data\smoke\media" `
  -NumFrames 8 -MaxPixels 1179648 -SystemPrompt "You are a helpful assistant." -ClockCapMHz 1600
```

Expected, in order: the preflight line, `clock cap 1600 MHz applied`, the pacer armed, the
model load, `LoRA injection successful: 144 modules wrapped, 15,335,424 trainable LoRA params`,
twenty `train/avg_loss` lines (the smoke set is learnable: the loss falls below 1.0 by the
second epoch), a synchronous DCP save of about 33 GB under
`<OutputRoot>\cosmos3_reasoner\cosmos_video_conversation_sft\<JobName>\checkpoints\epoch_2`,
`Cosmos Framework video-conversation SFT training completed successfully`, the receipt under
`<OutputRoot>\evidence\`, and `default clocks restored`. About 47 s for the first step and 1.3 s
per step after warm-up on the 64 GB unit; about 1.1 s per step on the 128 GB unit.

## Variant 4 — Export the merged model

```powershell
$ROOT = "C:\cosmos3"; $VENV = "$ROOT\venv"; $PY = "$VENV\Scripts\python.exe"
$RUN = "$ROOT\out\cosmos3_reasoner\cosmos_video_conversation_sft\smoke_20"
& $PY -m cosmos_framework.scripts.export_vlm_dcp --checkpoint-path "$RUN\checkpoints\epoch_2" --config-file "$RUN\config.yaml" --output-dir "$RUN\export" --base-model-path-or-uri "$ROOT\models\Cosmos3-Nano-VLM"
& $PY -c "import json, sys; m = json.load(open(sys.argv[1]))['weight_map']; print(len(m), 'tensors;', sum('lora' in k.lower() for k in m), 'lora keys')" "$RUN\export\model.safetensors.index.json"
```

Expected: 750 tensors, 5 shards, 16.3 GiB, `0 lora keys`. The export is a complete Hugging Face
directory that the evaluator below, transformers, or a vLLM server can load.

## Variant 5 — Evaluate

Reference: `references/training-overrides.md` § Evaluating. Copy
`assets/evaluate_binary.toml`, set `results_dir`, the annotation and media paths and
`model.model_name` (the export directory, or the converted base model for a baseline), keep the
`[vision]` block in step with what the model was trained on (frames and pixel budget), then:

```powershell
$ROOT = "C:\cosmos3"; $VENV = "$ROOT\venv"
$env:RANK = "0"; $env:WORLD_SIZE = "1"; $env:LOCAL_RANK = "0"; $env:PYTHONIOENCODING = "utf-8"
& "$VENV\Scripts\cosmos-reasoner-evaluate.exe" --config "$ROOT\out\eval_smoke.toml"
```

Expected: one `results.json` row per record (`id`, `question`, `response`, `gt`, `task`) under
`results_dir`, and the summary with overall and balanced accuracy, precision, recall and F1. The
binary scorer needs both classes present in the ground truth; `WEIGHTED_SCORE 0` in the log is
a known reporting artifact, read `overall.accuracy`. The data loader needs one worker with
prefetch and the spawn context (the asset's defaults) or zero workers with prefetch disabled;
the mixed setting fails at start.

## Verify the output

- Environment: the three Block 2 checks; `nvidia-smi` shows the process while a step runs.
- Training: the LoRA injection line with 144 modules; without it the run is a full-parameter
  update and will run out of memory. A loss that sits near 0.34 on a yes/no set and never falls
  means the model is learning the answer prior, not the video: check the two attention keys and
  the pixel budget, then run the smoke set.
- Export: `0 lora keys` in the index; the directory loads with transformers.
- Evaluation: `results.json` row count equals the record count; the metrics summary exists.
- Thermal: the receipt's `temp_c_max` and the pacer's pause count; a run that is several times
  slower than the table above is a latched GPU (`references/thermal-operating-point.md`).

## Reference files

| File | Content |
|---|---|
| `references/windows-setup.md` | The environment: pins, dependency markers, FFmpeg and TorchCodec routes, the DLL `.pth`, the model conversion |
| `references/training-overrides.md` | The eleven overrides and what each prevents, the attention keys, the environment block, inert overrides, the evaluator TOML |
| `references/thermal-operating-point.md` | The limiter scripts, the measured operating point, the failure modes (hard reset, latched low-power state, clock limiter) |
| `references/validation.md` | What was validated where: QA's lifecycle run on #303, the ITS fine-tuning ladder, the W14 history |
| `assets/video_sft_nano_windows.toml` | The recipe (main's Nano video SFT with the Windows deviations) |
| `assets/evaluate_binary.toml` | Evaluator template for yes/no video QA |
| `assets/smoke/` | The synthetic smoke annotations; `scripts/make_smoke_set.py` writes the clips |
| `assets/requirements-windows-arm64.txt` | The framework dependency set with the Windows-on-Arm markers |

## Things not obvious from the docs

- **Why sdpa here when Linux uses cosmos.** The `cosmos` adapter gives the vision tower one
  varlen attention call per block; only FlashAttention serves varlen, and FlashAttention has no
  Windows build. On main the sdpa path receives the attention mask, so it trains correctly; on
  checkouts older than August 2026 it trained silently to the label prior (W14).
- **`trainer.max_iter` on the command line is accepted and ignored.** The recipe's
  `scheduler.cycle_lengths`, `num_epochs` and `steps_per_epoch` decide the length; count the
  logged iterations.
- **Checkpoint memory.** Both DCP paths stage the whole state dict through host memory. #303
  makes saves synchronous and serial on Windows; a save still needs about the state-dict size
  free. Close everything else during saves. LoRA-only checkpoints are a follow-up.
- **pyarrow, datasets, av, obstore, grpcio have no win_arm64 wheels.** The dependency file
  marks them out; #303 makes pyarrow optional in the framework. Parquet features fail with a
  clear `ModuleNotFoundError` where used.
- **Two policy objects.** Overrides for the policy go to `model.config.policy.*`; the
  top-level `policy.*` key is accepted and not read by the model. The launcher sets both for the
  attention setting; do the same for `backbone.*` if you ever override it.
- **`IMAGINAIRE_OUTPUT_ROOT`** defaults to a drive-relative `/tmp` path on Windows and degrades
  silently; the launcher always sets it. `PYTHONIOENCODING=utf-8` prevents a cp1252 console
  from killing the run on a logged box-drawing character.
- **Gloo at world size 1 is mandatory** (the trainer runs an unconditional barrier), which is
  why the five rendezvous variables and `USE_LIBUV=0` are set even for one process.
- **Administrator rights are not required**; `LongPathsEnabled` off is fine for this layout.

## Related skills

| Skill | When to use |
|---|---|
| `../cosmos3-setup/SKILL.md` | Linux and container setup |
| `../cosmos3-post-training/SKILL.md` | The Linux post-training recipes this skill ports |
| `../cosmos3-env-troubleshoot/SKILL.md` | CUDA and import failures that are not Windows-specific |
| `../cosmos3-inference/SKILL.md` | Serving and inference after export |
