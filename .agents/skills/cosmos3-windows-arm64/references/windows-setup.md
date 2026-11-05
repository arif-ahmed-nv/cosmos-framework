# The Windows-on-Arm environment, pin by pin

Everything here was validated on RTX Spark N1X units (Windows 11 on Arm, GB10-class GPU with
unified memory) between September and November 2026, on native Windows: no WSL, no containers,
no emulation. The x64 Windows route (RTX PRO 6000) shares the framework-side guards but not the
wheels; it is not described here.

## Interpreter

ARM64 CPython 3.13, managed by uv (`uv python install 3.13` on an ARM64 host installs the ARM64
build). `platform.machine()` must print `ARM64`. An x64 CPython under emulation is the most
common first-day mistake: everything installs, `torch.cuda.is_available()` is False, and pip
pulls the CPU-only `win_arm64` torch from PyPI if torch is not installed first.

Create the venv with `--seed` so pip exists, and `--no-config` so a cosmos-framework checkout's
`[tool.uv]` pin does not apply to this standalone environment.

## Torch

Install torch before the dependency file, from NVIDIA's Windows-on-Arm index, with
`--only-binary=:all:` so nothing is built from source by accident:

| Package | Pin | Where |
|---|---|---|
| torch | `2.15.0+cu134` | `https://pypi.nvidia.com/nvtorch_oot/` (stable; what the PyTorch "additional platforms" page lists for RTX Spark) |
| torchvision | `0.29.0+cu134` | same |
| torchaudio | `2.11.0+cu134` | same |

Skill note (not in the source): the earlier validation stack used the nightly
`torch 2.15.0.dev20260902+cu134` from `https://pypi.nvidia.com/nvtorch_oot_nightly/`, paired with
a TorchCodec built against it. The stable pin above is the target for the hackathon stack;
`validation.md` says which pairing each number was measured on. Nightlies after 2026-11-04
dropped DLL preloading and need `torch\lib` on the DLL search path, which the `.pth` helper below
provides anyway.

## The dependency file

`assets/requirements-windows-arm64.txt` is the framework's dependency set as pinned for
Windows, with `platform_machine` markers:

- skipped on ARM64, no `win_arm64` wheel: `pyarrow`, `obstore`, `datasets` (needs pyarrow),
  `av` (no cp313 wheel), `grpcio`; `duckdb` is installed in pyarrow's place for the mining
  scripts of the TAO skills;
- the x64 torch lines are marked out; torch comes from the index above;
- `cryptography==46.0.3` is the abi3 wheel that installs without an OpenSSL source build.

With PR #303 the framework imports pyarrow optionally, so the reasoner config graph imports
without it; Parquet features fail with a clear `ModuleNotFoundError` where they are used.

## The framework

`main` with PR #303 ("make training importable and runnable on native Windows"). Until it
merges: `git fetch origin pull/303/head:windows-compat && git checkout windows-compat` in a
clone of `NVIDIA/cosmos-framework`. Install editable with `--no-deps` (the dependency file
already satisfied everything). Checkouts without #303 fail on this platform at import (pyarrow),
at the first training step (attention backend) or at the first checkpoint save (statvfs, libuv,
pinned staging); do not patch them, update.

## TorchCodec

The trainer decodes video on the GPU through TorchCodec and refuses CPU fallback; a wheel for
your exact torch is mandatory. Three routes, in order of preference:

1. **A wheel built for your torch (win_arm64, CUDA 13.4, cp313).** NVIDIA's Windows-on-Arm
   PyTorch team builds TorchCodec wheels in its CI (internal distribution today; public
   distribution through the same channel as torch is in preparation). These wheels are
   self-contained: they bundle the FFmpeg 8 libraries, libjpeg and libpng, so no FFmpeg or
   GStreamer install is needed. Install with `--no-deps` and verify with
   `from torchcodec.decoders import VideoDecoder`.
2. **Build from source** against the installed torch: clone `pytorch/torchcodec`, install the
   FFmpeg 9 Windows ARM64 shared GPL build
   (`https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n9.0-latest-winarm64-gpl-shared-9.0.zip`)
   and the GStreamer 1.28 MSVC ARM64 runtime (for `jpeg8.dll` and `png16.dll`), build with CUDA
   enabled, and register FFmpeg's `bin` and GStreamer's `msvc_arm64\bin` with the `.pth` helper.
   The validated build was `torchcodec 0.17.0a0` with CUDA 13.4 NVDEC video, nvJPEG, CPU JPEG
   and PNG; WebP, AVIF, GIF and HEIC omitted. Budget an hour.
3. Internal NVIDIA users: the prebuilt wheel of the earlier validation (`0.17.0a0+1ebfad0`,
   built against the 2026-09-02 nightly) exists; it pairs only with that torch.

## The DLL search path

Python 3.13 on Windows does not resolve extension-module DLL dependencies from `PATH`.
`scripts/configure_windows_arm64_dll_paths.py --venv <venv>` writes one `.pth` file into the
venv's site-packages that calls `os.add_dll_directory` for torch's `lib`, the CUDA ARM64 `bin`
and cuDNN directories if a toolkit is installed, GStreamer's `msvc_arm64\bin` if present, and
anything passed with `--extra-dll-dir`, and keeps the handles alive in every interpreter the
venv starts, including spawned workers and console scripts. An import failure for
`libtorchcodec_image.dll` means one of those directories is missing from the file.

## The model

`nvidia/Cosmos3-Nano` is gated on Hugging Face; accept the gate and export `HF_TOKEN`. The
converter (`python -m cosmos_framework.scripts.convert_model_to_vlm_safetensors --checkpoint-path
<release dir> -o <VLM dir>`) runs on the CPU, overlays the Cosmos3 vision tower and language
model onto a Qwen3-VL-shaped HF directory, and fetches the `Qwen/Qwen3-VL-8B-Instruct` processor
(about 16 GB of cache). Keep the venv's `Scripts` on `PATH` while it runs so a nested `uv` call
resolves. Expected: 750 tensors in 4 shards, 16.3 GiB. Disk budget: release 33 GB, cache 16 GB,
converted 16 GB, plus 33 GB per DCP checkpoint.

## Memory and GPU readings

`nvidia-smi` on WDDM shows the dedicated segment (8 GiB) and not the CUDA budget; the preflight
prints the budget (101.5 GiB on the 128 GB unit; about 45 GiB on the 64 GB unit). Under WSL the
readings differ again; this skill does not cover WSL.

## Verify

`scripts/preflight_windows_arm64.py --allocation-gib N` (ARM64, torch, CUDA, a real allocation of
N GiB), the three import lines of Variant 1 Block 2, and a one-step training smoke before any
real run.
