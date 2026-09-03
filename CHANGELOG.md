# Changelog

All notable changes to this fork are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/). Upstream history before the fork lives in
[michaelfeil/infinity](https://github.com/michaelfeil/infinity/releases).

## [Unreleased]

### Added

- `docs/benchmarks/cpu/`: load generator, parity checks and an int8 quantization script for the
  CPU image, with the measured effect of `--pad-to-multiple-of`, OpenVINO bf16 and static int8,
  and a note on why bf16 needs a graph with fused `LayerNormalization`.

### Fixed

- `--dtype int8` on CPU (`torch.quantization.quantize_dynamic`) probes the int8 kernels in a
  separate interpreter first: on CPUs where they raise an illegal-instruction fault (seen on
  GitHub's Windows runners) the model now stays fp32 with a warning instead of the process dying
  without a Python exception.
- The cache path of offline-optimized ONNX graphs is derived from the model path without its
  root or drive. For a local model directory on Windows the optimized graph was written next to
  the model instead of into the cache.

### Changed

- The `-cpu` image is built natively per platform (`ubuntu-24.04-arm` for arm64 instead of QEMU)
  with the layer cache kept in ghcr (`buildcache-<platform>` tags); the test jobs install CPU
  torch wheels and share one virtualenv cache.
- `requirements_install_from_poetry.sh` installs the torch and torchvision versions from the lock
  file instead of the newest CPU wheels, and `--keep-venv` reuses an existing `.venv`.

## [0.1.1] - 2026-09-03

### Fixed

- The `optimum` engine failed with `KeyError: 'last_hidden_state'` on ONNX exports of
  sentence-transformers models (e.g. `BAAI/bge-m3`), whose hidden-state output is named
  `token_embeddings`. Both names are accepted; any other graph uses its first output, as optimum did.

## [0.1.0] - 2026-09-03

### Added

- `--pad-to-multiple-of` (env `INFINITY_PAD_TO_MULTIPLE_OF`, per model): pads every batch to a
  multiple of N tokens so the set of distinct `(batch, sequence)` shapes stays finite. Backends
  that cache kernels per shape (OpenVINO CPU plugin, oneDNN, torch CPU) otherwise grow without
  bound; a production deployment went from 12 GiB to 34 GiB RSS in a month. Applies to the
  optimum embedder and cross-encoder, the torch cross-encoder and sentence-transformers models.
- `--onnx-provider-options` (env `INFINITY_ONNX_PROVIDER_OPTIONS`, per model): JSON object merged
  into the onnxruntime execution provider options, e.g. `{"num_of_threads": 8}` for OpenVINO or
  `{"trt_fp16_enable": false}` for TensorRT.
- Per-model rerank token limits `--max-query-tokens`, `--max-tokens-per-doc`,
  `--max-pair-tokens` (upstream [#666](https://github.com/michaelfeil/infinity/pull/666)).
- `linux/arm64` builds of the `-cpu` image (upstream
  [#665](https://github.com/michaelfeil/infinity/pull/665)).
- Images published to `ghcr.io/elxreno/infinity:<version>-cpu` on every `v*` tag, scanned with
  Trivy; Renovate keeps the lock file and GitHub Actions current.

### Changed

- The `optimum` engine talks to onnxruntime directly. `optimum`/`optimum-onnx` are gone: they only
  wrapped `InferenceSession` and onnxruntime's own graph optimizer, and `optimum-onnx` pins
  `transformers<4.58`. The engine name, the CLI flags and the ONNX file selection are unchanged;
  the offline graph optimization now uses `onnxruntime.transformers.optimizer` with the same
  recipe. The `optimum` extra installs `onnxruntime` and `onnx`.
- Dependencies refreshed: Python >=3.12, torch 2.14, transformers 5.16, sentence-transformers 6.0,
  huggingface-hub 1.x, onnxruntime 1.24 / onnxruntime-openvino 1.24.1, numpy 2, typer 0.27 /
  click 8.5, fastapi 0.141, pytest 9, mypy 1.20. The CPU image is based on `ubuntu:24.04`.
- `--bettertransformer` is a no-op and defaults to off: BetterTransformer was removed from
  transformers 5 and optimum 2; torch's scaled dot product attention is the default anyway.
- `aiohttp`, used to fetch image and audio inputs by URL, is declared explicitly in the `server`,
  `vision` and `audio` extras instead of arriving as a transitive dependency.
- CLIP and CLAP text inputs are truncated to the smaller of the text tower's positions and the
  tokenizer's `model_max_length`; transformers 5 dropped `config.max_length`, and RoBERTa-style
  text towers (CLAP) reserve two positions, so a 514-token input used to crash the model.
- Models whose remote code still imports `transformers.models.clip.modeling_clip.clip_loss`
  (jinaai/jina-clip-v2 at the time of writing) do not load on transformers 5; the corresponding
  test is marked `xfail` until the upstream repository updates its code.
- `colpali-engine` is no longer part of the `vision` extra: every release pins an old torch. Install
  it manually to use the colpali engine.
- Classification scores honour the model's default activation (sigmoid for multi-label heads,
  softmax otherwise) instead of always applying softmax; `raw_scores=true` returns logits
  (builds on upstream [#662](https://github.com/michaelfeil/infinity/pull/662)).
- CI lints on Python 3.12 and 3.13; Python 3.9 legs are gone.

### Fixed

- ONNX models with external weights (`*.onnx_data`, every model above 2 GB) load again from the
  Hugging Face cache: onnxruntime >= 1.23 rejects external data whose canonical path leaves the
  model directory, and the cache stores files as symlinks into `blobs/`. The snapshot is now
  hardlinked into a plain directory under `HF_HOME/hub/infinity_onnx/materialized/` first.
- A cached `*_optimized.onnx` that fails to load no longer crashes startup; the unoptimized model
  is used instead. Previously only the first start fell back, the next one died.
- The CPU image no longer downloads the ~3 GB `cuda-toolkit` that torch >= 2.10 declares as a
  dependency; the requirements filter drops it before installing the CPU wheels.
- HF tokenizers reject `pad_to_multiple_of` when the truncation length is not a multiple of it
  (XLM-R models report 514 positions); the truncation length is now rounded down.
- A failed disk-cache write no longer kills the writer thread (upstream
  [#669](https://github.com/michaelfeil/infinity/pull/669)).
- Classifier checkpoints without `config.pad_token_id` adopt the tokenizer's value (upstream
  [#670](https://github.com/michaelfeil/infinity/pull/670)).
- Requests fail fast with `EngineUnhealthyError`, carrying the root cause, when a model worker
  stops, instead of hanging until the client times out and every later request answering 429
  (upstream [#668](https://github.com/michaelfeil/infinity/pull/668)).
- `test_batch_handler` no longer applies marks to a fixture, which pytest 9 rejects.

[Unreleased]: https://github.com/ElXreno/infinity/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ElXreno/infinity/compare/1eb4396...v0.1.0
