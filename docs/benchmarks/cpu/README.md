# CPU benchmarks: padding, OpenVINO bf16 and int8

Throughput of the `-cpu` image (`optimum` engine, onnxruntime-openvino 1.24,
`OpenVINOExecutionProvider`) for two request shapes: an embedder receiving batches of
mixed-length texts, and a reranker scoring a short query against a few hundred short texts per
request. Measured on 2026-09-03 with `ghcr.io/elxreno/infinity:0.1.0-cpu` against
`michaelf34/infinity:0.0.77-cpu`.

## Setup

- Host: AMD Ryzen 7 8845H (Zen 4, `avx512_bf16` and `avx512_vnni`, no AMX), podman with
  `--cpus 8 --memory 24g`, one container at a time.
- Models: `ElXreno/LaBSE-en-ru-onnx` (BERT base, 768d) as the embedder and
  `keisuke-miyako/bge-reranker-v2-m3-onnx-f32` (XLM-RoBERTa large, fp32 export) as the reranker,
  batch sizes 16 and 4 unless stated otherwise, `--pooling-method cls`,
  `INFINITY_ONNX_DISABLE_OPTIMIZE=true`, `TOKENIZERS_PARALLELISM=true`.
- Texts: random word sequences drawn from `docs/assets/multilingual_calibration.utf8`
  (`corpus.py`), so the token lengths are realistic and nothing here is real data. The absolute
  numbers depend on the corpus; the ratios between configurations are what matters.
- Load: `workload.py mixed` (embedding batches of 8 texts of 3-25 or 5-120 words, reranking of
  8 documents of 40-300 words and 4 documents of 300-900 words) and `workload.py titles`
  (200 texts of 3-12 words per rerank request, 1/2/4 concurrent clients). Deterministic seeds,
  p50/p95 latency per request, throughput in texts or documents per second.

```bash
BF16='{"load_config": "{\"CPU\": {\"INFERENCE_PRECISION_HINT\": \"bf16\"}}"}'
./run.sh ghcr.io/elxreno/infinity:0.1.0-cpu fp32 --pad-to-multiple-of 16 --pad-to-multiple-of 16
./run.sh ghcr.io/elxreno/infinity:0.1.0-cpu bf16 --pad-to-multiple-of 16 --pad-to-multiple-of 16 \
  --onnx-provider-options "$BF16" --onnx-provider-options "$BF16"
python3 compare.py embeddings embeddings-fp32.json embeddings-bf16.json
WORKLOAD=titles RERANK_BATCH=16 ./run.sh ... && python3 compare.py scores scores-fp32.json scores-bf16.json
```

`run.sh` expects the models in a podman volume mounted as `HF_HOME` (`HF_CACHE_VOLUME`, default
`infinity-hf-cache`); the first run downloads them.

## Mixed workload: upstream vs fork vs bf16

| stage                                 | upstream 0.0.77 | fork 0.1.0, pad 16 | fork 0.1.0, pad 16, bf16 |
| ------------------------------------- | --------------- | ------------------ | ------------------------ |
| embed short b8 c1 (3-25 words)        | 107.5 ms, 67.7/s | 113.3 ms, 65.6/s  | 65.0 ms, 114.8/s (x1.7)  |
| embed mixed b8 c1 (5-120 words)       | 433.9 ms, 18.1/s | 436.2 ms, 17.9/s  | 229.1 ms, 34.4/s (x1.9)  |
| embed mixed b8 c4                     | 1672 ms, 18.9/s | 1701 ms, 18.6/s    | 831 ms, 37.9/s (x2.0)    |
| embed single c1                       | 30.3 ms, 33.5/s | 30.6 ms, 33.3/s    | 20.9 ms, 48.1/s (x1.4)   |
| rerank 8 docs c1 (40-300 words)       | 2707 ms, 2.9/s  | 2703 ms, 2.9/s     | 1237 ms, 6.5/s (x2.2)    |
| rerank 8 docs c4                      | 10582 ms, 3.0/s | 10852 ms, 2.9/s    | 4829 ms, 6.5/s (x2.2)    |
| rerank 4 long docs c1 (300-900 words) | 7824 ms, 0.5/s  | 6013 ms, 0.7/s     | 2867 ms, 1.5/s (x3.0)    |
| anonymous RSS after the run           | 5195 MiB        | 4243 MiB           | 3811 MiB                 |

The refactoring (bare onnxruntime instead of optimum, transformers 5) does not change
throughput: the fork in fp32 returns embeddings bit-identical to upstream (cosine 1.000000) at
the same speed, long rerank pairs got faster (per-pair tokenization) and memory went down.
`--pad-to-multiple-of 16` costs about 3% on very short texts and nothing measurable elsewhere;
it is what keeps the set of input shapes, and therefore the OpenVINO kernel cache, finite.

## Reranker: batch size, padding, bf16, int8

`workload.py titles`, documents per second at 1 / 2 / 4 concurrent clients.

| reranker config                              | c=1  | c=2  | c=4  | anon RSS |
| -------------------------------------------- | ---- | ---- | ---- | -------- |
| b4, pad 64, fp32                             | 19.5 | 19.7 | 19.6 | 4053 MiB |
| b16, pad 16, fp32                            | 28.8 | 26.3 | 27.7 | 4876 MiB |
| **b16, pad 16, bf16**                        | **62.9** | **61.4** | **60.8** | 3898 MiB |
| b16, pad 16, int8 (MatMul/Gemm, percentile)  | 77.5 | 81.6 | 80.9 | 4705 MiB |

- Batch size and padding do not change the scores at all (b4/pad 64 vs b16/pad 16: Spearman
  1.0000, max score difference 0). A finer padding bucket is worth +45% on short texts because
  fewer pad tokens are computed; a bigger batch mostly costs memory.
- bf16 (`INFERENCE_PRECISION_HINT=bf16` on the OpenVINO CPU plugin, passed through
  `--onnx-provider-options`) gives x2.2 with no change to the model files. Note that
  onnxruntime's OpenVINO EP pins `inference_precision` to f32 on CPU by default and its `precision`
  option does not accept bf16; `load_config` is the only way in, and upstream infinity had no
  flag to set it.
- int8 is another +28% on top of bf16, but see below.

### Parity

- Embeddings (LaBSE), fp32 vs bf16, 5 texts: cosine 0.99994-0.99996, max abs diff 0.0017.
- Rerank scores, 10 queries x 20 texts, fp32 vs bf16: Spearman 0.9980 (min 0.9970), top-5
  overlap 1.00, max |delta score| 0.0037, max |delta logit| 0.125.
- fp32 vs int8: Spearman 0.78 (min 0.47), top-5 overlap 0.70, max |delta logit| 6.8.

## int8 (static QDQ quantization with onnxruntime)

`quantize_int8.py`: per-channel, QUInt8 activations / QInt8 weights, calibration on synthetic
texts from the same corpus, reranker only. The speed is there, the ranking quality is not:

- onnxruntime quantizes the `Add` of the additive attention mask by default. Its constant is
  -3.4e38, so the scale is 1e36 and the softmax turns into noise (Spearman 0.05); histogram
  calibrators fail on that tensor with `Too many bins for data range`.
  `op_types_to_quantize=["MatMul", "Gemm"]` avoids it.
- Even then the rank correlation drops to 0.78-0.91 depending on the calibrator and corpus, and
  irrelevant documents move from 0.00 to 0.03-0.10, which is where relevance thresholds usually
  sit. Production-grade int8 for XLM-RoBERTa large needs accuracy-aware quantization (NNCF,
  SmoothQuant, excluding sensitive layers) and calibration on real inputs.
- Practicalities: `huggingface_hub` 1.x raises `IncompleteSnapshotError` from
  `snapshot_download(local_files_only=True)` on a cache that only holds the ONNX files; `onnx`
  >= 1.22 refuses external data with more than one hard link (the fork materializes model dirs
  with hard links), so quantize a plain copy; models above 2 GB need
  `quant_pre_process(save_as_external_data=True)` and `quantize_static(use_external_data_format=True)`;
  the Percentile calibrator needs every calibration sample to have the same shape.

## bf16 and old ONNX exports

bf16 is only as good as the graph. The `onnx/` folder shipped with `BAAI/bge-m3` is an opset-11
export from PyTorch 2.1 with every LayerNorm decomposed into `ReduceMean / Sub / Pow / Sqrt / Div`;
under bf16 the variance is computed in bf16 and the embeddings come out with cosine 0.22-0.40 to
fp32, while the same model re-exported at opset 18 with the fused `LayerNormalization` op gives
0.99998 ([ElXreno/bge-m3-onnx](https://huggingface.co/ElXreno/bge-m3-onnx)). Check the parity of
every model before enabling bf16, not just the throughput.
