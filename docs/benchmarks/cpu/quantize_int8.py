"""Static int8 (QDQ) quantization of an ONNX embedder or cross-encoder with onnxruntime.

    quantize_int8.py <hf-repo> <output-dir> rerank|embed [minmax|percentile|entropy] [samples]

Run inside the infinity image with the HF cache mounted as HF_HOME and this directory plus
`docs/assets` mounted (or `BENCH_CORPUS` pointing at a text file, see corpus.py). The output
directory gets the tokenizer and config files of the repo plus `model.onnx` (+ `model.onnx_data`),
so it can be served with `--model-id <output-dir>`. See README.md for what this does to ranking
quality.
"""

import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from corpus import text
from huggingface_hub import snapshot_download
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from transformers import AutoTokenizer

REPO = sys.argv[1]
OUT = Path(sys.argv[2])
KIND = sys.argv[3]
METHOD = sys.argv[4] if len(sys.argv) > 4 else "minmax"
SAMPLES = int(sys.argv[5]) if len(sys.argv) > 5 else 128
HUB = Path(os.environ.get("HF_HOME", "/data")) / "hub"


class Reader(CalibrationDataReader):
    def __init__(self, tokenizer, input_names):
        rng = random.Random(7)
        self.batches = []
        for _ in range(SAMPLES):
            # histogram calibrators stack every batch into one array, so all samples share a shape
            if KIND == "rerank":
                enc = tokenizer(
                    text(rng, 3, 8),
                    text(rng, 3, 30),
                    truncation=True,
                    max_length=64,
                    padding="max_length",
                    return_tensors="np",
                    return_token_type_ids=False,
                )
            else:
                enc = tokenizer(
                    [text(rng, 3, 60) for _ in range(4)],
                    truncation=True,
                    max_length=96,
                    padding="max_length",
                    return_tensors="np",
                )
            feed = {k: v.astype(np.int64) for k, v in enc.items() if k in input_names}
            if "token_type_ids" in input_names and "token_type_ids" not in feed:
                feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
            self.batches.append(feed)
        self.it = iter(self.batches)

    def get_next(self):
        return next(self.it, None)


try:
    snapshot = Path(snapshot_download(REPO, local_files_only=True))
except Exception:
    # huggingface_hub 1.x raises IncompleteSnapshotError on a cache that only holds the ONNX files
    snapshot = sorted((HUB / f"models--{REPO.replace('/', '--')}" / "snapshots").iterdir())[0]
materialized = HUB / "infinity_onnx" / "materialized" / REPO
model_root = sorted(materialized.iterdir())[0] if materialized.exists() else snapshot
onnx_files = sorted(model_root.rglob("*.onnx"))
model_path = [p for p in onnx_files if "quant" not in p.name][0]
print(f"source {model_path} ({model_path.resolve().stat().st_size / 1e6:.0f} MB)", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
for src in snapshot.iterdir():
    if src.is_file() and ".onnx" not in src.name:
        shutil.copy2(src.resolve(), OUT / src.name)

# onnx >= 1.22 refuses external data files with more than one hard link, so work on plain copies
work = OUT / "src"
work.mkdir(exist_ok=True)
for src in model_path.parent.iterdir():
    if src.name.startswith(model_path.name):
        dst = work / src.name
        if not dst.exists() or dst.stat().st_size != src.resolve().stat().st_size:
            shutil.copyfile(src.resolve(), dst)
model_path = work / model_path.name

graph = onnx.load(model_path.as_posix(), load_external_data=False).graph
input_names = [i.name for i in graph.input]
del graph
print(f"inputs {input_names}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(snapshot.as_posix())

pre = OUT / "model_preprocessed.onnx"
t0 = time.perf_counter()
quant_pre_process(
    model_path.as_posix(),
    pre.as_posix(),
    skip_symbolic_shape=True,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    external_data_location="model_preprocessed.onnx_data",
)
print(f"preprocessed in {time.perf_counter() - t0:.0f}s", flush=True)

method = {
    "minmax": CalibrationMethod.MinMax,
    "percentile": CalibrationMethod.Percentile,
    "entropy": CalibrationMethod.Entropy,
}[METHOD]
t0 = time.perf_counter()
quantize_static(
    model_input=pre.as_posix(),
    model_output=(OUT / "model.onnx").as_posix(),
    calibration_data_reader=Reader(tokenizer, input_names),
    quant_format=QuantFormat.QDQ,
    per_channel=True,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    calibrate_method=method,
    # the additive attention mask is -3.4e38: quantizing that Add (ORT default) wrecks softmax
    op_types_to_quantize=["MatMul", "Gemm"],
    use_external_data_format=True,
    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
)
print(f"quantized ({METHOD}, {SAMPLES} samples) in {time.perf_counter() - t0:.0f}s", flush=True)
pre.unlink()
for extra in OUT.glob("model_preprocessed.onnx*"):
    extra.unlink()
shutil.rmtree(work)
for p in sorted(OUT.iterdir()):
    print(f"  {p.name} {p.stat().st_size / 1e6:.0f} MB")
