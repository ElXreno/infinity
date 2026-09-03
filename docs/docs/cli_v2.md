# CLI v2 Documentation

The current version of Infinity uses the following arguments in its CLI:
```bash
$ infinity_emb v2 --help
```

```
                                                                                                                        
 Usage: infinity_emb v2 [OPTIONS]                                                                                       
                                                                                                                        
 Infinity API ♾️  cli v2. MIT License. Copyright (c) 2023-now Michael Feil                                              
                                                                                                                        
 Multiple Model CLI Playbook:                                                                                           
                                                                                                                        
 - 1. cli options can be overloaded i.e. `v2 --model-id model/id1 --model-id model/id2 --batch-size 8 --batch-size 4`   
                                                                                                                        
 - 2. or adapt the defaults by setting ENV Variables separated by `;`: INFINITY_MODEL_ID="model/id1;model/id2;" &&      
 INFINITY_BATCH_SIZE="8;4;"                                                                                             
                                                                                                                        
 - 3. single items are broadcasted to `--model-id` length, making `v2 --model-id model/id1 --model-id/id2 --batch-size  
 8` both models have batch-size 8.                                                                                      
                                                                                                                        
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --model-id                                                   <str>                       Huggingface model repo id.  │
│                                                                                          Subset of possible models:  │
│                                                                                          https://huggingface.co/mod… │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_MODEL_ID`]        │
│                                                                                          [default:                   │
│                                                                                          michaelfeil/bge-small-en-v… │
│ --served-model-name                                          <str>                       the nickname for the API,   │
│                                                                                          under which the model_id    │
│                                                                                          can be selected             │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_SERVED_MODEL_NAM… │
│ --batch-size                                                 <int>                       maximum batch size for      │
│                                                                                          inference                   │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_BATCH_SIZE`]      │
│                                                                                          [default: 32]               │
│ --revision                                                   <str>                       huggingface  model repo     │
│                                                                                          revision.                   │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_REVISION`]        │
│ --trust-remote-code            --no-trust-remote-code                                    if potential remote         │
│                                                                                          modeling code from          │
│                                                                                          huggingface repo is         │
│                                                                                          trusted.                    │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_TRUST_REMOTE_COD… │
│                                                                                          [default:                   │
│                                                                                          trust-remote-code]          │
│ --engine                                                     <torch|ctranslate2|optimum  Which backend to use.       │
│                                                              |neuron|debugengine>        `torch` uses Pytorch        │
│                                                                                          GPU/CPU, optimum uses ONNX  │
│                                                                                          on GPU/CPU/NVIDIA-TensorRT, │
│                                                                                          `CTranslate2` uses          │
│                                                                                          torch+ctranslate2 on        │
│                                                                                          CPU/GPU.                    │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_ENGINE`]          │
│                                                                                          [default: optimum]          │
│ --model-warmup                 --no-model-warmup                                         if model should be warmed   │
│                                                                                          up after startup, and       │
│                                                                                          before ready.               │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_MODEL_WARMUP`]    │
│                                                                                          [default: model-warmup]     │
│ --vector-disk-cache            --no-vector-disk-cache                                    If hash(request)/results    │
│                                                                                          should be cached to SQLite  │
│                                                                                          for latency improvement.    │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_VECTOR_DISK_CACH… │
│                                                                                          [default:                   │
│                                                                                          vector-disk-cache]          │
│ --device                                                     <cpu|cuda|mps|tensorrt|xla  device to use for computing │
│                                                              |auto>                      the model forward pass.     │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_DEVICE`]          │
│                                                                                          [default: auto]             │
│ --device-id                                                  <str>                       device id defines the model │
│                                                                                          placement. e.g. `0,1` will  │
│                                                                                          place the model on          │
│                                                                                          MPS/CUDA/GPU 0 and 1 each   │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_DEVICE_ID`]       │
│ --lengths-via-tokenize         --no-lengths-via-tokenize                                 if True, returned tokens is │
│                                                                                          based on actual tokenizer   │
│                                                                                          count. If false, uses       │
│                                                                                          len(input) as proxy.        │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_LENGTHS_VIA_TOKE… │
│                                                                                          [default:                   │
│                                                                                          lengths-via-tokenize]       │
│ --dtype                                                      <float32|float16|bfloat16|  dtype for the model         │
│                                                              int8|fp8|auto>              weights.                    │
│                                                                                          [env var: `INFINITY_DTYPE`] │
│                                                                                          [default: auto]             │
│ --embedding-dtype                                            <float32|int8|uint8|binary  dtype post-forward pass. If │
│                                                              |ubinary>                   != `float32`, using         │
│                                                                                          Post-Forward Static         │
│                                                                                          quantization.               │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_EMBEDDING_DTYPE`] │
│                                                                                          [default: float32]          │
│ --pooling-method                                             <mean|cls|auto>             overwrite the pooling       │
│                                                                                          method if inferred          │
│                                                                                          incorrectly.                │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_POOLING_METHOD`]  │
│                                                                                          [default: auto]             │
│ --compile                      --no-compile                                              Enable usage of             │
│                                                                                          `torch.compile(dynamic=Tru… │
│                                                                                          if engine relies on it.     │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_COMPILE`]         │
│                                                                                          [default: compile]          │
│ --bettertransformer            --no-bettertransformer                                    No-op kept for              │
│                                                                                          compatibility:              │
│                                                                                          BetterTransformer was       │
│                                                                                          removed from transformers 5 │
│                                                                                          and optimum 2, torch SDPA   │
│                                                                                          is used instead.            │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_BETTERTRANSFORME… │
│                                                                                          [default:                   │
│                                                                                          bettertransformer]          │
│ --preload-only                 --no-preload-only                                         If true, only downloads     │
│                                                                                          models and verifies setup,  │
│                                                                                          then exit. Recommended for  │
│                                                                                          pre-caching the download in │
│                                                                                          a Dockerfile.               │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_PRELOAD_ONLY`]    │
│                                                                                          [default: no-preload-only]  │
│ --host                                                       <str>                       host for the FastAPI        │
│                                                                                          uvicorn server              │
│                                                                                          [env var: `INFINITY_HOST`]  │
│                                                                                          [default: 0.0.0.0]          │
│ --port                                                       <int>                       port for the FastAPI        │
│                                                                                          uvicorn server              │
│                                                                                          [env var: `INFINITY_PORT`]  │
│                                                                                          [default: 7997]             │
│ --url-prefix                                                 <str>                       prefix for all routes of    │
│                                                                                          the FastAPI uvicorn server. │
│                                                                                          Useful if you run behind a  │
│                                                                                          proxy / cascaded API.       │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_URL_PREFIX`]      │
│ --redirect-slash                                             <str>                       where to redirect `/`       │
│                                                                                          requests to.                │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_REDIRECT_SLASH`]  │
│                                                                                          [default: /docs]            │
│ --log-level                                                  <critical|error|warning|in  console log level.          │
│                                                              fo|debug|trace>             [env var:                   │
│                                                                                          `INFINITY_LOG_LEVEL`]       │
│                                                                                          [default: info]             │
│ --permissive-cors              --no-permissive-cors                                      whether to allow permissive │
│                                                                                          cors.                       │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_PERMISSIVE_CORS`] │
│                                                                                          [default:                   │
│                                                                                          no-permissive-cors]         │
│ --api-key                                                    <str>                       api_key used for            │
│                                                                                          authentication headers.     │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_API_KEY`]         │
│ --proxy-root-path                                            <str>                       Proxy prefix for the        │
│                                                                                          application. See:           │
│                                                                                          https://fastapi.tiangolo.c… │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_PROXY_ROOT_PATH`] │
│ --onnx-disable-optimize        --no-onnx-disable-optimize                                Disable onnx optimization   │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_ONNX_DISABLE_OPT… │
│                                                                                          [default:                   │
│                                                                                          onnx-disable-optimize]      │
│ --onnx-do-not-prefer-quant…    --no-onnx-do-not-prefer-q…                                Do not use quantized onnx   │
│                                                                                          models by default if        │
│                                                                                          available                   │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_ONNX_DO_NOT_PREF… │
│                                                                                          [default:                   │
│                                                                                          onnx-do-not-prefer-quantiz… │
│ --pad-to-multiple-of                                         <int>                       Pad every batch to a        │
│                                                                                          sequence length that is a   │
│                                                                                          multiple of this value, 0   │
│                                                                                          disables. Bounds the number │
│                                                                                          of distinct input shapes,   │
│                                                                                          which keeps memory flat on  │
│                                                                                          backends that cache kernels │
│                                                                                          per shape (OpenVINO,        │
│                                                                                          oneDNN, torch CPU) at the   │
│                                                                                          cost of a few percent of    │
│                                                                                          extra compute. Suggested:   │
│                                                                                          8-16 for embedders, 32-64   │
│                                                                                          for rerankers.              │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_PAD_TO_MULTIPLE_… │
│                                                                                          [default: 0]                │
│ --onnx-provider-options                                      <str>                       JSON object merged into the │
│                                                                                          provider_options of the     │
│                                                                                          onnxruntime execution       │
│                                                                                          provider, e.g.              │
│                                                                                          '{"num_of_threads": 8}' for │
│                                                                                          OpenVINO or                 │
│                                                                                          '{"trt_fp16_enable":        │
│                                                                                          false}' for TensorRT. Empty │
│                                                                                          string passes nothing.      │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_ONNX_PROVIDER_OP… │
│ --max-query-tokens                                           <int>                       Rerank ceiling:             │
│                                                                                          head-truncate the query to  │
│                                                                                          at most N tokens before     │
│                                                                                          scoring. A client may       │
│                                                                                          request fewer. Unset        │
│                                                                                          disables the limit.         │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_MAX_QUERY_TOKENS… │
│ --max-tokens-per-doc                                         <int>                       Rerank ceiling:             │
│                                                                                          head-truncate each document │
│                                                                                          to at most N tokens before  │
│                                                                                          scoring. A client may       │
│                                                                                          request fewer. Unset        │
│                                                                                          disables the limit.         │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_MAX_TOKENS_PER_D… │
│ --max-pair-tokens                                            <int>                       Rerank ceiling on the       │
│                                                                                          joined (query, document)    │
│                                                                                          pair, in tokens. A client   │
│                                                                                          may request fewer. Unset    │
│                                                                                          disables the limit.         │
│                                                                                          [env var:                   │
│                                                                                          `INFINITY_MAX_PAIR_TOKENS`] │
│ --help                                                                                   Show this message and exit. │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```
Note: This doc is auto-generated. Do not edit this file directly.
