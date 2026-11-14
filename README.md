# tsfm-prediction

A small HTTP service that forecasts time series for NebulaHome packages (energy, climate…).
It runs [Timestral](https://github.com/mistralai/timestral) by Mistral AI, a family of
probabilistic time-series foundation models, through a native Rust port of the network:
no Python, no PyTorch, no ONNX at run time. The image is one binary on distroless
(48 MB unpacked), and it idles at about 3.5 MB of RAM.

- One model in memory at a time, one forecast at a time, a bounded queue in front.
- Four quality levels chosen per request, mapped to the four model sizes.
- Weights are downloaded from Hugging Face on first use and cached in a volume.
- The model is unloaded after 5 idle minutes, and the memory really goes back to the OS.

## Why a native Rust port

The brief asked for the first runtime that works, in this order: (1) Rust with a native port
of the network loading the published safetensors, (2) Rust + ONNX Runtime, (3) Python.

Path 1 was taken. Timestral's inference code is about 1 500 lines of Python built from
ordinary parts: linear layers, RMSNorm, SiLU/SwiGLU, partial RoPE, masked scaled dot-product
attention over time and over variates, a sigmoid gate, a monotone quantile head, and Welford
statistics in float64 for the scaling. All of it ports directly. Matrix products use the
[`gemm`](https://crates.io/crates/gemm) crate; the rest is plain Rust. The port follows
`timestral` at commit `a534bdb` file by file (`model.py`, `layers.py`, `masks.py`,
`scaler.py`, `forecaster.py`), including the forecaster defaults: symmetric averaging,
non-negative clipping, the context cap on quantiles, block-by-block decoding with the
median fed back, and the advancing refine frame. Paths 2 and 3 were not needed.

Two things make the memory low:

- The weights are never copied: `model.safetensors` is memory-mapped and read in place.
  They show as file-backed memory while the model is loaded, and `munmap` returns them
  the moment the model is dropped.
- Activations above 4 MB get their own mapping (`mallopt(M_MMAP_THRESHOLD)`), and
  `malloc_trim` runs after every forecast, so glibc does not keep freed blocks around.
  Without this, 30 MB stayed resident after a long-context forecast with the 28M model.

The image is built on glibc (`distroless/cc`) rather than a static musl binary: musl's
allocator made multi-threaded inference 1.5 to 3 times slower.

### Parity with the official implementation

`scripts/parity.py` sends the same inputs to the service and to the official
`TimestralForecaster` (same weights, CPU, float32), asks for all 21 quantiles and compares
them. The cases cover a plain series, missing values with a context that is not a whole
number of patches, a 3-block horizon, 3 variates, a non-negative series (clipping),
a 5-step context, past and known covariates (one of them a 0/1 indicator), a known
covariate over two blocks, and a full 32 768-step context whose second block hits the cap.

The figure is `max |rust - python| / (std(context) + |python|)` over every quantile,
step and variate of a case. Results on 2026-11-14:

| case | low (5M) | medium (28M) | high (123M) | xhigh (209M) |
|---|---|---|---|---|
| univariate, T=2000, H=96 | 1.9e-6 | 1.5e-6 | 5.2e-7 | 8.2e-7 |
| missing values, T=1007, H=64 | 1.6e-6 | 4.9e-7 | 5.4e-7 | 5.1e-7 |
| three blocks, T=500, H=2100 | 1.4e-5 | 9.6e-5 | 2.2e-5 | 2.4e-5 |
| 3 variates, T=777, H=50 | 1.8e-6 | 1.4e-6 | 2.5e-6 | 4.7e-7 |
| non-negative, T=300, H=40 | 1.1e-6 | 3.1e-7 | 4.0e-7 | 3.5e-7 |
| short context, T=5, H=10 | 1.2e-6 | 2.3e-6 | 1.5e-6 | 8.3e-7 |
| 2 past + 2 known covariates, T=600, H=100 | 1.7e-6 | 1.6e-6 | 1.9e-6 | 1.4e-6 |
| known covariate over two blocks, T=400, H=1100 | 2.1e-5 | 1.1e-5 | 1.3e-5 | 9.0e-6 |
| full context, T=32768, H=1500 | 1.0e-5 | 2.7e-6 | 7.6e-6 | 8.7e-6 |

On the median alone the worst case is 7.5e-6.

Single-block forecasts agree to a few millionths of the series' spread. Over several
blocks the fed-back medians carry float32 rounding forward, and the far tails (q0.01 and
q0.99, which `sinh` can push to 1 000 times the median) show it most.

## Contract

All bodies are JSON (`content-type: application/json`).

### `POST /v1/forecast`

| field | type | |
|---|---|---|
| `quality` | `"low"`, `"medium"`, `"high"`, `"xhigh"` | default `"medium"`; 5M, 28M, 123M, 209M |
| `context` | array of variates, each an array of numbers; `null` = missing | a flat array is one variate; all variates the same length, 1 to 32 768 steps, at least one value |
| `horizon` | integer | steps to forecast, 1 to 4 096 |
| `quantiles` | array of numbers | optional, default `[0.1, 0.5, 0.9]`; each one of 0.01, 0.05, 0.1, 0.15 … 0.9, 0.95, 0.99 |
| `pastCovariates` | array of rows like `context` | optional; known over the context only, each row as long as the context |
| `knownCovariates` | array of rows like `context` | optional; known over context and horizon, each row `context + horizon` long; rows of only 0 and 1 are treated as indicators |

Variates of one request are forecast jointly; covariates condition the forecast and are
not forecast. At most 32 rows in total (variates plus covariates). Numbers must fit in
float32. Unknown fields are refused.

```json
{
  "quality": "low",
  "context": [[21.5, 21.7, null, 22.0, 22.4, 22.1, 21.8, 21.6]],
  "horizon": 3,
  "quantiles": [0.1, 0.5, 0.9],
  "pastCovariates": [[0, 0, 1, 1, 1, 0, 0, 0]],
  "knownCovariates": [[5.1, 5.0, 4.8, 4.9, 5.3, 6.0, 6.4, 6.2, 5.9, 5.5, 5.2]]
}
```

Answer `200`:

```json
{
  "model": "mistralai/Timestral-5M",
  "quantiles": [0.1, 0.5, 0.9],
  "forecast": [[[21.266823, 21.190586, 21.171677], [21.54477, 21.617207, 21.653446], [21.86371, 22.056873, 22.144987]]],
  "queuedMs": 0,
  "inferenceMs": 12
}
```

`forecast` is variate → quantile (in the order asked) → steps. `queuedMs` is the time from
arrival to the start of inference, including a model load or download; `inferenceMs` is the
forecast itself. A batch of independent series in one call is not supported: send one
request per series (they queue and run back to back on the same model).

### `GET /v1/status`

```json
{
  "loadedModel": "mistralai/Timestral-28M",
  "loadedQuality": "medium",
  "unloadInSeconds": 287,
  "queueDepth": 0,
  "running": false,
  "rssBytes": 116293632,
  "modelsOnDisk": ["low", "medium"],
  "maxQuality": "xhigh"
}
```

`loadedModel`, `loadedQuality` and `unloadInSeconds` are `null` when no model is loaded;
`queueDepth` counts waiting requests, not the running one.

### `GET /healthz`

`200 {"status":"ok"}` while the server accepts connections.

### Errors

Every error is `{"code": "...", "message": "..."}`:

| code | status | when |
|---|---|---|
| `invalid_input` | 400 (413 for a body over 16 MB, 415 without the JSON content type, 422 for a wrong JSON type) | the request breaks the contract above |
| `quality_unavailable` | 403 | `quality` is above `TSFM_MAX_QUALITY` |
| `queue_full` | 503 | `TSFM_QUEUE_SIZE` requests are already waiting; answered at once |
| `shutting_down` | 503 | the server got SIGTERM |
| `timeout` | 504 | no answer within `TSFM_TIMEOUT_SECS` |
| `model_download_failed` | 502 | the weights are not cached and Hugging Face could not be reached |
| `model_load_failed` | 500 | the cached files are not a valid Timestral checkpoint |
| `forecast_failed` | 422 | the model produced NaN (a degenerate input) |
| `internal` | 500 | a bug |

## Behaviour

- **Queue.** A single worker runs one forecast at a time. Waiting requests for the model
  already in memory go first, so requests for the same model run back to back; otherwise
  the oldest request goes and its model replaces the loaded one.
- **Starvation bound.** The oldest waiting request is served next once 4 later requests
  have been served before it, or once it has waited 30 seconds.
- **Disconnects and timeouts.** A client that disconnects or times out while queued takes
  its job out of the queue. A forecast already running is finished and its answer dropped.
- **Idle unload.** After `TSFM_IDLE_SECS` without a request the model is dropped.
- **Cache.** `TSFM_CACHE_DIR/Timestral-<size>/{config.json,model.safetensors}`. Each file is
  downloaded to a temporary name, checked against its `Content-Length` and renamed, so the
  cache never holds a partial file under its real name. Cached models need no network.
- **Shutdown.** On SIGTERM the server stops accepting connections, answers queued requests
  with `shutting_down`, lets the running forecast finish and exits.
- **Network.** It listens on `0.0.0.0:TSFM_PORT`, so it works on a bridge network (by
  service name or alias) and on the host network (loopback). It has no authentication:
  keep it on an internal network.

## Configuration

Environment variables only.

| variable | default | |
|---|---|---|
| `TSFM_PORT` | `8080` | listening port |
| `TSFM_CACHE_DIR` | `/models` | weights cache; mount a volume here |
| `TSFM_IDLE_SECS` | `300` | unload the model after this many idle seconds |
| `TSFM_QUEUE_SIZE` | `16` | waiting requests before `queue_full` |
| `TSFM_THREADS` | `0` = available cores, at most 4 | threads for one forecast |
| `TSFM_MAX_QUALITY` | `xhigh` | highest quality served; `low` or `medium` on a weak machine |
| `TSFM_TIMEOUT_SECS` | `600` | longest wait for an answer, queue included |
| `HF_ENDPOINT` | `https://huggingface.co` | Hugging Face endpoint or mirror |
| `HF_TOKEN` | none | Hugging Face token, if needed |

On thread count: on a 24-thread host, 2 to 4 threads gave the best times for typical
requests; more threads made them slower and helped long contexts only a little.

```sh
docker build -t tsfm-prediction:dev .
docker run -d -p 8080:8080 -v tsfm-models:/models -e TSFM_MAX_QUALITY=medium tsfm-prediction:dev
curl -s localhost:8080/v1/forecast -H 'content-type: application/json' \
  -d '{"quality":"low","context":[1,2,3,4,5,6,7,8,9,10],"horizon":5}'
```

`compose.yaml` shows the service on an internal network.

## Memory and speed

Measured on 2026-11-14 in Docker Desktop (WSL2, x86-64, 24 threads), `TSFM_THREADS=4`,
from `/proc/<pid>/status`. "Loaded" is after forecasts on a 2 000-step context; "peak" is
the high-water mark during a forecast on a full 32 768-step context; "after unload" is once
the idle timer dropped the model.

| quality | model | weights | idle, no model | loaded | of which weights (file-backed) | peak | after unload |
|---|---|---|---|---|---|---|---|
| low | Timestral-5M | 21 MB | 3.6 MB | 25.5 MB | 24.0 MB | 42.9 MB | 5.5 MB |
| medium | Timestral-28M | 111 MB | 3.3 MB | 113.6 MB | 111.7 MB | 144.7 MB | 5.3 MB |
| high | Timestral-123M | 490 MB | 3.6 MB | 485.9 MB | 483.6 MB | 535.1 MB | 7.2 MB |
| xhigh | Timestral-209M | 837 MB | 3.5 MB | 824.5 MB | 822.3 MB | 883.2 MB | 6.7 MB |

The weights are file-backed pages: under a container memory limit the kernel can evict
them and read them back, so a limit below the "loaded" figure still works, only slower.
Give the container the "peak" figure of the largest quality you allow for full speed.

Time for one forecast, 2 000-step context, 96-step horizon (`inferenceMs`), next to the
official Python package on the same inputs with 4 CPUs:

| quality | this service | official Python (PyTorch, CPU) |
|---|---|---|
| low | 12 ms | 16 ms |
| medium | 31 ms | 39 ms |
| high | 107 ms | 279 ms |
| xhigh | 183 ms | 565 ms |

A full 32 768-step context takes about 0.2 s (low) to 3 s (xhigh) per 96-step horizon.
The first request of a quality also pays the load: well under a second from the cache,
or the download (21 MB to 837 MB) on first use.

## Development

```sh
cargo test        # queue, scheduling, idle unload, validation, HTTP contract; no network, no weights
cargo clippy --all-targets -- -D warnings
```

No Rust toolchain on the machine? Run the same in `rust:1.100-slim-trixie` with the
folder mounted. The tests use a fake model behind the same `Model`/`Loader` traits as the
real one.

### Running the parity check

Not part of `cargo test`: it needs the weights and PyTorch.

```sh
docker build -t tsfm-prediction:dev .
docker build -f scripts/parity.Dockerfile -t tsfm-parity scripts
docker network create tsfm-net
docker run -d --name tsfm --network tsfm-net -p 8080:8080 -v tsfm-models:/models tsfm-prediction:dev
# one request downloads the weights into the volume, where the reference reads them too
curl -s localhost:8080/v1/forecast -H 'content-type: application/json' -d '{"quality":"low","context":[1,2,3],"horizon":1}'
docker run --rm --network tsfm-net --cpus 4 -v tsfm-models:/models:ro tsfm-parity \
  --url http://tsfm:8080 --quality low --model-dir /models/Timestral-5M
```

Use `medium` with `Timestral-28M` and so on for the other sizes. Exit code 0 means every
case is within `--rtol` (default `5e-4`; the worst measured is 9.6e-5).

## Licences

- This code, including the port of the Timestral network, is under the Apache License 2.0,
  like the [timestral](https://github.com/mistralai/timestral) code it is ported from
  (© Mistral AI).
- The Timestral **weights** are not part of this repository or the image. The service
  downloads them from Hugging Face, where Mistral AI publishes them under a **Modified MIT
  licence** ([5M](https://huggingface.co/mistralai/Timestral-5M/blob/main/LICENSE),
  [28M](https://huggingface.co/mistralai/Timestral-28M/blob/main/LICENSE),
  [123M](https://huggingface.co/mistralai/Timestral-123M/blob/main/LICENSE),
  [209M](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE)). Among its
  terms: it does not apply to a company (or employer) whose consolidated monthly revenue
  exceeded USD 20 million in the preceding month, which needs a commercial licence from
  Mistral AI. Read the licence before using the weights.
