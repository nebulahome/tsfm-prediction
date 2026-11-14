"""Parity check: the service against the official Python implementation of Timestral.

Sends the same inputs to a running tsfm-prediction service and to `timestral.TimestralForecaster`
(loaded from the same `config.json` + `model.safetensors`), asks for all 21 quantiles and
compares them. Python runs with the CPU count of its container, Rust with the service's threads.
Needs `pip install numpy torch` and the `timestral` package; see the README.

    python parity.py --url http://localhost:8080 --quality low --model-dir /models/Timestral-5M
"""

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import torch
from timestral import TimestralForecaster

LEVELS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]


def series(rng, n, period=24, level=0.0, noise=1.0):
    t = np.arange(n)
    return level + 10 * np.sin(2 * math.pi * t / period) + rng.normal(0, noise, n).cumsum() * 0.3


def cases():
    rng = np.random.default_rng(0)
    yield "univariate T=2000 H=96", dict(context=series(rng, 2000)[None], horizon=96)
    gappy = series(rng, 1007, period=48, level=50)
    gappy[[3, 4, 5, 200, 201, 650]] = np.nan
    gappy[:40] = np.nan
    yield "missing values, T=1007 not a patch multiple, H=64", dict(context=gappy[None], horizon=64)
    yield "three blocks, T=500 H=2100", dict(context=series(rng, 500, period=12)[None], horizon=2100)
    yield "multivariate V=3 T=777 H=50", dict(context=np.stack([series(rng, 777, p) for p in (24, 24, 168)]), horizon=50)
    load = np.abs(series(rng, 300, level=20)) + 1
    yield "non-negative (clipped) T=300 H=40", dict(context=load[None], horizon=40)
    yield "short context T=5 H=10", dict(context=np.array([[1.0, 2.0, 3.0, 2.5, 2.0]]), horizon=10)
    t, h = 600, 100
    temp = series(rng, t + h, period=24, level=15)
    daylight = (np.sin(2 * math.pi * np.arange(t + h) / 24) > 0).astype(float)
    past = np.stack([series(rng, t, 24), series(rng, t, 7)])
    past[1, 100:120] = np.nan
    target = 0.5 * temp[:t] + 3 * daylight[:t] + rng.normal(0, 0.5, t)
    yield "covariates: 2 past + 2 known (one 0/1), T=600 H=100", dict(
        context=target[None], horizon=h, past=past, known=np.stack([temp, daylight]))
    t, h = 400, 1100
    known = series(rng, t + h, period=24)
    yield "known covariate over two blocks, T=400 H=1100", dict(
        context=(known[:t] * 2 + rng.normal(0, 0.3, t))[None], horizon=h, known=known[None])
    yield "full context T=32768 H=1500 (cap reached in block 2)", dict(context=series(rng, 32768, period=96)[None], horizon=1500)


def rows(a):
    return [[None if np.isnan(v) else float(v) for v in r] for r in a]


def service(url, quality, case):
    body = {"quality": quality, "context": rows(case["context"]), "horizon": case["horizon"], "quantiles": LEVELS}
    if "past" in case:
        body["pastCovariates"] = rows(case["past"])
    if "known" in case:
        body["knownCovariates"] = rows(case["known"])
    req = urllib.request.Request(url + "/v1/forecast", json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"service error {e.code}: {e.read().decode()}")
    # variate -> quantile -> step  to  variate, step, quantile
    return np.array(out["forecast"], dtype=np.float64).transpose(0, 2, 1), out["inferenceMs"]


def reference(fc, case):
    f32 = lambda a: None if a is None else torch.tensor(np.asarray(a, dtype=np.float32)[None])
    t = time.time()
    q = fc.forecast(f32(case["context"]), case["horizon"], past_covariates=f32(case.get("past")),
                    known_covariates=f32(case.get("known"))).quantiles[0].double().numpy()
    return q, (time.time() - t) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--quality", default="low")
    ap.add_argument("--model-dir", required=True, help="folder with config.json and model.safetensors")
    ap.add_argument("--rtol", type=float, default=5e-4, help="tolerance on |rust - python| / (std(context) + |python|)")
    args = ap.parse_args()
    fc = TimestralForecaster.from_pretrained(args.model_dir)
    worst = 0.0
    for name, case in cases():
        got, ms = service(args.url, args.quality, case)
        want, ref_ms = reference(fc, case)
        assert got.shape == want.shape, (got.shape, want.shape)
        # Difference relative to the series' spread plus the value itself: float32 rounding of a
        # far tail (q0.99 can be 1000x the median after sinh) stays relative to its own size.
        scale = np.nanstd(case["context"], axis=-1)[:, None, None] + np.abs(want) + 1e-6
        err = np.abs(got - want) / scale
        med = LEVELS.index(0.5)
        print(f"{name:55s} all={err.max():.1e} q0.5={err[..., med].max():.1e} "
              f"q0.1={err[..., 2].max():.1e} q0.9={err[..., 18].max():.1e}  rust {ms} ms, python {ref_ms:.0f} ms")
        worst = max(worst, err.max())
    print(f"worst relative difference {worst:.2e} (tolerance {args.rtol:.0e})")
    sys.exit(0 if worst <= args.rtol else 1)


if __name__ == "__main__":
    main()
