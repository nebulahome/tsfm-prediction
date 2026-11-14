//! The forecaster around the network, ported from `timestral/forecaster.py` with its defaults:
//! block-by-block decoding with median feedback, the advancing refine frame, the context cap
//! on the quantiles, symmetric averaging and non-negative clipping.

use crate::model::{causal_patch_stats, clamp_nonfinite, Role, Timestral};

/// One series to forecast. `NaN` marks a missing value everywhere.
#[derive(Clone, Debug, Default)]
pub struct Input {
    /// Target variates, each `t` long.
    pub context: Vec<Vec<f32>>,
    pub horizon: usize,
    /// Covariates known over the context only, each `t` long.
    pub past: Vec<Vec<f32>>,
    /// Covariates known over the context and the horizon, each `t + horizon` long.
    pub known: Vec<Vec<f32>>,
}

const QUANTILE_REAL_CAP_K: f32 = 1e4;
const EXTREME_VALUE_THRESHOLD: f32 = 1e30;

/// `TimestralForecaster.forecast` for one series: quantiles `(variates, horizon, Q)`.
pub fn forecast(m: &Timestral, inp: &Input) -> Result<Vec<f32>, String> {
    let c = &m.cfg;
    if (!inp.past.is_empty() || !inp.known.is_empty()) && !c.covariate_enabled {
        return Err("this model was not trained with covariates".into());
    }
    if !inp.known.is_empty() && !c.supports_known_covariates() {
        return Err("this model cannot condition on known-future covariates".into());
    }
    let nq = c.quantile_knots.len();
    let (v, h) = (inp.context.len(), inp.horizon);
    let masks: Vec<Vec<bool>> = inp
        .context
        .iter()
        .map(|r| r.iter().map(|x| x.is_finite()).collect())
        .collect();
    // The two passes run side by side only while their activations are small (about 4 MB of
    // residual stream each), so a big request never doubles its peak memory.
    let tokens = (inp.context.len() + inp.past.len() + inp.known.len())
        * (inp.context[0].len().min(c.context_length) / c.patch_size + 33);
    let (plain, mirrored) = if tokens * c.d_model <= 1 << 20 {
        rayon::join(
            || forecast_raw(m, inp, &masks, 1.0),
            || forecast_raw(m, inp, &masks, -1.0),
        )
    } else {
        (
            forecast_raw(m, inp, &masks, 1.0),
            forecast_raw(m, inp, &masks, -1.0),
        )
    };
    let (plain, mirrored) = (plain?, mirrored?);
    let mut out = vec![0f32; v * h * nq];
    for (step, (a, b)) in out
        .chunks_mut(nq)
        .zip(plain.chunks(nq).zip(mirrored.chunks(nq)))
    {
        for j in 0..nq {
            step[j] = 0.5 * (a[j] + -b[nq - 1 - j]);
        }
    }
    // Clip at 0 the variates whose observed context is entirely non-negative.
    for (i, row) in inp.context.iter().enumerate() {
        let observed = row.iter().filter(|x| x.is_finite());
        if observed.clone().next().is_some() && observed.clone().all(|&x| x >= 0.0) {
            out[i * h * nq..(i + 1) * h * nq]
                .iter_mut()
                .for_each(|q| *q = q.max(0.0));
        }
    }
    Ok(out)
}

/// `_forecast_raw`: one pass on `sign * inputs` (covariates negated with it).
fn forecast_raw(
    m: &Timestral,
    inp: &Input,
    masks: &[Vec<bool>],
    sign: f32,
) -> Result<Vec<f32>, String> {
    let c = &m.cfg;
    let ps = c.patch_size;
    let nq = c.quantile_knots.len();
    let (v, h) = (inp.context.len(), inp.horizon);
    let t0 = inp.context[0].len();

    // _assemble_rows: targets forward-filled, covariates with NaN as missing.
    let mut rows: Vec<(Vec<f32>, Vec<bool>)> = Vec::new();
    for (x, mk) in inp.context.iter().zip(masks) {
        let mut last = 0f32;
        let vals = x.iter().zip(mk).map(|(&x, &o)| {
            if o {
                last = sign * x;
            }
            last
        });
        rows.push((vals.collect(), mk.clone()));
    }
    let cov = |r: &[f32]| -> (Vec<f32>, Vec<bool>) {
        (
            r.iter()
                .map(|&x| if x.is_nan() { 0.0 } else { sign * x })
                .collect(),
            r.iter().map(|x| !x.is_nan()).collect(),
        )
    };
    let mut roles = vec![Role::Target; v];
    for r in &inp.past {
        rows.push(cov(r));
        roles.push(Role::Past);
    }
    for r in &inp.known {
        rows.push(cov(&r[..t0]));
        roles.push(Role::Known);
    }
    let nr = rows.len();
    // Keep the most recent `context_length` steps, then left-pad to whole patches.
    let keep = t0.min(c.context_length);
    let pad = (ps - keep % ps) % ps;
    let mut t = keep + pad;
    let mut ctx = vec![0f32; nr * t];
    let mut obs = vec![false; nr * t];
    for (i, (x, mk)) in rows.iter().enumerate() {
        ctx[i * t + pad..(i + 1) * t].copy_from_slice(&x[t0 - keep..]);
        obs[i * t + pad..(i + 1) * t].copy_from_slice(&mk[t0 - keep..]);
    }
    let cap_rows: Vec<(Vec<f32>, Vec<bool>)> = (0..v)
        .map(|i| {
            (
                ctx[i * t..(i + 1) * t].to_vec(),
                obs[i * t..(i + 1) * t].to_vec(),
            )
        })
        .collect();
    let roles = if nr > v { Some(roles) } else { None };
    let hp = h.div_ceil(ps) * ps;
    let mut kf = vec![0f32; nr * hp];
    let mut kfm = vec![false; nr * hp];
    for (j, r) in inp.known.iter().enumerate() {
        let i = v + inp.past.len() + j;
        let (vals, mk) = cov(&r[t0..]);
        kf[i * hp..i * hp + h].copy_from_slice(&vals);
        kfm[i * hp..i * hp + h].copy_from_slice(&mk);
    }

    // _block_ar_decode
    let per_shot = 32.min(c.block_size);
    let med = c.median_index();
    let mut blocks: Vec<Vec<f32>> = vec![Vec::with_capacity(hp * nq); nr];
    let mut remaining = h.div_ceil(ps);
    let mut offset = 0;
    while remaining > 0 {
        let block = per_shot.min(remaining);
        let blen = block * ps;
        let (fv, fm): (Vec<f32>, Vec<bool>) = if inp.known.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            (0..nr)
                .flat_map(|i| (offset..offset + blen).map(move |k| (i, k)))
                .map(|(i, k)| (kf[i * hp + k], kfm[i * hp + k]))
                .unzip()
        };
        let future = if inp.known.is_empty() {
            None
        } else {
            Some((&fv[..], &fm[..]))
        };
        let mut q = m.predict_block(nr, &ctx, &obs, t, block, roles.as_deref(), future);
        refine_block(m, &mut q, &ctx, &obs, t, nr, blen, roles.as_deref());
        for (i, b) in blocks.iter_mut().enumerate() {
            b.extend_from_slice(&q[i * blen * nq..(i + 1) * blen * nq]);
        }
        remaining -= block;
        offset += blen;
        if remaining == 0 {
            break;
        }
        // Feedback: targets carry their median, known covariates their real future, past ones a blank.
        let nt = t + blen;
        let mut nctx = vec![0f32; nr * nt];
        let mut nobs = vec![false; nr * nt];
        for i in 0..nr {
            nctx[i * nt..i * nt + t].copy_from_slice(&ctx[i * t..(i + 1) * t]);
            nobs[i * nt..i * nt + t].copy_from_slice(&obs[i * t..(i + 1) * t]);
            for k in 0..blen {
                let (val, o) = match roles.as_ref().map_or(Role::Target, |r| r[i]) {
                    Role::Target => {
                        let x = q[(i * blen + k) * nq + med];
                        (
                            if x.is_finite() && x.abs() < EXTREME_VALUE_THRESHOLD {
                                x
                            } else {
                                0.0
                            },
                            true,
                        )
                    }
                    Role::Known => (fv[i * blen + k], fm[i * blen + k]),
                    Role::Past => (0.0, false),
                };
                nctx[i * nt + t + k] = val;
                nobs[i * nt + t + k] = o;
            }
        }
        // Past the cap the oldest steps are dropped.
        let keep = if nt > c.context_length {
            ps.max(c.context_length / ps * ps)
        } else {
            nt
        };
        ctx = (0..nr)
            .flat_map(|i| nctx[i * nt + nt - keep..(i + 1) * nt].to_vec())
            .collect();
        obs = (0..nr)
            .flat_map(|i| nobs[i * nt + nt - keep..(i + 1) * nt].to_vec())
            .collect();
        t = keep;
    }

    // Targets only, capped to [min - k * s, max + k * s] of the observed context.
    let mut out = Vec::with_capacity(v * h * nq);
    for (i, (x, mk)) in cap_rows.iter().enumerate() {
        let observed = || x.iter().zip(mk).filter(|p| *p.1).map(|p| *p.0);
        let hi = observed().fold(f32::NEG_INFINITY, f32::max);
        let lo = observed().fold(f32::INFINITY, f32::min);
        let (hi, lo) = (
            if hi.is_finite() { hi } else { 0.0 },
            if lo.is_finite() { lo } else { 0.0 },
        );
        let s = *causal_patch_stats(x, mk, ps, c.minimum_scale)
            .1
            .last()
            .unwrap()
            * QUANTILE_REAL_CAP_K;
        for &q in &blocks[i][..h * nq] {
            if q.is_nan() {
                return Err("the forecast produced NaN values; the input is degenerate".into());
            }
            out.push(q.max(lo - s).min(hi + s));
        }
    }
    Ok(out)
}

/// `_refine_block`: re-denormalize horizon patch `j > 0` of the target rows with the context
/// statistics plus the block's earlier medians (the advancing refine frame).
fn refine_block(
    m: &Timestral,
    q: &mut [f32],
    ctx: &[f32],
    obs: &[bool],
    t: usize,
    rows: usize,
    blen: usize,
    roles: Option<&[Role]>,
) {
    let c = &m.cfg;
    let ps = c.patch_size;
    let nq = c.quantile_knots.len();
    let n_pred = blen / ps;
    if n_pred <= 1 {
        return;
    }
    let med = c.median_index();
    for i in 0..rows {
        if roles.is_some_and(|r| r[i] != Role::Target) {
            continue;
        }
        let (x, mk) = (&ctx[i * t..(i + 1) * t], &obs[i * t..(i + 1) * t]);
        let (locs, scales) = causal_patch_stats(x, mk, ps, c.minimum_scale);
        let (loc0, scale0) = (*locs.last().unwrap(), *scales.last().unwrap());
        let qi = &mut q[i * blen * nq..(i + 1) * blen * nq];
        let sinh_median: Vec<f64> = (0..blen)
            .map(|k| (qi[k * nq + med] as f64 - loc0 as f64) / scale0 as f64)
            .collect();
        let (lj, sj) = advancing_block_frames(x, mk, &sinh_median, ps, c.minimum_scale);
        // Patch 0 keeps the anchor frame, but goes through the same arithmetic as in Python.
        for p in 0..n_pred {
            let (l, s) = if p == 0 {
                (loc0, scale0)
            } else {
                (lj[p] as f32, sj[p] as f32)
            };
            for k in p * ps..(p + 1) * ps {
                let step = &mut qi[k * nq..(k + 1) * nq];
                for v in step.iter_mut() {
                    *v = (*v - loc0) / scale0 * s + l;
                }
                clamp_nonfinite(step);
            }
        }
    }
}

type Welford = (f64, f64, f64);

/// `welford_combine`; `(0, 0, 0)` is the identity.
fn combine(a: Welford, b: Welford) -> Welford {
    let w = a.0 + b.0;
    let ws = w.max(f64::MIN_POSITIVE);
    let delta = b.1 - a.1;
    let mean = a.1 + delta * b.0 / ws;
    let m2 = a.2 + b.2 + delta * delta * a.0 * b.0 / ws;
    if w == 0.0 {
        (w, 0.0, 0.0)
    } else {
        (w, mean, m2)
    }
}

/// `advancing_block_frames` for one row: per-patch `(loc, scale)` of a decoded block.
fn advancing_block_frames(
    x: &[f32],
    mk: &[bool],
    sinh_median: &[f64],
    ps: usize,
    min_scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_ctx = x.len() / ps;
    let n_pred = sinh_median.len() / ps;
    let mut st: Vec<Welford> = (0..n_ctx)
        .map(|p| {
            let (xs, ms) = (&x[p * ps..(p + 1) * ps], &mk[p * ps..(p + 1) * ps]);
            let cnt: f64 = ms.iter().map(|&o| if o { 1.0 } else { 0.0 }).sum();
            let mean = xs
                .iter()
                .zip(ms)
                .map(|(&v, &o)| if o { v as f64 } else { 0.0 })
                .sum::<f64>()
                / cnt.max(1.0);
            let m2: f64 = xs
                .iter()
                .zip(ms)
                .map(|(&v, &o)| if o { (v as f64 - mean).powi(2) } else { 0.0 })
                .sum();
            if cnt == 0.0 {
                (0.0, 0.0, 0.0)
            } else {
                (cnt, mean, m2)
            }
        })
        .collect();
    st.resize(n_ctx + n_pred, (0.0, 0.0, 0.0));
    // Hillis-Steele inclusive scan, the same merge order as `cumulative_welford`.
    let mut d = 1;
    while d < st.len() {
        st = (0..st.len())
            .map(|i| combine(if i >= d { st[i - d] } else { (0.0, 0.0, 0.0) }, st[i]))
            .collect();
        d *= 2;
    }
    let mut cur = st[n_ctx - 1];
    let offset = cur.1;
    let clip = (f32::MAX as f64).sqrt() / 1e3;
    let (mut locs, mut scales) = (Vec::with_capacity(n_pred), Vec::with_capacity(n_pred));
    for p in 0..n_pred {
        let var = (cur.2 / (cur.0 - 1.0).max(1.0)).max(0.0);
        let scale = var.sqrt().max(min_scale);
        locs.push(cur.1);
        scales.push(scale);
        let folded: Vec<f64> = sinh_median[p * ps..(p + 1) * ps]
            .iter()
            .map(|&z| (z * scale + cur.1 - offset).clamp(-clip, clip) + offset)
            .collect();
        let b_mean = folded.iter().sum::<f64>() / ps as f64;
        let b_m2 = folded.iter().map(|f| (f - b_mean).powi(2)).sum();
        cur = combine(cur, (ps as f64, b_mean, b_m2));
    }
    (locs, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welford_scan_matches_direct_stats() {
        let x: Vec<f32> = (0..64)
            .map(|i| (i as f32 * 0.3).sin() * 5.0 + i as f32)
            .collect();
        let mk = vec![true; 64];
        let (l, s) = advancing_block_frames(&x, &mk, &[0.0; 64], 32, 1e-6);
        let mean = x.iter().map(|&v| v as f64).sum::<f64>() / 64.0;
        let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / 63.0;
        assert!((l[0] - mean).abs() < 1e-9 && (s[0] - var.sqrt()).abs() < 1e-9);
    }
}
