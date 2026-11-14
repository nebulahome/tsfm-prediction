//! The Timestral network, ported from `timestral/model.py`, `layers.py`, `masks.py` and the
//! causal scaler of `scaler.py` (https://github.com/mistralai/timestral, Apache-2.0).
//!
//! Weights stay in the memory-mapped `model.safetensors` (no copy): they count as file-backed
//! memory while loaded and are handed back to the OS by `munmap` when the model is dropped.
//! Activations use the packed layout of the Python code: one row per variate, `S` tokens per
//! row (`n` encoder patches then `h` decoder queries), `D` floats per token.

use std::{collections::HashMap, fs::File, path::Path};

use gemm::Parallelism;
use memmap2::Mmap;
use rayon::prelude::*;
use serde::Deserialize;

#[cfg(target_endian = "big")]
compile_error!("safetensors weights are little-endian and are read in place");

/// Row role in a covariate forecast (`ROLE_*` in model.py).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Target,
    Past,
    Known,
}

fn d_theta() -> f64 {
    10000.0
}
fn d_half() -> f64 {
    0.5
}
fn d_eps() -> f64 {
    1e-6
}
fn d_gate() -> String {
    "sdpa_sigmoid".into()
}
fn d_true() -> bool {
    true
}
fn d_two() -> f64 {
    2.0
}

/// `config.json` of a checkpoint (`TimestralConfig`), unknown keys rejected as in Python.
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub patch_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub mlp_hidden_dim: usize,
    pub block_schedule: Vec<String>,
    pub quantile_knots: Vec<f64>,
    pub block_size: usize,
    pub context_length: usize,
    #[serde(default = "d_theta")]
    pub rope_theta: f64,
    #[serde(default = "d_half")]
    pub rope_partial_factor: f64,
    #[serde(default = "d_eps")]
    pub norm_eps: f64,
    #[serde(default)]
    pub attn_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "d_gate")]
    pub time_attn_gate: String,
    #[serde(default = "d_gate")]
    pub variate_attn_gate: String,
    #[serde(default = "d_eps")]
    pub minimum_scale: f64,
    #[serde(default = "d_true")]
    pub use_asinh: bool,
    #[serde(default)]
    pub stabilize_with_global: bool,
    #[serde(default = "d_true")]
    pub covariate_enabled: bool,
    // Splitting context and horizon queries is only a speed-up; the result is the same.
    #[serde(default = "d_true")]
    #[allow(dead_code)]
    pub split_context_horizon_attn: bool,
    #[serde(default = "d_true")]
    pub attend_missing_patches: bool,
    #[serde(default = "d_two")]
    pub quantile_width_factor: f64,
}

impl Config {
    fn axes(kind: &str) -> Option<&'static [bool]> {
        match kind {
            "time" => Some(&[true]),
            "variate" => Some(&[false]),
            "time_variate" => Some(&[true, false]),
            "variate_time" => Some(&[false, true]),
            _ => None,
        }
    }

    pub fn median_index(&self) -> usize {
        self.quantile_knots.iter().position(|&q| q == 0.5).unwrap()
    }

    /// Whether a known covariate's future can reach a target (`supports_known_covariates`).
    pub fn supports_known_covariates(&self) -> bool {
        if !self.covariate_enabled {
            return false;
        }
        let ops: Vec<bool> = self
            .block_schedule
            .iter()
            .flat_map(|k| Self::axes(k).unwrap().iter().copied())
            .collect();
        (0..ops.len())
            .any(|i| ops[i] && ops[i + 1..].iter().any(|&t| t) && !ops[i + 1..].iter().all(|&t| t))
    }

    fn check(&self) -> Result<(), String> {
        let hd = self.d_model / self.num_heads.max(1);
        let rope = (hd as f64 * self.rope_partial_factor) as usize;
        let ok = self.num_heads > 0
            && self.d_model.is_multiple_of(self.num_heads)
            && rope > 0
            && rope.is_multiple_of(2)
            && hd <= 256
            && self.patch_size > 0
            && self.block_size > 0
            && self.block_schedule.iter().all(|k| Self::axes(k).is_some())
            && [&self.time_attn_gate, &self.variate_attn_gate]
                .iter()
                .all(|g| *g == "none" || *g == "sdpa_sigmoid")
            && self.quantile_knots.contains(&0.5)
            && self.quantile_knots.windows(2).all(|w| w[0] < w[1]);
        if !ok {
            return Err("config.json is not a valid Timestral configuration".into());
        }
        // shortcut: biases and the global stabilizer are off in every published checkpoint; port them if one ships with them
        if self.attn_bias || self.mlp_bias || self.stabilize_with_global {
            return Err("config.json uses attn_bias, mlp_bias or stabilize_with_global, which this port does not implement".into());
        }
        Ok(())
    }
}

/// A tensor inside the mapped file: offset and length in f32 units from the data start.
#[derive(Clone, Copy)]
struct T {
    off: usize,
    len: usize,
}

struct Resid {
    hidden: (T, Option<T>),
    output: (T, Option<T>),
    residual: (T, Option<T>),
    dims: (usize, usize, usize),
}

struct Attn {
    is_time: bool,
    norm: T,
    qkv: T,
    wo: T,
    gate: Option<T>,
}

struct Layer {
    attns: Vec<Attn>,
    mlp_norm: T,
    w12: T,
    w3: T,
}

pub struct Timestral {
    pub cfg: Config,
    map: Mmap,
    base: usize,
    embed: Resid,
    query: T,
    layers: Vec<Layer>,
    out_norm: T,
    head: Resid,
    inv_freq: Vec<f32>,
}

#[derive(Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

impl Timestral {
    /// Map `model.safetensors` and check every tensor's name, dtype and shape (`strict=True`).
    pub fn load(dir: &Path) -> Result<Self, String> {
        let cfg: Config = serde_json::from_slice(
            &std::fs::read(dir.join("config.json")).map_err(|e| format!("config.json: {e}"))?,
        )
        .map_err(|e| format!("config.json: {e}"))?;
        cfg.check()?;
        let file = File::open(dir.join("model.safetensors"))
            .map_err(|e| format!("model.safetensors: {e}"))?;
        // SAFETY: the cache file is only ever replaced by rename, never written in place.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| format!("model.safetensors: {e}"))?;
        let bad = |m: &str| format!("model.safetensors: {m}");
        if map.len() < 8 {
            return Err(bad("truncated"));
        }
        let hlen = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        let base = 8usize
            .checked_add(hlen)
            .filter(|&b| b <= map.len())
            .ok_or_else(|| bad("truncated header"))?;
        let mut header: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&map[8..base]).map_err(|e| bad(&e.to_string()))?;
        header.remove("__metadata__");
        if !(map.as_ptr() as usize + base).is_multiple_of(4) {
            return Err(bad("tensor data is not 4-byte aligned"));
        }
        let data_len = map.len() - base;
        let mut take = |name: &str, shape: &[usize]| -> Result<T, String> {
            let v = header
                .remove(name)
                .ok_or_else(|| bad(&format!("missing tensor {name}")))?;
            let e: Entry = serde_json::from_value(v).map_err(|e| bad(&e.to_string()))?;
            let [a, b] = e.data_offsets;
            let len: usize = shape.iter().product();
            if e.dtype != "F32"
                || e.shape != shape
                || b < a
                || b - a != len * 4
                || b > data_len
                || a % 4 != 0
            {
                return Err(bad(&format!("tensor {name} is not F32 {shape:?}")));
            }
            Ok(T { off: a / 4, len })
        };
        let (d, ps, nq, hid) = (
            cfg.d_model,
            cfg.patch_size,
            cfg.quantile_knots.len(),
            cfg.mlp_hidden_dim,
        );
        let mut resid =
            |p: &str, i: usize, h: usize, o: usize, bias: bool| -> Result<Resid, String> {
                let mut lin = |n: &str, o: usize, i: usize| -> Result<(T, Option<T>), String> {
                    let w = take(&format!("{p}.{n}.weight"), &[o, i])?;
                    let b = if bias {
                        Some(take(&format!("{p}.{n}.bias"), &[o])?)
                    } else {
                        None
                    };
                    Ok((w, b))
                };
                Ok(Resid {
                    hidden: lin("hidden_layer", h, i)?,
                    output: lin("output_layer", o, h)?,
                    residual: lin("residual_layer", o, i)?,
                    dims: (i, h, o),
                })
            };
        let embed = resid("representation.patch_proj", 2 * ps, 4 * d, d, true)?;
        let head = resid("head.projection", d, 4 * d, ps * nq, false)?;
        let query = take("representation.forecast_query", &[d])?;
        let out_norm = take("transformer.out_norm.scale", &[d])?;
        let mut layers = Vec::new();
        for (li, kind) in cfg.block_schedule.iter().enumerate() {
            let p = format!("transformer.layers.{li}");
            let mut attns = Vec::new();
            for (ai, &is_time) in Config::axes(kind).unwrap().iter().enumerate() {
                let gated = if is_time {
                    &cfg.time_attn_gate
                } else {
                    &cfg.variate_attn_gate
                } == "sdpa_sigmoid";
                attns.push(Attn {
                    is_time,
                    norm: take(&format!("{p}.attn_norms.{ai}.scale"), &[d])?,
                    qkv: take(&format!("{p}.attns.{ai}.wQKV.weight"), &[3 * d, d])?,
                    wo: take(&format!("{p}.attns.{ai}.wO.weight"), &[d, d])?,
                    gate: if gated {
                        Some(take(&format!("{p}.attns.{ai}._gate.weight"), &[d, d])?)
                    } else {
                        None
                    },
                });
            }
            layers.push(Layer {
                attns,
                mlp_norm: take(&format!("{p}.mlp_norm.scale"), &[d])?,
                w12: take(&format!("{p}.mlp.w12.weight"), &[2 * hid, d])?,
                w3: take(&format!("{p}.mlp.w3.weight"), &[d, hid])?,
            });
        }
        if let Some(extra) = header.keys().next() {
            return Err(bad(&format!("unexpected tensor {extra}")));
        }
        // SinCosRoPE._build_inv_freq, in f32 like torch.
        let hd = d / cfg.num_heads;
        let rot = (hd as f64 * cfg.rope_partial_factor) as usize;
        let mut inv_freq = vec![0f32; hd / 2];
        for (i, f) in inv_freq.iter_mut().take(rot / 2).enumerate() {
            *f = 1.0 / (cfg.rope_theta as f32).powf((2 * i) as f32 / rot as f32);
        }
        Ok(Self {
            cfg,
            map,
            base,
            embed,
            query,
            layers,
            out_norm,
            head,
            inv_freq,
        })
    }

    fn w(&self, t: T) -> &[f32] {
        // SAFETY: bounds and alignment were checked in `load`; the map lives as long as self.
        unsafe {
            std::slice::from_raw_parts(
                (self.map.as_ptr().add(self.base) as *const f32).add(t.off),
                t.len,
            )
        }
    }

    fn bias(&self, t: Option<T>) -> Option<&[f32]> {
        t.map(|t| self.w(t))
    }

    /// `ResidualBlock`: `output(silu(hidden(x))) + residual(x)` on `m` tokens.
    fn resid(&self, r: &Resid, x: &[f32], m: usize, out: &mut [f32]) {
        let (i, h, o) = r.dims;
        let mut hid = vec![0f32; m * h];
        linear(
            x,
            m,
            i,
            self.w(r.hidden.0),
            h,
            self.bias(r.hidden.1),
            &mut hid,
            false,
        );
        hid.iter_mut().for_each(|v| *v = silu(*v));
        linear(
            &hid,
            m,
            h,
            self.w(r.output.0),
            o,
            self.bias(r.output.1),
            out,
            false,
        );
        let mut res = vec![0f32; m * o];
        linear(
            x,
            m,
            i,
            self.w(r.residual.0),
            o,
            self.bias(r.residual.1),
            &mut res,
            false,
        );
        out.iter_mut().zip(&res).for_each(|(a, b)| *a += b);
    }

    /// `TimestralModel.predict_block`: quantiles `(rows, h * patch_size, Q)` in raw space.
    ///
    /// `ctx`/`obs` are `(rows, t)` with `t` a multiple of the patch size. `future` holds the
    /// known-future values of this block `(rows, h * patch_size)`, zeros on other rows.
    pub fn predict_block(
        &self,
        rows: usize,
        ctx: &[f32],
        obs: &[bool],
        t: usize,
        h: usize,
        roles: Option<&[Role]>,
        future: Option<(&[f32], &[bool])>,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let (ps, d, nq) = (c.patch_size, c.d_model, c.quantile_knots.len());
        let ll = t / ps;
        let roles = roles.filter(|r| c.covariate_enabled && r.iter().any(|&x| x != Role::Target));
        let role = |i: usize| roles.map_or(Role::Target, |r| r[i]);
        // With covariates the encoder also covers the horizon, so known-future values are keys.
        let n = if roles.is_some() { ll + h } else { ll };
        let w = n * ps;
        let s = n + h;

        let mut x = vec![0f32; rows * s * d];
        let mut key_valid = vec![false; rows * n];
        let mut anchor = vec![(0f32, 0f32); rows];
        let mut any_obs = vec![false; n];
        for i in 0..rows {
            let mut vals = vec![0f32; w];
            let mut m = vec![false; w];
            vals[..t].copy_from_slice(&ctx[i * t..(i + 1) * t]);
            m[..t].copy_from_slice(&obs[i * t..(i + 1) * t]);
            if let (true, Some((fv, fm))) = (roles.is_some(), future) {
                vals[t..].copy_from_slice(&fv[i * h * ps..(i + 1) * h * ps]);
                m[t..].copy_from_slice(&fm[i * h * ps..(i + 1) * h * ps]);
            }
            let (loc, scale, scaled) = if role(i) == Role::Known {
                known_scaling(&vals, &m, c, n)
            } else {
                causal_scaling(&vals, &m, c)
            };
            anchor[i] = (loc[ll - 1], scale[ll - 1]);
            // PatchRepresentation: [values, missing] per patch.
            let mut patches = vec![0f32; n * 2 * ps];
            for p in 0..n {
                let row = &mut patches[p * 2 * ps..(p + 1) * 2 * ps];
                row[..ps].copy_from_slice(&scaled[p * ps..(p + 1) * ps]);
                for k in 0..ps {
                    row[ps + k] = if m[p * ps + k] { 0.0 } else { 1.0 };
                }
                let observed = m[p * ps..(p + 1) * ps].iter().any(|&b| b);
                any_obs[p] |= observed;
                // compute_enc_key_validity: observed, or the row has started.
                let started = p > 0 && key_valid[i * n + p - 1];
                key_valid[i * n + p] = observed || (c.attend_missing_patches && started);
            }
            let xr = &mut x[i * s * d..(i + 1) * s * d];
            self.resid(&self.embed, &patches, n, &mut xr[..n * d]);
            for a in 0..h {
                xr[(n + a) * d..(n + a + 1) * d].copy_from_slice(self.w(self.query));
            }
        }

        // RoPE tables: one series, so every row shares the positions (patch - first observed patch).
        let first = any_obs.iter().position(|&b| b).unwrap_or(0) as i64;
        let hd = d / c.num_heads;
        let mut cos = vec![0f32; s * hd];
        let mut sin = vec![0f32; s * hd];
        for tok in 0..s {
            let p = if tok < n {
                tok as i64
            } else {
                (ll + tok - n) as i64
            };
            let pos = (p - first) as f32;
            for e in 0..hd {
                let f = self.inv_freq[e % (hd / 2)] * pos;
                cos[tok * hd + e] = f.cos();
                sin[tok * hd + e] = f.sin();
            }
        }
        let ctx = Ctx {
            rows,
            n,
            ll,
            s,
            key_valid: &key_valid,
            roles,
            cos: &cos,
            sin: &sin,
        };

        for layer in &self.layers {
            for a in &layer.attns {
                if a.is_time {
                    for i in 0..rows {
                        self.time_attention(a, &mut x[i * s * d..(i + 1) * s * d], &ctx, i);
                    }
                } else {
                    self.variate_attention(a, &mut x, &ctx);
                }
            }
            for i in 0..rows {
                self.mlp(layer, &mut x[i * s * d..(i + 1) * s * d], s);
            }
        }

        // Head on the decoder tokens, then denormalize in the frame of the last context patch.
        let hp = h * ps;
        let mut out = vec![0f32; rows * hp * nq];
        let mut normed = vec![0f32; h * d];
        let mut raw = vec![0f32; h * ps * nq];
        for i in 0..rows {
            rms_norm(
                &x[(i * s + n) * d..(i + 1) * s * d],
                self.w(self.out_norm),
                c.norm_eps as f32,
                &mut normed,
            );
            self.resid(&self.head, &normed, h, &mut raw);
            let (loc, scale) = anchor[i];
            for step in 0..hp {
                let q = &mut out[(i * hp + step) * nq..(i * hp + step + 1) * nq];
                monotone_quantiles(
                    &raw[step * nq..(step + 1) * nq],
                    &c.quantile_knots,
                    c.quantile_width_factor as f32,
                    q,
                );
                for v in q.iter_mut() {
                    let z = if c.use_asinh { v.sinh() } else { *v };
                    *v = z * scale + loc;
                }
                clamp_nonfinite(q);
            }
        }
        out
    }

    fn time_attention(&self, a: &Attn, xr: &mut [f32], cx: &Ctx, row: usize) {
        let c = &self.cfg;
        let (d, nh) = (c.d_model, c.num_heads);
        let hd = d / nh;
        let s = cx.s;
        let mut nrm = vec![0f32; s * d];
        rms_norm(xr, self.w(a.norm), c.norm_eps as f32, &mut nrm);
        let (mut q, mut k, v) = self.qkv(a, &nrm, s);
        for tok in 0..s {
            let (cs, sn) = (
                &cx.cos[tok * hd..(tok + 1) * hd],
                &cx.sin[tok * hd..(tok + 1) * hd],
            );
            for hh in 0..nh {
                let o = tok * d + hh * hd;
                rope(&mut q[o..o + hd], cs, sn);
                rope(&mut k[o..o + hd], cs, sn);
            }
        }
        let valid = &cx.key_valid[row * cx.n..(row + 1) * cx.n];
        let bidir = cx.roles.is_some_and(|r| r[row] == Role::Known);
        let (n, ll) = (cx.n, cx.ll);
        let allowed = |qi: usize, kj: usize| -> bool {
            if qi < n {
                // Encoder: causal over valid keys (all valid keys for known covariates), self always.
                kj == qi || (kj < n && valid[kj] && (bidir || kj <= qi))
            } else if kj < n {
                // Decoder to encoder: valid context patches only.
                kj < ll && valid[kj]
            } else {
                kj <= qi // decoder chain is causal
            }
        };
        let key_end = |q_end: usize| {
            if q_end <= n {
                if bidir {
                    n
                } else {
                    q_end
                }
            } else {
                s
            }
        };
        let mut att = vec![0f32; s * d];
        attend(&q, &k, &v, nh, hd, &mut att, &allowed, &key_end);
        drop((q, k, v));
        self.gate_and_project(a, &nrm, att, s, xr);
    }

    fn variate_attention(&self, a: &Attn, x: &mut [f32], cx: &Ctx) {
        let c = &self.cfg;
        let (d, nh) = (c.d_model, c.num_heads);
        let hd = d / nh;
        let (r, s, n) = (cx.rows, cx.s, cx.n);
        // variate_firewall: covariates never read targets; known-future rows never read causal rows.
        let role = |i: usize| cx.roles.map_or(Role::Target, |ro| ro[i]);
        let forbid = |qi: usize, kj: usize| {
            cx.roles.is_some()
                && ((role(qi) != Role::Target && role(kj) == Role::Target)
                    || (role(qi) == Role::Known && role(kj) != Role::Known))
        };
        const SLOTS: usize = 64;
        for s0 in (0..s).step_by(SLOTS) {
            let cs = SLOTS.min(s - s0);
            let m = cs * r;
            let mut tok = vec![0f32; m * d];
            for si in 0..cs {
                for i in 0..r {
                    tok[(si * r + i) * d..(si * r + i + 1) * d]
                        .copy_from_slice(&x[(i * s + s0 + si) * d..(i * s + s0 + si + 1) * d]);
                }
            }
            let mut nrm = vec![0f32; m * d];
            rms_norm(&tok, self.w(a.norm), c.norm_eps as f32, &mut nrm);
            let (q, k, v) = self.qkv(a, &nrm, m);
            let mut att = vec![0f32; m * d];
            let scale = 1.0 / (hd as f32).sqrt();
            att.par_chunks_mut(r * d).enumerate().for_each(|(si, out)| {
                let slot = s0 + si;
                let mut p = vec![0f32; r];
                for hh in 0..nh {
                    for qi in 0..r {
                        let qv = &q[(si * r + qi) * d + hh * hd..][..hd];
                        let mut mx = f32::NEG_INFINITY;
                        for kj in 0..r {
                            let ok = kj == qi
                                || ((slot >= n || cx.key_valid[kj * n + slot]) && !forbid(qi, kj));
                            p[kj] = if ok {
                                dot(qv, &k[(si * r + kj) * d + hh * hd..][..hd]) * scale
                            } else {
                                f32::NEG_INFINITY
                            };
                            mx = mx.max(p[kj]);
                        }
                        softmax_in_place(&mut p, mx);
                        let o = &mut out[qi * d + hh * hd..][..hd];
                        o.fill(0.0);
                        for kj in 0..r {
                            if p[kj] != 0.0 {
                                let vv = &v[(si * r + kj) * d + hh * hd..][..hd];
                                o.iter_mut().zip(vv).for_each(|(a, b)| *a += p[kj] * b);
                            }
                        }
                    }
                }
            });
            drop((q, k, v));
            // Project, then add back into the rows of each slot.
            let mut delta = tok;
            delta.fill(0.0);
            self.gate_and_project(a, &nrm, att, m, &mut delta);
            for si in 0..cs {
                for i in 0..r {
                    let dst = &mut x[(i * s + s0 + si) * d..(i * s + s0 + si + 1) * d];
                    dst.iter_mut()
                        .zip(&delta[(si * r + i) * d..(si * r + i + 1) * d])
                        .for_each(|(a, b)| *a += b);
                }
            }
        }
    }

    /// Fused projection split into token-major `(m, heads, head_dim)` q, k, v.
    /// The weight layout is `(qkv, head_dim, n_heads)`, as in `Attention._qkv`.
    fn qkv(&self, a: &Attn, nrm: &[f32], m: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let d = self.cfg.d_model;
        let nh = self.cfg.num_heads;
        let hd = d / nh;
        let mut fused = vec![0f32; m * 3 * d];
        linear(nrm, m, d, self.w(a.qkv), 3 * d, None, &mut fused, false);
        let mut out = [vec![0f32; m * d], vec![0f32; m * d], vec![0f32; m * d]];
        for (which, dst) in out.iter_mut().enumerate() {
            for tok in 0..m {
                let src = &fused[tok * 3 * d + which * d..tok * 3 * d + (which + 1) * d];
                for hh in 0..nh {
                    for e in 0..hd {
                        dst[tok * d + hh * hd + e] = src[e * nh + hh];
                    }
                }
            }
        }
        let [q, k, v] = out;
        (q, k, v)
    }

    /// `out * sigmoid(gate(inputs))`, then `wO`, accumulated into `dst` (the residual add).
    fn gate_and_project(
        &self,
        a: &Attn,
        nrm: &[f32],
        mut att: Vec<f32>,
        m: usize,
        dst: &mut [f32],
    ) {
        let d = self.cfg.d_model;
        if let Some(g) = a.gate {
            let mut gate = vec![0f32; m * d];
            linear(nrm, m, d, self.w(g), d, None, &mut gate, false);
            att.iter_mut()
                .zip(&gate)
                .for_each(|(o, g)| *o *= sigmoid(*g));
        }
        let mut proj = vec![0f32; m * d];
        linear(&att, m, d, self.w(a.wo), d, None, &mut proj, false);
        dst.iter_mut().zip(&proj).for_each(|(a, b)| *a += b);
    }

    /// Pre-norm `FusedSwiGLU` residual: `x + w3(gate * silu(up))`.
    fn mlp(&self, l: &Layer, xr: &mut [f32], s: usize) {
        let (d, hid) = (self.cfg.d_model, self.cfg.mlp_hidden_dim);
        let mut nrm = vec![0f32; s * d];
        rms_norm(xr, self.w(l.mlp_norm), self.cfg.norm_eps as f32, &mut nrm);
        let mut h12 = vec![0f32; s * 2 * hid];
        linear(&nrm, s, d, self.w(l.w12), 2 * hid, None, &mut h12, false);
        drop(nrm);
        let mut act = vec![0f32; s * hid];
        for tok in 0..s {
            let (g, u) = h12[tok * 2 * hid..(tok + 1) * 2 * hid].split_at(hid);
            for j in 0..hid {
                act[tok * hid + j] = g[j] * silu(u[j]);
            }
        }
        drop(h12);
        let mut proj = vec![0f32; s * d];
        linear(&act, s, hid, self.w(l.w3), d, None, &mut proj, false);
        xr.iter_mut().zip(&proj).for_each(|(a, b)| *a += b);
    }
}

struct Ctx<'a> {
    rows: usize,
    n: usize,
    ll: usize,
    s: usize,
    key_valid: &'a [bool],
    roles: Option<&'a [Role]>,
    cos: &'a [f32],
    sin: &'a [f32],
}

/// Masked multi-head attention over one row's `s` tokens, query chunks in parallel.
fn attend(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    nh: usize,
    hd: usize,
    out: &mut [f32],
    allowed: &(dyn Fn(usize, usize) -> bool + Sync),
    key_end: &(dyn Fn(usize) -> usize + Sync),
) {
    const CHUNK: usize = 16;
    let d = nh * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    out.par_chunks_mut(CHUNK * d)
        .enumerate()
        .for_each(|(ci, o)| {
            let q0 = ci * CHUNK;
            let cq = o.len() / d;
            let ke = key_end(q0 + cq);
            let mut sc = vec![0f32; cq * ke];
            for hh in 0..nh {
                // scores = Q_h[q0..q0+cq] @ K_h[..ke]^T
                unsafe {
                    gemm::gemm(
                        cq,
                        ke,
                        hd,
                        sc.as_mut_ptr(),
                        1,
                        ke as isize,
                        false,
                        q.as_ptr().add(q0 * d + hh * hd),
                        1,
                        d as isize,
                        k.as_ptr().add(hh * hd),
                        d as isize,
                        1,
                        0.0,
                        1.0,
                        false,
                        false,
                        false,
                        Parallelism::None,
                    );
                }
                for qi in 0..cq {
                    let row = &mut sc[qi * ke..(qi + 1) * ke];
                    let mut mx = f32::NEG_INFINITY;
                    for (kj, val) in row.iter_mut().enumerate() {
                        *val = if allowed(q0 + qi, kj) {
                            *val * scale
                        } else {
                            f32::NEG_INFINITY
                        };
                        mx = mx.max(*val);
                    }
                    softmax_in_place(row, mx);
                }
                // out_h = P @ V_h[..ke]
                unsafe {
                    gemm::gemm(
                        cq,
                        hd,
                        ke,
                        o.as_mut_ptr().add(hh * hd),
                        1,
                        d as isize,
                        false,
                        sc.as_ptr(),
                        1,
                        ke as isize,
                        v.as_ptr().add(hh * hd),
                        1,
                        d as isize,
                        0.0,
                        1.0,
                        false,
                        false,
                        false,
                        Parallelism::None,
                    );
                }
            }
        });
}

fn softmax_in_place(p: &mut [f32], mx: f32) {
    let mut sum = 0f32;
    for v in p.iter_mut() {
        *v = if *v == f32::NEG_INFINITY {
            0.0
        } else {
            (*v - mx).exp()
        };
        sum += *v;
    }
    p.iter_mut().for_each(|v| *v /= sum);
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `out (m, n) = x (m, k) @ w (n, k)^T + bias`; `acc` adds to `out` instead.
pub fn linear(
    x: &[f32],
    m: usize,
    k: usize,
    w: &[f32],
    n: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    acc: bool,
) {
    assert!(x.len() >= m * k && w.len() == n * k && out.len() >= m * n);
    if m == 0 {
        return;
    }
    // Tiles of at least ~1M multiply-adds, split over output columns first (disjoint weight
    // rows per thread), then over tokens. gemm's own threading is far slower on these shapes.
    let tasks = rayon::current_num_threads().min((m * n * k) >> 20).max(1);
    let nt = tasks.min(n.div_ceil(32));
    let mt = (tasks / nt).min(m.div_ceil(16)).max(1);
    let (nc, mc) = (n.div_ceil(nt), m.div_ceil(mt));
    let dst = SendPtr(out.as_mut_ptr());
    let tile = |t: usize| {
        let (i0, j0) = ((t / nt) * mc, (t % nt) * nc);
        if i0 >= m || j0 >= n {
            return;
        }
        let (mm, nn) = (mc.min(m - i0), nc.min(n - j0));
        // SAFETY: sizes asserted above; tiles are disjoint; strides describe row-major x/out and w read as its transpose.
        unsafe {
            gemm::gemm(
                mm,
                nn,
                k,
                dst.get().add(i0 * n + j0),
                1,
                n as isize,
                acc,
                x.as_ptr().add(i0 * k),
                1,
                k as isize,
                w.as_ptr().add(j0 * k),
                k as isize,
                1,
                1.0,
                1.0,
                false,
                false,
                false,
                Parallelism::None,
            );
        }
    };
    if nt * mt == 1 {
        tile(0);
    } else {
        (0..nt * mt).into_par_iter().for_each(tile);
    }
    if let Some(b) = bias {
        out[..m * n]
            .chunks_mut(n)
            .for_each(|r| r.iter_mut().zip(b).for_each(|(o, b)| *o += b));
    }
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
// SAFETY: only used to hand disjoint output tiles to rayon tasks.
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    fn get(self) -> *mut f32 {
        self.0
    }
}

fn rms_norm(x: &[f32], scale: &[f32], eps: f32, out: &mut [f32]) {
    let d = scale.len();
    for (xi, oi) in x.chunks(d).zip(out.chunks_mut(d)) {
        let ms = (xi.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / d as f64) as f32;
        let r = 1.0 / (ms + eps).sqrt();
        for j in 0..d {
            oi[j] = xi[j] * r * scale[j];
        }
    }
}

/// `x * cos + rotate_half(x) * sin` for one head.
fn rope(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = x.len() / 2;
    let mut buf = [0f32; 256];
    let orig = &mut buf[..x.len()];
    orig.copy_from_slice(x);
    for e in 0..x.len() {
        let rot = if e < half {
            -orig[e + half]
        } else {
            orig[e - half]
        };
        x[e] = orig[e] * cos[e] + rot * sin[e];
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// `monotone_quantiles`: the median is free, the others fan out by cumulated softplus widths.
fn monotone_quantiles(raw: &[f32], levels: &[f64], width_factor: f32, out: &mut [f32]) {
    let m = levels.iter().position(|&l| l == 0.5).unwrap();
    let inc = |i: usize| {
        let w = if i > m {
            levels[i] - levels[i - 1]
        } else {
            levels[i + 1] - levels[i]
        } as f32;
        width_factor * w * softplus(raw[i])
    };
    let med = raw[m];
    out[m] = med;
    let mut up = 0f32;
    for (i, o) in out.iter_mut().enumerate().skip(m + 1) {
        up += inc(i);
        *o = med + up;
    }
    let mut down = 0f32;
    for (i, o) in out[..m].iter_mut().enumerate().rev() {
        down += inc(i);
        *o = med - down;
    }
    out.iter_mut()
        .for_each(|v| *v = v.clamp(-f32::MAX, f32::MAX));
}

/// `clamp_nonfinite`: +/-inf becomes the nearest finite value of the step (0 if none); NaN stays.
pub fn clamp_nonfinite(q: &mut [f32]) {
    let hi = q
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let lo = q
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::INFINITY, f32::min);
    for v in q.iter_mut() {
        if *v == f32::INFINITY {
            *v = hi;
        } else if *v == f32::NEG_INFINITY {
            *v = lo;
        }
        if v.is_infinite() {
            *v = 0.0;
        }
    }
}

/// `compute_causal_statistics` + `causal_patch_loc_scale`: per-patch `(loc, scale)` from every
/// observed value up to the end of the patch, accumulated in f64 as in Python.
pub fn causal_patch_stats(
    x: &[f32],
    m: &[bool],
    ps: usize,
    min_scale: f64,
) -> (Vec<f32>, Vec<f32>) {
    let (mut cw, mut cx, mut m2, mut prev) = (0f64, 0f64, 0f64, 0f64);
    let mut loc = Vec::with_capacity(x.len() / ps);
    let mut scale = Vec::with_capacity(x.len() / ps);
    for t in 0..x.len() {
        let w = if m[t] { 1.0 } else { 0.0 };
        let xv = x[t] as f64;
        cw += w;
        cx += w * xv;
        let den = cw.max(1.0);
        let mean = cx / den;
        m2 += (xv - prev) * (xv - mean) * w;
        prev = mean;
        if (t + 1) % ps == 0 {
            let var = m2 / (den - 1.0).max(1.0);
            loc.push(mean as f32);
            scale.push(var.sqrt().max(min_scale) as f32);
        }
    }
    (loc, scale)
}

/// `TimestralModel.scale`: causal per-patch stats and `asinh((x - loc) / scale)` on observed steps.
fn causal_scaling(x: &[f32], m: &[bool], c: &Config) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let ps = c.patch_size;
    let (loc, scale) = causal_patch_stats(x, m, ps, c.minimum_scale);
    let scaled = (0..x.len())
        .map(|t| {
            if !m[t] {
                return 0.0;
            }
            let z = (x[t] - loc[t / ps]) / scale[t / ps].max(f32::EPSILON);
            if c.use_asinh {
                z.asinh()
            } else {
                z
            }
        })
        .collect();
    (loc, scale, scaled)
}

/// `_known_cov_scaling` + `instance_norm_stats`: whole-row stats; 0/1 indicator rows keep
/// `loc=0, scale=1` and skip the asinh squash.
fn known_scaling(x: &[f32], m: &[bool], c: &Config, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let cnt = m.iter().filter(|&&b| b).count();
    let nf = (cnt as f32).max(1.0);
    let loc = (x
        .iter()
        .zip(m)
        .filter(|p| *p.1)
        .map(|(&v, _)| v as f64)
        .sum::<f64>() as f32)
        / nf;
    let var = (x
        .iter()
        .zip(m)
        .map(|(&v, &o)| if o { ((v - loc) as f64).powi(2) } else { 0.0 })
        .sum::<f64>() as f32)
        / nf;
    let mut scale = var.sqrt().max(c.minimum_scale as f32);
    let binary = cnt > 0 && x.iter().zip(m).all(|(&v, &o)| v == 0.0 || v == 1.0 || !o);
    let loc = if binary { 0.0 } else { loc };
    if binary {
        scale = 1.0;
    }
    let scaled = (0..x.len())
        .map(|t| {
            if !m[t] {
                return 0.0;
            }
            let z = (x[t] - loc) / scale.max(f32::EPSILON);
            if c.use_asinh && !binary {
                z.asinh()
            } else {
                z
            }
        })
        .collect();
    (vec![loc; n], vec![scale; n], scaled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_matches_naive() {
        let (m, k, n) = (5, 7, 3);
        let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.37).sin()).collect();
        let w: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.11).cos()).collect();
        let b = [1.0, -2.0, 0.5];
        let mut out = vec![0f32; m * n];
        linear(&x, m, k, &w, n, Some(&b), &mut out, false);
        for i in 0..m {
            for j in 0..n {
                let want: f32 = (0..k).map(|t| x[i * k + t] * w[j * k + t]).sum::<f32>() + b[j];
                assert!((out[i * n + j] - want).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn quantiles_are_monotone_and_finite() {
        let levels = [0.1, 0.5, 0.9];
        let mut out = [0f32; 3];
        monotone_quantiles(&[-3.0, 1.0, 50.0], &levels, 3.0, &mut out);
        assert!(out[0] < out[1] && out[1] < out[2]);
        let mut q = [f32::NEG_INFINITY, 1.0, f32::INFINITY];
        clamp_nonfinite(&mut q);
        assert_eq!(q, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn causal_stats_ignore_missing() {
        let (loc, scale) =
            causal_patch_stats(&[1.0, 99.0, 3.0, 5.0], &[true, false, true, true], 2, 1e-6);
        assert_eq!(loc, vec![1.0, 3.0]);
        assert!((scale[1] - 2.0).abs() < 1e-6);
    }
}
