//! HTTP contract: `POST /v1/forecast`, `GET /v1/status`, `GET /healthz`. Everything a client
//! sends is checked here before it reaches the queue.

use std::{fmt, sync::Arc, time::Duration};

use axum::{
    extract::{rejection::JsonRejection, DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Deserializer, Serialize};

use crate::{forecast::Input, hub, queue::Scheduler};

/// The 21 quantile levels of every Timestral checkpoint.
pub const LEVELS: [f64; 21] = [
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
    0.85, 0.9, 0.95, 0.99,
];
pub const MAX_CONTEXT: usize = 32768;
pub const MAX_HORIZON: usize = 4096;
/// Target variates plus covariate rows in one request.
pub const MAX_ROWS: usize = 32;
pub const MAX_BODY_BYTES: usize = 16 << 20;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Quality {
    Low,
    Medium,
    High,
    Xhigh,
}

impl Quality {
    pub const ALL: [Quality; 4] = [Quality::Low, Quality::Medium, Quality::High, Quality::Xhigh];

    pub fn model_name(self) -> &'static str {
        match self {
            Quality::Low => "Timestral-5M",
            Quality::Medium => "Timestral-28M",
            Quality::High => "Timestral-123M",
            Quality::Xhigh => "Timestral-209M",
        }
    }

    pub fn repo(self) -> String {
        format!("mistralai/{}", self.model_name())
    }

    pub fn parse(s: &str) -> Option<Self> {
        serde_json::from_value(serde_json::Value::String(s.to_ascii_lowercase())).ok()
    }
}

impl fmt::Display for Quality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format!("{self:?}").to_ascii_lowercase())
    }
}

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }
    pub fn invalid(m: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_input", m)
    }
    pub fn queue_full() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "queue_full",
            "the queue is full, retry later",
        )
    }
    pub fn shutting_down() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "shutting_down",
            "the server is shutting down",
        )
    }
    pub fn timeout() -> Self {
        Self::new(
            StatusCode::GATEWAY_TIMEOUT,
            "timeout",
            "no answer within the request timeout",
        )
    }
    pub fn download_failed(m: String) -> Self {
        Self::new(StatusCode::BAD_GATEWAY, "model_download_failed", m)
    }
    pub fn load_failed(m: String) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "model_load_failed", m)
    }
    pub fn forecast_failed(m: String) -> Self {
        Self::new(StatusCode::UNPROCESSABLE_ENTITY, "forecast_failed", m)
    }
    pub fn internal(m: &str) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", m)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({ "code": self.code, "message": self.message })),
        )
            .into_response()
    }
}

/// Rows of numbers, `null` = missing (stored as NaN). A flat array is one row.
#[derive(Debug, Default)]
pub struct Rows(pub Vec<Vec<f32>>);

impl<'de> Deserialize<'de> for Rows {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Cell {
            Value(Option<f64>),
            Row(Vec<Option<f64>>),
        }
        let cell = |v: Option<f64>| -> Result<f32, D::Error> {
            match v {
                None => Ok(f32::NAN),
                Some(x) if (x as f32).is_finite() => Ok(x as f32),
                Some(x) => Err(serde::de::Error::custom(format!(
                    "{x} is outside the float32 range"
                ))),
            }
        };
        let cells = Vec::<Cell>::deserialize(d)?;
        if cells.is_empty() {
            return Ok(Rows(Vec::new()));
        }
        if cells.iter().all(|c| matches!(c, Cell::Value(_))) {
            let row = cells
                .into_iter()
                .map(|c| {
                    if let Cell::Value(v) = c {
                        cell(v)
                    } else {
                        unreachable!()
                    }
                })
                .collect::<Result<_, _>>()?;
            return Ok(Rows(vec![row]));
        }
        cells
            .into_iter()
            .map(|c| match c {
                Cell::Row(r) => r.into_iter().map(cell).collect(),
                Cell::Value(_) => Err(serde::de::Error::custom(
                    "expected an array of numbers or an array of arrays, not a mix",
                )),
            })
            .collect::<Result<_, _>>()
            .map(Rows)
    }
}

fn default_quality() -> Quality {
    Quality::Medium
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ForecastRequest {
    #[serde(default = "default_quality")]
    pub quality: Quality,
    pub context: Rows,
    pub horizon: usize,
    #[serde(default)]
    pub quantiles: Option<Vec<f64>>,
    #[serde(default)]
    pub past_covariates: Option<Rows>,
    #[serde(default)]
    pub known_covariates: Option<Rows>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ForecastResponse {
    model: String,
    quantiles: Vec<f64>,
    /// variate -> quantile -> step
    forecast: Vec<Vec<Vec<f32>>>,
    queued_ms: u64,
    inference_ms: u64,
}

/// Check a request against the contract; returns the job and the indices of the asked quantiles.
pub fn validate(
    req: ForecastRequest,
    max_quality: Quality,
) -> Result<(Quality, Input, Vec<usize>), ApiError> {
    if req.quality > max_quality {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "quality_unavailable",
            format!(
                "quality {} is above this server's maximum, {max_quality}",
                req.quality
            ),
        ));
    }
    let context = req.context.0;
    let t = context.first().map_or(0, Vec::len);
    if t == 0 || context.iter().any(|r| r.len() != t) {
        return Err(ApiError::invalid(
            "context must be one or more non-empty variates of equal length",
        ));
    }
    if t > MAX_CONTEXT {
        return Err(ApiError::invalid(format!(
            "context is {t} steps long, the maximum is {MAX_CONTEXT}"
        )));
    }
    if !context.iter().flatten().any(|x| !x.is_nan()) {
        return Err(ApiError::invalid("context has no values, only nulls"));
    }
    let h = req.horizon;
    if !(1..=MAX_HORIZON).contains(&h) {
        return Err(ApiError::invalid(format!(
            "horizon must be between 1 and {MAX_HORIZON}"
        )));
    }
    let asked = req.quantiles.unwrap_or_else(|| vec![0.1, 0.5, 0.9]);
    if asked.is_empty() {
        return Err(ApiError::invalid("quantiles must not be empty"));
    }
    let idx = asked
        .iter()
        .map(|q| {
            LEVELS
                .iter()
                .position(|l| (l - q).abs() < 1e-9)
                .ok_or_else(|| ApiError::invalid(format!("quantile {q} is not one of {LEVELS:?}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let past = req.past_covariates.unwrap_or_default().0;
    let known = req.known_covariates.unwrap_or_default().0;
    if past.iter().any(|r| r.len() != t) {
        return Err(ApiError::invalid(format!(
            "each pastCovariates row must be {t} steps long, like the context"
        )));
    }
    if known.iter().any(|r| r.len() != t + h) {
        return Err(ApiError::invalid(format!(
            "each knownCovariates row must be {} steps long, context plus horizon",
            t + h
        )));
    }
    if context.len() + past.len() + known.len() > MAX_ROWS {
        return Err(ApiError::invalid(format!(
            "at most {MAX_ROWS} variates and covariates in total"
        )));
    }
    Ok((
        req.quality,
        Input {
            context,
            horizon: h,
            past,
            known,
        },
        idx,
    ))
}

#[derive(Clone)]
pub struct App {
    pub sched: Scheduler,
    pub hub: Arc<hub::Hub>,
    pub max_quality: Quality,
    pub timeout: Duration,
}

pub fn router(app: App) -> Router {
    Router::new()
        .route("/v1/forecast", post(forecast))
        .route("/v1/status", get(status))
        .route(
            "/healthz",
            get(|| async { Json(serde_json::json!({ "status": "ok" })) }),
        )
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(app)
}

async fn forecast(
    State(app): State<App>,
    body: Result<Json<ForecastRequest>, JsonRejection>,
) -> Result<Json<ForecastResponse>, ApiError> {
    let Json(req) = body.map_err(|e| ApiError::new(e.status(), "invalid_input", e.body_text()))?;
    let (quality, input, idx) = validate(req, app.max_quality)?;
    let (v, h) = (input.context.len(), input.horizon);
    // Dropping the ticket (client gone) takes the job out of the queue.
    let mut ticket = app.sched.submit(quality, input)?;
    let done = ticket.wait(app.timeout).await?;
    let nq = LEVELS.len();
    let forecast = (0..v)
        .map(|i| {
            idx.iter()
                .map(|&j| {
                    (0..h)
                        .map(|k| done.forecast[(i * h + k) * nq + j])
                        .collect()
                })
                .collect()
        })
        .collect();
    Ok(Json(ForecastResponse {
        model: quality.repo(),
        quantiles: idx.iter().map(|&i| LEVELS[i]).collect(),
        forecast,
        queued_ms: done.queued_ms,
        inference_ms: done.inference_ms,
    }))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StatusResponse {
    loaded_model: Option<String>,
    loaded_quality: Option<Quality>,
    unload_in_seconds: Option<u64>,
    queue_depth: usize,
    running: bool,
    rss_bytes: Option<u64>,
    models_on_disk: Vec<Quality>,
    max_quality: Quality,
}

async fn status(State(app): State<App>) -> Json<StatusResponse> {
    let s = app.sched.status();
    Json(StatusResponse {
        loaded_model: s.loaded.map(Quality::repo),
        loaded_quality: s.loaded,
        unload_in_seconds: s.unload_in.map(|d| d.as_secs_f64().ceil() as u64),
        queue_depth: s.queue_depth,
        running: s.running,
        rss_bytes: crate::queue::rss_bytes(),
        models_on_disk: Quality::ALL
            .into_iter()
            .filter(|&q| app.hub.present(q))
            .collect(),
        max_quality: app.max_quality,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue::{Loader, Model};
    use axum::body::{to_bytes, Body};
    use serde_json::{json, Value};
    use tower::ServiceExt;

    fn check(body: Value, max: Quality) -> Result<(Quality, Input, Vec<usize>), ApiError> {
        validate(
            serde_json::from_value(body).map_err(|e| ApiError::invalid(e.to_string()))?,
            max,
        )
    }

    fn code(body: Value) -> &'static str {
        check(body, Quality::Xhigh)
            .expect_err("should be refused")
            .code
    }

    #[test]
    fn accepts_flat_and_multivariate_context_with_nulls() {
        let (q, inp, idx) = check(
            json!({ "context": [1, null, 3], "horizon": 4 }),
            Quality::Xhigh,
        )
        .unwrap();
        assert_eq!(
            (q, inp.context.len(), idx),
            (Quality::Medium, 1, vec![2, 10, 18])
        );
        assert!(inp.context[0][1].is_nan());
        let (_, inp, idx) = check(json!({ "quality": "low", "context": [[1, 2], [3, null]], "horizon": 2, "quantiles": [0.5, 0.99],
            "pastCovariates": [5, 6], "knownCovariates": [[0, 1, 1, 0]] }), Quality::Xhigh).unwrap();
        assert_eq!(
            (inp.context.len(), inp.past.len(), inp.known[0].len(), idx),
            (2, 1, 4, vec![10, 20])
        );
    }

    #[test]
    fn refuses_bad_shapes_lengths_and_values() {
        let long = vec![1.0; MAX_CONTEXT + 1];
        for body in [
            json!({ "context": [], "horizon": 1 }),
            json!({ "context": [[1, 2], [3]], "horizon": 1 }),
            json!({ "context": [null, null], "horizon": 1 }),
            json!({ "context": long, "horizon": 1 }),
            json!({ "context": [1, 2], "horizon": 0 }),
            json!({ "context": [1, 2], "horizon": MAX_HORIZON + 1 }),
            json!({ "context": [1, 2], "horizon": 1, "quantiles": [0.42] }),
            json!({ "context": [1, 2], "horizon": 1, "quantiles": [] }),
            json!({ "context": [1, 2], "horizon": 1, "pastCovariates": [[1, 2, 3]] }),
            json!({ "context": [1, 2], "horizon": 1, "knownCovariates": [[1, 2]] }),
            json!({ "context": vec![vec![1.0]; MAX_ROWS + 1], "horizon": 1 }),
        ] {
            assert_eq!(code(body), "invalid_input");
        }
        for body in [
            json!({ "context": [1, [2]], "horizon": 1 }),
            json!({ "context": [1e300], "horizon": 1 }),
            json!({ "context": ["1"], "horizon": 1 }),
            json!({ "context": [1], "horizon": 1, "extra": true }),
            json!({ "context": [1], "horizon": -1 }),
            json!({ "context": [1], "horizon": 1, "quality": "ultra" }),
        ] {
            assert!(serde_json::from_value::<ForecastRequest>(body).is_err());
        }
    }

    #[test]
    fn quality_above_the_cap_is_unavailable() {
        let e = check(
            json!({ "quality": "high", "context": [1], "horizon": 1 }),
            Quality::Medium,
        )
        .err()
        .unwrap();
        assert_eq!(
            (e.code, e.status),
            ("quality_unavailable", StatusCode::FORBIDDEN)
        );
        assert!(check(
            json!({ "quality": "medium", "context": [1], "horizon": 1 }),
            Quality::Medium
        )
        .is_ok());
        assert_eq!(Quality::parse("XHIGH"), Some(Quality::Xhigh));
    }

    struct Echo;
    struct EchoModel;
    impl Loader for Echo {
        fn load(&self, _: Quality) -> Result<Box<dyn Model>, ApiError> {
            Ok(Box::new(EchoModel))
        }
    }
    impl Model for EchoModel {
        /// Quantile j of step k of variate i is `100 i + k + j / 100`.
        fn forecast(&self, inp: &Input) -> Result<Vec<f32>, String> {
            let h = inp.horizon;
            Ok((0..inp.context.len() * h * 21)
                .map(|n| (100 * (n / (h * 21)) + (n / 21) % h) as f32 + (n % 21) as f32 / 100.0)
                .collect())
        }
    }

    async fn call(app: &Router, req: axum::http::Request<Body>) -> (StatusCode, Value) {
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        (
            status,
            serde_json::from_slice(&to_bytes(resp.into_body(), usize::MAX).await.unwrap()).unwrap(),
        )
    }

    fn post(body: String) -> axum::http::Request<Body> {
        axum::http::Request::post("/v1/forecast")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap()
    }

    #[tokio::test]
    async fn http_contract() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let sched = Scheduler::start(Arc::new(Echo), pool, 4, Duration::from_secs(60));
        let hub = Arc::new(hub::Hub {
            cache_dir: "/nonexistent".into(),
            endpoint: String::new(),
            token: None,
        });
        let app = router(App {
            sched,
            hub,
            max_quality: Quality::High,
            timeout: Duration::from_secs(5),
        });

        let (s, v) = call(
            &app,
            post(
                json!({ "context": [[1, 2], [3, 4]], "horizon": 2, "quantiles": [0.5, 0.01] })
                    .to_string(),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["model"], "mistralai/Timestral-28M");
        assert_eq!(
            v["forecast"],
            json!([[[0.1, 1.1], [0.0, 1.0]], [[100.1, 101.1], [100.0, 101.0]]])
        );
        assert!(v["queuedMs"].is_u64() && v["inferenceMs"].is_u64());

        let (s, v) = call(&app, post("{not json".into())).await;
        assert_eq!(
            (s, v["code"].as_str()),
            (StatusCode::BAD_REQUEST, Some("invalid_input"))
        );
        let (s, v) = call(&app, post(format!("[{}]", "1,".repeat(MAX_BODY_BYTES / 2)))).await;
        assert_eq!(
            (s, v["code"].as_str()),
            (StatusCode::PAYLOAD_TOO_LARGE, Some("invalid_input"))
        );
        let (s, v) = call(
            &app,
            post(json!({ "quality": "xhigh", "context": [1], "horizon": 1 }).to_string()),
        )
        .await;
        assert_eq!(
            (s, v["code"].as_str()),
            (StatusCode::FORBIDDEN, Some("quality_unavailable"))
        );

        let (s, v) = call(
            &app,
            axum::http::Request::get("/v1/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(
            (
                v["loadedQuality"].as_str(),
                v["queueDepth"].as_u64(),
                v["maxQuality"].as_str()
            ),
            (Some("medium"), Some(0), Some("high"))
        );
        let (s, v) = call(
            &app,
            axum::http::Request::get("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!((s, v["status"].as_str()), (StatusCode::OK, Some("ok")));
    }
}
