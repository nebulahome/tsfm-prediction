//! tsfm-prediction: a small HTTP service that forecasts time series with Timestral.

#![allow(clippy::too_many_arguments)] // the numeric kernels take shapes and strides explicitly

mod api;
mod forecast;
mod hub;
mod model;
mod queue;

use std::{
    env,
    io::{Read, Write},
    net::TcpStream,
    process::ExitCode,
    str::FromStr,
    sync::Arc,
    time::Duration,
};

use api::{ApiError, Quality, LEVELS};
use queue::{Loader, Model, Scheduler};

struct TimestralLoader(Arc<hub::Hub>);

impl Loader for TimestralLoader {
    fn load(&self, q: Quality) -> Result<Box<dyn Model>, ApiError> {
        let dir = self.0.ensure(q).map_err(ApiError::download_failed)?;
        let m = model::Timestral::load(&dir).map_err(ApiError::load_failed)?;
        if m.cfg.quantile_knots != LEVELS {
            return Err(ApiError::load_failed(format!(
                "{} has unexpected quantile levels",
                q.model_name()
            )));
        }
        Ok(Box::new(m))
    }
}

impl Model for model::Timestral {
    fn forecast(&self, input: &forecast::Input) -> Result<Vec<f32>, String> {
        forecast::forecast(self, input)
    }
}

fn var<T: FromStr>(name: &str, default: T) -> Result<T, String> {
    match env::var(name) {
        Ok(v) if !v.is_empty() => v.parse().map_err(|_| format!("{name}={v} is not valid")),
        _ => Ok(default),
    }
}

/// `tsfm-prediction healthcheck`: the image has no curl, so the binary checks itself.
fn healthcheck() -> ExitCode {
    let port: u16 = var("TSFM_PORT", 8080).unwrap_or(8080);
    let ok = (|| -> std::io::Result<bool> {
        let mut s =
            TcpStream::connect_timeout(&([127, 0, 0, 1], port).into(), Duration::from_secs(3))?;
        s.set_read_timeout(Some(Duration::from_secs(3)))?;
        s.write_all(b"GET /healthz HTTP/1.0\r\n\r\n")?;
        let mut buf = [0u8; 16];
        s.read_exact(&mut buf)?;
        Ok(buf.starts_with(b"HTTP/1.1 200") || buf.starts_with(b"HTTP/1.0 200"))
    })();
    if ok.unwrap_or(false) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn main() -> ExitCode {
    if env::args().nth(1).as_deref() == Some("healthcheck") {
        return healthcheck();
    }
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let port: u16 = var("TSFM_PORT", 8080)?;
    let idle = Duration::from_secs(var("TSFM_IDLE_SECS", 300)?);
    let queue_size: usize = var("TSFM_QUEUE_SIZE", 16)?;
    let threads: usize = var("TSFM_THREADS", 0)?;
    // Default: up to 4 threads. Measured on a 24-thread host, more threads made typical
    // requests slower (idle workers spinning next to busy ones) and long ones barely faster.
    let threads = if threads == 0 {
        std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(4)
    } else {
        threads
    };
    let timeout = Duration::from_secs(var("TSFM_TIMEOUT_SECS", 600)?);
    let max = var("TSFM_MAX_QUALITY", "xhigh".to_string())?;
    let max_quality = Quality::parse(&max).ok_or(format!(
        "TSFM_MAX_QUALITY={max} is not one of low, medium, high, xhigh"
    ))?;
    let hub = Arc::new(hub::Hub {
        cache_dir: var("TSFM_CACHE_DIR", "/models".to_string())?.into(),
        endpoint: var("HF_ENDPOINT", "https://huggingface.co".to_string())?,
        token: env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()),
    });

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("infer-{i}"))
        .build()
        .map_err(|e| e.to_string())?;
    let sched = Scheduler::start(
        Arc::new(TimestralLoader(hub.clone())),
        pool,
        queue_size,
        idle,
    );
    let app = api::router(api::App {
        sched: sched.clone(),
        hub: hub.clone(),
        max_quality,
        timeout,
    });

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await.map_err(|e| format!("port {port}: {e}"))?;
        eprintln!("listening on :{port}, cache {}, {threads} threads, max quality {max_quality}, idle unload {}s", hub.cache_dir.display(), idle.as_secs());
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                use tokio::signal::unix::{signal, SignalKind};
                let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
                tokio::select! {
                    _ = term.recv() => {}
                    _ = tokio::signal::ctrl_c() => {}
                }
                eprintln!("shutting down");
                sched.shutdown();
            })
            .await
            .map_err(|e| e.to_string())
    })
}
