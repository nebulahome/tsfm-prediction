//! The request queue and its single worker: one model in memory, one forecast at a time.
//!
//! Scheduling: a waiting request for the model already loaded goes first, so requests for the
//! same model run back to back; otherwise the oldest request goes and its model is swapped in.
//! Starvation bound: the oldest request is served next once `MAX_BYPASS` later requests have
//! been served before it, or once it has waited `MAX_BYPASS_WAIT`.

use std::{
    collections::VecDeque,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{Arc, Condvar, Mutex, MutexGuard},
    thread,
    time::{Duration, Instant},
};

use tokio::sync::oneshot;

use crate::{
    api::{ApiError, Quality},
    forecast::Input,
};

pub const MAX_BYPASS: u32 = 4;
pub const MAX_BYPASS_WAIT: Duration = Duration::from_secs(30);

/// A loaded forecaster: quantiles `(variates, horizon, 21)` for one input.
pub trait Model: Send + Sync {
    fn forecast(&self, input: &Input) -> Result<Vec<f32>, String>;
}

/// Fetches (if needed) and loads the model of a quality level.
pub trait Loader: Send + Sync + 'static {
    fn load(&self, q: Quality) -> Result<Box<dyn Model>, ApiError>;
}

pub struct Done {
    pub forecast: Vec<f32>,
    pub queued_ms: u64,
    pub inference_ms: u64,
}

struct Job {
    id: u64,
    quality: Quality,
    input: Input,
    enqueued: Instant,
    bypassed: u32,
    reply: oneshot::Sender<Result<Done, ApiError>>,
}

struct State {
    jobs: VecDeque<Job>,
    next_id: u64,
    loaded: Option<Quality>,
    running: bool,
    last_used: Instant,
    shutdown: bool,
}

struct Shared {
    st: Mutex<State>,
    cv: Condvar,
    queue_size: usize,
    idle: Duration,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.st.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[derive(Clone)]
pub struct Scheduler(Arc<Shared>);

pub struct Status {
    pub loaded: Option<Quality>,
    pub unload_in: Option<Duration>,
    pub queue_depth: usize,
    pub running: bool,
}

impl Scheduler {
    /// Start the worker thread. `pool` runs the forecasts (its size is the intra-op thread count).
    pub fn start(
        loader: Arc<dyn Loader>,
        pool: rayon::ThreadPool,
        queue_size: usize,
        idle: Duration,
    ) -> Self {
        let st = State {
            jobs: VecDeque::new(),
            next_id: 0,
            loaded: None,
            running: false,
            last_used: Instant::now(),
            shutdown: false,
        };
        let shared = Arc::new(Shared {
            st: Mutex::new(st),
            cv: Condvar::new(),
            queue_size,
            idle,
        });
        let s = shared.clone();
        thread::Builder::new()
            .name("worker".into())
            .spawn(move || worker(s, loader, pool))
            .expect("spawn worker");
        Self(shared)
    }

    /// Queue a job, or refuse at once when the queue is full.
    pub fn submit(&self, quality: Quality, input: Input) -> Result<Ticket, ApiError> {
        let (reply, rx) = oneshot::channel();
        let mut st = self.0.lock();
        if st.shutdown {
            return Err(ApiError::shutting_down());
        }
        if st.jobs.len() >= self.0.queue_size {
            return Err(ApiError::queue_full());
        }
        let id = st.next_id;
        st.next_id += 1;
        st.jobs.push_back(Job {
            id,
            quality,
            input,
            enqueued: Instant::now(),
            bypassed: 0,
            reply,
        });
        self.0.cv.notify_all();
        Ok(Ticket {
            id,
            rx,
            sched: self.clone(),
        })
    }

    pub fn status(&self) -> Status {
        let st = self.0.lock();
        let unload_in = st.loaded.map(|_| {
            if st.running {
                self.0.idle
            } else {
                self.0.idle.saturating_sub(st.last_used.elapsed())
            }
        });
        Status {
            loaded: st.loaded,
            unload_in,
            queue_depth: st.jobs.len(),
            running: st.running,
        }
    }

    /// Refuse new jobs, fail the queued ones, let the running one finish.
    pub fn shutdown(&self) {
        let mut st = self.0.lock();
        st.shutdown = true;
        for j in st.jobs.drain(..) {
            let _ = j.reply.send(Err(ApiError::shutting_down()));
        }
        self.0.cv.notify_all();
    }
}

/// A queued job. Dropping it (client gone, timeout) takes the job out of the queue.
pub struct Ticket {
    id: u64,
    rx: oneshot::Receiver<Result<Done, ApiError>>,
    sched: Scheduler,
}

impl Ticket {
    pub async fn wait(&mut self, timeout: Duration) -> Result<Done, ApiError> {
        match tokio::time::timeout(timeout, &mut self.rx).await {
            Ok(Ok(r)) => r,
            Ok(Err(_)) => Err(ApiError::internal("the worker dropped the job")),
            Err(_) => Err(ApiError::timeout()),
        }
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        let id = self.id;
        self.sched.0.lock().jobs.retain(|j| j.id != id);
    }
}

/// Resident set size of this process, from `/proc`.
pub fn rss_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let kb: u64 = s
        .lines()
        .find(|l| l.starts_with("VmRSS:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    Some(kb * 1024)
}

/// Hand freed heap pages back to the OS; glibc keeps them in its arenas otherwise.
fn release_memory() {
    #[cfg(target_env = "gnu")]
    {
        extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        // SAFETY: a plain glibc call with no pointers involved.
        unsafe { malloc_trim(0) };
    }
}

/// Serve blocks of 4 MB and more (activations) with their own mapping, unmapped on free.
/// glibc otherwise raises this threshold as big blocks are freed and keeps them in arenas,
/// which malloc_trim did not fully return (30 MB kept after a long-context forecast).
/// Smaller blocks (gemm's per-call packing buffers) stay on the heap, where reuse is fast.
fn tune_allocator() {
    #[cfg(target_env = "gnu")]
    {
        extern "C" {
            fn mallopt(param: i32, value: i32) -> i32;
        }
        const M_MMAP_THRESHOLD: i32 = -3;
        // SAFETY: a plain glibc call with no pointers involved.
        unsafe { mallopt(M_MMAP_THRESHOLD, 4 << 20) };
    }
}

/// Index of the next job to run, given the model in memory.
fn pick(jobs: &mut VecDeque<Job>, loaded: Option<Quality>, now: Instant) -> Option<usize> {
    let front = jobs.front()?;
    let starving =
        front.bypassed >= MAX_BYPASS || now.duration_since(front.enqueued) >= MAX_BYPASS_WAIT;
    let i = match loaded {
        Some(q) if !starving => jobs.iter().position(|j| j.quality == q).unwrap_or(0),
        _ => 0,
    };
    jobs.iter_mut().take(i).for_each(|j| j.bypassed += 1);
    Some(i)
}

fn worker(sh: Arc<Shared>, loader: Arc<dyn Loader>, pool: rayon::ThreadPool) {
    tune_allocator();
    let mut model: Option<(Quality, Box<dyn Model>)> = None;
    loop {
        let job = {
            let mut st = sh.lock();
            loop {
                if st.shutdown {
                    return;
                }
                if let Some(i) = pick(&mut st.jobs, model.as_ref().map(|m| m.0), Instant::now()) {
                    st.running = true;
                    break st.jobs.remove(i).unwrap();
                }
                if model.is_none() {
                    st = sh.cv.wait(st).unwrap_or_else(|e| e.into_inner());
                    continue;
                }
                let idle_for = st.last_used.elapsed();
                if idle_for >= sh.idle {
                    model = None; // munmap: the weights go back to the OS here
                    st.loaded = None;
                    release_memory();
                    eprintln!("model unloaded after {}s idle", idle_for.as_secs());
                    continue;
                }
                st = sh
                    .cv
                    .wait_timeout(st, sh.idle - idle_for)
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
        };

        let result = (|| {
            if model.as_ref().map(|m| m.0) != Some(job.quality) {
                model = None; // one model in memory at a time: drop the old one before loading
                sh.lock().loaded = None;
                let t = Instant::now();
                model = Some((job.quality, loader.load(job.quality)?));
                sh.lock().loaded = Some(job.quality);
                eprintln!(
                    "loaded {} in {} ms",
                    job.quality.model_name(),
                    t.elapsed().as_millis()
                );
            }
            let m = &model.as_ref().unwrap().1;
            let start = Instant::now();
            let out = pool.install(|| catch_unwind(AssertUnwindSafe(|| m.forecast(&job.input))));
            let forecast = match out {
                Ok(r) => r.map_err(ApiError::forecast_failed)?,
                Err(_) => return Err(ApiError::internal("the forecast panicked")),
            };
            Ok(Done {
                forecast,
                queued_ms: (start - job.enqueued).as_millis() as u64,
                inference_ms: start.elapsed().as_millis() as u64,
            })
        })();
        let _ = job.reply.send(result);
        release_memory();
        let mut st = sh.lock();
        st.running = false;
        st.last_used = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    /// Records loads, unloads and runs; each forecast blocks until the test lets it go.
    struct Fake {
        log: Arc<Mutex<Vec<String>>>,
        gate: Arc<Mutex<mpsc::Receiver<()>>>,
    }
    struct FakeModel {
        q: Quality,
        log: Arc<Mutex<Vec<String>>>,
        gate: Arc<Mutex<mpsc::Receiver<()>>>,
    }
    impl Loader for Fake {
        fn load(&self, q: Quality) -> Result<Box<dyn Model>, ApiError> {
            self.log.lock().unwrap().push(format!("load {q:?}"));
            Ok(Box::new(FakeModel {
                q,
                log: self.log.clone(),
                gate: self.gate.clone(),
            }))
        }
    }
    impl Model for FakeModel {
        fn forecast(&self, input: &Input) -> Result<Vec<f32>, String> {
            self.gate.lock().unwrap().recv().unwrap();
            self.log
                .lock()
                .unwrap()
                .push(format!("run {}", input.horizon));
            Ok(vec![0.0; input.context.len() * input.horizon * 21])
        }
    }
    impl Drop for FakeModel {
        fn drop(&mut self) {
            self.log
                .lock()
                .unwrap()
                .push(format!("unload {:?}", self.q));
        }
    }

    fn setup(
        queue: usize,
        idle: Duration,
    ) -> (Scheduler, Arc<Mutex<Vec<String>>>, mpsc::Sender<()>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let (tx, rx) = mpsc::channel();
        let loader = Arc::new(Fake {
            log: log.clone(),
            gate: Arc::new(Mutex::new(rx)),
        });
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        (Scheduler::start(loader, pool, queue, idle), log, tx)
    }

    /// `horizon` doubles as a job tag in the log.
    fn input(tag: usize) -> Input {
        Input {
            context: vec![vec![1.0; 8]],
            horizon: tag,
            ..Default::default()
        }
    }

    fn wait_for(log: &Arc<Mutex<Vec<String>>>, n: usize) {
        let t = Instant::now();
        while log.lock().unwrap().len() < n {
            assert!(
                t.elapsed() < Duration::from_secs(5),
                "timed out: {:?}",
                log.lock().unwrap()
            );
            thread::sleep(Duration::from_millis(5));
        }
    }

    async fn run_all(tickets: Vec<Ticket>) {
        for mut t in tickets {
            t.wait(Duration::from_secs(5)).await.unwrap();
        }
    }

    #[tokio::test]
    async fn same_model_requests_run_back_to_back() {
        let (s, log, go) = setup(16, Duration::from_secs(60));
        let first = s.submit(Quality::Low, input(1)).unwrap();
        wait_for(&log, 1); // low loaded, job 1 blocked in the model
        let rest: Vec<Ticket> = [
            (Quality::High, 2),
            (Quality::Low, 3),
            (Quality::High, 4),
            (Quality::Low, 5),
        ]
        .into_iter()
        .map(|(q, tag)| s.submit(q, input(tag)).unwrap())
        .collect();
        for _ in 0..5 {
            go.send(()).unwrap();
        }
        run_all(std::iter::once(first).chain(rest).collect()).await;
        let got = log.lock().unwrap().clone();
        assert_eq!(
            got,
            [
                "load Low",
                "run 1",
                "run 3",
                "run 5",
                "unload Low",
                "load High",
                "run 2",
                "run 4"
            ]
        );
    }

    #[tokio::test]
    async fn oldest_request_is_not_bypassed_forever() {
        let (s, log, go) = setup(16, Duration::from_secs(60));
        let first = s.submit(Quality::Low, input(1)).unwrap();
        wait_for(&log, 1);
        let mut tickets = vec![first, s.submit(Quality::High, input(100)).unwrap()];
        for tag in 2..=7 {
            tickets.push(s.submit(Quality::Low, input(tag)).unwrap());
        }
        for _ in 0..8 {
            go.send(()).unwrap();
        }
        run_all(tickets).await;
        let got = log.lock().unwrap().clone();
        // Job 100 waits behind exactly MAX_BYPASS later low-quality jobs, then runs.
        assert_eq!(
            got[..8],
            [
                "load Low",
                "run 1",
                "run 2",
                "run 3",
                "run 4",
                "run 5",
                "unload Low",
                "load High"
            ]
        );
        assert_eq!(got[8], "run 100");
    }

    #[test]
    fn pick_prefers_loaded_model_then_fifo() {
        let mk = |q, enq| {
            let (reply, _) = oneshot::channel();
            Job {
                id: 0,
                quality: q,
                input: Input::default(),
                enqueued: enq,
                bypassed: 0,
                reply,
            }
        };
        let now = Instant::now();
        let mut jobs: VecDeque<Job> = [mk(Quality::High, now), mk(Quality::Low, now)].into();
        assert_eq!(pick(&mut jobs, None, now), Some(0));
        assert_eq!(pick(&mut jobs, Some(Quality::Low), now), Some(1));
        assert_eq!(jobs[0].bypassed, 1);
        assert_eq!(pick(&mut jobs, Some(Quality::Medium), now), Some(0));
        // Waiting past the time bound wins over the loaded model.
        jobs[0].enqueued = now - MAX_BYPASS_WAIT;
        assert_eq!(pick(&mut jobs, Some(Quality::Low), now), Some(0));
        assert_eq!(pick(&mut VecDeque::new(), None, now), None);
    }

    #[tokio::test]
    async fn model_is_unloaded_after_idle_period() {
        let (s, log, go) = setup(4, Duration::from_millis(150));
        go.send(()).unwrap();
        let mut t = s.submit(Quality::Medium, input(1)).unwrap();
        t.wait(Duration::from_secs(5)).await.unwrap();
        assert_eq!(s.status().loaded, Some(Quality::Medium));
        assert!(s.status().unload_in.unwrap() <= Duration::from_millis(150));
        wait_for(&log, 3);
        assert_eq!(log.lock().unwrap()[2], "unload Medium");
        assert_eq!(s.status().loaded, None);
    }

    #[tokio::test]
    async fn full_queue_refuses_at_once_and_dropped_jobs_never_run() {
        let (s, log, go) = setup(2, Duration::from_secs(60));
        let first = s.submit(Quality::Low, input(1)).unwrap();
        wait_for(&log, 1); // running, so the queue is empty again
        let gone = s.submit(Quality::Low, input(2)).unwrap();
        let kept = s.submit(Quality::Low, input(3)).unwrap();
        assert_eq!(
            s.submit(Quality::Low, input(4)).err().unwrap().code,
            "queue_full"
        );
        assert_eq!(s.status().queue_depth, 2);
        drop(gone); // the client went away
        assert_eq!(s.status().queue_depth, 1);
        go.send(()).unwrap();
        go.send(()).unwrap();
        run_all(vec![first, kept]).await;
        assert_eq!(*log.lock().unwrap(), ["load Low", "run 1", "run 3"]);
    }

    #[tokio::test]
    async fn timeout_and_shutdown_answer_with_their_codes() {
        let (s, log, go) = setup(4, Duration::from_secs(60));
        let mut first = s.submit(Quality::Low, input(1)).unwrap();
        wait_for(&log, 1);
        let mut queued = s.submit(Quality::Low, input(2)).unwrap();
        assert_eq!(
            queued
                .wait(Duration::from_millis(20))
                .await
                .err()
                .unwrap()
                .code,
            "timeout"
        );
        s.shutdown();
        assert_eq!(
            s.submit(Quality::Low, input(3)).err().unwrap().code,
            "shutting_down"
        );
        assert_eq!(
            queued
                .wait(Duration::from_secs(5))
                .await
                .err()
                .unwrap()
                .code,
            "shutting_down"
        );
        go.send(()).unwrap();
        first.wait(Duration::from_secs(5)).await.unwrap(); // the running job still finishes
    }
}
