//! stdin/stdout loop: reader thread, brain worker thread, deadline fallback.

use std::io::{self, BufRead, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::brain::Brain;
use crate::io::proto::{Action, Hello, Obs, Ready, ServerMsg};
use crate::Shared;

pub const BOT_NAME: &str = "automata";

static LOG: AtomicBool = AtomicBool::new(false);

pub fn log_enabled() -> bool {
    LOG.load(Ordering::Relaxed)
}

/// Writes one line to stderr when MECH_LOG=1.
pub fn log(msg: &str) {
    if log_enabled() {
        let _ = writeln!(io::stderr(), "[automata] {msg}");
    }
}

/// Time the Brain may still spend on the current observation.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    deadline: Option<Instant>,
}

impl Budget {
    pub fn unlimited() -> Budget {
        Budget { deadline: None }
    }
    pub fn until(deadline: Instant) -> Budget {
        Budget { deadline: Some(deadline) }
    }
    pub fn from_now_ms(ms: f64) -> Budget {
        Budget::until(Instant::now() + Duration::from_secs_f64(ms.max(0.0) / 1000.0))
    }
    pub fn is_unlimited(&self) -> bool {
        self.deadline.is_none()
    }
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
    pub fn remaining(&self) -> Duration {
        match self.deadline {
            None => Duration::MAX,
            Some(d) => d.saturating_duration_since(Instant::now()),
        }
    }
    /// f64::INFINITY when unlimited.
    pub fn remaining_ms(&self) -> f64 {
        match self.deadline {
            None => f64::INFINITY,
            Some(_) => self.remaining().as_secs_f64() * 1000.0,
        }
    }
    pub fn expired(&self) -> bool {
        matches!(self.deadline, Some(d) if Instant::now() >= d)
    }
}

/// Answer deadline used by the IO thread, measured from reading the observation.
const IO_FRACTION: f64 = 0.70;
/// Deadline handed to the Brain.
const BRAIN_FRACTION: f64 = 0.60;

struct Job {
    obs: Box<Obs>,
    budget: Budget,
}

enum WorkerMsg {
    Act(i64, Action),
    Panicked(i64),
}

/// Reads stdin lines. Lines that arrive together (already buffered) are sent as one batch.
fn spawn_reader() -> Receiver<Vec<(Instant, String)>> {
    let (tx, rx) = mpsc::channel();
    thread::Builder::new()
        .name("stdin".into())
        .spawn(move || {
            let mut rd = io::BufReader::with_capacity(1 << 20, io::stdin());
            let mut buf = Vec::with_capacity(1 << 16);
            let mut batch = Vec::new();
            let mut eof = false;
            while !eof {
                loop {
                    buf.clear();
                    match rd.read_until(b'\n', &mut buf) {
                        Ok(0) | Err(_) => {
                            eof = true;
                            break;
                        }
                        Ok(_) => {
                            let s = String::from_utf8_lossy(&buf).trim().to_string();
                            if !s.is_empty() {
                                batch.push((Instant::now(), s));
                            }
                        }
                    }
                    if rd.buffer().is_empty() {
                        break;
                    }
                }
                if !batch.is_empty() && tx.send(std::mem::take(&mut batch)).is_err() {
                    break;
                }
            }
        })
        .expect("spawn reader");
    rx
}

fn new_brain(hello: &Hello, shared: &Arc<Shared>) -> Option<Brain> {
    match catch_unwind(AssertUnwindSafe(|| Brain::new(hello, shared.clone()))) {
        Ok(b) => Some(b),
        Err(_) => {
            log("Brain::new panicked");
            None
        }
    }
}

fn spawn_worker(hello: Arc<Hello>, shared: Arc<Shared>, brain: Option<Brain>) -> (Sender<Job>, Receiver<WorkerMsg>) {
    let (jtx, jrx) = mpsc::channel::<Job>();
    let (dtx, drx) = mpsc::channel::<WorkerMsg>();
    thread::Builder::new()
        .name("brain".into())
        .stack_size(64 << 20)
        .spawn(move || {
            let mut brain = brain;
            while let Ok(mut job) = jrx.recv() {
                while let Ok(j) = jrx.try_recv() {
                    job = j;
                }
                let tick = job.obs.tick;
                if brain.is_none() {
                    brain = new_brain(&hello, &shared);
                }
                let Some(b) = brain.as_mut() else {
                    let _ = dtx.send(WorkerMsg::Panicked(tick));
                    continue;
                };
                match catch_unwind(AssertUnwindSafe(|| b.act(&job.obs, &job.budget))) {
                    Ok(mut a) => {
                        a.tick = tick;
                        if dtx.send(WorkerMsg::Act(tick, a)).is_err() {
                            break;
                        }
                    }
                    Err(_) => {
                        log(&format!("Brain::act panicked at tick {tick}"));
                        let _ = dtx.send(WorkerMsg::Panicked(tick));
                        brain = new_brain(&hello, &shared);
                    }
                }
            }
        })
        .expect("spawn worker");
    (jtx, drx)
}

struct Output {
    max_line: usize,
    last_move: Option<[f64; 2]>,
    last_boost: Option<bool>,
    last_aim: Option<[f64; 2]>,
}

impl Output {
    fn write_line<T: serde::Serialize>(&self, msg: &T) {
        if let Ok(s) = serde_json::to_string(msg) {
            let out = io::stdout();
            let mut lock = out.lock();
            let _ = lock.write_all(s.as_bytes());
            let _ = lock.write_all(b"\n");
            let _ = lock.flush();
        }
    }

    fn send_action(&mut self, mut a: Action) {
        self.last_move = a.move_;
        self.last_boost = a.boost;
        if a.aim.is_some() {
            self.last_aim = a.aim;
        }
        let mut s = serde_json::to_string(&a).unwrap_or_default();
        if s.len() > self.max_line && a.debug.is_some() {
            a.debug = None;
            s = serde_json::to_string(&a).unwrap_or_default();
        }
        if s.is_empty() {
            return;
        }
        let out = io::stdout();
        let mut lock = out.lock();
        let _ = lock.write_all(s.as_bytes());
        let _ = lock.write_all(b"\n");
        let _ = lock.flush();
    }

    fn fallback(&self, tick: i64) -> Action {
        Action { tick, move_: self.last_move, boost: self.last_boost, aim: self.last_aim, ..Default::default() }
    }
}

fn silence_panics() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if log_enabled() {
            default(info);
        }
    }));
}

/// Peeks the tick of a line that did not parse as an observation.
fn peek_obs_tick(line: &str) -> Option<i64> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    if v.get("type")?.as_str()? != "obs" {
        return None;
    }
    v.get("tick")?.as_f64().map(|t| t as i64)
}

fn parse_hello(line: &str) -> Option<Hello> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    if v.get("type").and_then(|t| t.as_str()) != Some("hello") {
        return None;
    }
    match serde_json::from_value::<Hello>(v.clone()) {
        Ok(h) => Some(h),
        Err(e) => {
            log(&format!("hello parse error: {e}"));
            let mut h = Hello::default();
            if let Some(d) = v.pointer("/map/dir").and_then(|d| d.as_str()) {
                h.map.dir = d.to_string();
            }
            if let Some(s) = v.get("slot").and_then(|s| s.as_f64()) {
                h.slot = s as i64;
            }
            if let Some(c) = v.get("config") {
                if let Ok(c) = serde_json::from_value(c.clone()) {
                    h.config = c;
                }
            }
            Some(h)
        }
    }
}

/// The process main loop. Returns the exit code.
pub fn run() -> i32 {
    LOG.store(std::env::var("MECH_LOG").map(|v| v == "1").unwrap_or(false), Ordering::Relaxed);
    silence_panics();
    let start = Instant::now();
    let rx = spawn_reader();

    let mut pending: Vec<(Instant, String)> = Vec::new();
    let hello = 'h: loop {
        let Ok(batch) = rx.recv() else { return 0 };
        let mut it = batch.into_iter();
        while let Some((_, line)) = it.next() {
            if let Some(h) = parse_hello(&line) {
                pending.extend(it);
                break 'h h;
            }
            log("ignoring line before hello");
        }
    };
    let shared = Arc::new(Shared::load(Path::new(&hello.map.dir), &hello.config));
    let loaded = start.elapsed();
    let hello = Arc::new(hello);
    let brain = new_brain(&hello, &shared);
    let tick_ms = if hello.config.bot.tick_budget_ms > 0.0 { hello.config.bot.tick_budget_ms } else { 50.0 };
    let io_wait = Duration::from_secs_f64(tick_ms * IO_FRACTION / 1000.0);
    let brain_wait = Duration::from_secs_f64(tick_ms * BRAIN_FRACTION / 1000.0);
    let max_line = if hello.config.bot.max_line_bytes > 0.0 { hello.config.bot.max_line_bytes as usize } else { 65536 };
    let (jtx, drx) = spawn_worker(hello.clone(), shared, brain);

    let mut out = Output { max_line, last_move: None, last_boost: None, last_aim: None };
    out.write_line(&Ready::new(BOT_NAME));
    log(&format!("ready after {:?} (shared loaded after {:?})", start.elapsed(), loaded));

    let (mut answered, mut fallbacks, mut panics) = (0u64, 0u64, 0u64);
    loop {
        let mut lines = std::mem::take(&mut pending);
        if lines.is_empty() {
            let Ok(first) = rx.recv() else { break };
            lines = first;
        }
        while let Ok(x) = rx.try_recv() {
            lines.extend(x);
        }
        // Newest observation wins; "end" anywhere ends the process.
        let mut newest: Option<(Instant, Result<Box<Obs>, i64>)> = None;
        let mut ended = false;
        for (t, line) in lines {
            match serde_json::from_str::<ServerMsg>(&line) {
                Ok(ServerMsg::Obs(o)) => newest = Some((t, Ok(o))),
                Ok(ServerMsg::End(e)) => {
                    log(&format!("end: place {} kills {}", e.place, e.kills));
                    ended = true;
                }
                Ok(_) => {}
                Err(err) => {
                    if let Some(tick) = peek_obs_tick(&line) {
                        log(&format!("obs parse error at tick {tick}: {err}"));
                        newest = Some((t, Err(tick)));
                    } else if line.contains("\"end\"") {
                        ended = true;
                    }
                }
            }
        }
        if ended {
            break;
        }
        let Some((t_read, parsed)) = newest else { continue };
        let obs = match parsed {
            Ok(o) => o,
            Err(tick) => {
                out.send_action(out.fallback(tick));
                fallbacks += 1;
                continue;
            }
        };
        let tick = obs.tick;
        let deadline = t_read + io_wait;
        if jtx.send(Job { obs, budget: Budget::until(t_read + brain_wait) }).is_err() {
            out.send_action(out.fallback(tick));
            fallbacks += 1;
            continue;
        }
        let mut action = None;
        loop {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            match drx.recv_timeout(deadline - now) {
                Ok(WorkerMsg::Act(t, a)) if t == tick => {
                    action = Some(a);
                    break;
                }
                Ok(WorkerMsg::Panicked(t)) if t == tick => {
                    panics += 1;
                    break;
                }
                Ok(_) => continue,
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        match action {
            Some(a) => {
                answered += 1;
                out.send_action(a)
            }
            None => {
                fallbacks += 1;
                log(&format!("fallback at tick {tick}"));
                out.send_action(out.fallback(tick))
            }
        }
    }
    log(&format!("exit: answered {answered}, fallbacks {fallbacks}, panics {panics}"));
    let _ = io::stdout().flush();
    0
}
