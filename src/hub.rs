//! Weights from the Hugging Face Hub into the cache directory, one folder per model.

use std::{fs, io, path::PathBuf, time::Duration};

use crate::api::Quality;

const FILES: [&str; 2] = ["model.safetensors", "config.json"];

pub struct Hub {
    pub cache_dir: PathBuf,
    pub endpoint: String,
    pub token: Option<String>,
}

impl Hub {
    pub fn dir(&self, q: Quality) -> PathBuf {
        self.cache_dir.join(q.model_name())
    }

    pub fn present(&self, q: Quality) -> bool {
        FILES.iter().all(|f| self.dir(q).join(f).is_file())
    }

    /// The model folder, downloading what is missing. Each file goes to a temp name and is
    /// renamed when complete, so a folder never holds a partial file under its real name.
    pub fn ensure(&self, q: Quality) -> Result<PathBuf, String> {
        let dir = self.dir(q);
        if self.present(q) {
            return Ok(dir);
        }
        fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(30))
            .timeout_read(Duration::from_secs(120))
            .build();
        for f in FILES {
            let path = dir.join(f);
            if path.is_file() {
                continue;
            }
            let url = format!(
                "{}/{}/resolve/main/{f}",
                self.endpoint.trim_end_matches('/'),
                q.repo()
            );
            eprintln!("downloading {url}");
            let mut req = agent.get(&url);
            if let Some(t) = &self.token {
                req = req.set("Authorization", &format!("Bearer {t}"));
            }
            let resp = req.call().map_err(|e| format!("{url}: {e}"))?;
            let expected: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());
            let tmp = dir.join(format!(".{f}.part"));
            let write = || -> io::Result<()> {
                let mut out = fs::File::create(&tmp)?;
                let got = io::copy(&mut resp.into_reader(), &mut out)?;
                if expected.is_some_and(|n| n != got) {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "download cut short",
                    ));
                }
                out.sync_all()?;
                fs::rename(&tmp, &path)
            };
            write().map_err(|e| {
                let _ = fs::remove_file(&tmp);
                format!("{url}: {e}")
            })?;
        }
        Ok(dir)
    }
}
