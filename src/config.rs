use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Runtime configuration, loaded from a JSON file and overridable from the CLI.
///
/// Every field has a sensible default, so a missing (or partial) config file
/// is fine — only the values present in the file override the defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Full URL of an OpenAI-compatible chat completions endpoint.
    pub endpoint: String,
    /// Optional bearer token; sent as `Authorization: Bearer ...` when non-empty.
    pub api_key: String,
    /// Model name passed to the endpoint (ignored by servers with a single model).
    pub model: String,
    pub temperature: f64,
    /// Cap for generated tokens per request. Must leave room for the JSON reply.
    pub max_tokens: u32,
    /// Per-request timeout in seconds.
    pub timeout_secs: u64,
    /// Extra attempts after the first failure (per batch).
    pub retries: u32,
    /// Maximum number of subtitle cues sent in one request.
    pub batch_size: usize,
    /// Maximum total characters of source text per request.
    pub max_batch_chars: usize,
    /// Ask llama.cpp-style servers to skip the reasoning phase
    /// (`chat_template_kwargs.enable_thinking = false`).
    pub disable_thinking: bool,
    /// Optional replacement for the built-in system prompt.
    pub system_prompt: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:8001/v1/chat/completions".to_string(),
            api_key: String::new(),
            model: "qwen".to_string(),
            temperature: 0.2,
            max_tokens: 2048,
            timeout_secs: 180,
            retries: 2,
            batch_size: 8,
            max_batch_chars: 1500,
            disable_thinking: true,
            system_prompt: None,
        }
    }
}

impl Config {
    /// Load configuration from `explicit` when given, otherwise from the first
    /// existing file among `./srt2bg.json` and `~/.config/srt2bg/config.json`,
    /// otherwise built-in defaults.
    pub fn load(explicit: Option<&str>) -> Result<Self, String> {
        if let Some(path) = explicit {
            return Self::from_file(path);
        }

        let mut candidates: Vec<PathBuf> = vec![PathBuf::from("srt2bg.json")];
        if let Some(home) = std::env::var_os("HOME") {
            candidates.push(PathBuf::from(home).join(".config/srt2bg/config.json"));
        }

        for path in candidates {
            if path.is_file() {
                let path = path.display().to_string();
                return Self::from_file(&path);
            }
        }

        Ok(Self::default())
    }

    fn from_file(path: &str) -> Result<Self, String> {
        let raw = fs::read_to_string(path)
            .map_err(|e| format!("cannot read config '{path}': {e}"))?;
        let mut cfg: Self = serde_json::from_str(&raw)
            .map_err(|e| format!("invalid JSON in config '{path}': {e}"))?;

        // Guard against degenerate values that would break batching.
        cfg.batch_size = cfg.batch_size.max(1);
        cfg.max_batch_chars = cfg.max_batch_chars.max(50);
        if cfg.endpoint.trim().is_empty() {
            return Err(format!("config '{path}': \"endpoint\" must not be empty"));
        }
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let cfg = Config::default();
        assert!(cfg.batch_size >= 1);
        assert!(cfg.endpoint.starts_with("http"));
    }

    #[test]
    fn partial_config_overrides_defaults() {
        let cfg: Config = serde_json::from_str(r#"{"model":"llama3","batch_size":3}"#).unwrap();
        assert_eq!(cfg.model, "llama3");
        assert_eq!(cfg.batch_size, 3);
        // untouched fields fall back to defaults
        assert_eq!(cfg.temperature, Config::default().temperature);
        assert_eq!(cfg.endpoint, Config::default().endpoint);
    }

    #[test]
    fn degenerate_batch_size_is_clamped() {
        let dir = std::env::temp_dir().join("srt2bg_cfg_test.json");
        std::fs::write(&dir, r#"{"batch_size":0}"#).unwrap();
        let cfg = Config::load(Some(&dir.display().to_string())).unwrap();
        assert_eq!(cfg.batch_size, 1);
        let _ = std::fs::remove_file(&dir);
    }
}
