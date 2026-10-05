//! srt2bg — translate English SRT subtitles to Bulgarian via an
//! OpenAI-compatible local LLM endpoint.

mod config;
mod srt;
mod translate;

use config::Config;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use translate::Item;

const USAGE: &str = "\
srt2bg — translate English SRT subtitles to Bulgarian

USAGE:
    srt2bg [OPTIONS] <INPUT.srt>

ARGS:
    <INPUT.srt>            Source subtitle file (UTF-8)

OPTIONS:
    -o, --output <FILE>    Output file [default: <input>.bg.srt]
    -c, --config <FILE>    JSON config file [default: ./srt2bg.json, then
                           ~/.config/srt2bg/config.json]
    -m, --model <NAME>     Override model from config
    -e, --endpoint <URL>   Override endpoint from config
    -n, --batch-size <N>   Override cues per request from config
        --dry-run          Parse only: show cue count, do not translate
    -h, --help             Print this help
    -V, --version          Print version

CONFIG (all fields optional):
    { \"endpoint\": \"http://127.0.0.1:8001/v1/chat/completions\",
      \"api_key\": \"\", \"model\": \"qwen\", \"temperature\": 0.2,
      \"max_tokens\": 2048, \"timeout_secs\": 180, \"retries\": 2,
      \"batch_size\": 8, \"max_batch_chars\": 1500,
      \"disable_thinking\": true, \"system_prompt\": null }
";

#[derive(Debug, Default)]
struct Args {
    input: Option<String>,
    output: Option<String>,
    config: Option<String>,
    model: Option<String>,
    endpoint: Option<String>,
    batch_size: Option<usize>,
    dry_run: bool,
}

#[derive(Debug)]
enum CliAction {
    Run(Args),
    Help,
    Version,
}

fn parse_args<I: Iterator<Item = String>>(mut args: I) -> Result<CliAction, String> {
    let mut out = Args::default();

    while let Some(arg) = args.next() {
        let mut value_for = |flag: &str| {
            args.next().ok_or_else(|| format!("{flag} requires a value"))
        };

        match arg.as_str() {
            "-h" | "--help" => return Ok(CliAction::Help),
            "-V" | "--version" => return Ok(CliAction::Version),
            "-o" | "--output" => out.output = Some(value_for("--output")?),
            "-c" | "--config" => out.config = Some(value_for("--config")?),
            "-m" | "--model" => out.model = Some(value_for("--model")?),
            "-e" | "--endpoint" => out.endpoint = Some(value_for("--endpoint")?),
            "-n" | "--batch-size" => {
                let raw = value_for("--batch-size")?;
                let n: usize = raw
                    .parse()
                    .map_err(|_| format!("invalid --batch-size: {raw}"))?;
                if n == 0 {
                    return Err("--batch-size must be >= 1".to_string());
                }
                out.batch_size = Some(n);
            }
            "--dry-run" => out.dry_run = true,
            other if other.starts_with('-') => {
                return Err(format!("unknown option: {other}\n\n{USAGE}"));
            }
            other => {
                if out.input.is_some() {
                    return Err(format!("unexpected argument: {other}\n\n{USAGE}"));
                }
                out.input = Some(other.to_string());
            }
        }
    }

    if out.input.is_none() {
        return Err(format!("missing input file\n\n{USAGE}"));
    }
    Ok(CliAction::Run(out))
}

fn default_output_path(input: &Path) -> PathBuf {
    let stem = input.file_stem().map(|s| s.to_string_lossy().into_owned());
    let dir = input.parent().unwrap_or_else(|| Path::new(""));
    match stem {
        Some(stem) => dir.join(format!("{stem}.bg.srt")),
        None => dir.join("output.bg.srt"),
    }
}

fn run(args: Args) -> Result<(), String> {
    let input_path = PathBuf::from(args.input.as_ref().unwrap());

    let mut cfg = Config::load(args.config.as_deref())?;
    if let Some(model) = args.model {
        cfg.model = model;
    }
    if let Some(endpoint) = args.endpoint {
        cfg.endpoint = endpoint;
    }
    if let Some(n) = args.batch_size {
        cfg.batch_size = n;
    }

    let raw = std::fs::read_to_string(&input_path)
        .map_err(|e| format!("cannot read '{}': {e}", input_path.display()))?;

    let mut parsed = srt::parse(&raw)?;
    if parsed.skipped > 0 {
        eprintln!(
            "warning: skipped {} block(s) without a timing line",
            parsed.skipped
        );
    }
    eprintln!("parsed {} cue(s) from {}", parsed.cues.len(), input_path.display());

    if args.dry_run {
        return Ok(());
    }

    // Collect the text to translate, remembering which lines belong to which cue.
    let mut items: Vec<Item> = Vec::new();
    for (cue_idx, cue) in parsed.cues.iter().enumerate() {
        if cue.text.iter().all(|l| l.trim().is_empty()) {
            continue;
        }
        items.push(Item {
            index: cue_idx,
            text: cue.text.join("\n"),
        });
    }

    eprintln!(
        "translating {} cue(s) via {} (model '{}')",
        items.len(),
        cfg.endpoint,
        cfg.model
    );

    let translated = translate::translate_items(&cfg, &items);

    let mut failed = 0usize;
    for (item, text) in items.iter().zip(translated) {
        match text {
            Some(text) => parsed.cues[item.index].text = text.split('\n').map(str::to_string).collect(),
            None => {
                failed += 1;
                eprintln!("warning: cue {} left in English", item.index + 1);
            }
        }
    }

    let output_path = match args.output {
        Some(p) => PathBuf::from(p),
        None => default_output_path(&input_path),
    };
    std::fs::write(&output_path, srt::serialize(&parsed))
        .map_err(|e| format!("cannot write '{}': {e}", output_path.display()))?;

    eprintln!(
        "wrote {} ({} cue(s){})",
        output_path.display(),
        parsed.cues.len(),
        if failed > 0 {
            format!(", {failed} untranslated")
        } else {
            String::new()
        }
    );
    Ok(())
}

fn main() -> ExitCode {
    let action = parse_args(std::env::args().skip(1));

    match action {
        Ok(CliAction::Help) => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(CliAction::Version) => {
            println!("srt2bg {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(CliAction::Run(args)) => match run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::FAILURE
            }
        },
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(list: &[&str]) -> Result<CliAction, String> {
        parse_args(list.iter().map(|s| s.to_string()))
    }

    #[test]
    fn minimal_invocation() {
        let CliAction::Run(args) = parse(&["movie.srt"]).unwrap() else {
            panic!("expected Run");
        };
        assert_eq!(args.input.as_deref(), Some("movie.srt"));
        assert!(args.output.is_none());
    }

    #[test]
    fn full_invocation() {
        let CliAction::Run(args) = parse(&[
            "-o", "out.srt", "-c", "cfg.json", "-m", "qwen3", "-e",
            "http://localhost:9999/v1/chat/completions", "-n", "5", "--dry-run", "in.srt",
        ])
        .unwrap() else {
            panic!("expected Run");
        };
        assert_eq!(args.input.as_deref(), Some("in.srt"));
        assert_eq!(args.output.as_deref(), Some("out.srt"));
        assert_eq!(args.config.as_deref(), Some("cfg.json"));
        assert_eq!(args.model.as_deref(), Some("qwen3"));
        assert_eq!(
            args.endpoint.as_deref(),
            Some("http://localhost:9999/v1/chat/completions")
        );
        assert_eq!(args.batch_size, Some(5));
        assert!(args.dry_run);
    }

    #[test]
    fn help_version_and_errors() {
        assert!(matches!(parse(&["--help"]).unwrap(), CliAction::Help));
        assert!(matches!(parse(&["-V"]).unwrap(), CliAction::Version));
        assert!(parse(&[]).is_err()); // no input
        assert!(parse(&["--nope", "a.srt"]).is_err()); // unknown flag
        assert!(parse(&["a.srt", "b.srt"]).is_err()); // two inputs
        assert!(parse(&["-n", "0", "a.srt"]).is_err()); // zero batch size
        assert!(parse(&["-n", "abc", "a.srt"]).is_err()); // non-numeric
        assert!(parse(&["-o"]).is_err()); // missing value
    }

    #[test]
    fn default_output_name() {
        assert_eq!(
            default_output_path(Path::new("movie.srt")),
            PathBuf::from("movie.bg.srt")
        );
        assert_eq!(
            default_output_path(Path::new("/tmp/movie.en.srt")),
            PathBuf::from("/tmp/movie.en.bg.srt")
        );
    }
}
