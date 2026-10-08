//! srt2lang — translate English SRT subtitles into another language via an
//! OpenAI-compatible local LLM endpoint.

mod config;
mod lang;
mod srt;
mod translate;

use config::Config;
use lang::Lang;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use translate::Item;

const USAGE: &str = "\
srt2lang — translate English SRT subtitles into another language

USAGE:
    srt2lang [OPTIONS] <INPUT.srt>

ARGS:
    <INPUT.srt>            Source subtitle file (UTF-8)

OPTIONS:
    -l, --lang <CODE>      Target language, ISO 639-1 code (fr, bg, ru, …)
                           [default: from config, then bg]
    -o, --output <FILE>    Output file [default: <input>.<lang>.srt]
    -c, --config <FILE>    JSON config file [default: ./srt2lang.json, then
                           ~/.config/srt2lang/config.json]
    -m, --model <NAME>     Override model from config
    -e, --endpoint <URL>   Override endpoint from config
    -n, --batch-size <N>   Override cues per request from config
        --dry-run          Parse only: show cue count, do not translate
    -h, --help             Print this help
    -V, --version          Print version

CONFIG (all fields optional):
    { \"endpoint\": \"http://127.0.0.1:8001/v1/chat/completions\",
      \"api_key\": \"\", \"model\": \"gemma-3-4b-it\", \"temperature\": 0.2,
      \"max_tokens\": 2048, \"timeout_secs\": 180, \"retries\": 2,
      \"batch_size\": 8, \"max_batch_chars\": 1500,
      \"disable_thinking\": true, \"lang\": \"bg\", \"system_prompt\": null }
";

#[derive(Debug, Default)]
struct Args {
    input: Option<String>,
    output: Option<String>,
    config: Option<String>,
    model: Option<String>,
    endpoint: Option<String>,
    batch_size: Option<usize>,
    lang: Option<String>,
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
            "-l" | "--lang" => out.lang = Some(value_for("--lang")?),
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

/// `movie.srt` + `fr` → `movie.fr.srt`. The language keeps the output name
/// unambiguous once the same source can be translated several ways.
fn default_output_path(input: &Path, lang: &str) -> PathBuf {
    let stem = input.file_stem().map(|s| s.to_string_lossy().into_owned());
    let dir = input.parent().unwrap_or_else(|| Path::new(""));
    match stem {
        Some(stem) => dir.join(format!("{stem}.{lang}.srt")),
        None => dir.join(format!("output.{lang}.srt")),
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
    if let Some(code) = args.lang {
        cfg.lang = code;
    }
    // Resolve before any file or network work: a bad code should cost nothing.
    let target = Lang::resolve(&cfg.lang)?;

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
        "translating {} cue(s) via {} (model '{}', target {} / {})",
        items.len(),
        cfg.endpoint,
        cfg.model,
        target.code,
        target.name
    );

    let (translated, report) = translate::translate_items(&cfg, &items, &target);

    let mut failed = 0usize;
    for (item, text) in items.iter().zip(&translated) {
        match text {
            Some(text) => parsed.cues[item.index].text = text.split('\n').map(str::to_string).collect(),
            None => {
                failed += 1;
                eprintln!("warning: cue {} left in English", item.index + 1);
            }
        }
    }

    // Cues whose text came back byte-identical to the source: the model
    // "succeeded" but produced no translation at all.
    let unchanged: Vec<usize> = items
        .iter()
        .zip(translated.iter())
        .filter(|(item, t)| t.as_deref() == Some(item.text.as_str()))
        .map(|(item, _)| item.index + 1)
        .collect();

    // The Report keys degradations by 0-based file position (the same space
    // as `Item::index`), but cue numbers shown to the user are 1-based.
    let line_issues = cue_numbers(report.line_issues());
    let markup_issues = cue_numbers(report.markup_issues());
    let english_issues = cue_numbers(report.untranslated_issues());

    let output_path = match args.output {
        Some(p) => PathBuf::from(p),
        None => default_output_path(&input_path, &target.code),
    };
    std::fs::write(&output_path, srt::serialize(&parsed))
        .map_err(|e| format!("cannot write '{}': {e}", output_path.display()))?;

    // Quality summary: failures alone understate the picture, since a cue can
    // be "translated" yet still differ from what was asked for.
    eprintln!("--- quality ---");
    eprintln!("  translated          : {}", items.len() - failed);
    if !unchanged.is_empty() {
        eprintln!(
            "  unchanged (no-op)   : {}  -> {}",
            unchanged.len(),
            list_cues(&unchanged)
        );
    }
    if !english_issues.is_empty() {
        eprintln!(
            "  kept in English     : {}  -> {}",
            english_issues.len(),
            list_cues(&english_issues)
        );
    }
    if !line_issues.is_empty() {
        eprintln!(
            "  line structure drift: {}  -> {}",
            line_issues.len(),
            list_cues(&line_issues)
        );
    }
    if !markup_issues.is_empty() {
        eprintln!(
            "  markup repaired     : {}  -> {}",
            markup_issues.len(),
            list_cues(&markup_issues)
        );
    }
    if failed > 0 {
        eprintln!("  left in English     : {failed}");
    }
    if unchanged.is_empty()
        && english_issues.is_empty()
        && line_issues.is_empty()
        && markup_issues.is_empty()
        && failed == 0
    {
        eprintln!("  no issues detected");
    }

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

/// Convert 0-based file positions (as `Report` stores them) into the 1-based
/// cue numbers the SRT displays. Forgetting this is an easy way to print a
/// summary that is silently off by one — see the test below.
fn cue_numbers(positions: Vec<usize>) -> Vec<usize> {
    positions.into_iter().map(|i| i + 1).collect()
}

/// Render cue numbers (1-based) compactly: `14, 15, 20 +3`.
fn list_cues(cues: &[usize]) -> String {
    const SHOWN: usize = 8;
    if cues.len() <= SHOWN {
        cues.iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    } else {
        let head: Vec<String> = cues[..SHOWN].iter().map(|c| c.to_string()).collect();
        format!("{} +{}", head.join(", "), cues.len() - SHOWN)
    }
}

fn main() -> ExitCode {
    let action = parse_args(std::env::args().skip(1));

    match action {
        Ok(CliAction::Help) => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(CliAction::Version) => {
            println!("srt2lang {}", env!("CARGO_PKG_VERSION"));
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
    fn report_positions_are_printed_as_one_based_cue_numbers() {
        // `Report` stores 0-based file positions; cue 1 is position 0.
        // (translate tests pin `Report` to 0-based; this pins the conversion.)
        assert_eq!(cue_numbers(vec![0, 4, 119]), vec![1, 5, 120]);
        assert_eq!(cue_numbers(vec![]), Vec::<usize>::new());
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
            "http://localhost:9999/v1/chat/completions", "-n", "5", "-l", "fr", "--dry-run",
            "in.srt",
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
        assert_eq!(args.lang.as_deref(), Some("fr"));
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
    fn default_output_name_carries_the_target_language() {
        assert_eq!(
            default_output_path(Path::new("movie.srt"), "bg"),
            PathBuf::from("movie.bg.srt")
        );
        assert_eq!(
            default_output_path(Path::new("/tmp/movie.en.srt"), "fr"),
            PathBuf::from("/tmp/movie.en.fr.srt")
        );
        assert_eq!(
            default_output_path(Path::new("movie.srt"), "ru"),
            PathBuf::from("movie.ru.srt")
        );
    }
}
