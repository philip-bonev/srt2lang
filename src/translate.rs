//! Batched translation against an OpenAI-compatible chat completions endpoint.
//!
//! Consecutive cues are grouped into batches and sent as one JSON object
//! `{"1": "...", "2": "..."}`; the model must answer with the same keys and
//! values in the target language. Keys are always the **1-based position within the
//! batch** — never a file-global index — so request and reply can never drift
//! apart. Markup tags are replaced by `<tN>` placeholders before the request
//! and restored afterwards, so preservation does not depend on model
//! obedience. Failures fall back to single-cue requests, and cues that still
//! fail keep their original English text.

use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, HashSet};
use std::time::Duration;

use crate::config::Config;
use crate::lang::Lang;

/// Instructs the model to return a bare JSON object keyed like the input.
/// `{lang}` is substituted with the target language's English name.
const DEFAULT_SYSTEM_PROMPT: &str = "You are a professional subtitle translator. \
You receive a JSON object whose keys are sequential segment numbers starting at 1 and whose values are English subtitle lines (possibly multi-line, separated by \\n). \
Translate every value into natural, colloquial {lang} suitable for subtitles. \
Rules: keep the tone and register of the original; preserve line count within each value; \
copy any placeholder tokens standing for markup verbatim; \
keep proper names, numbers and onomatopoeia; do not renumber the keys; do not add explanations. \
Respond with ONLY a JSON object with exactly the same keys and the {lang} translations as values. \
No markdown, no code fences, no commentary.";

/// Subtitle batches are small; cap endpoint responses well below ureq's
/// generic 10 MiB default to bound memory when an endpoint misbehaves.
const MAX_RESPONSE_BYTES: u64 = 1_048_576;

/// One cue ready for translation: its index within the file and its text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    pub index: usize,
    pub text: String,
}

/// A cue that was translated, but not cleanly.
///
/// These are *not* failures — the text is usable — but they are exactly what
/// a user reviewing a subtitle file cares about, so they are counted and
/// summarised rather than left as scattered stderr lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Degradation {
    /// The model returned fewer (or more) lines than the source cue had.
    Lines { source: usize, result: usize },
    /// The model dropped or duplicated markup placeholders; tags were repaired
    /// where their original position was unambiguous.
    Markup { tags: String },
    /// The model returned the source text back instead of a translation —
    /// an echo, not an answer. See `looks_untranslated`.
    Untranslated,
}

/// Per-run quality report: every degradation, keyed by **0-based file
/// position** — the same space as `Item::index`. Callers printing cue numbers
/// must convert with `+ 1`, since SRT cues are numbered from 1.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Report {
    pub degradations: Vec<(usize, Degradation)>,
}

impl Report {
    fn push(&mut self, index: usize, kind: Degradation) {
        self.degradations.push((index, kind));
    }

    /// Cue indices whose line structure differs from the source.
    pub fn line_issues(&self) -> Vec<usize> {
        self.degradations
            .iter()
            .filter(|(_, d)| matches!(d, Degradation::Lines { .. }))
            .map(|(i, _)| *i)
            .collect()
    }

    /// Cue indices whose markup had to be repaired by guess.
    pub fn markup_issues(&self) -> Vec<usize> {
        self.degradations
            .iter()
            .filter(|(_, d)| matches!(d, Degradation::Markup { .. }))
            .map(|(i, _)| *i)
            .collect()
    }

    /// Cue indices the model answered in the source language rather than the
    /// target one.
    pub fn untranslated_issues(&self) -> Vec<usize> {
        self.degradations
            .iter()
            .filter(|(_, d)| matches!(d, Degradation::Untranslated))
            .map(|(i, _)| *i)
            .collect()
    }
}

/// True when `reply` gives no sign of having been translated at all.
///
/// This check exists because a small model sometimes echoes its input
/// verbatim. Such a reply passes *every* structural test — keys intact,
/// placeholders present, line counts matching — so without a language check
/// it is accepted silently and reported as a success while the cue stays in
/// English.
///
/// Single-word sources are excluded: `- Ooh.`, `- Cory?`, `Angelyne` are
/// correctly left alone, and flagging them would only buy useless retries.
fn looks_untranslated(source: &str, reply: &str, target: &Lang) -> bool {
    if ascii_word_runs(source) < 2 {
        return false;
    }
    // A non-Latin target is checked against its own script: precise, and
    // impossible to satisfy with English.
    match target.reply_in_target_script(reply) {
        Some(in_script) => !in_script,
        // Latin (or a script CLDR does not cover) has no characters of its
        // own to look for, so ask instead whether the reply says anything the
        // source did not.
        None => echoes_source(source, reply),
    }
}

/// Whether `reply` introduces no word the source did not already have — i.e.
/// it is the source, possibly reordered or lightly rephrased, rather than a
/// translation.
///
/// Word comparison rather than string equality, because models echo with
/// small edits (dropped markup, reflowed punctuation) — and because it is the
/// only test available for Latin-script targets, where "this reply contains
/// no Cyrillic" would flag every correct French translation.
///
/// A rendering built entirely from words the source also uses is
/// indistinguishable from an echo without a language detector, so it is
/// reported rather than silently passed.
fn echoes_source(source: &str, reply: &str) -> bool {
    let source_words = content_words(source);
    content_words(reply)
        .into_iter()
        .all(|word| source_words.contains(&word))
}

/// Lowercased alphabetic runs of two or more characters — content words that
/// can be compared across languages.
///
/// `is_alphabetic` rather than ASCII: splitting on non-ASCII would chop
/// `déjà` into `d`/`j` and `émission` into `mission`, so a perfectly good
/// French reply would look like it said nothing new.
fn content_words(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphabetic())
        .filter(|w| w.chars().count() >= 2)
        .map(|w| w.to_lowercase())
        .collect()
}

/// Count ASCII alphabetic runs in an English source. One-letter words matter
/// here: `I am` and `A dog` are ordinary translatable cues, not names.
fn ascii_word_runs(text: &str) -> usize {
    let (protected, _) = protect_markup(text);
    let text = strip_placeholder_residue(&protected);
    text.split(|c: char| !c.is_ascii_alphabetic())
        .filter(|w| !w.is_empty())
        .count()
}

/// Group consecutive items into batches, bounded by both `batch_size` cues
/// and `max_batch_chars` total characters (a single oversized cue forms its
/// own batch). Items whose text is empty/whitespace are skipped entirely.
pub fn build_batches(items: &[Item], batch_size: usize, max_batch_chars: usize) -> Vec<Vec<Item>> {
    let mut batches: Vec<Vec<Item>> = Vec::new();
    let mut current: Vec<Item> = Vec::new();
    let mut chars = 0usize;

    for item in items {
        if item.text.trim().is_empty() {
            continue;
        }
        let len = item.text.chars().count();
        let would_exceed =
            !current.is_empty() && (current.len() >= batch_size || chars + len > max_batch_chars);
        if would_exceed {
            batches.push(std::mem::take(&mut current));
            chars = 0;
        }
        chars += len;
        current.push(item.clone());
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

/// Translate all items, returning a vector aligned with the input:
/// `Some(translated)` on success, `None` when every fallback failed
/// (caller keeps the original text), plus a report of degraded cues.
pub fn translate_items(
    cfg: &Config,
    items: &[Item],
    target: &Lang,
) -> (Vec<Option<String>>, Report) {
    let mut results: Vec<Option<String>> = vec![None; items.len()];
    // `Item::index` is a file-cue position and can be sparse when timing-only
    // cues are skipped. Results, however, are part of this function's API and
    // must stay aligned with the input slice.
    let result_positions: BTreeMap<usize, usize> = items
        .iter()
        .enumerate()
        .map(|(position, item)| (item.index, position))
        .collect();
    let mut report = Report::default();

    let batches = build_batches(items, cfg.batch_size, cfg.max_batch_chars);
    let total_batches = batches.len();

    for (bi, batch) in batches.iter().enumerate() {
        eprintln!(
            "  batch {}/{} ({} cue{})",
            bi + 1,
            total_batches,
            batch.len(),
            if batch.len() == 1 { "" } else { "s" }
        );

        match translate_batch(cfg, batch, target, AcceptPartial::Reject, &mut report) {
            Ok(map) => {
                for item in batch {
                    if let Some(text) = map.get(&item.index) {
                        results[*result_positions
                            .get(&item.index)
                            .expect("batch item came from input slice")] = Some(text.clone());
                    }
                }
            }
            Err(err) => eprintln!("  batch {} failed: {err}", bi + 1),
        }

        // Fallback: retry each cue of this batch on its own.
        for item in batch {
            let result_position = *result_positions
                .get(&item.index)
                .expect("batch item came from input slice");
            if results[result_position].is_some() {
                continue;
            }
            match translate_batch(
                cfg,
                std::slice::from_ref(item),
                target,
                AcceptPartial::Accept,
                &mut report,
            ) {
                Ok(map) => {
                    if let Some(text) = map.get(&item.index) {
                        results[*result_positions
                            .get(&item.index)
                            .expect("batch item came from input slice")] = Some(text.clone());
                    }
                }
                Err(err) => eprintln!(
                    "  cue {} failed after retry, keeping original text: {err}",
                    item.index + 1
                ),
            }
        }
    }

    (results, report)
}

/// Send one batch and parse the reply into `file index -> translation`.
/// Retries the whole request `cfg.retries` extra times on transport errors,
/// non-2xx replies, or unparseable/empty model output.
///
/// `partial` decides what happens when the model drops markup placeholders:
/// `Reject` (batch calls) leaves the cue out so it is retried on its own,
/// `Accept` (the final single-cue fallback) keeps the translation.
fn translate_batch(
    cfg: &Config,
    batch: &[Item],
    target: &Lang,
    partial: AcceptPartial,
    report: &mut Report,
) -> Result<BTreeMap<usize, String>, String> {
    // Hide markup from the model; restored after the reply.
    let protected: Vec<(String, Vec<String>)> = batch
        .iter()
        .map(|item| protect_markup(&item.text))
        .collect();
    let texts: Vec<String> = protected.iter().map(|(t, _)| t.clone()).collect();

    let payload = build_payload(cfg, &texts, target);
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        match post_chat(cfg, &payload) {
            Ok(content) => match parse_translations(&content, batch.len()) {
                Ok(positional) => {
                    let (out, lost) =
                        assemble_reply(positional, batch, &protected, target, partial, report);
                    if partial == AcceptPartial::Reject {
                        for index in lost {
                            eprintln!(
                                "  cue {}: not usable as batch reply (markup/line count/untranslated), retrying alone",
                                index + 1
                            );
                        }
                    }
                    return Ok(out);
                }
                Err(err) if attempt <= cfg.retries => {
                    eprintln!("  retrying batch (attempt {}): {err}", attempt + 1);
                    sleep_backoff(attempt);
                }
                Err(err) => return Err(err),
            },
            Err(err) if attempt <= cfg.retries => {
                eprintln!("  retrying batch (attempt {}): {err}", attempt + 1);
                sleep_backoff(attempt);
            }
            Err(err) => return Err(err),
        }
    }
}

/// Whether a single-cue reply that lost markup is still worth keeping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AcceptPartial {
    /// Drop the cue so it can be retried on its own.
    Reject,
    /// Keep the translation and report the lost tags (final fallback).
    Accept,
}

/// Map a parsed reply back onto the batch: positional key (1..n) → file
/// index, with markup restored. Cues missing from the reply, or (when
/// `AcceptPartial::Reject`) cues whose placeholders the model dropped, are
/// omitted from the result and returned as lost. Cues that survive but are
/// imperfect are recorded in the report.
fn assemble_reply(
    positional: BTreeMap<usize, String>,
    batch: &[Item],
    protected: &[(String, Vec<String>)],
    target: &Lang,
    partial: AcceptPartial,
    report: &mut Report,
) -> (BTreeMap<usize, String>, Vec<usize>) {
    let mut out = BTreeMap::new();
    let mut lost = Vec::new();

    for (pos, text) in &positional {
        let (Some(item), Some((source, tags))) = (batch.get(pos - 1), protected.get(pos - 1))
        else {
            continue;
        };

        let text = normalize_model_text(text);
        let malformed_tags = malformed_placeholders(&text, tags);
        let source_lines = count_lines(&item.text);
        let result_lines = count_lines(&text);
        let merged = source_lines != result_lines;
        let untranslated = looks_untranslated(&item.text, &text, target);
        if (untranslated || !malformed_tags.is_empty() || merged)
            && partial == AcceptPartial::Reject
        {
            lost.push(item.index);
            continue;
        }
        if untranslated {
            report.push(item.index, Degradation::Untranslated);
            eprintln!(
                "  cue {}: reply reads as the English source, not a translation",
                item.index + 1
            );
        }
        if !malformed_tags.is_empty() {
            let (restored, unplaced) = restore_markup(&text, source, tags);
            report.push(
                item.index,
                Degradation::Markup {
                    tags: if unplaced.is_empty() {
                        malformed_tags.join(", ")
                    } else {
                        format!(
                            "{} (unplaced: {})",
                            malformed_tags.join(", "),
                            unplaced.join(", ")
                        )
                    },
                },
            );
            if unplaced.is_empty() {
                eprintln!(
                    "  cue {}: model damaged markup ({}), repaired where possible",
                    item.index + 1,
                    malformed_tags.join(", ")
                );
            } else {
                eprintln!(
                    "  cue {}: model damaged markup and position is unknown for {}, keeping translation",
                    item.index + 1,
                    unplaced.join(", ")
                );
            }
            out.insert(item.index, restored);
            continue;
        }
        if merged {
            report.push(
                item.index,
                Degradation::Lines {
                    source: source_lines,
                    result: result_lines,
                },
            );
            eprintln!(
                "  cue {}: model merged {} line(s) into {}, keeping translation",
                item.index + 1,
                source_lines,
                result_lines
            );
        }
        out.insert(item.index, restore_markup(&text, source, tags).0);
    }

    // Cues the reply never mentioned are lost too.
    for (pos, item) in batch.iter().enumerate() {
        if !positional.contains_key(&(pos + 1)) {
            lost.push(item.index);
        }
    }

    (out, lost)
}

fn sleep_backoff(attempt: u32) {
    std::thread::sleep(Duration::from_millis(300 * u64::from(attempt)));
}

/// Build the chat-completions request body. `texts` are keyed by their
/// 1-based position in the batch.
fn build_payload(cfg: &Config, texts: &[String], target: &Lang) -> Value {
    let mut input = Map::new();
    for (pos, text) in texts.iter().enumerate() {
        input.insert((pos + 1).to_string(), Value::String(text.clone()));
    }

    // A custom prompt may use `{lang}` too; substitute there as well so an
    // override does not silently pin the target language to Bulgarian.
    let system = cfg
        .system_prompt
        .clone()
        .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string())
        .replace("{lang}", target.name);

    let mut body = json!({
        "model": cfg.model,
        "messages": [
            { "role": "system", "content": system },
            { "role": "user", "content": Value::Object(input).to_string() }
        ],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stream": false,
    });

    // Lemonade/llama-server: without this the reasoning model spends every
    // token on `reasoning_content` and returns an empty `content`.
    if cfg.disable_thinking {
        body["chat_template_kwargs"] = json!({ "enable_thinking": false });
    }
    body
}

/// POST to the endpoint and return `choices[0].message.content`.
fn post_chat(cfg: &Config, payload: &Value) -> Result<String, String> {
    let mut request = ureq::post(&cfg.endpoint)
        .header("Content-Type", "application/json")
        .config()
        .timeout_global(Some(Duration::from_secs(cfg.timeout_secs)))
        .build();
    if !cfg.api_key.is_empty() {
        request = request.header("Authorization", format!("Bearer {}", cfg.api_key));
    }

    let mut response = request
        .send_json(payload)
        .map_err(|e| format!("request error: {e}"))?;

    let status = response.status();
    let body = response
        .body_mut()
        .with_config()
        .limit(MAX_RESPONSE_BYTES)
        .read_to_string()
        .map_err(|e| format!("cannot read response body: {e}"))?;

    if !(200..300).contains(&status.as_u16()) {
        return Err(format!("HTTP {status}: {}", truncate(body.trim(), 300)));
    }

    let value: Value =
        serde_json::from_str(&body).map_err(|_| format!("not JSON: {}", truncate(&body, 300)))?;

    let content = value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .map(str::to_string)
        .filter(|c| !c.trim().is_empty())
        .ok_or_else(|| "empty content in model reply (is thinking disabled?)".to_string())?;
    Ok(content)
}

/// Extract translations from raw model output, tolerating markdown fences
/// and surrounding prose. Returns `1-based position -> translation`.
///
/// Candidate objects are found by balanced-brace scanning rather than
/// "first `{` to last `}`": a model that emits two objects, or a `}` inside
/// trailing prose, would otherwise produce an unparseable slice. Findings
/// from every candidate are merged, since a reply may legitimately split
/// across objects.
///
/// Two passes. The second straightens curly quotes used as string
/// delimiters and runs only when the first pass came up short, because that
/// rewrite also touches quotation marks that are legitimate text in French,
/// German, Russian and Bulgarian alike, and must never see a healthy reply.
/// Entries from the first pass win on conflict.
fn parse_translations(content: &str, expected: usize) -> Result<BTreeMap<usize, String>, String> {
    let stripped = strip_fences(content);
    let mut found = collect_json_objects(stripped, expected);

    if found.len() < expected {
        // A curly quote left a string unterminated, so the closing `}` reads
        // as part of that string and the object never balances — extraction
        // finds nothing and the parser is never reached. Straightening the
        // whole reply first is the only way to get a candidate out of it.
        let straightened = straighten_curly_quotes(stripped);
        for (pos, text) in collect_json_objects(&straightened, expected) {
            found.entry(pos).or_insert(text);
        }
    }

    if found.is_empty() {
        return Err(format!(
            "no usable translations in reply: {}",
            truncate(content, 200)
        ));
    }
    Ok(found)
}

/// Parse every balanced object in `text` and harvest the usable cues.
fn collect_json_objects(text: &str, expected: usize) -> BTreeMap<usize, String> {
    let mut found: BTreeMap<usize, String> = BTreeMap::new();

    for candidate in json_object_candidates(text) {
        let Ok(parsed) = serde_json::from_str::<Value>(&repair_json(candidate)) else {
            continue;
        };
        let Some(obj) = parsed.as_object() else {
            continue;
        };

        for (key, value) in obj {
            if let Ok(pos) = key.parse::<usize>()
                && (1..=expected).contains(&pos)
                && let Some(text) = value.as_str()
                && !text.trim().is_empty()
                && !found.contains_key(&pos)
            {
                found.insert(pos, text.to_string());
            }
        }
        if found.len() == expected {
            break; // every cue accounted for
        }
    }
    found
}

/// Balanced `{...}` objects found in `text`.
///
/// Each opening brace is tried independently, so an unbalanced `{` in
/// surrounding prose cannot swallow the real object that follows it.
/// Braces inside string literals are not structural.
fn json_object_candidates(text: &str) -> Vec<&str> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();

    for open in (0..chars.len()).filter(|&i| chars[i] == '{') {
        let mut depth = 0usize;
        let mut in_string = false;
        let mut escaped = false;

        for (i, &ch) in chars.iter().enumerate().skip(open) {
            if in_string {
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    in_string = false;
                }
                continue;
            }
            match ch {
                '"' => in_string = true,
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        out.push(slice_by_chars(text, open, i));
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    out
}

/// Slice `text` from char index `from` through `to` inclusive.
fn slice_by_chars(text: &str, from: usize, to: usize) -> &str {
    let start = text
        .char_indices()
        .nth(from)
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    let end = text
        .char_indices()
        .nth(to + 1)
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    &text[start..end]
}

/// Every repair small models need before `serde_json` can read their reply.
///
/// Passes are independent, and each one must leave already-valid text
/// untouched — a repair that corrupts good JSON is worse than the failure it
/// fixes, because it converts a clean error into a silently wrong result.
fn repair_json(json: &str) -> String {
    repair_invalid_escapes(&repair_doubled_key_quotes(
        &escape_control_chars_in_strings(json),
    ))
}

/// Replace curly double quotes with straight ones.
///
/// Used only as a second pass over an already-failing reply: `“ ”` are
/// legitimate quotation marks in French, German, Russian and Bulgarian alike,
/// so rewriting them on a healthy reply would terminate strings early and
/// corrupt the values.
fn straighten_curly_quotes(text: &str) -> String {
    text.replace(['\u{201C}', '\u{201D}'], "\"")
}

/// Rewrite any backslash inside a string that does not introduce a legal
/// escape as a newline escape (`\n`).
///
/// The system prompt tells the model that multi-line cue values are joined by
/// `\n`, so an illegal escape is that separator written wrong — observed as
/// `мнения\за`, a backslash before Cyrillic text where a newline belonged.
/// serde_json rejects the whole reply over it, and one bad escape costs every
/// cue in the batch: that is how a single cue went untranslated through three
/// batch attempts *and* its single-cue retry.
///
/// Emitting a literal backslash instead would parse, but leave a visible `\`
/// on screen and still count as line drift. Interpreting it as a newline
/// matches the source structure in the observed cases, and any misjudgement
/// still surfaces through `assemble_reply`'s line-count check rather than
/// passing silently.
///
/// A legal escape is `\` followed by `" \ / b f n r t`, or `\u` plus four
/// hex digits — those are copied through untouched.
fn repair_invalid_escapes(json: &str) -> String {
    let chars: Vec<char> = json.chars().collect();
    let mut out = String::with_capacity(json.len());
    let mut in_string = false;
    let mut i = 0usize;

    while i < chars.len() {
        let ch = chars[i];

        if ch == '\\' && in_string {
            let next = chars.get(i + 1).copied();
            let valid = match next {
                Some(n) => {
                    matches!(n, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't')
                        || (n == 'u'
                            && (i + 2..i + 6)
                                .all(|k| chars.get(k).is_some_and(|c| c.is_ascii_hexdigit())))
                }
                None => false,
            };
            if valid {
                out.push('\\');
                out.push(next.expect("checked above"));
                i += 2;
            } else {
                out.push_str("\\n"); // the botched line separator
                i += 1;
            }
            continue;
        }

        if ch == '"' {
            in_string = !in_string;
        }
        out.push(ch);
        i += 1;
    }
    out
}

/// Remove the extra opening quote models sometimes put before object keys:
/// `{"1":"a.",""2":"b."}` instead of `{"1":"a.","2":"b."}`.
///
/// The pattern is `""` followed by digits and a closing quote — an opening
/// quote immediately before digits can only be a key delimiter, since a bare
/// `"` is not legal *inside* a JSON string (it must be `\"`). So this cannot
/// rewrite valid JSON; it only fires where the reply is already broken.
fn repair_doubled_key_quotes(json: &str) -> String {
    let chars: Vec<char> = json.chars().collect();
    let mut out = String::with_capacity(json.len());
    let mut i = 0usize;

    while i < chars.len() {
        if chars[i] == '"' && chars.get(i + 1) == Some(&'"') {
            let mut j = i + 2;
            while chars.get(j).is_some_and(|c| c.is_ascii_digit()) {
                j += 1;
            }
            if j > i + 2 && chars.get(j) == Some(&'"') {
                out.push('"'); // collapse the two opening quotes into one
                i += 2;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Escape raw control characters (U+0000–U+001F) that appear *inside* JSON
/// string literals — what small models emit for multi-line cue text instead
/// of writing `\n`. serde_json rejects those, so without this a batch of
/// ordinary two-line cues fails outright and every cue falls back to English.
///
/// Newlines and indentation *between* tokens are legal JSON whitespace and
/// are left untouched, so pretty-printed replies still parse. Text outside
/// strings is copied verbatim.
fn escape_control_chars_in_strings(json: &str) -> String {
    let mut out = String::with_capacity(json.len());
    let mut in_string = false;
    let mut escaped = false;

    for ch in json.chars() {
        if in_string && (ch as u32) < 0x20 {
            match ch {
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                other => out.push_str(&format!("\\u{:04x}", other as u32)),
            }
            escaped = false;
            continue;
        }

        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
        } else if ch == '"' {
            in_string = true;
        }
        out.push(ch);
    }
    out
}

/// Replace markup tags with `<tN>` placeholders so they survive the model.
///
/// Recognised: HTML-style tags (`<i>`, `</b>`, `<font color="red">`) and
/// ASS override blocks (`{\an8}`, `{\i1c&H0000FF&}`). Plain `<` without a
/// closing `>` (e.g. `3 < 5`) is left untouched.
fn protect_markup(text: &str) -> (String, Vec<String>) {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut tags: Vec<String> = Vec::new();
    let mut i = 0usize;

    while i < chars.len() {
        let close = match chars[i] {
            '<' => chars[i..]
                .iter()
                .position(|&c| c == '>')
                .filter(|&rel| is_supported_html_tag(&chars[i..=i + rel])),
            '{' if chars.get(i + 1) == Some(&'\\') => chars[i..].iter().position(|&c| c == '}'),
            _ => None,
        };
        if let Some(rel) = close {
            let end = i + rel; // index of the closing char
            let tag: String = chars[i..=end].iter().collect();
            tags.push(tag);
            out.push_str(&format!("<t{}>", tags.len() - 1));
            i = end + 1;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    (out, tags)
}

/// Subtitle markup is a small subset of HTML. Restricting protection to these
/// tags keeps literal text such as `Use <this> option` translatable.
fn is_supported_html_tag(tag: &[char]) -> bool {
    let inner: String = tag.iter().collect();
    let inner = inner.strip_prefix('<').and_then(|s| s.strip_suffix('>'));
    let Some(inner) = inner else {
        return false;
    };
    let inner = inner
        .trim_start()
        .strip_prefix('/')
        .unwrap_or(inner.trim_start());
    let name_end = inner
        .find(|c: char| c.is_ascii_whitespace() || c == '/')
        .unwrap_or(inner.len());
    let (name, rest) = inner.split_at(name_end);
    matches!(
        name.to_ascii_lowercase().as_str(),
        "b" | "i" | "u" | "s" | "font" | "br"
    ) && (rest.is_empty()
        || rest.starts_with('/')
        || rest.chars().next().is_some_and(char::is_whitespace))
}

/// Normalize model text to the internal LF representation before validation or
/// serialization. A trailing newline is a real extra subtitle line.
fn normalize_model_text(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

/// Number of text lines, including a trailing empty line.
fn count_lines(text: &str) -> usize {
    text.split('\n').count()
}

/// Tags whose `<tN>` placeholder was dropped or duplicated by the model.
/// Exactly one occurrence is required to restore every original tag exactly
/// once; duplicates would otherwise create malformed nested markup.
fn malformed_placeholders<'a>(text: &str, tags: &'a [String]) -> Vec<&'a str> {
    let lower = text.to_ascii_lowercase();
    tags.iter()
        .enumerate()
        .filter(|(idx, _)| lower.matches(&format!("<t{idx}>")).count() != 1)
        .map(|(_, tag)| tag.as_str())
        .collect()
}

/// Put the original tags back where their placeholders ended up, then drop
/// any placeholder-shaped token the model invented or we could not restore.
///
/// A tag whose placeholder the model dropped entirely is re-inserted at its
/// original position when that position is unambiguous — leading tags (only
/// whitespace before them in the source) go to the front, trailing tags to
/// the back. Tags that sat in the middle cannot be placed and are returned
/// so the caller can report them. Matching is case-insensitive: small models
/// sometimes lowercase `<t0>`.
fn restore_markup(text: &str, source: &str, tags: &[String]) -> (String, Vec<String>) {
    let mut out = text.to_string();
    let mut leading: Vec<&String> = Vec::new();
    let mut trailing: Vec<&String> = Vec::new();
    let mut unplaced: Vec<String> = Vec::new();

    for (idx, tag) in tags.iter().enumerate() {
        let needle = format!("<t{idx}>");
        if out
            .to_ascii_lowercase()
            .contains(&needle.to_ascii_lowercase())
        {
            out = replace_first_ascii_ignore_case(&out, &needle, tag);
            continue;
        }
        match placeholder_position(source, &needle) {
            Some(Position::Leading) => leading.push(tag),
            Some(Position::Trailing) => trailing.push(tag),
            None => unplaced.push(tag.clone()),
        }
    }

    // `protect_markup` numbers tags left-to-right, so index order is source
    // order: leading tags keep their relative order at the front, same for
    // trailing tags at the back.
    let mut result = String::with_capacity(out.len() + 32);
    for tag in &leading {
        result.push_str(tag);
    }
    result.push_str(&strip_placeholder_residue(&out));
    for tag in &trailing {
        result.push_str(tag);
    }
    (result, unplaced)
}

/// Where a placeholder sits in the protected source, if that position is
/// unambiguous: before any non-whitespace, or after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Position {
    Leading,
    Trailing,
}

fn placeholder_position(source: &str, needle: &str) -> Option<Position> {
    let at = source
        .to_ascii_lowercase()
        .find(&needle.to_ascii_lowercase())?;
    let before = &source[..at];
    let after = &source[at + needle.len()..];
    if before.trim().is_empty() {
        Some(Position::Leading)
    } else if after.trim().is_empty() {
        Some(Position::Trailing)
    } else {
        None
    }
}

/// Remove leftover `<tN>` / `</tN>` tokens after restoration.
///
/// Real placeholders are all consumed by `restore_markup`, so anything
/// matching the pattern afterwards is model junk — either hallucinated (the
/// source had no markup at all) or a closing-tag imitation like `</t0>`,
/// which is never something we emit.
fn strip_placeholder_residue(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;

    while i < chars.len() {
        if chars[i] == '<'
            && let Some(rel_end) = chars[i..].iter().position(|&c| c == '>')
        {
            let end = i + rel_end;
            let inner: String = chars[i + 1..end].iter().collect();
            if parse_placeholder_number(&inner).is_some() {
                // Every legitimate placeholder was consumed before this
                // pass. Any one left is a duplicate or model invention.
                i = end + 1;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// `t12`, `T12`, `/t12`, or `/T12` → `Some(12)`.
fn parse_placeholder_number(inner: &str) -> Option<usize> {
    let rest = inner.strip_prefix('/').unwrap_or(inner);
    let digits = rest.strip_prefix('t').or_else(|| rest.strip_prefix('T'))?;
    digits.parse().ok()
}

/// Replace the first case-insensitive occurrence of `needle` with `replacement`.
/// `needle` is ASCII, so byte offsets in the lowercased copy match the
/// original exactly (`to_ascii_lowercase` never changes byte length).
fn replace_first_ascii_ignore_case(haystack: &str, needle: &str, replacement: &str) -> String {
    let lower = haystack.to_ascii_lowercase();
    let needle_lower = needle.to_ascii_lowercase();
    let Some(at) = lower.find(&needle_lower) else {
        return haystack.to_string();
    };
    format!(
        "{}{}{}",
        &haystack[..at],
        replacement,
        &haystack[at + needle.len()..]
    )
}

/// Remove an optional leading/trailing markdown code fence.
fn strip_fences(content: &str) -> &str {
    let t = content.trim();
    let without_open = t
        .strip_prefix("```")
        .and_then(|rest| {
            // Drop an optional language tag line (e.g. ```json).
            let (first_line, remainder) = match rest.find('\n') {
                Some(n) => (&rest[..n], &rest[n + 1..]),
                None => ("", ""),
            };
            if first_line.trim().is_empty()
                || first_line.trim().chars().all(|c| c.is_ascii_alphanumeric())
            {
                Some(remainder)
            } else {
                None
            }
        })
        .unwrap_or(t);
    without_open
        .strip_suffix("```")
        .unwrap_or(without_open)
        .trim()
}

fn truncate(s: &str, max: usize) -> String {
    let trimmed = s.trim();
    let mut t: String = trimmed.chars().take(max).collect();
    if trimmed.chars().count() > max {
        t.push('…');
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    fn item(index: usize, text: &str) -> Item {
        Item {
            index,
            text: text.to_string(),
        }
    }

    /// Bulgarian — the default target, and what the Cyrillic fixtures in these
    /// tests were written against before the target became configurable.
    fn bg() -> Lang {
        Lang::resolve("bg").expect("built-in default always resolves")
    }

    #[test]
    fn batches_respect_count_and_char_limits() {
        let items: Vec<Item> = (0..10).map(|i| item(i, &"x".repeat(40))).collect();

        // count limit
        let batches = build_batches(&items, 3, 10_000);
        assert_eq!(
            batches.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![3, 3, 3, 1]
        );

        // char limit: 40 chars each, cap 100 -> 2 per batch (3rd would exceed)
        let batches = build_batches(&items, 8, 100);
        assert!(batches.iter().all(|b| b.len() == 2));
        assert_eq!(batches.len(), 5);
    }

    #[test]
    fn oversized_item_gets_own_batch() {
        let items = vec![item(0, &"y".repeat(500)), item(1, &"z".repeat(10))];
        let batches = build_batches(&items, 8, 100);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 1);
        assert_eq!(batches[1].len(), 1);
    }

    #[test]
    fn empty_items_are_skipped() {
        let items = vec![item(0, "   "), item(1, "Hello"), item(2, "\n")];
        let batches = build_batches(&items, 5, 1000);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 1);
        assert_eq!(batches[0][0].index, 1);
    }

    #[test]
    fn payload_keys_are_batch_positions_not_file_indices() {
        let cfg = Config::default();
        // File indices deliberately *not* starting at 0, as in a mid-file batch.
        let texts = vec!["A".to_string(), "B".to_string()];
        let payload = build_payload(&cfg, &texts, &bg());
        assert_eq!(
            payload["chat_template_kwargs"]["enable_thinking"],
            json!(false)
        );
        let user = payload["messages"][1]["content"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(user).unwrap();
        assert_eq!(parsed, json!({"1": "A", "2": "B"}));
    }

    #[test]
    fn translate_items_handles_sparse_file_indices() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!(
            "http://{}/v1/chat/completions",
            listener.local_addr().unwrap()
        );
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut header = Vec::new();
            loop {
                let mut byte = [0u8; 1];
                stream.read_exact(&mut byte).unwrap();
                header.extend_from_slice(&byte);
                if header.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            let header = String::from_utf8(header).unwrap();
            let header_lower = header.to_ascii_lowercase();
            assert!(header_lower.contains("authorization: bearer test-token\r\n"));
            let content_length = header_lower
                .lines()
                .find_map(|line| line.strip_prefix("content-length: "))
                .unwrap()
                .parse::<usize>()
                .unwrap();
            let mut request_body = vec![0; content_length];
            stream.read_exact(&mut request_body).unwrap();
            assert!(
                String::from_utf8(request_body)
                    .unwrap()
                    .contains(r#"\"1\":\"Hello\""#)
            );

            let body = json!({
                "choices": [{"message": {"content": r#"{"1":"Здравей"}"#}}]
            })
            .to_string();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
        });

        let cfg = Config {
            endpoint,
            api_key: "test-token".to_string(),
            retries: 0,
            batch_size: 1,
            ..Config::default()
        };
        // A timing-only cue at file position 0 leaves the translatable cue at
        // position 1. The returned vector must still have one entry.
        let items = vec![item(1, "Hello")];
        let (translations, report) = translate_items(&cfg, &items, &bg());
        server.join().unwrap();

        assert_eq!(translations, vec![Some("Здравей".to_string())]);
        assert!(report.degradations.is_empty());
    }

    #[test]
    fn parses_plain_json_reply() {
        let map = parse_translations(r#"{"1":"Здравей","2":"Довиждане"}"#, 2).unwrap();
        assert_eq!(map[&1], "Здравей");
        assert_eq!(map[&2], "Довиждане");
    }

    #[test]
    fn parses_fenced_reply_with_prose() {
        let reply = "Sure! Here you go:\n```json\n{\"1\": \"Здравей\"}\n```\nHope that helps.";
        let map = parse_translations(reply, 1).unwrap();
        assert_eq!(map[&1], "Здравей");
    }

    #[test]
    fn ignores_out_of_range_keys_and_fills_nothing_for_missing() {
        // A model that answers with keys 0..n-1 must not shift positions.
        let map = parse_translations(r#"{"0":"първи","1":"втори"}"#, 2).unwrap();
        assert_eq!(map[&1], "втори"); // 0 is out of range, dropped
        assert!(!map.contains_key(&0));
    }

    #[test]
    fn rejects_junk_reply() {
        assert!(parse_translations("I cannot translate that.", 1).is_err());
        assert!(parse_translations(r#"{"1":""}"#, 1).is_err());
        assert!(parse_translations(r#"["Здравей"]"#, 1).is_err());
        assert!(parse_translations(r#"{"9":"Здравей"}"#, 1).is_err()); // all out of range
    }

    #[test]
    fn protect_and_restore_markup_roundtrip() {
        let src = r#"<i>Go now</i> and {\an8}shout"#;
        let (protected, tags) = protect_markup(src);
        assert_eq!(protected, "<t0>Go now<t1> and <t2>shout");
        assert_eq!(tags, vec!["<i>", "</i>", r"{\an8}"]);

        // Simulate the model keeping placeholders verbatim.
        assert_eq!(
            restore_markup(&protected, &protected, &tags),
            (src.to_string(), vec![])
        );
    }

    #[test]
    fn restore_reinserts_dropped_leading_and_trailing_tags() {
        // Source: {\an8}Don't be late!  — placeholder at the very front.
        let (protected, tags) = protect_markup(r"{\an8}Don't be late!");
        assert_eq!(protected, "<t0>Don't be late!");
        // Model dropped the placeholder entirely.
        let (restored, unplaced) = restore_markup("Не закъснявай!", &protected, &tags);
        assert_eq!(restored, r"{\an8}Не закъснявай!");
        assert!(unplaced.is_empty());

        // Closing tag at the very back is appended.
        let (protected, tags) = protect_markup("<i>Go now</i>");
        let (restored, unplaced) = restore_markup("<t0>Хайде", &protected, &tags);
        assert_eq!(restored, "<i>Хайде</i>");
        assert!(unplaced.is_empty());
    }

    #[test]
    fn restore_reports_mid_text_tags_it_cannot_place() {
        let (protected, tags) = protect_markup("a <b>c</b> d");
        let (restored, unplaced) = restore_markup("x y", &protected, &tags);
        // `<b>` and `</b>` both sat mid-text → cannot re-insert.
        assert_eq!(unplaced, vec!["<b>", "</b>"]);
        assert_eq!(restored, "x y");
    }

    #[test]
    fn restore_tolerates_lowercased_placeholders() {
        let tags = vec!["<i>".to_string()];
        let (restored, _) = restore_markup("кажи <T0> нещо", "<t0>say this", &tags);
        assert_eq!(restored, "кажи <i> нещо");
    }

    #[test]
    fn protect_leaves_plain_angles_alone() {
        let (protected, tags) = protect_markup("if 3 < 5 then {ok}");
        assert_eq!(protected, "if 3 < 5 then {ok}"); // `{o...` is not `{\...`
        assert!(tags.is_empty());

        let (protected, tags) = protect_markup("Use <this> option");
        assert_eq!(protected, "Use <this> option");
        assert!(tags.is_empty());
    }

    #[test]
    fn assemble_maps_positions_to_file_indices() {
        // Mid-file batch: file indices 10..12, positions 1..3.
        let batch = vec![item(10, "a"), item(11, "b"), item(12, "c")];
        let protected = vec![
            ("a".to_string(), vec![]),
            ("b".to_string(), vec![]),
            ("c".to_string(), vec![]),
        ];
        let reply = BTreeMap::from([(1, "Първи".to_string()), (3, "Трети".to_string())]);

        let (out, lost) = assemble_reply(
            reply,
            &batch,
            &protected,
            &bg(),
            AcceptPartial::Reject,
            &mut Report::default(),
        );
        assert_eq!(out.get(&10).map(String::as_str), Some("Първи"));
        assert_eq!(out.get(&12).map(String::as_str), Some("Трети"));
        assert!(!out.contains_key(&11));
        assert_eq!(lost, vec![11]); // missing position 2
    }

    #[test]
    fn assemble_rejects_dropped_markup_then_accepts_it() {
        let batch = vec![item(0, "<i>Hello</i>")];
        let protected = vec![(
            "<t0>Hello<t1>".to_string(),
            vec!["<i>".to_string(), "</i>".to_string()],
        )];
        // Model kept the opening tag but dropped the closing one.
        let reply = BTreeMap::from([(1, "Здравей <t0>".to_string())]);

        let (out, lost) = assemble_reply(
            reply.clone(),
            &batch,
            &protected,
            &bg(),
            AcceptPartial::Reject,
            &mut Report::default(),
        );
        assert!(out.is_empty(), "batch reply must drop the damaged cue");
        assert_eq!(lost, vec![0]);

        // Final single-cue fallback keeps the translation; the surviving
        // placeholder is substituted and the dropped closing tag re-appended.
        let (out, lost) = assemble_reply(
            reply,
            &batch,
            &protected,
            &bg(),
            AcceptPartial::Accept,
            &mut Report::default(),
        );
        assert_eq!(out.get(&0).map(String::as_str), Some("Здравей <i></i>"));
        assert!(lost.is_empty());
    }

    #[test]
    fn duplicate_placeholders_are_retried_then_removed_in_final_fallback() {
        let batch = vec![item(0, "<i>Hello</i>")];
        let protected = vec![(
            "<t0>Hello<t1>".to_string(),
            vec!["<i>".to_string(), "</i>".to_string()],
        )];
        let reply = BTreeMap::from([(1, "<t0>Здравей<t0><t1>".to_string())]);

        let (out, lost) = assemble_reply(
            reply.clone(),
            &batch,
            &protected,
            &bg(),
            AcceptPartial::Reject,
            &mut Report::default(),
        );
        assert!(out.is_empty());
        assert_eq!(lost, vec![0]);

        let (out, _) = assemble_reply(
            reply,
            &batch,
            &protected,
            &bg(),
            AcceptPartial::Accept,
            &mut Report::default(),
        );
        assert_eq!(out.get(&0).map(String::as_str), Some("<i>Здравей</i>"));
    }

    #[test]
    fn malformed_placeholders_reports_dropped_and_duplicate_tags() {
        let tags = vec!["<i>".to_string(), "</i>".to_string(), r"{\an8}".to_string()];
        assert!(malformed_placeholders("<t0>x<t1><t2>", &tags).is_empty());
        // case-insensitive detection
        assert!(malformed_placeholders("<T0>x<T1><T2>", &tags).is_empty());
        assert_eq!(
            malformed_placeholders("<t0>x", &tags),
            vec!["</i>", r"{\an8}"]
        );
        assert_eq!(malformed_placeholders("no tags", &tags).len(), 3);
        assert_eq!(
            malformed_placeholders("<t0>x<t0><t1><t2>", &tags),
            vec!["<i>"]
        );
    }

    #[test]
    fn strips_hallucinated_placeholder_residue() {
        // Model invented tokens for a source with no markup at all.
        assert_eq!(strip_placeholder_residue("<t0>Събитието</t0>"), "Събитието");
        // Closing-tag imitation of a real placeholder.
        assert_eq!(
            strip_placeholder_residue(r"{\an8}Рано!</t0>"),
            r"{\an8}Рано!"
        );
        // In-range placeholders are consumed before stripping, so a surviving
        // out-of-range one goes too, while ordinary text is untouched.
        assert_eq!(strip_placeholder_residue("няма <t7> тук"), "няма  тук");
        assert_eq!(strip_placeholder_residue("няма <T0> тук"), "няма  тук");
        assert_eq!(
            strip_placeholder_residue("обикновен <b>текст</b>"),
            "обикновен <b>текст</b>"
        );
    }

    #[test]
    fn restore_markup_removes_residue_after_replacing_real_tags() {
        let tags = vec!["<i>".to_string(), "</i>".to_string()];
        // Real placeholders restored; the invented closing token dropped.
        let (restored, unplaced) = restore_markup("<t0>Здравей<t1></t0>", "<t0>Hi<t1>", &tags);
        assert_eq!(restored, "<i>Здравей</i>");
        assert!(unplaced.is_empty());
        // No tags in source, model hallucinated some: all residue removed.
        let (restored, unplaced) = restore_markup("<t0>Без тагове</t0>", "Без тагове", &[]);
        assert_eq!(restored, "Без тагове");
        assert!(unplaced.is_empty());
    }

    #[test]
    fn assemble_rejects_line_merges_then_accepts_them() {
        // Two-line cue answered with a single line.
        let batch = vec![item(0, "Line one\nLine two")];
        let protected = vec![("Line one\nLine two".to_string(), vec![])];
        let reply = BTreeMap::from([(1, "Един ред".to_string())]);

        let (out, lost) = assemble_reply(
            reply.clone(),
            &batch,
            &protected,
            &bg(),
            AcceptPartial::Reject,
            &mut Report::default(),
        );
        assert!(out.is_empty(), "merged reply must be retried alone");
        assert_eq!(lost, vec![0]);

        let (out, _) = assemble_reply(
            reply,
            &batch,
            &protected,
            &bg(),
            AcceptPartial::Accept,
            &mut Report::default(),
        );
        assert_eq!(out.get(&0).map(String::as_str), Some("Един ред"));
    }

    #[test]
    fn assemble_accepts_matching_line_counts() {
        let batch = vec![item(4, "Line one\nLine two")];
        let protected = vec![("Line one\nLine two".to_string(), vec![])];
        let reply = BTreeMap::from([(1, "Ред един\nРед два".to_string())]);
        let (out, lost) = assemble_reply(
            reply,
            &batch,
            &protected,
            &bg(),
            AcceptPartial::Reject,
            &mut Report::default(),
        );
        assert_eq!(out.get(&4).map(String::as_str), Some("Ред един\nРед два"));
        assert!(lost.is_empty());
    }

    #[test]
    fn assemble_normalizes_crlf_and_rejects_trailing_newlines() {
        let batch = vec![item(0, "Line one\nLine two")];
        let protected = vec![("Line one\nLine two".to_string(), vec![])];
        let reply = BTreeMap::from([(1, "Ред едно\r\nРед две".to_string())]);
        let (out, lost) = assemble_reply(
            reply,
            &batch,
            &protected,
            &bg(),
            AcceptPartial::Reject,
            &mut Report::default(),
        );
        assert_eq!(out.get(&0).map(String::as_str), Some("Ред едно\nРед две"));
        assert!(lost.is_empty());

        let reply = BTreeMap::from([(1, "Ред едно\nРед две\n".to_string())]);
        let (out, lost) = assemble_reply(
            reply,
            &batch,
            &protected,
            &bg(),
            AcceptPartial::Reject,
            &mut Report::default(),
        );
        assert!(out.is_empty());
        assert_eq!(lost, vec![0]);
    }

    #[test]
    fn extracts_balanced_objects_from_noisy_replies() {
        // Two objects side by side: the old "first { to last }" slice was
        // `{"1":"a"},{"2":"b"}` → trailing characters.
        let map = parse_translations(r#"{"1":"a"} {"2":"b"}"#, 2).unwrap();
        assert_eq!(map[&1], "a");
        assert_eq!(map[&2], "b");

        // A closing brace in trailing prose must not extend the slice.
        let map = parse_translations(r#"{"1":"Здравей"} here is } that"#, 1).unwrap();
        assert_eq!(map[&1], "Здравей");

        // Braces inside string values are not structural.
        let map = parse_translations(r#"{"1":"кодът } е } тук"}"#, 1).unwrap();
        assert_eq!(map[&1], "кодът } е } тук");

        // Prose containing an unbalanced { before the real object.
        let map = parse_translations("note {oops\n{\"1\":\"да\"}", 1).unwrap();
        assert_eq!(map[&1], "да");
    }

    #[test]
    fn prefers_the_candidate_with_the_most_usable_cues() {
        // First object is junk, second holds real translations.
        let reply = r#"{"1":""} {"1":"Здравей","2":"Добър"}"#;
        let map = parse_translations(reply, 2).unwrap();
        assert_eq!(map[&1], "Здравей");
        assert_eq!(map[&2], "Добър");
    }

    #[test]
    fn repairs_invalid_escape_sequences() {
        // The exact reply that cost cue 53: a backslash before Cyrillic text
        // where a newline was intended. serde_json rejects the whole batch.
        let broken = r#"{"1":"- Хората имат много мнения\за това кой си ти."}"#;
        let fixed = repair_json(broken);
        let parsed: Value = serde_json::from_str(&fixed).expect("must parse now");
        assert_eq!(
            parsed["1"],
            json!("- Хората имат много мнения\nза това кой си ти.")
        );
        // Interpreting it as a separator rather than a stray character also
        // restores the two-line structure the source had.
        assert_eq!(parsed["1"].as_str().unwrap().lines().count(), 2);

        // Every legal escape survives untouched.
        for good in [
            r#"{"1":"a\nb"}"#,
            r#"{"1":"tab\there"}"#,
            r#"{"1":"quote\"inside"}"#,
            r#"{"1":"back\\slash"}"#,
            r#"{"1":"sl/ash"}"#,
            r#"{"1":"Здравей"}"#,
            r#"{"1":"\u0417дравей"}"#,
        ] {
            assert_eq!(repair_json(good), good, "must not rewrite {good}");
        }

        // `\u` not followed by four hex digits is not a legal escape. Only
        // the backslash is rewritten — the `uZZ` that followed is model text
        // and is kept, since eating characters would be a second guess about
        // intent on top of the first.
        let bad_u = repair_invalid_escapes(r#"{"1":"\uZZ"}"#);
        assert_eq!(bad_u, r#"{"1":"\nuZZ"}"#);

        // A trailing backslash with nothing after it is also invalid.
        let trailing = repair_invalid_escapes(r#"{"1":"x\"#);
        assert_eq!(trailing, r#"{"1":"x\n"#);

        // Backslashes outside strings are not the model's escape usage and
        // are left for serde_json to complain about honestly.
        let outside = repair_invalid_escapes(r#"\z{"1":"a"}"#);
        assert_eq!(outside, r#"\z{"1":"a"}"#);
    }

    #[test]
    fn straightens_curly_quotes_only_when_the_straight_parse_failed() {
        // Observed killing batch 21 three attempts running: values closed
        // with ” or “ instead of ".
        let curly_right = "\"1\": \"- Джеф Гласър\\nсе оказыва...”,\n  \"2\": \"лъжец.\"";
        let reply = format!("{{{curly_right}}}");
        let map = parse_translations(&reply, 2).expect("fallback must recover it");
        assert_eq!(map[&1], "- Джеф Гласър\nсе оказыва...");
        assert_eq!(map[&2], "лъжец.");

        // A left curly quote used as the closer is handled the same way.
        let curly_left = r#"{"1":"Искат\nда интервюират Анжели.“}"#;
        let map = parse_translations(curly_left, 1).expect("fallback must recover it");
        assert_eq!(map[&1], "Искат\nда интервюират Анжели.");

        // Healthy replies are never rewritten — “ ” are legitimate quotation
        // marks in the target language, and straightening them mid-value would
        // terminate the string early and corrupt the text.
        let healthy = r#"{"1":"Той каза “не” наистина"}"#;
        assert_eq!(repair_json(healthy), healthy);
        let map = parse_translations(healthy, 1).unwrap();
        assert_eq!(map[&1], "Той каза “не” наистина");

        // Same guard for a Latin-script target: a French reply using the same
        // quotation marks must come through untouched.
        let french = r#"{"1":"Il a dit “non” vraiment"}"#;
        assert_eq!(
            parse_translations(french, 1).unwrap()[&1],
            "Il a dit “non” vraiment"
        );
    }

    #[test]
    fn repairs_doubled_key_quotes() {
        // Observed from the 2B model: every key after the first gained a
        // second opening quote, which serde_json rejects outright.
        let broken = r#"{"1":"- Благодаря.",""2":"- Да, тя е била.",""3":"- Кори?"}"#;
        let fixed = repair_json(broken);
        let parsed: Value = serde_json::from_str(&fixed).unwrap();
        assert_eq!(parsed["2"], json!("- Да, тя е била."));
        assert_eq!(parsed["3"], json!("- Кори?"));

        // The first key is unaffected and valid replies are left alone.
        let good = r#"{"1":"a","2":"b"}"#;
        assert_eq!(repair_json(good), good);

        // Escaped quotes inside a value are not key delimiters: `\""` here is
        // an escaped quote followed by the string's closing quote.
        let quoted = r#"{"1":"he said \"5\" ok"}"#;
        assert_eq!(repair_json(quoted), quoted);

        // `""` not followed by digits is not a key delimiter and is left
        // alone — e.g. a doubled quote mid-value in an already-broken reply.
        assert_eq!(
            repair_doubled_key_quotes(r#"{"1":"she said ""hi""}"#),
            r#"{"1":"she said ""hi""}"#
        );
        assert_eq!(
            repair_doubled_key_quotes(r#"{"1":"a","2":"b"}"#),
            r#"{"1":"a","2":"b"}"#
        );
    }

    #[test]
    fn repairs_raw_newlines_inside_json_strings() {
        // Exactly what the model returned for cue 12 of the Angelyne file:
        // a raw newline inside the value, and an escaped quote at the start.
        let broken = "{\n  \"1\": \"\\\"Миссията на\nLA billboard diva Angelyne\",\n  \"2\": \"Angelyne,\nпочистваш ли си?\"\n}";
        let fixed = escape_control_chars_in_strings(broken);
        let parsed: Value = serde_json::from_str(&fixed).expect("repair must yield valid JSON");
        assert_eq!(
            parsed["1"].as_str().unwrap().lines().count(),
            2,
            "line structure survives"
        );
        assert!(parsed["1"].as_str().unwrap().starts_with('"'));
        assert_eq!(parsed["2"].as_str().unwrap().lines().count(), 2);
    }

    #[test]
    fn escape_control_chars_leaves_valid_json_untouched() {
        let pretty = "{\n  \"1\": \"Здравей\\nсвет\",\n  \"2\": \"текст\"\n}";
        assert_eq!(escape_control_chars_in_strings(pretty), pretty);
    }

    #[test]
    fn escape_control_chars_preserves_escaped_quotes_and_tabs() {
        let src = "{\"1\": \"рече \\\"здра\\\"\\tтук\\r\\nкрај\"}";
        assert_eq!(escape_control_chars_in_strings(src), src);
    }

    #[test]
    fn escape_control_chars_handles_nul_and_other_controls() {
        let broken = "{\"1\": \"a\u{0001}b\"}";
        let fixed = escape_control_chars_in_strings(broken);
        let parsed: Value = serde_json::from_str(&fixed).unwrap();
        assert_eq!(parsed["1"].as_str().unwrap(), "a\u{0001}b");
    }

    #[test]
    fn detects_echoed_english_but_spares_names_and_interjections() {
        // Verbatim echo of a multi-word sentence: no Cyrillic at all.
        assert!(looks_untranslated(
            "featuring the striking image",
            "<t0>featuring the striking image<t1>",
            &bg()
        ));
        assert!(looks_untranslated(
            "what they see.",
            "what they see.",
            &bg()
        ));
        assert!(looks_untranslated(
            "- I love Angelyne",
            "- I love Angelyne",
            &bg()
        ));

        // Correctly left in Latin script — names, numbers, interjections.
        assert!(!looks_untranslated("- Angelyne.", "- Angelyne.", &bg()));
        assert!(!looks_untranslated("Oh.", "Oh.", &bg()));
        assert!(!looks_untranslated("2019", "2019", &bg()));
        assert!(looks_untranslated("I am", "I am", &bg()));
        assert!(looks_untranslated("A dog", "A dog", &bg()));
        assert!(!looks_untranslated("Paris", "Пари", &bg()));

        // A real translation always contains Cyrillic.
        assert!(!looks_untranslated(
            "featuring the striking image",
            "<t0>със striking образ<t1>",
            &bg()
        ));
    }

    #[test]
    fn latin_targets_reject_english_echoes_and_accept_real_translations() {
        // The case a hardcoded Cyrillic check got wrong: every correct French
        // reply would have been rejected, sending each cue through a pointless
        // single-cue retry and reporting good text as untranslated.
        let fr = Lang::resolve("fr").unwrap();

        assert!(looks_untranslated(
            "I am not a woman.",
            "I am not a woman.",
            &fr
        ));
        // Edited and reordered English still says nothing new.
        assert!(looks_untranslated("what they see.", "they see what.", &fr));
        assert!(looks_untranslated(
            "featuring the striking image",
            "<t0>featuring the striking image<t1>",
            &fr
        ));

        assert!(!looks_untranslated(
            "I am not a woman.",
            "Je ne suis pas une femme.",
            &fr
        ));
        // Accented words must count as words, or French looks empty and every
        // reply falls into the echo bucket.
        assert!(!looks_untranslated(
            "It was déjà vu here.",
            "C'était déjà vu ici.",
            &fr
        ));
        // Short sources stay exempt whatever the target.
        assert!(!looks_untranslated("- Ooh.", "- Ooh.", &fr));
        assert!(!looks_untranslated("Angelyne", "Angelyne", &fr));
    }

    #[test]
    fn prompt_names_the_target_language() {
        let cfg = Config::default();
        let payload = build_payload(&cfg, &["Hi".to_string()], &Lang::resolve("fr").unwrap());
        let system = payload["messages"][0]["content"].as_str().unwrap();
        assert!(system.contains("French"), "{system}");
        assert!(
            !system.contains("{lang}"),
            "placeholder must be substituted"
        );
        assert!(
            !system.contains("Bulgarian"),
            "default must not leak through"
        );

        // A custom prompt gets the same substitution.
        let custom = Config {
            system_prompt: Some("Translate to {lang} only.".to_string()),
            ..Config::default()
        };
        let payload = build_payload(&custom, &["Hi".to_string()], &Lang::resolve("ru").unwrap());
        let system = payload["messages"][0]["content"].as_str().unwrap();
        assert_eq!(system, "Translate to Russian only.");
    }

    #[test]
    fn ascii_word_runs_counts_one_letter_words() {
        assert_eq!(ascii_word_runs("featuring the striking image"), 4);
        assert_eq!(ascii_word_runs("- Ooh."), 1);
        assert_eq!(ascii_word_runs("I am"), 2);
        assert_eq!(ascii_word_runs("A dog"), 2);
        assert_eq!(ascii_word_runs("2019"), 0);
        assert_eq!(ascii_word_runs("<t0>featuring the image<t1>"), 3);
    }

    #[test]
    fn content_words_are_case_insensitive_and_survive_accents() {
        let source = content_words("It was déjà vu.");
        assert!(source.contains("déjà"));
        assert!(source.contains("vu"));
        assert!(!source.contains("déjà vu")); // split into separate words

        // Wholly different vocabulary: a translation, not an echo.
        assert!(!echoes_source(
            "I am not a woman.",
            "Je ne suis pas une femme."
        ));
        // Same words, different order: still an echo.
        assert!(echoes_source("what they see.", "see what they."));
        // Case and inflection of the *source* words must not save a reply.
        assert!(echoes_source("What They See.", "what they see."));
    }

    #[test]
    fn report_counts_line_and_markup_degradations() {
        let mut report = Report::default();
        report.push(
            4,
            Degradation::Lines {
                source: 2,
                result: 1,
            },
        );
        report.push(
            7,
            Degradation::Markup {
                tags: "{\\an8}".into(),
            },
        );
        report.push(
            9,
            Degradation::Lines {
                source: 3,
                result: 1,
            },
        );

        assert_eq!(report.line_issues(), vec![4, 9]);
        assert_eq!(report.markup_issues(), vec![7]);
        assert_eq!(report.degradations.len(), 3);
        assert!(Report::default().degradations.is_empty());
    }

    #[test]
    fn assemble_reply_records_degradations_for_kept_cues() {
        // Two-line source answered with one line, in Accept mode.
        let batch = vec![item(3, "Line one\nLine two")];
        let protected = vec![("Line one\nLine two".to_string(), vec![])];
        let reply = BTreeMap::from([(1, "Един ред".to_string())]);
        let mut report = Report::default();

        let (out, _) = assemble_reply(
            reply,
            &batch,
            &protected,
            &bg(),
            AcceptPartial::Accept,
            &mut report,
        );
        assert!(out.contains_key(&3), "cue is still translated");
        assert_eq!(report.line_issues(), vec![3]);
        assert_eq!(
            report.degradations,
            vec![(
                3,
                Degradation::Lines {
                    source: 2,
                    result: 1
                }
            )]
        );
    }

    #[test]
    fn assemble_reply_records_nothing_in_reject_mode() {
        // Rejected cues are retried, not kept — they must not pollute the report.
        let batch = vec![item(3, "Line one\nLine two")];
        let protected = vec![("Line one\nLine two".to_string(), vec![])];
        let reply = BTreeMap::from([(1, "Един ред".to_string())]);
        let mut report = Report::default();

        let (out, lost) = assemble_reply(
            reply,
            &batch,
            &protected,
            &bg(),
            AcceptPartial::Reject,
            &mut report,
        );
        assert!(out.is_empty());
        assert_eq!(lost, vec![3]);
        assert!(report.degradations.is_empty());
    }

    #[test]
    fn assemble_reply_rejects_echoed_english_for_retry_then_keeps_it() {
        // A verbatim echo: valid JSON, placeholders intact, line count
        // matching — the language check is the only thing that catches it.
        let batch = vec![item(3, "featuring the striking image")];
        let protected = vec![(
            "featuring the striking image".to_string(),
            vec!["<i>".to_string(), "</i>".to_string()],
        )];
        let reply = BTreeMap::from([(1, "<t0>featuring the striking image<t1>".to_string())]);

        // Batch pass: rejected so the cue is retried on its own.
        let mut report = Report::default();
        let (out, lost) = assemble_reply(
            reply.clone(),
            &batch,
            &protected,
            &bg(),
            AcceptPartial::Reject,
            &mut report,
        );
        assert!(out.is_empty(), "echoed English must not be accepted");
        assert_eq!(lost, vec![3], "cue is queued for a single retry");
        assert!(
            report.degradations.is_empty(),
            "rejections are not reported"
        );

        // Single-cue pass: nothing left to fall back to, so keep it — but
        // record that it is English, so the summary is honest about it.
        let mut report = Report::default();
        let (out, lost) = assemble_reply(
            reply,
            &batch,
            &protected,
            &bg(),
            AcceptPartial::Accept,
            &mut report,
        );
        assert!(out.contains_key(&3), "kept rather than dropped");
        assert!(lost.is_empty(), "Accept mode never drops a cue");
        assert_eq!(report.untranslated_issues(), vec![3]);
        assert_eq!(report.degradations, vec![(3, Degradation::Untranslated)]);
    }

    #[test]
    fn assemble_reply_accepts_a_properly_translated_cue() {
        let batch = vec![item(3, "featuring the striking image")];
        let protected = vec![(
            "featuring the striking image".to_string(),
            vec!["<i>".to_string(), "</i>".to_string()],
        )];
        let reply = BTreeMap::from([(1, "<t0>със strike образ<t1>".to_string())]);

        let mut report = Report::default();
        let (out, _lost) = assemble_reply(
            reply,
            &batch,
            &protected,
            &bg(),
            AcceptPartial::Reject,
            &mut report,
        );
        assert!(out.contains_key(&3));
        assert!(report.degradations.is_empty());
    }

    #[test]
    fn strip_fences_variants() {
        assert_eq!(strip_fences("```json\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(strip_fences("{\"a\":1}"), "{\"a\":1}");
        assert_eq!(strip_fences("```\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(strip_fences("{\"a\":\"```\"}"), "{\"a\":\"```\"}");
    }

    #[test]
    fn payload_disables_thinking_and_carries_values() {
        let cfg = Config::default();
        assert!(cfg.disable_thinking);
        let payload = build_payload(&cfg, &["Hi".to_string()], &bg());
        assert_eq!(payload["model"], json!(cfg.model));

        let mut no_think = cfg.clone();
        no_think.disable_thinking = false;
        let payload = build_payload(&no_think, &["Hi".to_string()], &bg());
        assert!(payload.get("chat_template_kwargs").is_none());
    }
}
