//! Batched translation against an OpenAI-compatible chat completions endpoint.
//!
//! Consecutive cues are grouped into batches and sent as one JSON object
//! `{"1": "...", "2": "..."}`; the model must answer with the same keys and
//! Bulgarian values. Keys are always the **1-based position within the
//! batch** — never a file-global index — so request and reply can never drift
//! apart. Markup tags are replaced by `<tN>` placeholders before the request
//! and restored afterwards, so preservation does not depend on model
//! obedience. Failures fall back to single-cue requests, and cues that still
//! fail keep their original English text.

use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::time::Duration;

use crate::config::Config;

/// Instructs the model to return a bare JSON object keyed like the input.
const DEFAULT_SYSTEM_PROMPT: &str = "You are a professional subtitle translator. \
You receive a JSON object whose keys are sequential segment numbers starting at 1 and whose values are English subtitle lines (possibly multi-line, separated by \\n). \
Translate every value into natural, colloquial Bulgarian suitable for subtitles. \
Rules: keep the tone and register of the original; preserve line count within each value; \
copy any placeholder tokens standing for markup verbatim; \
keep proper names, numbers and onomatopoeia; do not renumber the keys; do not add explanations. \
Respond with ONLY a JSON object with exactly the same keys and the Bulgarian translations as values. \
No markdown, no code fences, no commentary.";

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
    /// The model dropped markup tags; they were re-inserted by guess.
    Markup { tags: String },
}

/// Per-run quality report: every degradation, by file cue index.
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
}

/// Group consecutive items into batches, bounded by both `batch_size` cues
/// and `max_batch_chars` total characters (a single oversized cue forms its
/// own batch). Items whose text is empty/whitespace are skipped entirely.
pub fn build_batches(
    items: &[Item],
    batch_size: usize,
    max_batch_chars: usize,
) -> Vec<Vec<Item>> {
    let mut batches: Vec<Vec<Item>> = Vec::new();
    let mut current: Vec<Item> = Vec::new();
    let mut chars = 0usize;

    for item in items {
        if item.text.trim().is_empty() {
            continue;
        }
        let len = item.text.chars().count();
        let would_exceed = !current.is_empty()
            && (current.len() >= batch_size || chars + len > max_batch_chars);
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
/// `Some(bulgarian)` on success, `None` when every fallback failed
/// (caller keeps the original text), plus a report of degraded cues.
pub fn translate_items(cfg: &Config, items: &[Item]) -> (Vec<Option<String>>, Report) {
    let mut results: Vec<Option<String>> = vec![None; items.len()];
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

        match translate_batch(cfg, batch, AcceptPartial::Reject, &mut report) {
            Ok(map) => {
                for item in batch {
                    if let Some(text) = map.get(&item.index) {
                        results[item.index] = Some(text.clone());
                    }
                }
            }
            Err(err) => eprintln!("  batch {} failed: {err}", bi + 1),
        }

        // Fallback: retry each cue of this batch on its own.
        for item in batch {
            if results[item.index].is_some() {
                continue;
            }
            match translate_batch(
                cfg,
                std::slice::from_ref(item),
                AcceptPartial::Accept,
                &mut report,
            ) {
                Ok(map) => {
                    if let Some(text) = map.get(&item.index) {
                        results[item.index] = Some(text.clone());
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
    partial: AcceptPartial,
    report: &mut Report,
) -> Result<BTreeMap<usize, String>, String> {
    // Hide markup from the model; restored after the reply.
    let protected: Vec<(String, Vec<String>)> =
        batch.iter().map(|item| protect_markup(&item.text)).collect();
    let texts: Vec<String> = protected.iter().map(|(t, _)| t.clone()).collect();

    let payload = build_payload(cfg, &texts);
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        match post_chat(cfg, &payload) {
            Ok(content) => match parse_translations(&content, batch.len()) {
                Ok(positional) => {
                    let (out, lost) =
                        assemble_reply(positional, batch, &protected, partial, report);
                    if partial == AcceptPartial::Reject {
                        for index in lost {
                            eprintln!(
                                "  cue {}: structure not preserved (markup/line count), retrying alone",
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
    partial: AcceptPartial,
    report: &mut Report,
) -> (BTreeMap<usize, String>, Vec<usize>) {
    let mut out = BTreeMap::new();
    let mut lost = Vec::new();

    for (pos, text) in &positional {
        let (Some(item), Some((source, tags))) =
            (batch.get(pos - 1), protected.get(pos - 1))
        else {
            continue;
        };

        let dropped: Vec<&str> = missing_placeholders(text, tags);
        let source_lines = count_lines(&item.text);
        let result_lines = count_lines(text);
        let merged = source_lines != result_lines;
        if (!dropped.is_empty() || merged) && partial == AcceptPartial::Reject {
            lost.push(item.index);
            continue;
        }
        if !dropped.is_empty() {
            let (restored, unplaced) = restore_markup(text, source, tags);
            report.push(
                item.index,
                Degradation::Markup {
                    tags: if unplaced.is_empty() {
                        dropped.join(", ")
                    } else {
                        format!("{} (unplaced: {})", dropped.join(", "), unplaced.join(", "))
                    },
                },
            );
            if unplaced.is_empty() {
                eprintln!(
                    "  cue {}: model dropped markup ({}), re-inserted at original position",
                    item.index + 1,
                    dropped.join(", ")
                );
            } else {
                eprintln!(
                    "  cue {}: model dropped markup and position is unknown for {}, keeping translation",
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
        out.insert(item.index, restore_markup(text, source, tags).0);
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
fn build_payload(cfg: &Config, texts: &[String]) -> Value {
    let mut input = Map::new();
    for (pos, text) in texts.iter().enumerate() {
        input.insert((pos + 1).to_string(), Value::String(text.clone()));
    }

    let system = cfg
        .system_prompt
        .clone()
        .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string());

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
/// trailing prose, would otherwise produce an unparseable slice. The
/// candidate yielding the most usable cues wins.
fn parse_translations(content: &str, expected: usize) -> Result<BTreeMap<usize, String>, String> {
    let stripped = strip_fences(content);
    let mut found: BTreeMap<usize, String> = BTreeMap::new();

    for candidate in json_object_candidates(stripped) {
        let repaired = escape_control_chars_in_strings(candidate);
        let Ok(parsed) = serde_json::from_str::<Value>(&repaired) else {
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
                        && !found.contains_key(&pos) {
                            found.insert(pos, text.to_string());
                        }
        }
        if found.len() == expected {
            break; // every cue accounted for
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
            '<' => chars[i..].iter().position(|&c| c == '>'),
            '{' if chars.get(i + 1) == Some(&'\\') => {
                chars[i..].iter().position(|&c| c == '}')
            }
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

/// Number of text lines, ignoring a trailing newline.
fn count_lines(text: &str) -> usize {
    text.lines().count()
}

/// Tags whose `<tN>` placeholder is absent from the model's reply, reported
/// with the original tag text so warnings are readable.
fn missing_placeholders<'a>(text: &str, tags: &'a [String]) -> Vec<&'a str> {
    let lower = text.to_ascii_lowercase();
    tags.iter()
        .enumerate()
        .filter(|(idx, _)| !lower.contains(&format!("<t{idx}>")))
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
        if out.to_ascii_lowercase().contains(&needle.to_ascii_lowercase()) {
            out = replace_ascii_ignore_case(&out, &needle, tag);
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
    result.push_str(&strip_placeholder_residue(&out, tags.len()));
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
fn strip_placeholder_residue(text: &str, tag_count: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;

    while i < chars.len() {
        if chars[i] == '<'
            && let Some(rel_end) = chars[i..].iter().position(|&c| c == '>') {
                let end = i + rel_end;
                let inner: String = chars[i + 1..end].iter().collect();
                if let Some(num) = parse_placeholder_number(&inner) {
                    // `</tN>` is never ours; `<tN>` beyond our tags is invented.
                    if inner.starts_with('/') || num >= tag_count {
                        i = end + 1;
                        continue;
                    }
                }
            }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// `t12` or `/t12` → `Some(12)`.
fn parse_placeholder_number(inner: &str) -> Option<usize> {
    let rest = inner.strip_prefix('/').unwrap_or(inner);
    let digits = rest.strip_prefix('t')?;
    digits.parse().ok()
}

/// Replace every case-insensitive occurrence of `needle` with `replacement`.
/// `needle` is ASCII, so byte offsets in the lowercased copy match the
/// original exactly (`to_ascii_lowercase` never changes byte length).
fn replace_ascii_ignore_case(haystack: &str, needle: &str, replacement: &str) -> String {
    let lower = haystack.to_ascii_lowercase();
    let needle_lower = needle.to_ascii_lowercase();
    let mut out = String::with_capacity(haystack.len());
    let mut last = 0usize;
    while let Some(found) = lower[last..].find(&needle_lower) {
        let at = last + found;
        out.push_str(&haystack[last..at]);
        out.push_str(replacement);
        last = at + needle.len();
    }
    out.push_str(&haystack[last..]);
    out
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

    fn item(index: usize, text: &str) -> Item {
        Item { index, text: text.to_string() }
    }

    #[test]
    fn batches_respect_count_and_char_limits() {
        let items: Vec<Item> = (0..10).map(|i| item(i, &"x".repeat(40))).collect();

        // count limit
        let batches = build_batches(&items, 3, 10_000);
        assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), vec![3, 3, 3, 1]);

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
        let payload = build_payload(&cfg, &texts);
        assert_eq!(payload["chat_template_kwargs"]["enable_thinking"], json!(false));
        let user = payload["messages"][1]["content"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(user).unwrap();
        assert_eq!(parsed, json!({"1": "A", "2": "B"}));
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
        assert_eq!(restore_markup(&protected, &protected, &tags), (src.to_string(), vec![]));
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

        let (out, lost) = assemble_reply(reply, &batch, &protected, AcceptPartial::Reject, &mut Report::default());
        assert_eq!(out.get(&10).map(String::as_str), Some("Първи"));
        assert_eq!(out.get(&12).map(String::as_str), Some("Трети"));
        assert!(!out.contains_key(&11));
        assert_eq!(lost, vec![11]); // missing position 2
    }

    #[test]
    fn assemble_rejects_dropped_markup_then_accepts_it() {
        let batch = vec![item(0, "<i>Hello</i>")];
        let protected = vec![("<t0>Hello<t1>".to_string(), vec!["<i>".to_string(), "</i>".to_string()])];
        // Model kept the opening tag but dropped the closing one.
        let reply = BTreeMap::from([(1, "Здравей <t0>".to_string())]);

        let (out, lost) = assemble_reply(reply.clone(), &batch, &protected, AcceptPartial::Reject, &mut Report::default());
        assert!(out.is_empty(), "batch reply must drop the damaged cue");
        assert_eq!(lost, vec![0]);

        // Final single-cue fallback keeps the translation; the surviving
        // placeholder is substituted and the dropped closing tag re-appended.
        let (out, lost) = assemble_reply(reply, &batch, &protected, AcceptPartial::Accept, &mut Report::default());
        assert_eq!(out.get(&0).map(String::as_str), Some("Здравей <i></i>"));
        assert!(lost.is_empty());
    }

    #[test]
    fn missing_placeholders_reports_dropped_tags() {
        let tags = vec!["<i>".to_string(), "</i>".to_string(), r"{\an8}".to_string()];
        assert!(missing_placeholders("<t0>x<t1><t2>", &tags).is_empty());
        // case-insensitive detection
        assert!(missing_placeholders("<T0>x<T1><T2>", &tags).is_empty());
        assert_eq!(missing_placeholders("<t0>x", &tags), vec!["</i>", r"{\an8}"]);
        assert_eq!(missing_placeholders("no tags", &tags).len(), 3);
    }

    #[test]
    fn strips_hallucinated_placeholder_residue() {
        // Model invented tokens for a source with no markup at all.
        assert_eq!(strip_placeholder_residue("<t0>Събитието</t0>", 0), "Събитието");
        // Closing-tag imitation of a real placeholder.
        assert_eq!(strip_placeholder_residue(r"{\an8}Рано!</t0>", 1), r"{\an8}Рано!");
        // In-range placeholders are consumed before stripping, so a surviving
        // out-of-range one goes too, while ordinary text is untouched.
        assert_eq!(strip_placeholder_residue("няма <t7> тук", 2), "няма  тук");
        assert_eq!(strip_placeholder_residue("обикновен <b>текст</b>", 0), "обикновен <b>текст</b>");
    }

    #[test]
    fn restore_markup_removes_residue_after_replacing_real_tags() {
        let tags = vec!["<i>".to_string(), "</i>".to_string()];
        // Real placeholders restored; the invented closing token dropped.
        let (restored, unplaced) =
            restore_markup("<t0>Здравей<t1></t0>", "<t0>Hi<t1>", &tags);
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

        let (out, lost) = assemble_reply(reply.clone(), &batch, &protected, AcceptPartial::Reject, &mut Report::default());
        assert!(out.is_empty(), "merged reply must be retried alone");
        assert_eq!(lost, vec![0]);

        let (out, _) = assemble_reply(reply, &batch, &protected, AcceptPartial::Accept, &mut Report::default());
        assert_eq!(out.get(&0).map(String::as_str), Some("Един ред"));
    }

    #[test]
    fn assemble_accepts_matching_line_counts() {
        let batch = vec![item(4, "Line one\nLine two")];
        let protected = vec![("Line one\nLine two".to_string(), vec![])];
        let reply = BTreeMap::from([(1, "Ред един\nРед два".to_string())]);
        let (out, lost) = assemble_reply(reply, &batch, &protected, AcceptPartial::Reject, &mut Report::default());
        assert_eq!(out.get(&4).map(String::as_str), Some("Ред един\nРед два"));
        assert!(lost.is_empty());
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
    fn repairs_raw_newlines_inside_json_strings() {
        // Exactly what the model returned for cue 12 of the Angelyne file:
        // a raw newline inside the value, and an escaped quote at the start.
        let broken = "{\n  \"1\": \"\\\"Миссията на\nLA billboard diva Angelyne\",\n  \"2\": \"Angelyne,\nпочистваш ли си?\"\n}";
        let fixed = escape_control_chars_in_strings(broken);
        let parsed: Value =
            serde_json::from_str(&fixed).expect("repair must yield valid JSON");
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
    fn report_counts_line_and_markup_degradations() {
        let mut report = Report::default();
        report.push(4, Degradation::Lines { source: 2, result: 1 });
        report.push(7, Degradation::Markup { tags: "{\\an8}".into() });
        report.push(9, Degradation::Lines { source: 3, result: 1 });

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

        let (out, _) =
            assemble_reply(reply, &batch, &protected, AcceptPartial::Accept, &mut report);
        assert!(out.contains_key(&3), "cue is still translated");
        assert_eq!(report.line_issues(), vec![3]);
        assert_eq!(
            report.degradations,
            vec![(3, Degradation::Lines { source: 2, result: 1 })]
        );
    }

    #[test]
    fn assemble_reply_records_nothing_in_reject_mode() {
        // Rejected cues are retried, not kept — they must not pollute the report.
        let batch = vec![item(3, "Line one\nLine two")];
        let protected = vec![("Line one\nLine two".to_string(), vec![])];
        let reply = BTreeMap::from([(1, "Един ред".to_string())]);
        let mut report = Report::default();

        let (out, lost) =
            assemble_reply(reply, &batch, &protected, AcceptPartial::Reject, &mut report);
        assert!(out.is_empty());
        assert_eq!(lost, vec![3]);
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
        let payload = build_payload(&cfg, &["Hi".to_string()]);
        assert_eq!(payload["model"], json!(cfg.model));

        let mut no_think = cfg.clone();
        no_think.disable_thinking = false;
        let payload = build_payload(&no_think, &["Hi".to_string()]);
        assert!(payload.get("chat_template_kwargs").is_none());
    }
}
