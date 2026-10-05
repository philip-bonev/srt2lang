# AGENTS.md

## Project

`srt2bg` — a Rust console application that translates English SRT subtitle files
into Bulgarian.

Input: an `.srt` file. Output: the same file with every cue's text translated to
Bulgarian; timestamps, cue order and cue indices are preserved.

## Build & test

Rust/cargo live in `~/.cargo/bin` and are **not** on the default `PATH` used by
non-interactive shells. Prefix every shell command with:

```sh
export PATH="/Users/fi4o/.cargo/bin:/bin:/usr/bin:/sbin:/usr/sbin:/usr/local/bin:/usr/local/sbin:/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"
```

Commands (run from the repo root):

```sh
cargo build                 # debug build
cargo test                  # unit tests (srt parsing, config, batching, response parsing)
cargo run -- movie.srt      # translate, writes movie.bg.srt
```

Edition: 2024 (rustc 1.99+).

## Translation backend

Translation is done by an **OpenAI-compatible chat completions endpoint**, not a
dedicated translation API. At the time of writing this is a local Lemonade
server (llama-server, Qwen3.5-0.8B) on `http://127.0.0.1:8001/v1/chat/completions`.

Endpoint details that matter:

- No auth by default; `api_key` in config is optional and sent as a Bearer token.
- The model is a **reasoning model**: without intervention it spends all tokens
  on `reasoning_content` and returns empty `content`. Requests must include
  `"chat_template_kwargs": {"enable_thinking": false}` (config flag
  `disable_thinking`, default `true`).
- The reply content may be wrapped in ```` ```json ```` fences even when
  `response_format` is requested — strip fences and locate the outer `{...}`
  before parsing.
- `response_format: {"type":"json_object"}` is not relied upon; prompts demand
  raw JSON and the parser tolerates extra prose/fences.

## Translation strategy: batched blocks

Consecutive cues are grouped into batches (config `batch_size` cues and
`max_batch_chars` characters, whichever is hit first) and sent as **one**
request containing a JSON object `{ "1": "line", "2": "line", ... }`. The model
returns the same keys with Bulgarian values. This preserves cross-cue context
and cuts request count.

**Keys are the 1-based position within the batch, never a file-global index.**
Early versions keyed by file index and decoded by position, which silently
shifted every translation by one — see `payload_keys_are_batch_positions_not_file_indices`
and `assemble_maps_positions_to_file_indices`.

### Markup protection

Tags are hidden before the request and restored afterwards, so preservation
does not depend on model obedience:

1. `protect_markup` replaces `<i>`, `</b>`, `{\an8}`-style tags with `<tN>`
   placeholders (plain `3 < 5` is left alone).
2. The model translates placeholder-bearing text.
3. `restore_markup` substitutes placeholders back (case-insensitively),
   **strips hallucinated residue** (`<t0>`, `</t0>` the model invented — a
   prompt that names concrete examples reliably makes an 0.8B model emit them),
   and **re-inserts dropped tags** at their original position when that
   position is unambiguous (leading → front, trailing → back; mid-text tags
   cannot be placed and are reported).

### Fallback ladder

1. Batch request; parse the returned JSON object (fences/prose tolerated).
2. A cue is **rejected** if it is missing from the reply, dropped a
   placeholder, changed its line count (SRT line structure is display
   structure), or **came back with no Bulgarian** → retried as a single-cue
   request.
3. The single-cue reply is **accepted** even if imperfect (better a
   translation with a warning than English text), with the markup repair
   above. If it still has no Bulgarian, `Degradation::Untranslated` is
   recorded so the summary says so.
4. Still nothing → keep the original English text and report a warning.

### Reply validation

Structural checks are not enough: a small model sometimes **echoes its input
verbatim**, placeholders and all. Such a reply has correct keys, intact
placeholders and matching line counts — it passes every structural test while
remaining English. `looks_untranslated` therefore requires Cyrillic in any
reply to a multi-word Latin-script cue. Sources of fewer than two Latin words
(`- Ooh.`, `- Cory?`, `Angelyne`, `2019`) are exempt: those are correctly left
in Latin script, and flagging them only buys useless retries.

`translate_items` returns a `Report` alongside the results. It records every
cue that was accepted *but* imperfect — `Lines`, `Markup`, `Untranslated` —
because a run's failures alone understate the picture: most degraded cues are
accepted, not rejected. `main.rs` prints these as a `--- quality ---` summary.
Rejections are deliberately *not* recorded; they are retried, so counting them
would double-count the eventual outcome.

`Report` keys entries by **0-based file position**, matching `Item::index`.
SRT cues are numbered from 1, so anything user-facing must go through
`cue_numbers()` — printing the raw positions yields a summary that is
silently off by one, which is easy to miss because the counts stay correct.

### Malformed JSON from small models

Three failures were observed in practice and are repaired before `serde_json`
is called, both covered by unit tests:

- **Raw newlines inside string values** — the model writes a literal `\n`
  where the JSON spec wants `\\n`. `escape_control_chars_in_strings` rewrites
  them, leaving newlines *between* tokens alone so pretty-printed replies
  still parse.
- **Naive `{`…`}` extraction** — "first `{` to last `}`" produces an
  unparseable slice when the reply holds two objects, a `}` in trailing
  prose, or an unbalanced `{` in prose. `json_object_candidates` instead
  balances each opening brace independently (skipping braces inside string
  literals) and the findings are merged.
- **A doubled quote before object keys** — `{"1":"a.",""2":"b."}`. Every
  batch attempt failed until `repair_doubled_key_quotes` collapsed `""<digits>"`
  to `"<digits>"`. A bare `"` is never legal inside a JSON string (it must be
  `\"`), so the pattern cannot occur in valid JSON and the repair is safe.

All three run through `repair_json`, the single entry point called before
`serde_json`. New repairs belong there — each pass must leave already-valid
text untouched, since a repair that corrupts good JSON turns a clean error
into a silently wrong result.

Failed batches retry `config.retries` times before falling back. Progress and
warnings go to stderr and name the cue number, so a run's decisions are
auditable.

## Configuration

JSON config file, discovered in order:

1. `--config <path>` (CLI)
2. `./srt2bg.json`
3. `~/.config/srt2bg/config.json`
4. built-in defaults

All fields are optional (`#[serde(default)]`); the file only overrides what it
contains. Fields: `endpoint`, `api_key`, `model`, `temperature`, `max_tokens`,
`timeout_secs`, `retries`, `batch_size`, `max_batch_chars`, `disable_thinking`,
`system_prompt`.

CLI overrides exist for the common ones (`--model`, `--endpoint`,
`--batch-size`); see `--help`.

## Code layout

- `src/main.rs` — CLI parsing, wiring, progress output
- `src/config.rs` — `Config` struct, JSON loading, defaults
- `src/srt.rs` — SRT parsing/serialization (pure functions, unit-tested)
- `src/translate.rs` — batching, prompt building, HTTP call, response parsing

## Conventions & cautions

- **Preserve the file's original line endings** (CRLF vs LF) and strip a UTF-8
  BOM on read; write UTF-8 without BOM. Verified end-to-end against a
  BOM + CRLF input.
- Keep `srt.rs` free of I/O and network code so it stays unit-testable.
- Never translate timestamps, cue indices, or markup tags (`<i>`, `<b>`,
  `{\an8}`): timestamps/indices never leave the parser, tags are hidden behind
  `<tN>` placeholders (see "Markup protection").
- **Do not put concrete placeholder examples in the system prompt** — the
  small model copies them into output for untagged cues. Describe them
  generically instead.
- Progress/warnings go to **stderr**; only the translated SRT may go to stdout.
- Do not commit generated output (`*.bg.srt`) — it is in `.gitignore`.
- Batch requests must be resilient: a local server can be busy or restarting
  (`/health` returns 200 when ready).
- `cargo clippy --all-targets` must stay warning-free; run it after changes.

## Known model limitation

Translation *quality* is bounded by whatever model the config points at. The
verified setup uses Qwen3.5-**0.8B**, which frequently merges two cue lines
into one, mistranslates idioms, and invents placeholder-shaped tokens. The
pipeline detects and repairs structure (lines, tags) but cannot make an 0.8B
model fluent — expect to raise `model`/`endpoint` in `srt2bg.json` for
production-quality Bulgarian.
