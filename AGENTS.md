# AGENTS.md

## Project

`srt2lang` — a Rust console application that translates English SRT subtitle files
into a target language chosen with `-l/--lang` (default `bg`, Bulgarian).

Input: an `.srt` file. Output: the same file with every cue's text translated to
the target language; timestamps, cue order and cue indices are preserved.

## Build & test

Rust/cargo live in `~/.cargo/bin` and are **not** on the default `PATH` used by
non-interactive shells. Prefix every shell command with:

```sh
export PATH="$HOME/.cargo/bin:/bin:/usr/bin:/sbin:/usr/sbin:/usr/local/bin:/usr/local/sbin:/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"
```

Commands (run from the repo root):

```sh
cargo build                 # debug build   -> target/debug/srt2lang
cargo build -r              # release build -> target/release/srt2lang (optimized; this is the one to benchmark)
cargo test                  # unit tests (srt parsing, config, batching, response parsing)
cargo run -- movie.srt      # translate, writes movie.bg.srt (default lang bg)
cargo run -- movie.srt -l fr    # ... writes movie.fr.srt
```

`-r` is cargo's shorthand for `--release`. Live runs and timing comparisons
should use `target/release/srt2lang`; the debug binary is several times slower
and would skew any measurement.

Edition: 2024 (rustc 1.99+).

## Target language

`-l, --lang <CODE>` (config `lang`, default `bg`) selects what to translate
into: `fr`, `ru`, `bg`, `de`, … Three-letter codes (`fra`) work too, case and
surrounding whitespace are ignored, and an unknown code is a hard error before
any file or network work. `src/lang.rs` resolves the code through the
**`iso639` crate**, which supplies both the English name used in the system
prompt (`French`, not `fr` — a 4B model follows the name far more reliably)
and the language's ISO 15924 script from CLDR. There is no table of our own to
keep in sync.

The output name carries the code, so one source can be translated several ways
without collisions: `movie.srt -l fr` → `movie.fr.srt`.

### Reply validation is script-aware

`looks_untranslated(source, reply, target)` branches on the target's script:

- **Non-Latin target** (Cyrillic, Greek, Arabic, Hebrew, Han, kana, Hangul,
  Thai, Devanagari) — the reply must contain a character of that script.
  Precise, and impossible to satisfy with English.
- **Latin target, or a script CLDR does not cover** — there are no
  distinctive characters to look for, so it falls back to `echoes_source`: the
  reply must introduce at least one word the source did not already have.
  Comparison is over lowercased runs matched with Unicode `is_alphabetic`, not
  ASCII, so `déjà` and `émission` count as words — an ASCII split would reduce
  a good French reply to fragments and flag it as untranslated.

**The fallback direction is the important part.** An unknown script must
degrade to the echo check, never to "not in the target script": a language
wrongly assumed non-Latin flags *every* correct reply, sending each cue
through a pointless single-cue retry. That is exactly what a hardcoded
Cyrillic test did to `fr` before this existed.

Known limit: for Latin targets this catches verbatim and lightly-edited
echoes, but not an English *paraphrase* — that would need a language detector.

## Translation backend

Translation is done by an **OpenAI-compatible chat completions endpoint**, not a
dedicated translation API. This is a local Lemonade server (llama-server) on
`http://127.0.0.1:8001/v1/chat/completions`.

**Default model: `gemma-3-4b-it`** (config `model`). See "Model choice" below
for why, and for the measurements behind it.

Endpoint details that matter:

- No auth by default; `api_key` in config is optional and sent as a Bearer token.
- The `model` field is **not validated** by llama-server — any string is
  accepted — so it documents intent rather than selecting anything. The model
  actually served is whichever one Lemonade has loaded (`lemonade status`), and
  only one LLM fits at a time. Switching models means reloading Lemonade, not
  editing this config.
- Qwen3.5 is a **reasoning model**: without intervention it spends all tokens
  on `reasoning_content` and returns empty `content`. Those requests must
  include `"chat_template_kwargs": {"enable_thinking": false}` (config flag
  `disable_thinking`). That field is a llama-server extension, not part of the
  OpenAI API. Gemma's chat template has no such variable and ignores it —
  verified by running with and without — so the flag stays at its default
  `true`: harmless for Gemma, and the only setting that works for Qwen.
- The reply content may be wrapped in ```` ```json ```` fences even when
  `response_format` is requested — strip fences and locate the outer `{...}`
  before parsing.
- `response_format: {"type":"json_object"}` is not relied upon; prompts demand
  raw JSON and the parser tolerates extra prose/fences.

### Model choice

Measured on the same 120 cues from `Angelyne.S01E01`, identical input,
release build, `batch_size 4` / `max_batch_chars 400`:

| model | untranslated | coverage | line drift | runtime | binary |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 6 | 94.2% | 36 | 610 s | *pre-fix* |
| Qwen3.5-2B | 6 | 92.5% | 17 | 1544 s | current |
| **Gemma-3-4b-it** | **2** | **98.3%** | **2** | **651 s** | current |

The **2B vs Gemma rows are the valid head-to-head** — same binary, same input.
The 0.8B row predates reply validation and the JSON repairs, so treat it as a
rough historical baseline rather than a like-for-like comparison; it is kept
because it shows where the pipeline started.

Gemma wins on both quality and speed. The larger Qwen was ~2.4× *slower*
despite being only ~2× bigger, because it spent most of that time on retries:
it repeatedly echoed its input back verbatim (see "Reply validation") and
mangled JSON in other ways. Gemma never echoed once. It is also officially
multilingual with Bulgarian listed, which shows in fluency — the 0.8B produced
nonsense such as `Съма се като икон.`

**Recommendation: treat ~4B parameters as the minimum usable size.** Both
sub-4B models were qualitatively worse, not marginally worse — 17–36 cues per
120 came back with the wrong number of lines (the model merging two subtitle
lines into one, which breaks on-screen layout), and both needed heavy retry
traffic to reach ~93% coverage. At 4B those failures dropped to 2 and coverage
reached 98.3% with no retry storm. If a candidate model cannot hold line
structure, it is not usable for subtitles regardless of how fluent its
sentences sound.

Two caveats so this is not over-read:

- Only one 4B model was measured, and it differs from the Qwens in *family*
  as well as size — so "Gemma beat Qwen" and "4B beat 2B" are not separable
  from this data alone. The safe conclusion is the conservative one: below 4B
  the results are bad enough to rule those models out, and 4B is the smallest
  size demonstrated to work.
- Coverage and line drift are structural measures. They say nothing about
  idiomatic phrasing, so a fluent-sounding result still needs a human
  spot-check before use.

Translation quality remains bounded by whatever model the config points at;
the pipeline repairs *structure* (line counts, tags, JSON) but cannot make a
weak model fluent.

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
   prompt that names concrete examples reliably makes a small model emit
   them), and **re-inserts dropped tags** at their original position when that
   position is unambiguous (leading → front, trailing → back; mid-text tags
   cannot be placed and are reported). A closing tag whose opening partner the
   model dropped is stripped too (`strip_dangling_closers`) — `Дучезата</i>`
   with nothing open is junk — but only when the source itself had that
   partner; an unbalanced source keeps its shape. An opener without a closer
   is left alone: unclosed `<i>` idiomatically means "styled until end of
   cue".

### Fallback ladder

1. Batch request; parse the returned JSON object (fences/prose tolerated).
2. A cue is **rejected** if it is missing from the reply, dropped a
   placeholder, changed its line count (SRT line structure is display
   structure), **came back untranslated**, or **lost a number** that was in
   the source → retried as a single-cue request.
3. The single-cue reply is **accepted** even if imperfect (better a
   translation with a warning than English text), with the markup repair
   above, plus the two structural repairs below. If it still reads as the
   source, `Degradation::Untranslated` is recorded so the summary says so.
4. Still nothing → keep the original English text and report a warning.

### Structural repairs on an accepted reply

Two failures are worth *repairing* rather than just reporting, because both
leave a file that is structurally wrong even though the wording is fine.

- **Merged lines.** A source cue broken over two or three lines that comes
  back as one long line overflows the screen. `rewrap_lines` re-splits the
  translation into the source's line count at whitespace, so no tag or
  placeholder is ever cut in half. Recorded as `Degradation::Rewrapped`.
  Measured on `sample.movie.srt`: this took the file from **70 merged cues to
  0**, with timestamps and cue count untouched. If the text cannot be split
  that many ways it is left merged and reported as `Degradation::Lines`.
  The rewrap must run **before** the markup branch in `assemble_reply` — an
  earlier version let the markup path `continue` first, and the first movie2
  sample escaped with 35 merged cues that were never rebalanced.
- **Dropped numbers.** `Not 10.1. Not 10.2. Mach 10.` answered with a
  sentence that never mentions 10.1 or 10.2 has lost its meaning, not just
  its style. `missing_numbers` compares normalised numeric tokens — `5,000`
  and `5000` are the same number, as are `7.5` and `7,5` — so only genuine
  drops are flagged. Recorded as `Degradation::Numbers`.

`normalize_model_text` also strips trailing newlines before validation. A
terminator is not a subtitle line: counting it let a merged reply pass the
line-count check and serialize as text plus a blank line, which showed up in
the movie sample as a cue with an invisible extra line. On the raw movie run
that check flagged 11 cues, of which 9 were formatting differences that
normalisation correctly ignores and 2 were real content errors.


### Reply validation

Structural checks are not enough: a small model sometimes **echoes its input
verbatim**, placeholders and all. Such a reply has correct keys, intact
placeholders and matching line counts — it passes every structural test while
remaining English. `looks_untranslated` therefore checks the reply against the
**target language** — see "Target language" above for the script-aware rule and
why the fallback direction matters. Sources of fewer than two Latin words
(`- Ooh.`, `- Cory?`, `Angelyne`, `2019`) are exempt: those are correctly left
alone, and flagging them only buys useless retries.

`translate_items` returns a `Report` alongside the results. It records every
cue that was accepted *but* imperfect — `Lines`, `Rewrapped`, `Markup`,
`Numbers`, `Untranslated` — because a run's failures alone understate the
picture: most degraded cues are accepted, not rejected. `main.rs` prints these
as a `--- quality ---` summary. Rejections are deliberately *not* recorded;
they are retried, so counting them would double-count the eventual outcome.

The headline `translated` count in that summary is `items` minus the cues that
did not actually become a translation: ones with no surviving reply, ones the
model echoed back, and byte-identical no-ops (`Mach 10!`, `2019`). Without
subtracting the no-ops the count is inflated — they are correct output, but
they are not translations.

`Report` keys entries by **0-based file position**, matching `Item::index`.
SRT cues are numbered from 1, so anything user-facing must go through
`cue_numbers()` — printing the raw positions yields a summary that is
silently off by one, which is easy to miss because the counts stay correct.

### Malformed JSON from small models

Four failures were observed in practice and are repaired before `serde_json`
is called, all covered by unit tests:

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
- **An illegal escape sequence** — `мнения\за`, a backslash before Cyrillic
  text where a newline was meant. `repair_invalid_escapes` rewrites any
  backslash not followed by `" \ / b f n r t` or four hex digits after `u`
  **as a newline escape** (`\n`), not as a literal backslash: the system
  prompt tells the model that multi-line cue values are joined by `\n`, so
  that is the separator written wrong. Emitting `\` instead would parse but
  leave a visible `\` on screen and still count as line drift. One bad
  escape costs every cue in the batch, and it is what left a cue
  untranslated through three batch attempts *and* its single-cue retry.

All four run through `repair_json`, the single entry point called before
`serde_json`. New repairs belong there — each pass must leave already-valid
text untouched, since a repair that corrupts good JSON turns a clean error
into a deliberately wrong result.

A fifth failure is handled *outside* `repair_json`, because it stops the
braces from balancing in the first place:

- **Curly quotes as string delimiters** — the model closes a value with `”`
  or `“` instead of `"`, leaving the string unterminated. The closing `}`
  then reads as string content, `json_object_candidates` finds no balanced
  object, and the parser is never reached (observed killing a batch on three
  consecutive attempts). `parse_translations` therefore runs a **second pass
  over the whole reply** when the first found fewer cues than expected:
  `straighten_curly_quotes` converts `“ ”` to `"` and extraction is re-run,
  with first-pass entries winning on conflict. It is guarded on the first
  pass falling short precisely because Bulgarian text legitimately uses
  `“ ”` as quotation marks — rewriting a healthy reply would terminate
  strings early and corrupt the values.

Failed batches retry `config.retries` times before falling back. Progress and
warnings go to stderr and name the cue number, so a run's decisions are
auditable.

#### Known-but-unrepaired: a backslash before the closing quote

The model occasionally *escapes the terminator it meant to end on*:

```
"3": "- Анжели?\",
"4": "Прочети го."
```

`\"` is a **legal** escape, so no repair can recognise it by syntax alone: it
is exactly what a value legitimately containing a quote looks like
(`"He said \"hi\","`). Every rule that would rewrite it can therefore rewrite
a correct value, and because the rewrite closes the string early the result
can still parse — with a key or value silently mangled, which passes both the
line-count and language checks. That is the "silently wrong" outcome the
repairs exist to avoid, so this one is deliberately **not** repaired: the
`retrying batch (attempt 2)` path recovers it, observed costing one retry.

## Configuration

JSON config file, discovered in order:

1. `--config <path>` (CLI)
2. `./srt2lang.json`
3. `~/.config/srt2lang/config.json`
4. built-in defaults

All fields are optional (`#[serde(default)]`); the file only overrides what it
contains. Fields: `endpoint`, `api_key`, `model`, `temperature`, `max_tokens`,
`timeout_secs`, `retries`, `batch_size`, `max_batch_chars`, `disable_thinking`,
`lang`, `system_prompt`.

CLI overrides exist for the common ones (`--lang`, `--model`, `--endpoint`,
`--batch-size`); see `--help`. `system_prompt` may contain `{lang}`, which is
substituted with the target language's English name.

## Code layout

- `src/main.rs` — CLI parsing, wiring, progress output
- `src/config.rs` — `Config` struct, JSON loading, defaults
- `src/lang.rs` — ISO 639 code → name + script (`iso639` crate)
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
- Do not commit generated output (`*.??.srt`, `*.???.srt` — `movie.bg.srt`,
  `movie.fr.srt`) — it is in `.gitignore`.
- Batch requests must be resilient: a local server can be busy or restarting
  (`/health` returns 200 when ready).
- `cargo clippy --all-targets` must stay warning-free; run it after changes.

## Known model limitation

Translation *quality* is bounded by whatever model the config points at. The
pipeline detects and repairs *structure* — line counts, markup tags, malformed
JSON — but it cannot make a weak model fluent. Qwen3.5-0.8B, the original
target, frequently merged two cue lines into one, mistranslated idioms,
invented placeholder-shaped tokens, and sometimes echoed its input back
verbatim rather than translating at all.

The default is now `gemma-3-4b-it`, which measured markedly better on every
axis; see "Model choice" for the numbers. If you point `model` elsewhere,
expect to re-verify line drift and coverage rather than assume parity — and
note that no local model reaches professional quality, so spot-check output
before shipping it.
