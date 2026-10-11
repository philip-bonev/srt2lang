# srt2lang

Translate English SRT subtitle files into another language with a local LLM.

```sh
srt2lang movie.srt -l fr    # movie.srt → movie.fr.srt
srt2lang movie.srt          # default target: bg → movie.bg.srt
```

Target language is an ISO 639-1 code (`fr`, `bg`, `ru`, `de`, …). Cue indices,
timestamps, line structure and markup tags (`<i>`, `<b>`, `{\an8}`) are
preserved exactly.

Runs entirely offline against any OpenAI-compatible chat completions endpoint —
by default a local [Lemonade](https://github.com/) / llama-server instance.

---

## Why Gemma instead of Qwen

This project originally targeted **Qwen3.5** in small sizes. It did not hold up.

Measured on the same 120 cues of real subtitle data, same input, same release
build, `batch_size 4` / `max_batch_chars 400`:

| model | untranslated | coverage | line drift | runtime |
|---|---|---|---|---|
| Qwen3.5-0.8B | 6 / 120 | 94.2% | 36 / 120 | 610 s |
| Qwen3.5-2B | 6 / 120 | 92.5% | 17 / 120 | 1544 s |
| **Gemma-3-4b-it** | **2 / 120** | **98.3%** | **2 / 120** | **651 s** |

**Gemma is both better and faster** — roughly 2.4× faster than the Qwen 2B
despite being the larger model. The reason matters: Qwen spent most of its time
on *retries*.

### The Qwen failure modes

These are not stylistic nitpicks — they are structural failures that leave a
subtitle file broken or untranslated:

- **Echoing instead of translating.** The 2B model would return its own input
  verbatim — placeholders and all — as if that were the answer. The reply had
  correct keys, intact placeholders and matching line counts, so it looked
  perfectly valid while being 100% English. 18 of 120 cues came back this way,
  and before a language check was added the pipeline reported them as
  successes.
- **Mangled JSON.** Raw newlines inside string values, doubled quotes before
  object keys (`{"1":"a.",""2":"b."}`), illegal escapes (`\z` where a newline
  was meant), and stray prose around the object. One bad character failed an
  entire batch of four cues — a single illegal escape in the Qwen 2B run cost
  one cue through three batch attempts *and* its individual retry.
- **Merging lines.** Two subtitle lines collapsed into one in 36 of 120 cues
  at 0.8B and 17 of 120 at 2B. Subtitle lines are a display constraint; merging
  them breaks on-screen layout.

Gemma did none of the echoing, produced far fewer malformed replies, and held
line structure in 118 of 120 cues. It is also officially multilingual with
Bulgarian listed, which shows: the 0.8B Qwen produced nonsense like
`Съма се като икон.`, whereas Gemma's output reads like actual Bulgarian.

### Minimum model size: 4B

**Treat ~4B parameters as the floor for usable results.** Both sub-4B models
were qualitatively worse, not marginally worse — needing heavy retry traffic
just to reach ~93% coverage while still breaking line structure in 14–30% of
cues. A model that cannot hold line structure is unusable for subtitles no
matter how fluent its sentences sound.

Two honest caveats: only one 4B model was measured, and it differs from the
Qwens in *family* as well as size, so "Gemma beat Qwen" and "4B beat 2B" can't
be separated from this data alone. The conservative reading is the right one —
below 4B the results are bad enough to rule those models out, and 4B is the
smallest size demonstrated to work.

The table measures **Bulgarian** output specifically. Other targets run the
identical pipeline, but translation quality per language is a property of the
model, so spot-check a `fr` or `ru` run rather than assuming these numbers
carry over.

---

## Build

Rust lives in `~/.cargo/bin`, which is not on the default `PATH` for
non-interactive shells:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
```

```sh
cargo build -r          # release build -> target/release/srt2lang
cargo test              # unit tests
cargo clippy --all-targets
```

Use the **release** binary for real runs; the debug build is several times
slower and will skew any timing.

## Usage

```sh
./target/release/srt2lang movie.srt                 # writes movie.bg.srt
./target/release/srt2lang movie.srt -l fr           # writes movie.fr.srt
./target/release/srt2lang movie.srt -o out.srt      # explicit output path
./target/release/srt2lang movie.srt --dry-run       # show batches, send nothing
./target/release/srt2lang --help
```

Progress and warnings go to **stderr**; the translated SRT may be written to
stdout, so redirect stderr if you want clean logs:

```sh
./target/release/srt2lang movie.srt 2>run.log
```

## Configuration

JSON, discovered in this order:

1. `--config <path>`
2. `./srt2lang.json`
3. `~/.config/srt2lang/config.json`
4. built-in defaults

All fields are optional; the file only overrides what it contains.

```json
{
  "endpoint": "http://127.0.0.1:8001/v1/chat/completions",
  "api_key": "",
  "model": "gemma-3-4b-it",
  "temperature": 0.2,
  "max_tokens": 2048,
  "timeout_secs": 180,
  "retries": 2,
  "batch_size": 4,
  "max_batch_chars": 400,
  "disable_thinking": true,
  "lang": "bg"
}
```

Notes:

- `lang` — target language, ISO 639-1 (`fr`, `bg`, `ru`, …); `-l` overrides it.
- `endpoint` — any OpenAI-compatible `/v1/chat/completions`.
- `api_key` — optional; sent as a Bearer token when non-empty.
- `model` — **not validated** by llama-server, so it documents intent. The
  model actually served is whichever one Lemonade has loaded
  (`lemonade status`), and only one LLM fits at a time. Switching models means
  reloading Lemonade, not editing this file.
- `disable_thinking` — sends llama-server's `chat_template_kwargs:
  {"enable_thinking": false}`. Required for Qwen3.5 reasoning models, which
  otherwise spend every token on `reasoning_content` and return empty content.
  Gemma ignores the variable (verified both with and without), so leave it
  `true`: that default works for either model.
- `batch_size` / `max_batch_chars` — batches are cut at whichever limit is hit
  first. Larger batches mean fewer requests but weaker per-cue attention.

## How it works

**Batched requests.** Consecutive cues go out as one request containing
`{"1": "line", "2": "line", ...}` and come back keyed the same way, which
preserves cross-cue context and cuts request count.

**Markup protection.** Tags are swapped for `<tN>` placeholders before the
request and restored afterwards, so preservation doesn't depend on the model
obeying instructions. Hallucinated placeholder residue is stripped, and tags
the model dropped are re-inserted at their original position when that
position is unambiguous. A closing tag whose opening partner the model dropped
is removed too — `Дучезата</i>` with nothing open is junk — while an opener
without a closer stays (unclosed `<i>` means "styled until end of cue").
Repairs on an accepted reply run in a fixed order: line re-wrapping first,
markup restoration second, so a reply that is both merged *and* markup-damaged
still gets its line breaks rebalanced.

**Reply validation.** Structural checks alone are not enough — a reply can be
perfectly well-formed and entirely untranslated, because small models sometimes
echo their input back verbatim. For a non-Latin target (`ru`, `bg`, `el`, `ar`,
`zh`, …) the reply must contain a character of that script. For a Latin target
(`fr`, `de`, `es`, …) there is no script to check, so instead the reply must
say something the source did not — an English echo reuses only source words.
Short cues like `- Cory?` or `2019` are exempt either way.

**Fallback ladder.** A cue is rejected from a batch if it is missing, dropped
a placeholder, changed line count, came back untranslated, or lost a number
from the source — then retried individually. The individual retry is accepted
even if imperfect, because a translation with a warning beats English text.
Anything still untranslated is kept as-is and reported.

**Structural repairs.** Two accepted-reply failures are repaired rather than
merely reported. A source cue broken over two lines that comes back as one
long line is re-split at whitespace to the source's line count (`rewrap_lines`),
so the on-screen layout survives; the movie sample went from 70 merged cues to
0 this way. A reply that drops a number present in the source — `Not 10.1. Not
10.2. Mach 10.` answered without ever mentioning 10.1 or 10.2 — is treated as
a content error and retried. Numbers are compared after normalising `5,000` to
`5000` and `7,5` to `7.5`, so only real drops are flagged.

**Reply normalisation.** Model text is normalised to LF and trailing newlines
are stripped before validation. A terminator is not a subtitle line: counting
one lets a merged reply pass the line count check and then serialise as text
followed by an invisible blank line.

**Malformed JSON repairs.** Five known small-model failure patterns are
repaired before or during parsing: raw newlines in strings, naive brace
extraction, doubled key quotes, illegal escape sequences (rewritten as line
breaks), and curly quotes used to close a string. Each repair is guarded so
it only ever runs on a reply that already failed to parse. A sixth pattern —
a backslash before the model's own closing quote — is deliberately *not*
repaired: it is syntactically identical to a correct escaped quote, so any
rule that "fixes" it risks parsing with mangled text. The retry ladder
handles it instead.

## Quality report

Every run ends with a summary, because "did not fail" and "is actually good"
are different questions:

```
--- quality ---
  translated          : 114
  unchanged (no-op)   : 1  -> 92
  kept in English     : 2  -> 391, 1284
  line structure drift: 1  -> 54
  line breaks rebalanced: 67 -> 13, 63, 106 +64
  numbers dropped      : 2  -> 664, 1135
  markup repaired     : 15 -> 1, 2, 3, 4, 25, 26, 27, 28 +7
  left in English     : 1
wrote out.srt (1441 cue(s), 7 untranslated)
```

Cue numbers are 1-based, matching the SRT. Failures alone understate the
picture — most degraded cues are *accepted*, not rejected, so they only show
up here.

`translated` counts only cues that actually became a translation: cues with no
surviving reply, cues the model echoed back, and byte-identical no-ops such as
`Mach 10!` or `2019` are all excluded. Without that subtraction the headline
count is inflated by cues that were never translated.

## Limitations

- Quality is bounded by the model. The pipeline repairs **structure**; it
  cannot make a weak model fluent.
- Coverage and line drift are structural measures and say nothing about
  idiomatic phrasing. Spot-check output before shipping it.
- Timestamps, cue order and indices are preserved byte-for-byte; original
  line endings (CRLF/LF) and BOM handling are also preserved.
