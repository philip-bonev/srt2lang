# srt2bg

Translate English SRT subtitle files into Bulgarian with a local LLM.

`movie.srt` → `movie.bg.srt`, with cue indices, timestamps, line structure and
markup tags (`<i>`, `<b>`, `{\an8}`) preserved exactly.

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

---

## Build

Rust lives in `~/.cargo/bin`, which is not on the default `PATH` for
non-interactive shells:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
```

```sh
cargo build -r          # release build -> target/release/srt2bg
cargo test              # unit tests
cargo clippy --all-targets
```

Use the **release** binary for real runs; the debug build is several times
slower and will skew any timing.

## Usage

```sh
./target/release/srt2bg movie.srt                 # writes movie.bg.srt
./target/release/srt2bg movie.srt -o out.srt      # explicit output path
./target/release/srt2bg movie.srt --dry-run       # show batches, send nothing
./target/release/srt2bg --help
```

Progress and warnings go to **stderr**; the translated SRT may be written to
stdout, so redirect stderr if you want clean logs:

```sh
./target/release/srt2bg movie.srt 2>run.log
```

## Configuration

JSON, discovered in this order:

1. `--config <path>`
2. `./srt2bg.json`
3. `~/.config/srt2bg/config.json`
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
  "disable_thinking": true
}
```

Notes:

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
position is unambiguous.

**Reply validation.** Structural checks alone are not enough — a reply can be
perfectly well-formed and entirely untranslated. Any reply to a multi-word
English cue must contain Cyrillic. Short cues like `- Cory?` or `2019` are
exempt, since those are correctly left in Latin script.

**Fallback ladder.** A cue is rejected from a batch if it is missing, dropped
a placeholder, changed line count, or came back without Bulgarian — then
retried individually. The individual retry is accepted even if imperfect,
because a translation with a warning beats English text. Anything still
untranslated is kept as-is and reported.

**Malformed JSON repairs.** Five known small-model failure patterns are
repaired before or during parsing: raw newlines in strings, naive brace
extraction, doubled key quotes, illegal escape sequences (rewritten as line
breaks), and curly quotes used to close a string. Each repair is guarded so
it only ever runs on a reply that already failed to parse.

## Quality report

Every run ends with a summary, because "did not fail" and "is actually good"
are different questions:

```
--- quality ---
  translated          : 119
  unchanged (no-op)   : 1  -> 92
  line structure drift: 2  -> 54, 106
  markup repaired     : 15 -> 1, 2, 3, 4, 25, 26, 27, 28 +7
  left in English     : 1
wrote out.srt (120 cue(s), 1 untranslated)
```

Cue numbers are 1-based, matching the SRT. Failures alone understate the
picture — most degraded cues are *accepted*, not rejected, so they only show
up here.

## Limitations

- Quality is bounded by the model. The pipeline repairs **structure**; it
  cannot make a weak model fluent.
- Coverage and line drift are structural measures and say nothing about
  idiomatic phrasing. Spot-check output before shipping it.
- Timestamps, cue order and indices are preserved byte-for-byte; original
  line endings (CRLF/LF) and BOM handling are also preserved.
