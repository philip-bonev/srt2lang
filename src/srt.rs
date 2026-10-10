//! SRT parsing and serialization.
//!
//! Pure functions only — no I/O, no network — so everything here is
//! unit-testable. Timestamps, cue indices and markup tags are never touched:
//! only the text lines of a cue are handed out for translation.

/// One subtitle cue: index line, timing line, and the text lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cue {
    /// The index line exactly as read (synthesized when the file omits it).
    pub index: String,
    /// The `HH:MM:SS,mmm --> HH:MM:SS,mmm` line exactly as read.
    pub timing: String,
    /// Text lines of the cue (may be empty for timing-only cues).
    pub text: Vec<String>,
}

/// Line ending style of the source file, preserved on write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Newline {
    Lf,
    Crlf,
}

impl Newline {
    pub fn as_str(self) -> &'static str {
        match self {
            Newline::Lf => "\n",
            Newline::Crlf => "\r\n",
        }
    }
}

/// A parsed subtitle file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Srt {
    pub cues: Vec<Cue>,
    pub newline: Newline,
    /// Number of blocks skipped because they had no timing line.
    pub skipped: usize,
}

/// Parse SRT content. Tolerant of a UTF-8 BOM, mixed line endings, and
/// blocks without a timing line (those are skipped and counted).
pub fn parse(input: &str) -> Result<Srt, String> {
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);

    let newline = if input.contains("\r\n") {
        Newline::Crlf
    } else {
        Newline::Lf
    };

    let mut cues = Vec::new();
    let mut skipped = 0usize;
    let mut block: Vec<&str> = Vec::new();
    // Sequential fallback for files that omit index lines.
    let mut next_index = 1usize;

    let flush = |block: &mut Vec<&str>,
                 cues: &mut Vec<Cue>,
                 skipped: &mut usize,
                 next_index: &mut usize| {
        if block.iter().all(|l| l.trim().is_empty()) {
            block.clear();
            return;
        }
        // Locate a valid timing line: either first or second line of the block.
        let timing_pos = block
            .iter()
            .position(|l| is_timing_line(l))
            .filter(|&p| p <= 1);
        let Some(timing_pos) = timing_pos else {
            *skipped += 1;
            block.clear();
            return;
        };
        let index = if timing_pos == 0 {
            *next_index += 1;
            (*next_index - 1).to_string()
        } else {
            let index = block[0].to_string();
            if let Ok(explicit) = index.trim().parse::<usize>() {
                *next_index = explicit.saturating_add(1);
            }
            index
        };
        let timing = block[timing_pos].to_string();
        let text: Vec<String> = block[timing_pos + 1..]
            .iter()
            .map(|l| (*l).to_string())
            .collect();
        cues.push(Cue {
            index,
            timing,
            text,
        });
        block.clear();
    };

    for line in input.lines() {
        if line.trim().is_empty() {
            flush(&mut block, &mut cues, &mut skipped, &mut next_index);
        } else {
            // `.lines()` already strips the `\r` of CRLF files.
            block.push(line);
        }
    }
    flush(&mut block, &mut cues, &mut skipped, &mut next_index);

    if cues.is_empty() {
        return Err("no subtitle cues found (file is empty or not an SRT)".to_string());
    }

    Ok(Srt {
        cues,
        newline,
        skipped,
    })
}

/// Accept SRT timing lines only. A literal `-->` in dialogue must not turn a
/// malformed block into a cue.
fn is_timing_line(line: &str) -> bool {
    let Some((start, rest)) = line.split_once("-->") else {
        return false;
    };
    let end = rest.split_whitespace().next().unwrap_or_default();
    is_timestamp(start.trim()) && is_timestamp(end)
}

fn is_timestamp(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 12
        && bytes[2] == b':'
        && bytes[5] == b':'
        && bytes[8] == b','
        && bytes
            .iter()
            .enumerate()
            .all(|(i, byte)| matches!(i, 2 | 5 | 8) || byte.is_ascii_digit())
}

/// Serialize cues back to SRT text using the file's original line endings.
/// Never emits a BOM.
pub fn serialize(srt: &Srt) -> String {
    let nl = srt.newline.as_str();
    let mut out = String::new();
    for cue in &srt.cues {
        out.push_str(&cue.index);
        out.push_str(nl);
        out.push_str(&cue.timing);
        out.push_str(nl);
        for line in &cue.text {
            out.push_str(line);
            out.push_str(nl);
        }
        out.push_str(nl);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "1\n00:00:01,000 --> 00:00:03,000\nHello there!\n\n2\n00:00:04,000 --> 00:00:06,000\nHow are you?\nFine, thanks.\n\n";

    #[test]
    fn parses_basic_file() {
        let srt = parse(SAMPLE).unwrap();
        assert_eq!(srt.cues.len(), 2);
        assert_eq!(srt.cues[0].index, "1");
        assert_eq!(srt.cues[0].timing, "00:00:01,000 --> 00:00:03,000");
        assert_eq!(srt.cues[0].text, vec!["Hello there!"]);
        assert_eq!(srt.cues[1].text, vec!["How are you?", "Fine, thanks."]);
        assert_eq!(srt.newline, Newline::Lf);
        assert_eq!(srt.skipped, 0);
    }

    #[test]
    fn preserves_crlf_and_bom() {
        let src = "\u{feff}1\r\n00:00:01,000 --> 00:00:02,000\r\nHi\r\n\r\n";
        let srt = parse(src).unwrap();
        assert_eq!(srt.newline, Newline::Crlf);
        assert_eq!(srt.cues[0].text, vec!["Hi"]);
        let out = serialize(&srt);
        assert!(!out.contains('\u{feff}'), "BOM must not be re-emitted");
        assert!(out.contains("\r\n"));
        assert!(
            !out.replace("\r\n", "").contains('\n'),
            "mixed endings not allowed"
        );
    }

    #[test]
    fn keeps_markup_and_timestamps_verbatim() {
        let src = "7\n01:02:03,004 --> 01:02:04,005\n<i>Render this</i>\n{\\an8}Top line\n\n";
        let srt = parse(src).unwrap();
        assert_eq!(srt.cues[0].timing, "01:02:03,004 --> 01:02:04,005");
        assert_eq!(
            srt.cues[0].text,
            vec!["<i>Render this</i>", "{\\an8}Top line"]
        );
        assert_eq!(serialize(&srt), src);
    }

    #[test]
    fn skips_blocks_without_timing_line() {
        let src = "1\n00:00:01,000 --> 00:00:02,000\nReal cue\n\nGarbage without timing\n\n2\n00:00:03,000 --> 00:00:04,000\nSecond\n\n";
        let srt = parse(src).unwrap();
        assert_eq!(srt.cues.len(), 2);
        assert_eq!(srt.skipped, 1);
        assert_eq!(srt.cues[1].index, "2");
    }

    #[test]
    fn handles_missing_index_line() {
        let src = "00:00:01,000 --> 00:00:02,000\nNo index here\n\n";
        let srt = parse(src).unwrap();
        assert_eq!(srt.cues[0].index, "1");
        assert_eq!(srt.cues[0].text, vec!["No index here"]);
    }

    #[test]
    fn rejects_empty_input() {
        assert!(parse("").is_err());
        assert!(parse("\n\n").is_err());
        assert!(parse("garbage without timing\n\n").is_err());
        assert!(parse("1\na --> b\ntext\n\n").is_err());
    }

    #[test]
    fn rejects_non_timing_arrows_and_advances_synthetic_indices() {
        let src =
            "9\n00:00:01,000 --> 00:00:02,000\nFirst\n\n00:00:03,000 --> 00:00:04,000\nSecond\n\n";
        let srt = parse(src).unwrap();
        assert_eq!(srt.cues[0].index, "9");
        assert_eq!(srt.cues[1].index, "10");
        assert!(!is_timing_line("dialogue --> more dialogue"));
    }

    #[test]
    fn roundtrip_is_stable() {
        let srt = parse(SAMPLE).unwrap();
        assert_eq!(serialize(&srt), SAMPLE);
        let again = parse(&serialize(&srt)).unwrap();
        assert_eq!(again, srt);
    }
}
