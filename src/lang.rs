//! Target language: an ISO 639 code resolved to a name and a writing script.
//!
//! Two jobs, one lookup:
//!
//! - **The prompt** must say "into French", not "into `fr`" — a 4B model
//!   follows the name far more reliably than a bare code.
//! - **Reply validation** must check the right thing. A reply to a Bulgarian
//!   or Russian cue must contain Cyrillic; a reply to a French cue cannot,
//!   so for Latin-script targets validation falls back to detecting a verbatim
//!   echo of the source. See `Lang::reply_in_target_script`.
//!
//! `iso639` supplies both the English name and the language's ISO 15924 script
//! from CLDR, so there is no table here to fall out of date.

use iso639::{autonym, script};

/// A validated target language.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lang {
    /// The tag as given, normalised to lowercase (`fr`, `bg`, `zho`, …).
    pub code: String,
    /// English name for the prompt: `French`, `Bulgarian`.
    pub name: &'static str,
    /// ISO 15924 script of the language (`Latn`, `Cyrl`, `Jpan`, …) when the
    /// database has one; `None` for the many languages CLDR does not cover.
    pub script: Option<&'static str>,
}

impl Lang {
    /// Resolve an ISO 639-1 (or -3) code such as `fr`, `bg`, `ru`.
    pub fn resolve(code: &str) -> Result<Lang, String> {
        let tag = code.trim().to_ascii_lowercase();
        if tag.is_empty() {
            return Err("target language code must not be empty".to_string());
        }

        let record = autonym::get(&tag).ok_or_else(|| {
            format!("unknown language code '{tag}' — pass an ISO 639-1 code such as 'fr', 'bg' or 'ru'")
        })?;

        // Some languages are only listed under their 3-letter code in the
        // script database, so try both keys before giving up.
        let script = script::get(&tag)
            .or_else(|| script::get(record.tag3))
            .map(|entry| entry.script);

        Ok(Lang {
            code: tag,
            name: record.name,
            script,
        })
    }

    /// `Some(true)` when `reply` contains characters of the target script,
    /// `Some(false)` when it does not, and `None` when the script is either
    /// Latin — which has no characters of its own to look for — or unknown to
    /// the database.
    ///
    /// Callers treat `None` as "no script opinion", and fall back to echo
    /// detection rather than guessing. Guessing in the other direction would
    /// be the dangerous one: a language wrongly assumed non-Latin would flag
    /// every correct reply as untranslated.
    pub fn reply_in_target_script(&self, reply: &str) -> Option<bool> {
        let script = self.script?;
        Some(match script {
            "Cyrl" => in_ranges(reply, CYRILLIC),
            "Grek" => in_ranges(reply, GREEK),
            "Arab" => in_ranges(reply, ARABIC),
            "Hebr" => in_ranges(reply, HEBREW),
            "Thai" => in_ranges(reply, THAI),
            "Deva" => in_ranges(reply, DEVANAGARI),
            "Hans" | "Hant" | "Hani" => in_ranges(reply, HAN),
            // Japanese is commonly written with kana alone, so accept either —
            // a pure-kanji subtitle line is still Japanese enough for us.
            "Jpan" => in_ranges(reply, KANA) || in_ranges(reply, HAN),
            "Kore" => in_ranges(reply, HANGUL),
            _ => return None,
        })
    }
}

fn in_ranges(text: &str, ranges: &[(u32, u32)]) -> bool {
    text.chars()
        .any(|c| ranges.iter().any(|&(lo, hi)| (c as u32) >= lo && (c as u32) <= hi))
}

const CYRILLIC: &[(u32, u32)] = &[(0x0400, 0x04FF), (0x0500, 0x052F)];
const GREEK: &[(u32, u32)] = &[(0x0370, 0x03FF), (0x1F00, 0x1FFF)];
const ARABIC: &[(u32, u32)] = &[
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
];
const HEBREW: &[(u32, u32)] = &[(0x0590, 0x05FF), (0xFB1D, 0xFB4F)];
const THAI: &[(u32, u32)] = &[(0x0E00, 0x0E7F)];
const DEVANAGARI: &[(u32, u32)] = &[(0x0900, 0x097F)];
const HAN: &[(u32, u32)] = &[(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)];
const KANA: &[(u32, u32)] = &[(0x3040, 0x30FF), (0x31F0, 0x31FF)];
const HANGUL: &[(u32, u32)] = &[(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_known_codes_to_english_names() {
        assert_eq!(Lang::resolve("fr").unwrap().name, "French");
        assert_eq!(Lang::resolve("bg").unwrap().name, "Bulgarian");
        assert_eq!(Lang::resolve("ru").unwrap().name, "Russian");
    }

    #[test]
    fn accepts_uppercase_and_whitespace_and_three_letter_codes() {
        assert_eq!(Lang::resolve(" FR ").unwrap().code, "fr");
        assert_eq!(Lang::resolve("fra").unwrap().name, "French");
    }

    #[test]
    fn rejects_unknown_and_empty_codes() {
        assert!(Lang::resolve("zz").is_err());
        assert!(Lang::resolve("").is_err());
        assert!(Lang::resolve("   ").is_err());
        let err = Lang::resolve("zz").unwrap_err();
        assert!(err.contains("'zz'"), "{err}");
    }

    #[test]
    fn resolves_scripts_that_the_database_covers() {
        assert_eq!(Lang::resolve("ru").unwrap().script, Some("Cyrl"));
        assert_eq!(Lang::resolve("bg").unwrap().script, Some("Cyrl"));
        assert_eq!(Lang::resolve("fr").unwrap().script, Some("Latn"));
        assert_eq!(Lang::resolve("el").unwrap().script, Some("Grek"));
        assert_eq!(Lang::resolve("ja").unwrap().script, Some("Jpan"));
    }

    #[test]
    fn script_check_passes_real_translations_and_fails_english() {
        let ru = Lang::resolve("ru").unwrap();
        assert_eq!(ru.reply_in_target_script("Привет, мир!"), Some(true));
        assert_eq!(ru.reply_in_target_script("Hello, world!"), Some(false));

        let ja = Lang::resolve("ja").unwrap();
        // Kana alone or kanji alone both count.
        assert_eq!(ja.reply_in_target_script("こんにちは"), Some(true));
        assert_eq!(ja.reply_in_target_script("今日は"), Some(true));
        assert_eq!(ja.reply_in_target_script("Hello"), Some(false));
    }

    #[test]
    fn latin_targets_have_no_script_opinion() {
        // French is Latin-script: there is nothing distinctive to look for, so
        // callers must fall back to echo detection instead of flagging every
        // correct reply — which is exactly what a hardcoded Cyrillic test did.
        assert_eq!(Lang::resolve("fr").unwrap().reply_in_target_script("Bonjour!"), None);
        assert_eq!(Lang::resolve("de").unwrap().reply_in_target_script("Hallo!"), None);
    }

    #[test]
    fn unknown_scripts_degrade_to_no_opinion() {
        // Chinese has no row in CLDR's script table: safe fallback, never a
        // wrong "this reply is not in the target script".
        let zh = Lang::resolve("zh").unwrap();
        assert!(zh.reply_in_target_script("你好").is_none());
    }
}
