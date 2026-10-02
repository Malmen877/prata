//! Model suggestion for long recordings (GET /api/model-hint).
//!
//! Rule: when the audio is longer than `LONG_SECS` and the Snabb engine is
//! available, suggest Snabb (Klang Pianissimo), which is much faster than the
//! KB-Whisper models on long files. Otherwise no suggestion. The UI decides how
//! (and whether) to show it.

use serde::Serialize;

/// Recordings longer than this (seconds) get the Snabb suggestion.
pub const LONG_SECS: f64 = 15.0 * 60.0;

/// Reason text shown with the suggestion (exact string, agreed with the UI).
pub const REASON: &str = "Lång inspelning – Snabb går betydligt fortare";

#[derive(Debug, Serialize, PartialEq)]
pub struct Hint {
    /// Suggested model id, or `None`
    pub suggest: Option<&'static str>,
    /// Short Swedish explanation (empty without a suggestion)
    pub reason: String,
}

pub fn hint(duration_s: f64, current: &str, snabb_available: bool) -> Hint {
    let none = Hint { suggest: None, reason: String::new() };
    if !snabb_available || !duration_s.is_finite() || duration_s <= LONG_SECS || crate::is_snabb(current) {
        return none;
    }
    Hint {
        suggest: Some("snabb"),
        reason: REASON.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suggests_snabb_only_for_long_audio_when_available() {
        assert_eq!(hint(600.0, "small", true).suggest, None);
        assert_eq!(hint(900.0, "small", true).suggest, None);
        let h = hint(1800.0, "small", true);
        assert_eq!(h.suggest, Some("snabb"));
        assert_eq!(h.reason, "Lång inspelning – Snabb går betydligt fortare");
        assert_eq!(hint(1800.0, "snabb", true).suggest, None);
        assert_eq!(hint(1800.0, "large", false).suggest, None);
        assert_eq!(hint(f64::NAN, "small", true).suggest, None);
    }
}
