//! Utility functions for timing and debugging.

use std::time::Instant;

/// RAII timer that logs elapsed time on drop.
///
/// # Example
/// ```ignore
/// let _t = Timed::new("Tessellation");
/// // ... do work ...
/// // logs "Tessellation: 1.234s" when _t is dropped
/// ```
pub struct Timed {
    name: &'static str,
    start: Instant,
    level: log::Level,
}

impl Timed {
    /// Create a new timer that logs at INFO level.
    pub fn info(name: &'static str) -> Self {
        log::debug!("{}...", name);
        Self {
            name,
            start: Instant::now(),
            level: log::Level::Info,
        }
    }

    /// Create a new timer that logs at DEBUG level.
    pub fn debug(name: &'static str) -> Self {
        log::trace!("{}...", name);
        Self {
            name,
            start: Instant::now(),
            level: log::Level::Debug,
        }
    }
}

impl Drop for Timed {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        log::log!(self.level, "{}: {:.3?}", self.name, elapsed);
    }
}
