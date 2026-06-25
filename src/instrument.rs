//! Hot-path instrumentation that compiles to nothing unless the `tracing` feature
//! is enabled.
//!
//! The core imposes no runtime dependency: with the feature off these
//! macros expand to an empty block, so the selection / posting-read / counting /
//! hydration call sites read the same whether or not a host wires up `tracing`.

/// Emit a `tracing::debug!` event when the `tracing` feature is enabled; a no-op
/// otherwise. Arguments are not evaluated when the feature is off.
macro_rules! trace_debug {
    ($($arg:tt)*) => {{
        #[cfg(feature = "tracing")]
        {
            ::tracing::debug!($($arg)*);
        }
    }};
}

pub(crate) use trace_debug;
