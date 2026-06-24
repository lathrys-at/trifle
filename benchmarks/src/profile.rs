//! The work-done instrument: Σ(kept-posting cardinality) per query.
//!
//! trifle aims for flat latency as the corpus grows — bit-sliced overlap is
//! posting-size-independent. The quantity that *would* break that if it grew is the
//! total cardinality of
//! the postings fed to the counter for a query. trifle already emits it on its
//! hot-path `tracing` event (`sum_cardinality`, behind the `tracing` feature); this
//! is a minimal in-process [`Subscriber`] that captures that field, one sample per
//! [`search`](trifle::Index::search), so the harness can correlate the work-done
//! distribution with the latency distribution.
//!
//! It is installed only around the untimed `profile` pass (via
//! [`dispatcher::with_default`](tracing::dispatcher::with_default)), so the timed
//! latency runs pay nothing for it.

use std::sync::{Arc, Mutex};

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Dispatch, Event, Metadata, Subscriber};

/// The shared sink the collector writes into; cloned out before the collector is
/// moved into a [`Dispatch`], so samples survive `with_default`.
type Sink = Arc<Mutex<Vec<u64>>>;

/// Captures `sum_cardinality` from trifle's per-query candidate-generation event.
struct Collector {
    sink: Sink,
}

/// Pulls `sum_cardinality` out of an event's fields, ignoring everything else.
struct SumVisitor {
    value: Option<u64>,
}

impl Visit for SumVisitor {
    fn record_u64(&mut self, field: &Field, value: u64) {
        if field.name() == "sum_cardinality" {
            self.value = Some(value);
        }
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        if field.name() == "sum_cardinality" && value >= 0 {
            self.value = Some(value as u64);
        }
    }
    // Required; everything not numeric is irrelevant here.
    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
}

impl Subscriber for Collector {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        // No span tree is needed; a constant non-zero id satisfies the contract.
        Id::from_u64(1)
    }
    fn record(&self, _span: &Id, _values: &Record<'_>) {}
    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}
    fn event(&self, event: &Event<'_>) {
        let mut v = SumVisitor { value: None };
        event.record(&mut v);
        if let Some(card) = v.value {
            self.sink.lock().expect("sink not poisoned").push(card);
        }
    }
    fn enter(&self, _span: &Id) {}
    fn exit(&self, _span: &Id) {}
}

/// Run `f` with the collector installed on the current thread, returning the
/// Σ-cardinality samples it captured (one per query, in call order).
pub fn capture<T>(f: impl FnOnce() -> T) -> (T, Vec<u64>) {
    let sink: Sink = Arc::new(Mutex::new(Vec::new()));
    let dispatch = Dispatch::new(Collector { sink: sink.clone() });
    let out = tracing::dispatcher::with_default(&dispatch, f);
    let samples = std::mem::take(&mut *sink.lock().expect("sink not poisoned"));
    (out, samples)
}
