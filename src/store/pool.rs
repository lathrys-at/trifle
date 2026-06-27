//! The connection machinery both backends share: a single mutexed writer plus a
//! pool of read-only connections opened on demand.

use std::sync::{Mutex, PoisonError};

use rusqlite::Connection;

use crate::error::Result;

use super::Namespace;

/// Opens a fully-initialized read connection (WAL, mmap, `carray` registered) on demand.
type ConnFactory = Box<dyn Fn() -> Result<Connection> + Send + Sync>;

/// The connection state behind [`Sidecar`](super::Sidecar): the one write connection (serialized
/// behind a `Mutex`) plus a read pool opened on demand. `Sidecar` is a thin wrapper that delegates
/// its `write`/`read`/`namespace` to here.
pub(crate) struct Store {
    namespace: Namespace,
    write: Mutex<Connection>,
    pool: ReadPool,
}

impl Store {
    pub(crate) fn new(namespace: Namespace, write: Connection, read_factory: ConnFactory) -> Self {
        Store {
            namespace,
            write: Mutex::new(write),
            pool: ReadPool {
                factory: read_factory,
                idle: Mutex::new(Vec::new()),
            },
        }
    }

    /// The exclusive writer. A poisoned lock (a prior panic mid-write) is recovered
    /// rather than propagated: the panicked write's transaction rolled back on
    /// unwind, so the connection is still consistent and usable.
    pub(crate) fn write(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.write.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A pooled read-only connection.
    pub(crate) fn read(&self) -> Result<ReadConn<'_>> {
        self.pool.checkout()
    }

    pub(crate) fn namespace(&self) -> &Namespace {
        &self.namespace
    }
}

/// A pool of read-only connections opened on demand and returned on guard drop.
/// It self-bounds at the caller's live read concurrency (at most one checkout per
/// thread), so it grows to that width and no further without an explicit cap.
struct ReadPool {
    factory: ConnFactory,
    /// Idle connections available for checkout. A connection lives on exactly one
    /// of: this vector, or a live [`ReadConn`].
    idle: Mutex<Vec<Connection>>,
}

impl ReadPool {
    fn checkout(&self) -> Result<ReadConn<'_>> {
        let popped = self
            .idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop();
        let conn = match popped {
            Some(conn) => conn,
            None => (self.factory)()?,
        };
        Ok(ReadConn {
            pool: self,
            conn: Some(conn),
        })
    }

    fn checkin(&self, conn: Connection) {
        // Defensive rollback on check-in (not only on `CandidateStream::Drop`): a stream that
        // leaked its read transaction — `mem::forget`, `panic = "abort"`, or a double-panic that
        // bypasses `Drop` — would otherwise hand the next checkout a connection still pinning an
        // open WAL snapshot. Returning it to the pool clean keeps the snapshot lifetime bounded by
        // the checkout, not by some past leak (PROPOSAL §6 pool check-in rollback).
        if !conn.is_autocommit() {
            let _ = conn.execute_batch("ROLLBACK");
        }
        self.idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(conn);
    }
}

/// A borrowed read-only connection that returns itself to its pool on drop.
///
/// Handed out by the read pool ([`Sidecar`](super::Sidecar)'s reads); `Deref`s to the underlying
/// [`Connection`] and is held only for the duration of one read (or one [`CandidateStream`]).
///
/// [`CandidateStream`]: crate::CandidateStream
pub struct ReadConn<'p> {
    pool: &'p ReadPool,
    /// `Some` until drop; `take`n in `Drop` to hand the connection back.
    conn: Option<Connection>,
}

impl Drop for ReadConn<'_> {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            self.pool.checkin(conn);
        }
    }
}

impl std::ops::Deref for ReadConn<'_> {
    type Target = Connection;
    fn deref(&self) -> &Connection {
        self.conn.as_ref().expect("read guard connection taken")
    }
}
