//! The connection machinery both backends share: a single mutexed writer plus a
//! pool of read-only connections opened on demand.

use std::sync::{Mutex, PoisonError};

use rusqlite::Connection;

use crate::error::Result;

use super::Namespace;

/// Opens a fully-initialized connection (post-`init_conn`) on demand.
type ConnFactory = Box<dyn Fn() -> Result<Connection> + Send + Sync>;

/// The connection state common to both [`Sidecar`](super::Sidecar) and
/// [`Shared`](super::Shared): the one write connection (serialized behind a
/// `Mutex`) plus a read pool. Both backends are thin wrappers that delegate their
/// [`Backend`](super::Backend) methods here.
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
        self.idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(conn);
    }
}

/// A borrowed read-only connection that returns itself to its pool on drop.
///
/// Returned by [`Backend::read`](super::Backend::read); `Deref`s to the underlying
/// [`Connection`] and is held only for the duration of one read.
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
