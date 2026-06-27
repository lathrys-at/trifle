//! The store: trifle opens and owns its own SQLite file.

use std::path::Path;
use std::time::Duration;

use rusqlite::{Connection, OpenFlags};

use crate::error::{Error, Result};

use super::pool::{ReadConn, Store};
use super::{Namespace, configure_sqlite_perf, register_carray};

/// Per-connection memory-map ceiling. 1 GiB maps the whole store at every standard
/// scale and stays under the 64-bit `mmap_size` cap, so reads serve pages straight
/// from the mapped file and bypass the page cache.
const MMAP_SIZE_BYTES: i64 = 1024 * 1024 * 1024;

/// `busy_timeout = 0`: lock contention returns `SQLITE_BUSY` **immediately** rather than
/// blocking the calling thread to wait the lock out. Library code must never sleep/block the
/// caller; the fault is mapped to the retryable [`Error::Busy`](crate::Error::Busy) at the error
/// boundary and the application owns the backoff/retry.
const BUSY_TIMEOUT: Duration = Duration::ZERO;

/// trifle's owned SQLite file: one write connection (WAL, `synchronous=NORMAL`) plus a pool of
/// read-only connections that, under WAL, run concurrently with the writer. The caller passes a
/// path; trifle owns everything else.
///
/// This is the only store (rev v0.3 dropped the `Backend` trait and the `Shared` backend). An
/// [`Index`](crate::Index) holds one `Sidecar` and reads/writes through it.
pub struct Sidecar {
    store: Store,
}

impl Sidecar {
    /// Open (creating if absent) a sidecar at `path` with bare, unprefixed table
    /// names. trifle owns the file, so there are no neighbors to collide with.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or WAL cannot be enabled.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Sidecar::open_with_namespace(path, Namespace::bare())
    }

    /// Open a sidecar at `path` with an explicit [`Namespace`] — useful only if
    /// several independent trifle indexes must share one file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or WAL cannot be enabled.
    pub fn open_with_namespace(path: impl AsRef<Path>, namespace: Namespace) -> Result<Self> {
        configure_sqlite_perf();
        let path = path.as_ref().to_path_buf();

        let write = Connection::open(&path)?;
        setup_write_conn(&write)?;

        let read_path = path.clone();
        let factory = Box::new(move || -> Result<Connection> {
            configure_sqlite_perf();
            let conn = Connection::open_with_flags(
                &read_path,
                OpenFlags::SQLITE_OPEN_READ_ONLY
                    | OpenFlags::SQLITE_OPEN_URI
                    | OpenFlags::SQLITE_OPEN_NO_MUTEX,
            )?;
            setup_read_conn(&conn)?;
            Ok(conn)
        });

        Ok(Sidecar {
            store: Store::new(namespace, write, factory),
        })
    }

    /// Acquire the exclusive write connection. Blocks until the writer is free.
    pub(crate) fn write(&self) -> Result<std::sync::MutexGuard<'_, Connection>> {
        Ok(self.store.write())
    }

    /// Acquire a pooled read-only connection (returned to the pool on drop).
    pub(crate) fn read(&self) -> Result<ReadConn<'_>> {
        self.store.read()
    }

    /// The table-naming namespace for this store.
    pub(crate) fn namespace(&self) -> &Namespace {
        self.store.namespace()
    }
}

/// Per-connection setup shared by read connections (and the non-WAL part of the
/// writer): the no-wait busy policy, memory-map, and the `carray` vtab.
fn setup_read_conn(conn: &Connection) -> Result<()> {
    conn.busy_timeout(BUSY_TIMEOUT)?;
    conn.pragma_update(None, "mmap_size", MMAP_SIZE_BYTES)?;
    register_carray(conn)?;
    Ok(())
}

/// Write-connection setup: WAL + `synchronous=NORMAL` on top of the per-connection
/// essentials. WAL is what lets pooled reads run concurrently with the single
/// writer; it is persistent in the file header, so later read connections inherit
/// it. `NORMAL` may lose the last transaction(s) on power loss (never integrity),
/// which a rebuildable cache absorbs.
fn setup_write_conn(conn: &Connection) -> Result<()> {
    let mode: String = conn.query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0))?;
    if !mode.eq_ignore_ascii_case("wal") {
        return Err(Error::corrupt(format!(
            "could not enter WAL mode (got {mode:?})"
        )));
    }
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    setup_read_conn(conn)?;
    Ok(())
}
