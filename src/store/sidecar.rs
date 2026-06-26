//! The default backend: trifle opens and owns its own SQLite file.

use std::path::Path;
use std::time::Duration;

use rusqlite::{Connection, OpenFlags};

use crate::error::{Error, Result};

use super::pool::{ReadConn, Store};
use super::{Backend, Namespace, configure_sqlite_perf, register_carray};

/// Per-connection memory-map ceiling. 1 GiB maps the whole store at every standard
/// scale and stays under the 64-bit `mmap_size` cap, so reads serve pages straight
/// from the mapped file and bypass the page cache.
const MMAP_SIZE_BYTES: i64 = 1024 * 1024 * 1024;

/// A read or write that overlaps another connection's commit waits the lock out
/// rather than taking an instant `SQLITE_BUSY`.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// A backend that owns its own SQLite file: one write connection (WAL,
/// `synchronous=NORMAL`) plus a pool of read-only connections that, under WAL, run
/// concurrently with the writer. Full encapsulation — the caller passes a path.
///
/// This is the recommended backend, and the right choice whenever the source of
/// truth is foreign, remote, or expensive to reach.
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
}

impl Backend for Sidecar {
    type WriteGuard<'a> = std::sync::MutexGuard<'a, Connection>;
    type ReadGuard<'a> = ReadConn<'a>;

    fn write(&self) -> Result<Self::WriteGuard<'_>> {
        Ok(self.store.write())
    }

    fn read(&self) -> Result<Self::ReadGuard<'_>> {
        self.store.read()
    }

    fn namespace(&self) -> &Namespace {
        self.store.namespace()
    }
}

/// Per-connection setup shared by read connections (and the non-WAL part of the
/// writer): wait budget, memory-map, and the `carray` vtab.
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
