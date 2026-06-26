//! The opt-in backend: trifle's tables live, namespaced, inside a database the
//! caller owns and supplies connections to.

use rusqlite::Connection;

use crate::error::Result;

use super::pool::{ReadConn, Store};
use super::{Backend, Namespace, configure_sqlite_perf, register_carray};

/// A backend over a database the caller owns. trifle's tables live [namespaced](Namespace)
/// inside the caller's file; the caller supplies the write connection and a factory
/// for read-only connections.
///
/// Use only for a hard co-location requirement. In this mode the caller takes on:
/// single-writer serialization across the whole file, compatible WAL/pragma setup,
/// and not holding a transaction across trifle's [`rebuild`](crate::Index::rebuild)
/// swap. trifle touches only its own namespaced tables.
pub struct Shared {
    store: Store,
}

impl Shared {
    /// Wrap a caller-owned database.
    ///
    /// `write` is the single connection trifle serializes all writes through;
    /// `read_factory` opens read-only connections to the same database on demand.
    /// trifle registers its `carray` vtab on every connection but otherwise leaves
    /// the caller's WAL/pragma configuration untouched.
    ///
    /// # Errors
    ///
    /// Returns an error if the `carray` vtab cannot be registered on the writer.
    pub fn new<F>(namespace: Namespace, write: Connection, read_factory: F) -> Result<Self>
    where
        F: Fn() -> rusqlite::Result<Connection> + Send + Sync + 'static,
    {
        configure_sqlite_perf();
        register_carray(&write)?;
        let factory = Box::new(move || -> Result<Connection> {
            let conn = read_factory()?;
            register_carray(&conn)?;
            Ok(conn)
        });
        Ok(Shared {
            store: Store::new(namespace, write, factory),
        })
    }
}

impl Backend for Shared {
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

    fn init_conn(&self, conn: &Connection) -> Result<()> {
        register_carray(conn)
    }
}
