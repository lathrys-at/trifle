//! The crate error type.
//!
//! Every fallible operation in trifle returns [`Result`]. The variants separate
//! the failure classes a caller reasons about differently: a transient or
//! environmental store fault ([`Error::Sqlite`], [`Error::Busy`]), bad input the caller
//! can fix ([`Error::InvalidInput`], [`Error::Namespace`], [`Error::Schema`]), an
//! internal invariant violation that should be impossible ([`Error::Corrupt`],
//! [`Error::Posting`]), and a [`Writer`](crate::Writer) handle that can no longer maintain
//! its transaction ([`Error::WriterStranded`]) — the store is intact, re-acquire the writer.

/// A specialized [`Result`](std::result::Result) for trifle operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Anything that can go wrong in trifle.
///
/// `#[non_exhaustive]` so new variants can be added without a breaking change;
/// match with a wildcard arm.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The backing SQLite store returned a **non-transient** error (a real fault, not lock
    /// contention) — e.g. I/O, `SQLITE_CANTOPEN`, or a logic error. Transient faults
    /// (`SQLITE_BUSY`/`SQLITE_LOCKED`, and the `SQLITE_SCHEMA` re-prepare) are mapped to the
    /// retryable [`Error::Busy`] instead (see the [`From`] impl below), because the library
    /// never blocks the caller's thread to retry — it surfaces a retryable signal and the caller
    /// owns the backoff (audit OD1).
    #[error("sqlite: {0}")]
    Sqlite(#[source] rusqlite::Error),

    /// A roaring posting failed to serialize or deserialize. This is an internal
    /// invariant violation (trifle wrote the bytes it is reading back), surfaced
    /// rather than panicked so a corrupt store degrades to an error.
    #[error("posting codec: {0}")]
    Posting(#[source] std::io::Error),

    /// Caller input that trifle rejects rather than silently coercing — for
    /// example a query or segment that cannot be processed as given.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// A [`Namespace`](crate::store::Namespace) was constructed with an invalid or
    /// colliding table name.
    #[error("invalid namespace: {0}")]
    Namespace(String),

    /// The declared [`Schema`](crate::Schema) is invalid — for example no key field, a
    /// duplicate or identifier-unsafe field name, or no text field to index.
    #[error("invalid schema: {0}")]
    Schema(String),

    /// The store is internally inconsistent in a way that cannot be repaired in
    /// place (for example a posting references a segment id with no row, beyond
    /// what a pending fold explains). The cache is rebuildable: a
    /// [`rebuild`](crate::Index::rebuild) restores a consistent store.
    #[error("index inconsistent: {0}")]
    Corrupt(String),

    /// A **transient** condition: SQLite lock contention (`SQLITE_BUSY`/`SQLITE_LOCKED`), a
    /// statement needing a re-prepare after a concurrent schema change (`SQLITE_SCHEMA`), or a
    /// read racing a concurrent [`rebuild`](crate::Index::rebuild)'s id-reassignment. The
    /// library does **not** block the caller's thread to retry internally (`busy_timeout` is 0);
    /// unlike [`Corrupt`](Error::Corrupt) the store is fine — **retry the operation on a fresh
    /// [`reader`](crate::Index::reader)** (and, for a write, re-submit the batch) and it will
    /// succeed.
    #[error("transient (retry): {0}")]
    Busy(String),

    /// A [`Writer`](crate::Writer) lease can no longer maintain its transaction and is
    /// unusable — for example a [`commit`](crate::Writer::commit) whose durable `COMMIT`
    /// succeeded but whose follow-on `BEGIN` failed, or a write whose savepoint rollback
    /// faulted. **The store is intact** (neither corrupt nor lost-needing-retry); drop this
    /// writer and acquire a fresh one. When it comes from `commit()`, the just-committed
    /// batch **is durable** — do not retry it (that would double-apply).
    #[error("writer stranded (re-acquire): {0}")]
    WriterStranded(String),
}

impl Error {
    /// Construct an [`Error::Namespace`] from anything string-like.
    pub(crate) fn namespace(msg: impl Into<String>) -> Self {
        Error::Namespace(msg.into())
    }

    /// Construct an [`Error::Corrupt`] from anything string-like.
    pub(crate) fn corrupt(msg: impl Into<String>) -> Self {
        Error::Corrupt(msg.into())
    }

    /// Construct an [`Error::Busy`] (transient; retry on a fresh reader) from a string.
    pub(crate) fn busy(msg: impl Into<String>) -> Self {
        Error::Busy(msg.into())
    }

    /// Construct an [`Error::WriterStranded`] (re-acquire the writer; store intact) from a
    /// string.
    pub(crate) fn writer_stranded(msg: impl Into<String>) -> Self {
        Error::WriterStranded(msg.into())
    }

    /// Construct an [`Error::Schema`] from anything string-like.
    pub(crate) fn schema(msg: impl Into<String>) -> Self {
        Error::Schema(msg.into())
    }
}

impl From<rusqlite::Error> for Error {
    /// Classify a SQLite error at the `?` boundary: a **transient** fault — lock contention
    /// (`SQLITE_BUSY`/`SQLITE_LOCKED`) or a schema-change re-prepare (`SQLITE_SCHEMA`) — becomes
    /// the retryable [`Error::Busy`], so every store path (read checkout, the search body, and
    /// writes) surfaces one uniform "retry me" signal and the caller owns the backoff. The
    /// library never sleeps/blocks to retry internally (`busy_timeout` is 0 — audit OD1). Any
    /// other SQLite error is a real [`Error::Sqlite`] fault.
    fn from(e: rusqlite::Error) -> Self {
        if is_retryable(&e) {
            Error::Busy(format!(
                "transient store fault (retry on a fresh reader): {e}"
            ))
        } else {
            Error::Sqlite(e)
        }
    }
}

/// Whether a SQLite error is a transient, retryable fault — lock contention
/// (`SQLITE_BUSY`/`SQLITE_LOCKED`) or a statement that needs re-preparing after a concurrent
/// schema change (`SQLITE_SCHEMA`) — as opposed to a real store fault.
fn is_retryable(e: &rusqlite::Error) -> bool {
    matches!(
        e,
        rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error {
                code: rusqlite::ErrorCode::DatabaseBusy
                    | rusqlite::ErrorCode::DatabaseLocked
                    | rusqlite::ErrorCode::SchemaChanged,
                ..
            },
            _,
        )
    )
}
