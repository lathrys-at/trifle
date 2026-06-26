//! The crate error type.
//!
//! Every fallible operation in trifle returns [`Result`]. The variants separate
//! the three failure classes a caller reasons about differently: a transient or
//! environmental store fault ([`Error::Sqlite`]), bad input the caller can fix
//! ([`Error::InvalidInput`], [`Error::Namespace`]), and an internal invariant
//! violation that should be impossible ([`Error::Corrupt`], [`Error::Posting`]).

/// A specialized [`Result`](std::result::Result) for trifle operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Anything that can go wrong in trifle.
///
/// `#[non_exhaustive]` so new variants can be added without a breaking change;
/// match with a wildcard arm.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The backing SQLite store returned an error. After the internal busy-retry
    /// budget is exhausted, a persistent `SQLITE_BUSY`/`SQLITE_LOCKED` surfaces
    /// here too — it is environmental (another writer holds the file), not a bug.
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),

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

    /// Construct an [`Error::Schema`] from anything string-like.
    pub(crate) fn schema(msg: impl Into<String>) -> Self {
        Error::Schema(msg.into())
    }
}
