//! The runtime data model: the caller declares a [`Schema`] (roles, not SQL types) and
//! indexes [`Document`]s; a match comes back as a [`Match`].
//!
//! trifle does exactly one type-dependent thing — tokenize text — so the schema axis is
//! **roles**, not types, with one principled exception: the **key**'s shape is declared
//! (`Integer`/`Text`/`Blob`), because the key is the one field trifle *compares* (dedup /
//! replace / delete / return). Everything else is a **text field** (tokenized; its name
//! is the label returned on a match) with a per-field [`StorageMode`] choosing where its
//! text comes from on hydration.
//!
//! A document is a `key` plus a set of named segments (`label → text`) — the two-level
//! document→segment hierarchy. `flat()` and `chunked()` are ergonomic front-ends that
//! both lower to the same engine.

use std::collections::HashMap;

use rusqlite::types::Value;

use crate::error::{Error, Result};

/// A caller key — the lifecycle handle trifle compares (dedup / replace / delete) and
/// returns on a match. Its [`KeyShape`] is declared in the [`Schema`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    /// An integer key (e.g. Anki's note id) — a fast native integer column.
    Integer(i64),
    /// A text key.
    Text(String),
    /// An opaque byte key, compared by **memcmp**. trifle requires only that equal keys
    /// are byte-equal (any deterministic encoding satisfies this) and does not interpret
    /// their order.
    Blob(Vec<u8>),
}

impl Key {
    /// The integer value, if this is an [`Integer`](Key::Integer) key.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Key::Integer(i) => Some(*i),
            _ => None,
        }
    }
    /// The string value, if this is a [`Text`](Key::Text) key.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Key::Text(s) => Some(s),
            _ => None,
        }
    }
    /// The byte value, if this is a [`Blob`](Key::Blob) key.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Key::Blob(b) => Some(b),
            _ => None,
        }
    }

    /// Bind this key as a SQL value.
    pub(crate) fn to_value(&self) -> Value {
        match self {
            Key::Integer(i) => Value::Integer(*i),
            Key::Text(s) => Value::Text(s.clone()),
            Key::Blob(b) => Value::Blob(b.clone()),
        }
    }

    /// Read a key of the given shape from a SQL value.
    pub(crate) fn from_value(shape: KeyShape, v: Value) -> Result<Key> {
        match (shape, v) {
            (KeyShape::Integer, Value::Integer(i)) => Ok(Key::Integer(i)),
            (KeyShape::Text, Value::Text(s)) => Ok(Key::Text(s)),
            (KeyShape::Blob, Value::Blob(b)) => Ok(Key::Blob(b)),
            _ => Err(Error::corrupt("stored key does not match the declared key shape")),
        }
    }
}

impl From<i64> for Key {
    fn from(i: i64) -> Self {
        Key::Integer(i)
    }
}
impl From<&str> for Key {
    fn from(s: &str) -> Self {
        Key::Text(s.to_string())
    }
}
impl From<String> for Key {
    fn from(s: String) -> Self {
        Key::Text(s)
    }
}
impl From<Vec<u8>> for Key {
    fn from(b: Vec<u8>) -> Self {
        Key::Blob(b)
    }
}

/// The declared shape of the [`Key`] — the one place a SQL type is declared. It picks
/// the `doc.key` column type and settles comparability (native for int/text, memcmp for
/// blob).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyShape {
    /// `INTEGER` column, native integer comparison.
    Integer,
    /// `TEXT` column.
    Text,
    /// `BLOB` column, memcmp comparison.
    Blob,
}

impl KeyShape {
    /// The SQL column type for this shape.
    pub(crate) fn sql_type(self) -> &'static str {
        match self {
            KeyShape::Integer => "INTEGER",
            KeyShape::Text => "TEXT",
            KeyShape::Blob => "BLOB",
        }
    }
    fn code(self) -> u8 {
        match self {
            KeyShape::Integer => 1,
            KeyShape::Text => 2,
            KeyShape::Blob => 3,
        }
    }
}

/// Where a text field's text comes from when hydrating a match — chosen per field
/// because interning decoupled delete from text (delete reads the stored term-id set, so
/// storage mode affects only hydration, never delete correctness).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageMode {
    /// trifle stores the text (`seg.txt`) and returns it on a match.
    Stored,
    /// trifle calls the caller's [`TextResolver`](crate::store::TextResolver) for the
    /// text on hydration (it stores none).
    Resolver,
    /// trifle stores no text and returns none — the caller renders from `(key, label)`.
    CoordinatesOnly,
}

impl StorageMode {
    fn code(self) -> u8 {
        match self {
            StorageMode::Stored => 1,
            StorageMode::Resolver => 2,
            StorageMode::CoordinatesOnly => 3,
        }
    }
}

/// A document to index: a [`Key`] plus its named segments (`label → text`). Used by
/// [`rebuild`](crate::Index::rebuild) and the batch insert.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Document {
    /// The caller's key — the unit of retrieval and lifecycle.
    pub key: Key,
    /// The document's segments as `(label, text)` pairs; each `label` names a text field.
    pub segments: Vec<(String, String)>,
}

impl Document {
    /// Construct a document.
    pub fn new(key: impl Into<Key>, segments: Vec<(String, String)>) -> Self {
        Document {
            key: key.into(),
            segments,
        }
    }
}

/// One ranked match. Rank is conveyed by position in the returned `Vec<Match>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Match {
    /// The caller's document key.
    pub key: Key,
    /// The label of the segment that matched (the text field name).
    pub label: String,
    /// The `[first, last)` UTF-8 byte span of the matched region within
    /// [`text`](Self::text). `None` when no span could be located or text is unavailable.
    pub span: Option<(usize, usize)>,
    /// The matched segment's text. `None` for a `CoordinatesOnly` field, or a `Resolver`
    /// field whose resolver returned nothing.
    pub text: Option<String>,
}

/// A validated, immutable index schema.
///
/// Build with [`Schema::builder`], or the [`flat`](Schema::flat) / [`chunked`](Schema::chunked)
/// front-ends. The schema persists into `meta` and a **schema fingerprint** folds into the
/// drift check — so reinterpreting columns (e.g. flipping a field's storage mode) drops the
/// cache rather than silently serving a mis-indexed store.
#[derive(Clone, Debug)]
pub struct Schema {
    key_shape: KeyShape,
    /// Declared text fields: `label → storage mode`.
    fields: HashMap<String, StorageMode>,
    /// Storage mode for labels not explicitly declared (`flat()`); `None` rejects them.
    default_text: Option<StorageMode>,
    fingerprint: u64,
}

impl Schema {
    /// Start a [`SchemaBuilder`].
    pub fn builder() -> SchemaBuilder {
        SchemaBuilder {
            key: None,
            fields: Vec::new(),
            default_text: None,
        }
    }

    /// A flat schema: one integer key named `key` and a single default text field (any
    /// label accepted), stored. The closest analogue to the v0.1 fixed model.
    pub fn flat() -> Schema {
        Schema::builder()
            .key("key", KeyShape::Integer)
            .default_text(StorageMode::Stored)
            .build()
            .expect("the flat schema is always valid")
    }

    /// A chunked schema: an integer key named `key` with explicitly declared text
    /// fields, each `Stored`. Add fields with [`SchemaBuilder::text`].
    pub fn chunked() -> SchemaBuilder {
        Schema::builder().key("key", KeyShape::Integer)
    }

    /// The declared key shape.
    pub(crate) fn key_shape(&self) -> KeyShape {
        self.key_shape
    }
    /// The storage mode for a segment label (declared field, else the default), or `None`
    /// if the label is not accepted by this schema.
    pub(crate) fn storage_for(&self, label: &str) -> Option<StorageMode> {
        self.fields.get(label).copied().or(self.default_text)
    }
    /// Whether any field resolves its text through the [`TextResolver`](crate::store::TextResolver).
    pub(crate) fn needs_resolver(&self) -> bool {
        self.default_text == Some(StorageMode::Resolver)
            || self.fields.values().any(|m| *m == StorageMode::Resolver)
    }
    /// The schema fingerprint (semantic identity), folded into the drift check.
    pub(crate) fn fingerprint(&self) -> u64 {
        self.fingerprint
    }
}

/// Builder for a [`Schema`].
pub struct SchemaBuilder {
    key: Option<(String, KeyShape)>,
    fields: Vec<(String, StorageMode)>,
    default_text: Option<StorageMode>,
}

impl SchemaBuilder {
    /// Declare the key field (exactly one is required) and its shape.
    pub fn key(mut self, name: impl Into<String>, shape: KeyShape) -> Self {
        self.key = Some((name.into(), shape));
        self
    }

    /// Declare a `Stored` text field named `name`.
    pub fn text(self, name: impl Into<String>) -> Self {
        self.text_mode(name, StorageMode::Stored)
    }

    /// Declare a text field named `name` with the given [`StorageMode`].
    pub fn text_mode(mut self, name: impl Into<String>, storage: StorageMode) -> Self {
        self.fields.push((name.into(), storage));
        self
    }

    /// Accept any (undeclared) segment label with this storage mode — the open-label
    /// front-end used by [`Schema::flat`].
    pub fn default_text(mut self, storage: StorageMode) -> Self {
        self.default_text = Some(storage);
        self
    }

    /// Validate and finish the schema.
    ///
    /// # Errors
    ///
    /// [`Error::Schema`] if there is no key, a name is not identifier-safe, names
    /// collide, or there is no way to index any text (no declared field and no default).
    pub fn build(self) -> Result<Schema> {
        let (key_name, key_shape) = self
            .key
            .ok_or_else(|| Error::schema("schema has no key field"))?;

        // Every schema-derived name is interpolated into DDL / WHERE, so validate it as a
        // safe identifier (the new injection surface).
        crate::store::validate_ident(&key_name)?;
        let mut seen = std::collections::HashSet::new();
        seen.insert(key_name.clone());
        let mut fields = HashMap::new();
        for (name, mode) in &self.fields {
            crate::store::validate_ident(name)?;
            if !seen.insert(name.clone()) {
                return Err(Error::schema(format!("duplicate field name {name:?}")));
            }
            fields.insert(name.clone(), *mode);
        }
        if fields.is_empty() && self.default_text.is_none() {
            return Err(Error::schema(
                "schema declares no text field and no default — nothing to index",
            ));
        }

        let fingerprint = schema_fingerprint(&key_name, key_shape, &self.fields, self.default_text);
        Ok(Schema {
            key_shape,
            fields,
            default_text: self.default_text,
            fingerprint,
        })
    }
}

/// A stable FNV-1a over a canonical *semantic* encoding of the schema (names, the key
/// shape, field storage modes, the default) — **not** column layout (`sqlite_schema`
/// owns structure). The dangerous drift is same-tables / reinterpreted-columns, which
/// this catches.
fn schema_fingerprint(
    key_name: &str,
    key_shape: KeyShape,
    fields: &[(String, StorageMode)],
    default_text: Option<StorageMode>,
) -> u64 {
    let mut sorted: Vec<&(String, StorageMode)> = fields.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"schema-v1");
    bytes.extend_from_slice(&(key_name.len() as u64).to_le_bytes());
    bytes.extend_from_slice(key_name.as_bytes());
    bytes.push(key_shape.code());
    bytes.push(default_text.map_or(0, |m| m.code()));
    bytes.extend_from_slice(&(sorted.len() as u64).to_le_bytes());
    for (name, mode) in sorted {
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(mode.code());
    }
    crate::tokenize::fnv1a_64(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_schema_accepts_any_label_stored() {
        let s = Schema::flat();
        assert_eq!(s.key_shape(), KeyShape::Integer);
        assert_eq!(s.storage_for("anything"), Some(StorageMode::Stored));
        assert!(!s.needs_resolver());
    }

    #[test]
    fn chunked_schema_only_accepts_declared_fields() {
        let s = Schema::chunked()
            .text("front")
            .text_mode("back", StorageMode::CoordinatesOnly)
            .build()
            .unwrap();
        assert_eq!(s.storage_for("front"), Some(StorageMode::Stored));
        assert_eq!(s.storage_for("back"), Some(StorageMode::CoordinatesOnly));
        assert_eq!(s.storage_for("undeclared"), None);
    }

    #[test]
    fn build_rejects_no_key_and_dup_names_and_no_text() {
        assert!(Schema::builder().default_text(StorageMode::Stored).build().is_err());
        assert!(
            Schema::builder()
                .key("id", KeyShape::Integer)
                .build()
                .is_err(),
            "no text field and no default"
        );
        assert!(
            Schema::builder()
                .key("id", KeyShape::Integer)
                .text("a")
                .text("a")
                .build()
                .is_err(),
            "duplicate field"
        );
    }

    #[test]
    fn fingerprint_is_stable_and_semantic() {
        let a = Schema::chunked().text("front").text("back").build().unwrap();
        let b = Schema::chunked().text("back").text("front").build().unwrap();
        // Field declaration order does not change identity.
        assert_eq!(a.fingerprint(), b.fingerprint());
        // Storage mode is semantic identity.
        let c = Schema::chunked()
            .text("front")
            .text_mode("back", StorageMode::CoordinatesOnly)
            .build()
            .unwrap();
        assert_ne!(a.fingerprint(), c.fingerprint());
    }
}
