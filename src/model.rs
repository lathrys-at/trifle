//! The runtime data model: the caller declares a [`Schema`] (roles, not SQL types) and
//! indexes [`Document`]s; a match comes back as a [`Match`].
//!
//! trifle does exactly one type-dependent thing — tokenize text — so the schema axis is
//! **roles**, not types, with one principled exception: the **key**'s shape is declared
//! (`Integer`/`Text`/`Blob`), because the key is the one field trifle *compares* (dedup /
//! replace / delete / return). Everything else is a **text field** (tokenized; its name is
//! the label returned on a match). Every indexed text field's text is **stored** and is
//! always returned on a match.
//!
//! A document is a `key` plus a set of named segments (`label → text`). `flat()` and
//! `chunked()` are ergonomic front-ends that both lower to the same engine.
//!
//! Filtering is an opt-in raw-SQL fragment over the caller's *live* data — see
//! [`SqlFilter`](crate::SqlFilter) — not a stored attribute on the index.

use crate::hash::FxHashSet;

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

    /// Read a key of the given shape from a SQL value.
    pub(crate) fn from_value(shape: KeyShape, v: Value) -> Result<Key> {
        match (shape, v) {
            (KeyShape::Integer, Value::Integer(i)) => Ok(Key::Integer(i)),
            (KeyShape::Text, Value::Text(s)) => Ok(Key::Text(s)),
            (KeyShape::Blob, Value::Blob(b)) => Ok(Key::Blob(b)),
            _ => Err(Error::corrupt(
                "stored key does not match the declared key shape",
            )),
        }
    }
}

/// Bind a [`Key`] as a SQL parameter **without cloning** — `Text`/`Blob` bind borrowed
/// (v0.5; the old `to_value` cloned the payload on every bind).
impl rusqlite::ToSql for Key {
    fn to_sql(&self) -> rusqlite::Result<rusqlite::types::ToSqlOutput<'_>> {
        use rusqlite::types::{ToSqlOutput, ValueRef};
        Ok(match self {
            Key::Integer(i) => ToSqlOutput::Owned(Value::Integer(*i)),
            Key::Text(s) => ToSqlOutput::Borrowed(ValueRef::Text(s.as_bytes())),
            Key::Blob(b) => ToSqlOutput::Borrowed(ValueRef::Blob(b)),
        })
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
/// the `seg.key` column type and settles comparability (native for int/text, memcmp for
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

/// A document to index: a [`Key`] plus its named segments (`label → text`). Used by
/// [`rebuild`](crate::Index::rebuild) and the batch upsert. `#[non_exhaustive]`: construct with
/// [`Document::new`], not a struct literal.
#[non_exhaustive]
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

/// One ranked match. Rank is conveyed by position in the returned `Vec<Match>`; the absolute
/// relevance magnitude and its components ride along (v0.5, the derivation-§10 contract: "the
/// engine emits a score together with its components").
///
/// `#[non_exhaustive]` (fields may be added in minor releases) and not `Eq` (the score fields are
/// floats).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct Match {
    /// The caller's document key.
    pub key: Key,
    /// The label of the segment that matched (the text field name).
    pub label: String,
    /// The `[first, last)` UTF-8 byte span of the matched region within
    /// [`text`](Self::text). `None` when no span could be located.
    pub span: Option<(usize, usize)>,
    /// The matched segment's text. Every indexed text field is stored, so this is always
    /// present.
    pub text: String,
    /// trifle's length-corrected relevance magnitude in **nats**: `energy + count − length`
    /// (derivation §10), from the match's governing rank-view. **Cross-query comparable** by
    /// construction — the right input for thresholding or downstream fusion. It is *not*
    /// necessarily the within-query sort key (a starved query's results are ordered by
    /// reciprocal-rank fusion; rank is the `Vec` position either way), and it can be negative
    /// (the length null dominating a weak match).
    pub score: f64,
    /// The §10 **energy** component (nats): the matched grams' quantized logit-idf sum.
    pub energy: f64,
    /// The §10 **count-credit** component (nats): `Σ μ` over the matched non-floored grams.
    pub count: f64,
    /// The §10 **length-null** component (nats): the saturating chance-match debit subtracted
    /// from `energy + count`.
    pub length: f64,
}

/// A validated, immutable index schema.
///
/// Build with [`Schema::builder`], or the [`flat`](Schema::flat) / [`chunked`](Schema::chunked)
/// front-ends. The schema persists into `meta` and a **schema fingerprint** folds into the
/// drift check — so reinterpreting columns drops the cache rather than silently serving a
/// mis-indexed store.
#[derive(Clone, Debug)]
pub struct Schema {
    key_shape: KeyShape,
    /// Declared text-field labels (all stored + indexed).
    fields: FxHashSet<String>,
    /// Whether labels not explicitly declared are accepted (`flat()`); `false` rejects them.
    default_text: bool,
    fingerprint: u64,
}

impl Schema {
    /// Start a [`SchemaBuilder`].
    pub fn builder() -> SchemaBuilder {
        SchemaBuilder {
            key: None,
            fields: Vec::new(),
            default_text: false,
        }
    }

    /// A flat schema: one integer key named `key` and a single default text field (any
    /// label accepted) — the simplest shape.
    pub fn flat() -> Schema {
        Schema::builder()
            .key("key", KeyShape::Integer)
            .default_text()
            .build()
            .expect("the flat schema is always valid")
    }

    /// A chunked schema: an integer key named `key` with explicitly declared text
    /// fields. Add fields with [`SchemaBuilder::text`].
    pub fn chunked() -> SchemaBuilder {
        Schema::builder().key("key", KeyShape::Integer)
    }

    /// The declared key shape.
    pub(crate) fn key_shape(&self) -> KeyShape {
        self.key_shape
    }
    /// Whether a segment `label` is accepted by this schema (a declared field, or any
    /// label when the schema has a default text field). Accepted labels are always stored.
    pub(crate) fn accepts_label(&self, label: &str) -> bool {
        self.default_text || self.fields.contains(label)
    }
    /// The schema fingerprint (semantic identity), folded into the drift check.
    pub(crate) fn fingerprint(&self) -> u64 {
        self.fingerprint
    }
}

/// Builder for a [`Schema`].
pub struct SchemaBuilder {
    key: Option<(String, KeyShape)>,
    fields: Vec<String>,
    default_text: bool,
}

impl SchemaBuilder {
    /// Declare the key field (exactly one is required) and its shape.
    pub fn key(mut self, name: impl Into<String>, shape: KeyShape) -> Self {
        self.key = Some((name.into(), shape));
        self
    }

    /// Declare a text field named `name`. Its text is stored, indexed, and returned on a
    /// match.
    pub fn text(mut self, name: impl Into<String>) -> Self {
        self.fields.push(name.into());
        self
    }

    /// Accept any (undeclared) segment label as a stored text field — the open-label
    /// front-end used by [`Schema::flat`].
    pub fn default_text(mut self) -> Self {
        self.default_text = true;
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

        // Every schema-derived name is interpolated into DDL, so validate it as a safe
        // identifier (the injection surface).
        crate::store::validate_ident(&key_name)?;
        let mut seen = FxHashSet::default();
        seen.insert(key_name.clone());
        let mut fields = FxHashSet::default();
        for name in &self.fields {
            crate::store::validate_ident(name)?;
            if !seen.insert(name.clone()) {
                return Err(Error::schema(format!("duplicate field name {name:?}")));
            }
            fields.insert(name.clone());
        }
        if fields.is_empty() && !self.default_text {
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

/// A stable FNV-1a over a canonical *semantic* encoding of the schema (key name + shape,
/// the set of text-field names, the open-label default) — **not** column layout
/// (`sqlite_schema` owns structure). The dangerous drift is same-tables /
/// reinterpreted-columns, which this catches.
fn schema_fingerprint(
    key_name: &str,
    key_shape: KeyShape,
    fields: &[String],
    default_text: bool,
) -> u64 {
    let mut sorted: Vec<&String> = fields.iter().collect();
    sorted.sort();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"schema-v3");
    bytes.extend_from_slice(&(key_name.len() as u64).to_le_bytes());
    bytes.extend_from_slice(key_name.as_bytes());
    bytes.push(key_shape.code());
    bytes.push(default_text as u8);
    bytes.extend_from_slice(&(sorted.len() as u64).to_le_bytes());
    for name in sorted {
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
    }
    crate::tokenize::fnv1a_64(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_schema_accepts_any_label() {
        let s = Schema::flat();
        assert_eq!(s.key_shape(), KeyShape::Integer);
        assert!(s.accepts_label("anything"));
    }

    #[test]
    fn chunked_schema_only_accepts_declared_fields() {
        let s = Schema::chunked()
            .text("front")
            .text("back")
            .build()
            .unwrap();
        assert!(s.accepts_label("front"));
        assert!(s.accepts_label("back"));
        assert!(!s.accepts_label("undeclared"));
    }

    #[test]
    fn build_rejects_no_key_and_dup_names_and_no_text() {
        assert!(Schema::builder().default_text().build().is_err());
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
        let a = Schema::chunked()
            .text("front")
            .text("back")
            .build()
            .unwrap();
        let b = Schema::chunked()
            .text("back")
            .text("front")
            .build()
            .unwrap();
        // Field declaration order does not change identity.
        assert_eq!(a.fingerprint(), b.fingerprint());
        // The set of indexed fields is semantic identity.
        let c = Schema::chunked().text("front").build().unwrap();
        assert_ne!(a.fingerprint(), c.fingerprint());
    }
}
