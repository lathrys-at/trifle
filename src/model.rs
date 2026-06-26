//! The runtime data model: the caller declares a [`Schema`] (roles, not SQL types) and
//! indexes [`Document`]s; a match comes back as a [`Match`].
//!
//! trifle does exactly one type-dependent thing — tokenize text — so the schema axis is
//! **roles**, not types, with one principled exception: the **key**'s shape is declared
//! (`Integer`/`Text`/`Blob`), because the key is the one field trifle *compares* (dedup /
//! replace / delete / return). Everything else is a **text field** (tokenized; its name is
//! the label returned on a match). Every indexed text field's text is **stored** and is
//! always surfaced to the reranker and returned on a match; filterable **payload** columns
//! (Tier 2) are stored separately for filtering only and are never reranked or returned.
//!
//! A document is a `key` plus a set of named segments (`label → text`) — the two-level
//! document→segment hierarchy. `flat()` and `chunked()` are ergonomic front-ends that
//! both lower to the same engine.

use std::collections::HashSet;

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
            _ => Err(Error::corrupt(
                "stored key does not match the declared key shape",
            )),
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

/// The declared type of a **filterable** field (Tier 2 of the filtering ladder): it
/// materializes as a real, indexed `doc` column the search can `WHERE` against. Picks the
/// column's SQLite affinity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterType {
    /// Integer-affinity column.
    Int,
    /// Real-affinity column.
    Real,
    /// Text-affinity column.
    Text,
    /// A datetime, stored as a **sortable scalar** so the plain `<`/`>`/`=` comparisons
    /// order chronologically. Sugar for an epoch-`INTEGER` column (the canonical
    /// encoding); for ISO-8601 strings — which sort chronologically as text — declare
    /// [`Text`](FilterType::Text) instead. SQLite has no datetime type, so no special
    /// operators are needed: the sortable encoding *is* the mechanism.
    Timestamp,
}

impl FilterType {
    pub(crate) fn sql_type(self) -> &'static str {
        match self {
            FilterType::Int | FilterType::Timestamp => "INTEGER",
            FilterType::Real => "REAL",
            FilterType::Text => "TEXT",
        }
    }
    fn code(self) -> u8 {
        match self {
            FilterType::Int => 1,
            FilterType::Real => 2,
            FilterType::Text => 3,
            FilterType::Timestamp => 4,
        }
    }
}

/// A comparison operator for a [`Filter`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    /// `=`
    Eq,
    /// `<>`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
}

impl CmpOp {
    fn sql(self) -> &'static str {
        match self {
            CmpOp::Eq => "=",
            CmpOp::Ne => "<>",
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
        }
    }
}

/// A structured filter over **filterable** fields (Tier 2). Compiles to a parameterized
/// `WHERE` over the materialized `doc` columns and is applied as a doc-id set
/// intersection between candidate generation and rerank/hydration (it prunes before
/// ranking; it does not save the candidate-generation overlap work — that needs a
/// partition). Only declared-filterable fields are addressable, via this restricted
/// grammar — never arbitrary SQL.
#[derive(Clone, Debug)]
pub enum Filter {
    /// `field op value`.
    Cmp {
        /// The filterable field name.
        field: String,
        /// The comparison operator.
        op: CmpOp,
        /// The bound value.
        value: Value,
    },
    /// `field IN (values…)`.
    In {
        /// The filterable field name.
        field: String,
        /// The candidate values.
        values: Vec<Value>,
    },
    /// `field BETWEEN low AND high` (inclusive) — the clean form for a range, including a
    /// datetime range over a [`Timestamp`](FilterType::Timestamp) field.
    Between {
        /// The filterable field name.
        field: String,
        /// The inclusive lower bound.
        low: Value,
        /// The inclusive upper bound.
        high: Value,
    },
    /// `field IS NULL`.
    IsNull {
        /// The filterable field name.
        field: String,
    },
    /// `field LIKE pattern`. **Cost:** a *leading-wildcard* pattern (`"%x"`) forces a full
    /// scan — the index cannot help — the same hidden cliff as an un-indexed path. Prefer
    /// an anchored pattern (`"x%"`).
    Like {
        /// The filterable field name.
        field: String,
        /// The `LIKE` pattern.
        pattern: String,
    },
    /// A raw, parameterized SQL predicate fragment — the escape hatch. It exposes all of
    /// SQLite's expression language (date functions, arithmetic, …). It is spliced into the
    /// `WHERE` of a `SELECT … FROM doc`, so the materialized filterable columns are in
    /// scope; it is **not** sandboxed — a subquery could reference other tables, which is
    /// why the **fragment must be a trusted constant**, not built from untrusted input.
    /// **Costs:** *untyped* (a malformed fragment is a runtime error, not a compile error);
    /// the **fragment is the injection surface** (keep it a constant and bind data through
    /// `params`, never format values into it); and it **couples you to trifle's column
    /// names**. Advanced and may break across versions — prefer the structured variants.
    Sql {
        /// The SQL predicate fragment, with `?` placeholders for `params`.
        fragment: String,
        /// The bound parameters, in placeholder order.
        params: Vec<Value>,
    },
    /// Both sub-filters.
    And(Box<Filter>, Box<Filter>),
    /// Either sub-filter.
    Or(Box<Filter>, Box<Filter>),
}

impl Filter {
    /// `field op value`.
    pub fn cmp(field: impl Into<String>, op: CmpOp, value: impl Into<Value>) -> Filter {
        Filter::Cmp {
            field: field.into(),
            op,
            value: value.into(),
        }
    }
    /// `field = value`.
    pub fn eq(field: impl Into<String>, value: impl Into<Value>) -> Filter {
        Filter::cmp(field, CmpOp::Eq, value)
    }
    /// `field IN (values…)`.
    pub fn in_(field: impl Into<String>, values: impl IntoIterator<Item = Value>) -> Filter {
        Filter::In {
            field: field.into(),
            values: values.into_iter().collect(),
        }
    }
    /// `field BETWEEN low AND high` (inclusive).
    pub fn between(
        field: impl Into<String>,
        low: impl Into<Value>,
        high: impl Into<Value>,
    ) -> Filter {
        Filter::Between {
            field: field.into(),
            low: low.into(),
            high: high.into(),
        }
    }
    /// `field IS NULL`.
    pub fn is_null(field: impl Into<String>) -> Filter {
        Filter::IsNull {
            field: field.into(),
        }
    }
    /// `field LIKE pattern` (mind the leading-wildcard scan cost — see [`Filter::Like`]).
    pub fn like(field: impl Into<String>, pattern: impl Into<String>) -> Filter {
        Filter::Like {
            field: field.into(),
            pattern: pattern.into(),
        }
    }
    /// A raw parameterized SQL fragment over the filterable columns (the escape hatch —
    /// see [`Filter::Sql`] for the costs).
    pub fn sql(fragment: impl Into<String>, params: impl IntoIterator<Item = Value>) -> Filter {
        Filter::Sql {
            fragment: fragment.into(),
            params: params.into_iter().collect(),
        }
    }
    /// Conjoin two filters.
    pub fn and(self, other: Filter) -> Filter {
        Filter::And(Box::new(self), Box::new(other))
    }
    /// Disjoin two filters.
    pub fn or(self, other: Filter) -> Filter {
        Filter::Or(Box::new(self), Box::new(other))
    }

    /// Compile to a parameterized SQL predicate over `doc` columns, validating every
    /// referenced field against the schema's filterable set (the injection guard — only
    /// declared idents reach SQL). Returns the predicate and its bound parameters.
    pub(crate) fn compile(&self, schema: &Schema) -> Result<(String, Vec<Value>)> {
        let mut params = Vec::new();
        let sql = self.build(schema, &mut params)?;
        Ok((sql, params))
    }

    fn build(&self, schema: &Schema, params: &mut Vec<Value>) -> Result<String> {
        match self {
            Filter::Cmp { field, op, value } => {
                let col = schema.filter_column(field)?;
                params.push(value.clone());
                Ok(format!("\"{col}\" {} ?", op.sql()))
            }
            Filter::In { field, values } => {
                let col = schema.filter_column(field)?;
                if values.is_empty() {
                    return Ok("0".to_string()); // empty IN matches nothing
                }
                let marks = vec!["?"; values.len()].join(", ");
                for v in values {
                    params.push(v.clone());
                }
                Ok(format!("\"{col}\" IN ({marks})"))
            }
            Filter::Between { field, low, high } => {
                let col = schema.filter_column(field)?;
                params.push(low.clone());
                params.push(high.clone());
                Ok(format!("\"{col}\" BETWEEN ? AND ?"))
            }
            Filter::IsNull { field } => {
                let col = schema.filter_column(field)?;
                Ok(format!("\"{col}\" IS NULL"))
            }
            Filter::Like { field, pattern } => {
                let col = schema.filter_column(field)?;
                params.push(Value::Text(pattern.clone()));
                Ok(format!("\"{col}\" LIKE ?"))
            }
            Filter::Sql {
                fragment,
                params: p,
            } => {
                // The escape hatch: the fragment is trusted and not field-validated (it is
                // the injection surface); only its values are parameterized. It is spliced
                // into a `WHERE` over the `doc` table, so the filterable columns are in
                // scope — but it is not sandboxed (a subquery could name other tables),
                // hence the trusted-constant contract on `Filter::Sql`.
                for v in p {
                    params.push(v.clone());
                }
                Ok(format!("({fragment})"))
            }
            Filter::And(a, b) => Ok(format!(
                "({} AND {})",
                a.build(schema, params)?,
                b.build(schema, params)?
            )),
            Filter::Or(a, b) => Ok(format!(
                "({} OR {})",
                a.build(schema, params)?,
                b.build(schema, params)?
            )),
        }
    }
}

/// A document to index: a [`Key`] plus its named segments (`label → text`). Used by
/// [`rebuild`](crate::Index::rebuild) and the batch insert.
#[derive(Clone, Debug, PartialEq)]
pub struct Document {
    /// The caller's key — the unit of retrieval and lifecycle.
    pub key: Key,
    /// The document's segments as `(label, text)` pairs; each `label` names a text field.
    pub segments: Vec<(String, String)>,
    /// Values for the schema's **filterable** fields, as `(field, value)` pairs. Stored
    /// into the materialized `doc` columns (Tier 2); undeclared names are ignored.
    pub payload: Vec<(String, Value)>,
}

impl Document {
    /// Construct a document with no filterable payload.
    pub fn new(key: impl Into<Key>, segments: Vec<(String, String)>) -> Self {
        Document {
            key: key.into(),
            segments,
            payload: Vec::new(),
        }
    }

    /// Set the filterable-field payload.
    pub fn with_payload(mut self, payload: Vec<(String, Value)>) -> Self {
        self.payload = payload;
        self
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
    /// [`text`](Self::text). `None` when no span could be located.
    pub span: Option<(usize, usize)>,
    /// The matched segment's text. Every indexed text field is stored, so this is always
    /// present.
    pub text: String,
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
    /// Declared text-field labels (all stored + indexed).
    fields: HashSet<String>,
    /// Whether labels not explicitly declared are accepted (`flat()`); `false` rejects them.
    default_text: bool,
    /// Declared filterable fields (Tier 2): materialized as indexed `doc` columns, in
    /// declaration order (the order the columns are created).
    filterable: Vec<(String, FilterType)>,
    fingerprint: u64,
}

impl Schema {
    /// Start a [`SchemaBuilder`].
    pub fn builder() -> SchemaBuilder {
        SchemaBuilder {
            key: None,
            fields: Vec::new(),
            default_text: false,
            filterable: Vec::new(),
        }
    }

    /// A flat schema: one integer key named `key` and a single default text field (any
    /// label accepted). The closest analogue to the v0.1 fixed model.
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
    /// The `doc` column name for a declared filterable `field`, or an error if it is not
    /// declared filterable (the injection guard for a compiled `WHERE`).
    pub(crate) fn filter_column(&self, field: &str) -> Result<&str> {
        self.filterable
            .iter()
            .find(|(n, _)| n == field)
            .map(|(n, _)| n.as_str())
            .ok_or_else(|| Error::schema(format!("{field:?} is not a filterable field")))
    }
    /// The declared filterable fields (name + type), in declaration / column order.
    pub(crate) fn filterable_columns(&self) -> &[(String, FilterType)] {
        &self.filterable
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
    filterable: Vec<(String, FilterType)>,
}

impl SchemaBuilder {
    /// Declare the key field (exactly one is required) and its shape.
    pub fn key(mut self, name: impl Into<String>, shape: KeyShape) -> Self {
        self.key = Some((name.into(), shape));
        self
    }

    /// Declare a text field named `name`. Its text is stored, indexed, and surfaced to the
    /// reranker / returned on a match.
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

    /// Declare a **filterable** field of the given type (Tier 2). It materializes as an
    /// indexed `doc` column a search can `WHERE` against via a [`Filter`].
    pub fn filterable(mut self, name: impl Into<String>, ty: FilterType) -> Self {
        self.filterable.push((name.into(), ty));
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
        let mut seen = HashSet::new();
        seen.insert(key_name.clone());
        let mut fields = HashSet::new();
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
        // Filterable fields become `doc` columns: ident-safe, distinct, and not the
        // built-in `id`/`key` columns.
        for (name, _) in &self.filterable {
            crate::store::validate_ident(name)?;
            if name == "id" || name == "key" {
                return Err(Error::schema(format!(
                    "filterable field {name:?} collides with a built-in doc column"
                )));
            }
            if !seen.insert(name.clone()) {
                return Err(Error::schema(format!("duplicate field name {name:?}")));
            }
        }

        let fingerprint = schema_fingerprint(
            &key_name,
            key_shape,
            &self.fields,
            self.default_text,
            &self.filterable,
        );
        Ok(Schema {
            key_shape,
            fields,
            default_text: self.default_text,
            filterable: self.filterable,
            fingerprint,
        })
    }
}

/// A stable FNV-1a over a canonical *semantic* encoding of the schema (key name + shape,
/// the set of text-field names, the open-label default, and the filterable columns) —
/// **not** column layout (`sqlite_schema` owns structure). The dangerous drift is
/// same-tables / reinterpreted-columns, which this catches. (`schema-v2`: v0.2 dropped
/// per-field storage modes — all text fields are stored.)
fn schema_fingerprint(
    key_name: &str,
    key_shape: KeyShape,
    fields: &[String],
    default_text: bool,
    filterable: &[(String, FilterType)],
) -> u64 {
    let mut sorted: Vec<&String> = fields.iter().collect();
    sorted.sort();
    let mut filt: Vec<&(String, FilterType)> = filterable.iter().collect();
    filt.sort_by(|a, b| a.0.cmp(&b.0));
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"schema-v2");
    bytes.extend_from_slice(&(key_name.len() as u64).to_le_bytes());
    bytes.extend_from_slice(key_name.as_bytes());
    bytes.push(key_shape.code());
    bytes.push(default_text as u8);
    bytes.extend_from_slice(&(sorted.len() as u64).to_le_bytes());
    for name in sorted {
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
    }
    bytes.extend_from_slice(&(filt.len() as u64).to_le_bytes());
    for (name, ty) in filt {
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(ty.code());
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
        // A filterable column is semantic identity too.
        let d = Schema::chunked()
            .text("front")
            .text("back")
            .filterable("deck", FilterType::Int)
            .build()
            .unwrap();
        assert_ne!(a.fingerprint(), d.fingerprint());
    }

    #[test]
    fn filter_grammar_compiles_and_validates_fields() {
        let s = Schema::chunked()
            .text("body")
            .filterable("deck", FilterType::Int)
            .filterable("created", FilterType::Timestamp)
            .filterable("lang", FilterType::Text)
            .build()
            .unwrap();

        // Comparison, range, null, like, in. Column names are double-quoted (keyword-safe).
        assert_eq!(
            Filter::eq("deck", Value::Integer(3)).compile(&s).unwrap().0,
            "\"deck\" = ?"
        );
        let (sql, p) = Filter::between("created", Value::Integer(100), Value::Integer(200))
            .compile(&s)
            .unwrap();
        assert_eq!(sql, "\"created\" BETWEEN ? AND ?");
        assert_eq!(p.len(), 2);
        assert_eq!(
            Filter::is_null("lang").compile(&s).unwrap().0,
            "\"lang\" IS NULL"
        );
        assert_eq!(
            Filter::like("lang", "en%").compile(&s).unwrap().0,
            "\"lang\" LIKE ?"
        );
        assert_eq!(
            Filter::in_("deck", [Value::Integer(1), Value::Integer(2)])
                .compile(&s)
                .unwrap()
                .0,
            "\"deck\" IN (?, ?)"
        );

        // AND / OR nest with parentheses.
        let nested = Filter::eq("deck", Value::Integer(1)).and(Filter::like("lang", "en%"));
        assert_eq!(
            nested.compile(&s).unwrap().0,
            "(\"deck\" = ? AND \"lang\" LIKE ?)"
        );

        // The raw SQL hatch is spliced verbatim (parenthesized), values bound.
        let (sql, p) = Filter::sql(
            "deck > ? OR lang = ?",
            [Value::Integer(5), Value::Text("fr".into())],
        )
        .compile(&s)
        .unwrap();
        assert_eq!(sql, "(deck > ? OR lang = ?)");
        assert_eq!(p.len(), 2);

        // An undeclared field is rejected (the injection guard).
        assert!(
            Filter::eq("not_a_field", Value::Integer(1))
                .compile(&s)
                .is_err()
        );

        // A Timestamp field is an INTEGER column (epoch).
        assert_eq!(FilterType::Timestamp.sql_type(), "INTEGER");
    }
}
