//! What an operation did to the faces, edges and vertices of its inputs.

use std::collections::{HashMap, HashSet};

use occt_kernel::{self as contract};

use crate::Shape;
use crate::kernel::RawShape;

/// Links between an operation's input subshapes and its result.
///
/// This is what persistent naming builds on: keep a reference to an input face, and after
/// the rebuild ask what it became.
///
/// # Examples
/// ```
/// use occt::prelude::*;
/// # fn main() -> occt::Result<()> {
/// # let history = occt::History::default();
/// # let face: occt::Shape = Cuboid::new([1.0, 1.0, 1.0]).build()?.into();
/// assert!(history.modified(&face).is_empty());
/// assert!(!history.is_deleted(&face));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Default)]
pub struct History {
    modified: HashMap<Shape, Vec<Shape>>,
    generated: HashMap<Shape, Vec<Shape>>,
    deleted: HashSet<Shape>,
}

impl History {
    /// What `input` became, when it survives in changed form. Empty if it's unchanged,
    /// deleted, or unknown to this operation.
    #[must_use]
    pub fn modified(&self, input: &Shape) -> &[Shape] {
        self.modified.get(input).map_or(&[], Vec::as_slice)
    }

    /// What `input` gave rise to, such as the face a fillet made from an edge.
    #[must_use]
    pub fn generated(&self, input: &Shape) -> &[Shape] {
        self.generated.get(input).map_or(&[], Vec::as_slice)
    }

    /// Whether `input` has no counterpart in the result.
    #[must_use]
    pub fn is_deleted(&self, input: &Shape) -> bool {
        self.deleted.contains(input)
    }

    /// Follows `input`: what it became, or itself if the operation left it alone.
    /// Empty when deleted.
    #[must_use]
    pub fn trace(&self, input: &Shape) -> Vec<Shape> {
        if self.is_deleted(input) {
            return Vec::new();
        }
        match self.modified.get(input) {
            Some(outputs) => outputs.clone(),
            None => vec![input.clone()],
        }
    }

    pub(crate) fn from_raw(raw: contract::History<RawShape>) -> Self {
        let pairs = |list: Vec<(RawShape, Vec<RawShape>)>| {
            list.into_iter()
                .map(|(k, v)| {
                    (
                        Shape::from_raw(k),
                        v.into_iter().map(Shape::from_raw).collect(),
                    )
                })
                .collect()
        };
        Self {
            modified: pairs(raw.modified),
            generated: pairs(raw.generated),
            deleted: raw.deleted.into_iter().map(Shape::from_raw).collect(),
        }
    }
}

/// An operation's result together with its [`History`].
#[derive(Clone, Debug)]
pub struct Modeled<T = Shape> {
    /// The result.
    pub shape: T,
    /// How the input subshapes relate to it.
    pub history: History,
}

impl Modeled {
    pub(crate) fn from_raw(raw: contract::Modeled<RawShape>) -> Self {
        Self {
            shape: Shape::from_raw(raw.shape),
            history: History::from_raw(raw.history),
        }
    }
}
