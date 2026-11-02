//! Argument checks, one place for every message.
//!
//! The kernel contract leaves plain-argument validation to this crate, so that bad
//! input fails the same way whichever kernel is active.

use occt_kernel::{Axis, Error, Point3, Result, Vec3};

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

/// Requires a finite value.
pub(crate) fn finite(what: &str, value: f64) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid(format!("{what} must be finite, got {value}")))
    }
}

/// Requires a finite, strictly positive value.
pub(crate) fn positive(what: &str, value: f64) -> Result<f64> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(invalid(format!(
            "{what} must be finite and positive, got {value}"
        )))
    }
}

/// Requires a finite, zero or positive value.
pub(crate) fn non_negative(what: &str, value: f64) -> Result<f64> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(invalid(format!(
            "{what} must be finite and not negative, got {value}"
        )))
    }
}

/// Requires finite coordinates.
pub(crate) fn point(what: &str, value: Point3) -> Result<Point3> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid(format!("{what} must be finite, got {value:?}")))
    }
}

/// Requires finite components.
pub(crate) fn vector(what: &str, value: Vec3) -> Result<Vec3> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid(format!("{what} must be finite, got {value:?}")))
    }
}

/// Requires a finite origin. The direction of an [`Axis`] is finite by construction.
pub(crate) fn axis(what: &str, value: Axis) -> Result<Axis> {
    point(what, value.origin)?;
    Ok(value)
}
