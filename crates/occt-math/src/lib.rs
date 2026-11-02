//! Geometry value types shared by the occt-rs crates.
//!
//! Plain `Copy` data with no dependencies and no knowledge of any kernel, so
//! these types stay valid whichever backend does the modelling.

#![forbid(unsafe_code)]

use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// A position in 3D space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point3 {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
    /// Z coordinate.
    pub z: f64,
}

/// A displacement in 3D space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
}

impl Point3 {
    /// The origin.
    pub const ORIGIN: Self = Self::new(0.0, 0.0, 0.0);

    /// Creates a point.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Whether every coordinate is finite.
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Distance to another point.
    #[must_use]
    pub fn distance(self, other: Self) -> f64 {
        (other - self).length()
    }
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    /// Creates a vector.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Whether every component is finite.
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Dot product.
    #[must_use]
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product.
    #[must_use]
    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }
}

macro_rules! array_conversions {
    ($t:ty) => {
        impl From<[f64; 3]> for $t {
            fn from([x, y, z]: [f64; 3]) -> Self {
                Self::new(x, y, z)
            }
        }
        impl From<(f64, f64, f64)> for $t {
            fn from((x, y, z): (f64, f64, f64)) -> Self {
                Self::new(x, y, z)
            }
        }
        impl From<$t> for [f64; 3] {
            fn from(v: $t) -> Self {
                [v.x, v.y, v.z]
            }
        }
    };
}
array_conversions!(Point3);
array_conversions!(Vec3);

impl Add<Vec3> for Point3 {
    type Output = Point3;
    fn add(self, v: Vec3) -> Point3 {
        Point3::new(self.x + v.x, self.y + v.y, self.z + v.z)
    }
}
impl AddAssign<Vec3> for Point3 {
    fn add_assign(&mut self, v: Vec3) {
        *self = *self + v;
    }
}
impl Sub<Vec3> for Point3 {
    type Output = Point3;
    fn sub(self, v: Vec3) -> Point3 {
        Point3::new(self.x - v.x, self.y - v.y, self.z - v.z)
    }
}
impl SubAssign<Vec3> for Point3 {
    fn sub_assign(&mut self, v: Vec3) {
        *self = *self - v;
    }
}
impl Sub for Point3 {
    type Output = Vec3;
    fn sub(self, o: Point3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}
impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}
impl Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}
impl Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, s: f64) -> Vec3 {
        Vec3::new(self.x / s, self.y / s, self.z / s)
    }
}

/// A unit-length direction. The constructor guarantees the invariant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dir3(Vec3);

/// The vector had no direction: zero length or non-finite components.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZeroDirection;

impl fmt::Display for ZeroDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a direction needs a finite, nonzero vector")
    }
}
impl std::error::Error for ZeroDirection {}

impl Dir3 {
    /// +X.
    pub const X: Self = Self(Vec3::new(1.0, 0.0, 0.0));
    /// +Y.
    pub const Y: Self = Self(Vec3::new(0.0, 1.0, 0.0));
    /// +Z.
    pub const Z: Self = Self(Vec3::new(0.0, 0.0, 1.0));

    /// Normalises a vector.
    ///
    /// # Errors
    /// [`ZeroDirection`] if the vector has zero length or non-finite components.
    pub fn new(v: impl Into<Vec3>) -> Result<Self, ZeroDirection> {
        let v = v.into();
        let length = v.length();
        if v.is_finite() && length > 0.0 {
            Ok(Self(v / length))
        } else {
            Err(ZeroDirection)
        }
    }

    /// The unit vector.
    #[must_use]
    pub const fn vector(self) -> Vec3 {
        self.0
    }
}

impl TryFrom<Vec3> for Dir3 {
    type Error = ZeroDirection;
    fn try_from(v: Vec3) -> Result<Self, ZeroDirection> {
        Self::new(v)
    }
}
impl TryFrom<[f64; 3]> for Dir3 {
    type Error = ZeroDirection;
    fn try_from(v: [f64; 3]) -> Result<Self, ZeroDirection> {
        Self::new(v)
    }
}
impl From<Dir3> for Vec3 {
    fn from(d: Dir3) -> Vec3 {
        d.0
    }
}
impl Neg for Dir3 {
    type Output = Dir3;
    fn neg(self) -> Dir3 {
        Dir3(-self.0)
    }
}

/// A line in space: an origin and a direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Axis {
    /// A point on the line.
    pub origin: Point3,
    /// The direction of the line.
    pub direction: Dir3,
}

impl Axis {
    /// The world X axis.
    pub const X: Self = Self::new(Point3::ORIGIN, Dir3::X);
    /// The world Y axis.
    pub const Y: Self = Self::new(Point3::ORIGIN, Dir3::Y);
    /// The world Z axis.
    pub const Z: Self = Self::new(Point3::ORIGIN, Dir3::Z);

    /// Creates an axis.
    #[must_use]
    pub const fn new(origin: Point3, direction: Dir3) -> Self {
        Self { origin, direction }
    }
}

/// A right-handed coordinate frame: an origin, a Z direction and an X direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    /// The origin.
    pub origin: Point3,
    /// The Z axis direction.
    pub z: Dir3,
    /// The X axis direction, perpendicular to `z`.
    pub x: Dir3,
}

impl Frame {
    /// The world frame.
    pub const WORLD: Self = Self {
        origin: Point3::ORIGIN,
        z: Dir3::Z,
        x: Dir3::X,
    };

    /// Builds a frame from an origin, a Z direction and a hint for X. The hint is made
    /// perpendicular to `z`.
    ///
    /// # Errors
    /// [`ZeroDirection`] if the hint is parallel to `z`.
    pub fn new(origin: Point3, z: Dir3, x_hint: Vec3) -> Result<Self, ZeroDirection> {
        let zv = z.vector();
        let x = Dir3::new(x_hint - zv * x_hint.dot(zv))?;
        Ok(Self { origin, z, x })
    }

    /// A frame at `origin` with this Z and an arbitrary perpendicular X.
    #[must_use]
    pub fn from_normal(origin: Point3, z: Dir3) -> Self {
        let zv = z.vector();
        let hint = if zv.x.abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        Self::new(origin, z, hint).unwrap_or(Self {
            origin,
            z,
            x: Dir3::X,
        })
    }

    /// The Y axis direction.
    #[must_use]
    pub fn y(self) -> Dir3 {
        Dir3(self.z.vector().cross(self.x.vector()))
    }
}

/// An infinite plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    /// A point on the plane.
    pub origin: Point3,
    /// The plane normal.
    pub normal: Dir3,
}

impl Plane {
    /// The XY plane.
    pub const XY: Self = Self {
        origin: Point3::ORIGIN,
        normal: Dir3::Z,
    };
    /// The YZ plane.
    pub const YZ: Self = Self {
        origin: Point3::ORIGIN,
        normal: Dir3::X,
    };
    /// The ZX plane.
    pub const ZX: Self = Self {
        origin: Point3::ORIGIN,
        normal: Dir3::Y,
    };

    /// Creates a plane.
    #[must_use]
    pub const fn new(origin: Point3, normal: Dir3) -> Self {
        Self { origin, normal }
    }
}

/// An affine transform, stored as the top three rows of a 4x4 matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    /// Rows of the matrix; column 3 holds the translation.
    pub rows: [[f64; 4]; 3],
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    /// The identity.
    pub const IDENTITY: Self = Self {
        rows: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    };

    /// A translation.
    #[must_use]
    pub const fn translation(by: Vec3) -> Self {
        Self {
            rows: [
                [1.0, 0.0, 0.0, by.x],
                [0.0, 1.0, 0.0, by.y],
                [0.0, 0.0, 1.0, by.z],
            ],
        }
    }

    /// A rotation about an axis.
    #[must_use]
    pub fn rotation(axis: Axis, angle: Angle) -> Self {
        let Vec3 { x, y, z } = axis.direction.vector();
        let (s, c) = angle.as_radians().sin_cos();
        let t = 1.0 - c;
        let m = Self {
            rows: [
                [t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0.0],
                [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0.0],
                [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0.0],
            ],
        };
        Self::about(axis.origin, m)
    }

    /// A scale about a point, per axis. Non-uniform factors make it non-rigid.
    #[must_use]
    pub fn scale(center: Point3, factors: Vec3) -> Self {
        let m = Self {
            rows: [
                [factors.x, 0.0, 0.0, 0.0],
                [0.0, factors.y, 0.0, 0.0],
                [0.0, 0.0, factors.z, 0.0],
            ],
        };
        Self::about(center, m)
    }

    /// A reflection through a plane.
    #[must_use]
    pub fn mirror(plane: Plane) -> Self {
        let Vec3 { x, y, z } = plane.normal.vector();
        let m = Self {
            rows: [
                [1.0 - 2.0 * x * x, -2.0 * x * y, -2.0 * x * z, 0.0],
                [-2.0 * x * y, 1.0 - 2.0 * y * y, -2.0 * y * z, 0.0],
                [-2.0 * x * z, -2.0 * y * z, 1.0 - 2.0 * z * z, 0.0],
            ],
        };
        Self::about(plane.origin, m)
    }

    fn about(p: Point3, m: Self) -> Self {
        let to = Self::translation(p - Point3::ORIGIN);
        let from = Self::translation(Point3::ORIGIN - p);
        to.then_after(m).then_after(from)
    }

    /// The transform that applies `other` first and then `self`.
    #[must_use]
    pub fn then_after(self, other: Self) -> Self {
        let mut rows = [[0.0; 4]; 3];
        for (i, row) in rows.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (0..3)
                    .map(|k| self.rows[i][k] * other.rows[k][j])
                    .sum::<f64>()
                    + if j == 3 { self.rows[i][3] } else { 0.0 };
            }
        }
        Self { rows }
    }

    /// Applies the transform to a point.
    #[must_use]
    pub fn point(self, p: Point3) -> Point3 {
        let r = self.rows;
        Point3::new(
            r[0][0] * p.x + r[0][1] * p.y + r[0][2] * p.z + r[0][3],
            r[1][0] * p.x + r[1][1] * p.y + r[1][2] * p.z + r[1][3],
            r[2][0] * p.x + r[2][1] * p.y + r[2][2] * p.z + r[2][3],
        )
    }

    /// Applies the linear part of the transform to a vector.
    #[must_use]
    pub fn vector(self, v: Vec3) -> Vec3 {
        let r = self.rows;
        Vec3::new(
            r[0][0] * v.x + r[0][1] * v.y + r[0][2] * v.z,
            r[1][0] * v.x + r[1][1] * v.y + r[1][2] * v.z,
            r[2][0] * v.x + r[2][1] * v.y + r[2][2] * v.z,
        )
    }

    /// Whether all entries are finite.
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.rows.iter().flatten().all(|v| v.is_finite())
    }
}

/// An angle. Construct from the unit you have; no bare-radians confusion.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Angle(f64);

impl Angle {
    /// A full turn.
    pub const FULL_TURN: Self = Self(std::f64::consts::TAU);

    /// From radians.
    #[must_use]
    pub const fn radians(radians: f64) -> Self {
        Self(radians)
    }

    /// From degrees.
    #[must_use]
    pub const fn degrees(degrees: f64) -> Self {
        Self(degrees.to_radians())
    }

    /// The value in radians.
    #[must_use]
    pub const fn as_radians(self) -> f64 {
        self.0
    }

    /// The value in degrees.
    #[must_use]
    pub const fn as_degrees(self) -> f64 {
        self.0.to_degrees()
    }
}

/// An axis-aligned bounding box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Lowest corner.
    pub min: Point3,
    /// Highest corner.
    pub max: Point3,
}

impl Aabb {
    /// Edge lengths along each axis.
    #[must_use]
    pub fn size(self) -> Vec3 {
        self.max - self.min
    }

    /// The middle of the box.
    #[must_use]
    pub fn center(self) -> Point3 {
        self.min + self.size() / 2.0
    }

    /// Whether the point is inside or on the box.
    #[must_use]
    pub fn contains(self, p: Point3) -> bool {
        (self.min.x..=self.max.x).contains(&p.x)
            && (self.min.y..=self.max.y).contains(&p.y)
            && (self.min.z..=self.max.z).contains(&p.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_is_normalised_and_rejects_zero() {
        let d = Dir3::new([3.0, 0.0, 4.0]).unwrap();
        assert!((d.vector().length() - 1.0).abs() < 1e-12);
        assert_eq!(Dir3::new([0.0; 3]), Err(ZeroDirection));
        assert_eq!(Dir3::new([f64::NAN, 1.0, 0.0]), Err(ZeroDirection));
    }

    #[test]
    fn point_vector_arithmetic() {
        let p = Point3::new(1.0, 2.0, 3.0) + Vec3::new(1.0, 1.0, 1.0);
        assert_eq!(p, Point3::new(2.0, 3.0, 4.0));
        assert_eq!(p - Point3::new(2.0, 3.0, 3.0), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(
            Vec3::new(1.0, 0.0, 0.0).cross(Vec3::new(0.0, 1.0, 0.0)),
            Vec3::new(0.0, 0.0, 1.0)
        );
    }

    #[test]
    fn transforms_compose_and_reflect() {
        let t = Transform::translation(Vec3::new(1.0, 0.0, 0.0));
        let r = Transform::rotation(Axis::Z, Angle::degrees(90.0));
        let p = t.then_after(r).point(Point3::new(1.0, 0.0, 0.0));
        assert!((p.x - 1.0).abs() < 1e-12 && (p.y - 1.0).abs() < 1e-12);
        let about = Transform::rotation(
            Axis::new(Point3::new(1.0, 0.0, 0.0), Dir3::Z),
            Angle::degrees(180.0),
        );
        assert!(
            about
                .point(Point3::ORIGIN)
                .distance(Point3::new(2.0, 0.0, 0.0))
                < 1e-12
        );
        let m = Transform::mirror(Plane::new(Point3::new(0.0, 0.0, 1.0), Dir3::Z));
        assert!(m.point(Point3::ORIGIN).distance(Point3::new(0.0, 0.0, 2.0)) < 1e-12);
        assert!(
            Frame::from_normal(Point3::ORIGIN, Dir3::X)
                .x
                .vector()
                .dot(Vec3::new(1.0, 0.0, 0.0))
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn angle_units() {
        assert!((Angle::degrees(180.0).as_radians() - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn aabb_queries() {
        let b = Aabb {
            min: Point3::ORIGIN,
            max: Point3::new(2.0, 2.0, 2.0),
        };
        assert_eq!(b.center(), Point3::new(1.0, 1.0, 1.0));
        assert!(b.contains(Point3::new(2.0, 0.0, 1.0)));
        assert!(!b.contains(Point3::new(2.1, 0.0, 1.0)));
    }
}
