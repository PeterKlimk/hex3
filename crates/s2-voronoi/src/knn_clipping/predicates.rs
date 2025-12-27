use glam::DVec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sign {
    Neg,
    Zero,
    Pos,
}

/// Compute determinant of 3x3 matrix [a|b|c] using standard f64 arithmetic.
/// This is useful for debugging but not for predicate decisions.
#[inline]
pub fn det3_f64(a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let (ax, ay, az) = (a.x, a.y, a.z);
    let (bx, by, bz) = (b.x, b.y, b.z);
    let (cx, cy, cz) = (c.x, c.y, c.z);
    ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
}

/// Compute the exact sign of det([a|b|c]) using Shewchuk-style adaptive predicates.
///
/// This uses the `robust` crate's `orient3d` which internally:
/// 1. Tries a fast floating-point filter (resolves ~99% of cases)
/// 2. Falls back to exact expansion arithmetic if needed
///
/// For finite inputs, this always returns a definite sign (never uncertain).
#[inline]
pub fn det3_sign(a: DVec3, b: DVec3, c: DVec3) -> Sign {
    use robust::Coord3D;

    // robust::orient3d computes the signed volume of tetrahedron (p0, p1, p2, p3).
    // Equivalently: sign of det([p1-p0 | p2-p0 | p3-p0])
    // For det([a|b|c]), we use origin as p0, so it becomes det([a|b|c]).
    let result = robust::orient3d(
        Coord3D { x: 0.0, y: 0.0, z: 0.0 },
        Coord3D { x: a.x, y: a.y, z: a.z },
        Coord3D { x: b.x, y: b.y, z: b.z },
        Coord3D { x: c.x, y: c.y, z: c.z },
    );

    if result > 0.0 {
        Sign::Pos
    } else if result < 0.0 {
        Sign::Neg
    } else {
        Sign::Zero
    }
}
