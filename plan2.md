# Unit-sphere `f64` clipping error bounds (short)

“Unit sphere” fixes **scale**: \\(\\|p\\|\\le 1\\), \\(\\|n\\|\\approx 1\\), so typical dot/cross magnitudes are \\(O(1)\\). What still dominates error is **conditioning** (small denominators / near-parallel geometry). Let \\(u=2^{-53}\\) (i.e., `f64::EPSILON/2`).

A useful worst-case structure becomes:
\\[
\\text{position error} \\;\\lesssim\\; C\\,u\\sum_{k=1}^{N}\\kappa_k
\\]
where \\(\\kappa_k\\) is the step conditioning (you can upper-bound by \\(N\\kappa_{\\max}\\) for a single number, but it’s very conservative).

## Case A: vertex from two planes on the sphere
If you form a vertex as:
- \\(\\ell = n_a \\times n_b\\)
- \\(x = \\ell/\\|\\ell\\|\\) (choose sign separately)

Let \\(\\sin\\theta = \\|\\ell\\|\\). Then a forward bound has the form:
\\[
\\angle(\\hat x, x) \\;\\le\\; \\frac{C\\,u}{\\sin\\theta_{\\text{lo}}}
\\quad\\Rightarrow\\quad
\\|\\hat x - x\\| \\le \\angle(\\hat x, x)\\;,
\\]
because on the unit sphere, chord distance \\(\\le\\) angular distance (radians). Use a conservative lower bound
\\[
\\sin\\theta_{\\text{lo}} = \\max(\\|\\widehat\\ell\\| - E_\\ell,\\; \\text{EPS}),
\\]
with \\(E_\\ell = O(u)\\) from the `cross` arithmetic.

**Accumulation:** if every vertex is computed directly from plane normals (not from previous vertices), there’s effectively **no accumulation across clips**; each vertex has its own bound dominated by \\(1/\\sin\\theta\\).

## Case B: segment/lerp intersection (Sutherland–Hodgman style)
For half-plane clipping with:
\\[
s_i = n\\cdot p_i - d,\\quad
t = \\frac{s_0}{s_0-s_1},\\quad
x = p_0 + t(p_1-p_0),
\\]
on the unit sphere \\(|n\\cdot p|\\le 1\\) and typically \\(|d|\\le 1\\) if normalized.

A rigorous per-intersection bound has the shape:
\\[
\\|\\hat x - x\\| \\;\\le\\; C\\,u\\left(\\frac{1}{\\delta}+\\frac{1}{\\delta^2}\\right),
\\]
where \\(\\delta = |s_0-s_1|\\), and in practice you must use a **lower bound**
\\[
\\delta_{\\text{lo}} = \\max(|\\widehat{s_0-s_1}| - E_{\\text{den}},\\; 0).
\\]
If \\(\\delta_{\\text{lo}}\\) is small, the bound must blow up (the intersection is ill-conditioned).

**Accumulation:** if later clips consume previously computed vertices, a worst-case accumulated bound is:
\\[
R_N \\le \\sum_{k=1}^{N} r_k
\\quad\\text{(or)}\\quad
R_N \\le N\\,r_{\\max}.
\\]

## What “unit sphere” changes
- Makes “worst-case in terms of \\(u\\) and conditioning” meaningful (no unknown coordinate scale factors).
- Does **not** remove conditioning as the dominant term.
- “Accumulated error” is only a real issue when outputs are fed as inputs in later steps.

If you specify whether you’re in **Case A** (`normalize(cross)`) or **Case B** (segment intersection), the conditioning term and tight constants are unambiguous.
