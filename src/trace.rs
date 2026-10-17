//! Sub-pixel tracer.
//!
//! The quantized image says which color each pixel *mostly* is. Anti-aliased
//! edge pixels are blends of two colors, and the blend fraction tells us where
//! inside the pixel the true edge lies. We turn each connected region into a
//! soft membership field (1 inside, the blend fraction on the edge ring, 0
//! outside), extract its 0.5 iso-contour with marching squares (which lands the
//! edge at sub-pixel positions), smooth it while preserving corners, and fit
//! cubic Beziers with a least-squares fitter that picks as few nodes as the
//! error budget allows.

use crate::quantize::{rgb_to_lab, Lab};
use image::RgbaImage;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct TraceOptions {
    /// Regions smaller than this many pixels are dropped.
    pub min_area: usize,
    /// Corner detection: turning angle in degrees above which a point is a corner.
    pub corner_angle: f32,
    /// Whether to detect corners at all (else everything is smoothed).
    pub sharp_corners: bool,
    /// Contour smoothing radius in points (0 = none).
    pub smooth_radius: usize,
    /// Max fit error in pixels between the contour and the Bezier curve.
    pub fit_error: f64,
    /// Emit straight segments only.
    pub polygon_only: bool,
    /// Keep holes as holes (cut-out) or fill them and let smaller shapes paint on top (stacked).
    pub cutout: bool,
    /// Decimal places in output coordinates.
    pub precision: u32,
}

/// One output shape: a compound path in pixel coordinates.
#[derive(Clone, Debug)]
pub struct TracedShape {
    pub fill: [u8; 3],
    pub area: usize,
    /// SVG path data (absolute M/L/C/Z only).
    pub d: String,
    pub even_odd: bool,
    pub node_count: usize,
}

type Pt = (f64, f64);

// ---------------------------------------------------------------------------
// Labeling and blend fractions
// ---------------------------------------------------------------------------

const NONE: u16 = u16::MAX;

fn label_map(quant: &RgbaImage, palette: &[[u8; 3]]) -> Vec<u16> {
    let lut: HashMap<[u8; 3], u16> = palette.iter().enumerate().map(|(i, c)| (*c, i as u16)).collect();
    quant
        .as_raw()
        .par_chunks_exact(4)
        .map(|p| if p[3] < 128 { NONE } else { *lut.get(&[p[0], p[1], p[2]]).unwrap_or(&NONE) })
        .collect()
}

/// For every pixel: how much of it belongs to its own label (0..1) and which
/// neighboring label it is blended with (NONE if interior).
fn blend_fractions(src: &RgbaImage, labels: &[u16], pal_lab: &[Lab], w: u32, h: u32) -> (Vec<f32>, Vec<u16>) {
    let px = src.as_raw();
    let n = (w * h) as usize;
    let out: Vec<(f32, u16)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let me = labels[i];
            if me == NONE {
                return (0.0, NONE);
            }
            let (x, y) = ((i as u32 % w) as i32, (i as u32 / w) as i32);
            // Most different neighbor label among the 8 neighbors.
            let mut best: Option<(u16, f32)> = None;
            let a = pal_lab[me as usize];
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
                let (xx, yy) = (x + dx, y + dy);
                if xx < 0 || yy < 0 || xx >= w as i32 || yy >= h as i32 {
                    continue;
                }
                let j = (yy as u32 * w + xx as u32) as usize;
                let other = labels[j];
                if other == me {
                    continue;
                }
                let d = if other == NONE { 1.0 } else { a.dist2(pal_lab[other as usize]) };
                if best.map_or(true, |(_, bd)| d > bd) {
                    best = Some((other, d));
                }
            }
            let Some((other, _)) = best else { return (1.0, NONE) };
            let p = &px[i * 4..i * 4 + 4];
            let c = rgb_to_lab(p[0], p[1], p[2]);
            let b = if other == NONE {
                // Blended toward transparency: alpha is the fraction.
                return ((p[3] as f32 / 255.0).clamp(0.0, 1.0), other);
            } else {
                pal_lab[other as usize]
            };
            let (abx, aby, abz) = (a.l - b.l, a.a - b.a, a.b - b.b);
            let denom = abx * abx + aby * aby + abz * abz;
            if denom < 1e-9 {
                return (1.0, other);
            }
            let t = ((c.l - b.l) * abx + (c.a - b.a) * aby + (c.b - b.b) * abz) / denom;
            (t.clamp(0.0, 1.0), other)
        })
        .collect();
    out.into_iter().unzip()
}

// ---------------------------------------------------------------------------
// Connected components
// ---------------------------------------------------------------------------

struct Component {
    label: u16,
    area: usize,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    pixels: Vec<u32>,
}

fn components(labels: &[u16], w: u32, h: u32) -> (Vec<Component>, Vec<u32>) {
    let n = (w * h) as usize;
    let mut comp = vec![u32::MAX; n];
    let mut out = Vec::new();
    let mut stack: Vec<u32> = Vec::new();
    for start in 0..n {
        if comp[start] != u32::MAX || labels[start] == NONE {
            continue;
        }
        let id = out.len() as u32;
        let lab = labels[start];
        let mut c = Component { label: lab, area: 0, x0: u32::MAX, y0: u32::MAX, x1: 0, y1: 0, pixels: Vec::new() };
        comp[start] = id;
        stack.push(start as u32);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            c.area += 1;
            c.x0 = c.x0.min(x);
            c.y0 = c.y0.min(y);
            c.x1 = c.x1.max(x);
            c.y1 = c.y1.max(y);
            c.pixels.push(i);
            let mut visit = |j: u32| {
                let j = j as usize;
                if comp[j] == u32::MAX && labels[j] == lab {
                    comp[j] = id;
                    stack.push(j as u32);
                }
            };
            if x > 0 { visit(i - 1); }
            if x + 1 < w { visit(i + 1); }
            if y > 0 { visit(i - w); }
            if y + 1 < h { visit(i + w); }
        }
        out.push(c);
    }
    (out, comp)
}

/// Nesting depth of every component: 0 for regions touching the border or
/// contained by nothing, 1 for regions directly inside a depth-0 region, etc.
/// The parent is whichever component owns the pixel just left of the
/// component's leftmost pixel.
fn nesting_depth(comps: &[Component], comp_map: &[u32], w: u32) -> Vec<u32> {
    let n = comps.len();
    let mut parent = vec![u32::MAX; n];
    for (i, c) in comps.iter().enumerate() {
        if c.x0 == 0 {
            continue;
        }
        // Leftmost pixel of the component (any row).
        let Some(&px) = c.pixels.iter().find(|&&p| p % w == c.x0) else { continue };
        let left = comp_map[(px - 1) as usize];
        if left != u32::MAX && left as usize != i {
            parent[i] = left;
        }
    }
    let mut depth = vec![u32::MAX; n];
    for i in 0..n {
        if depth[i] != u32::MAX {
            continue;
        }
        let mut chain = vec![i];
        let mut cur = i;
        let mut d = 0u32;
        let mut guard = 0;
        loop {
            let p = parent[cur];
            if p == u32::MAX {
                break;
            }
            if depth[p as usize] != u32::MAX {
                d = depth[p as usize] + 1;
                break;
            }
            cur = p as usize;
            chain.push(cur);
            guard += 1;
            if guard > n {
                break; // cycle guard
            }
        }
        // chain holds [i, parent(i), ...] up to the first already-resolved ancestor; assign depths bottom-up.
        let base = if parent[*chain.last().unwrap()] == u32::MAX { 0 } else { d };
        for (k, &c) in chain.iter().rev().enumerate() {
            depth[c] = base + k as u32;
        }
    }
    depth
}

// ---------------------------------------------------------------------------
// Marching squares
// ---------------------------------------------------------------------------

/// Extract closed 0.5 iso-contours from a scalar field sampled at pixel
/// centers. `field` is (fw x fh), sample (i, j) sits at world (ox + i, oy + j).
fn iso_contours(field: &[f32], fw: usize, fh: usize, ox: f64, oy: f64) -> Vec<Vec<Pt>> {
    let iso = 0.5f32;
    let at = |i: usize, j: usize| field[j * fw + i];
    // Edge keys: horizontal edge between (i,j)-(i+1,j): key = (j*fw+i)*2; vertical (i,j)-(i,j+1): +1
    let hkey = |i: usize, j: usize| ((j * fw + i) * 2) as u64;
    let vkey = |i: usize, j: usize| ((j * fw + i) * 2 + 1) as u64;
    let lerp = |a: f32, b: f32| -> f64 {
        let d = b - a;
        if d.abs() < 1e-12 { 0.5 } else { ((iso - a) / d).clamp(0.0, 1.0) as f64 }
    };
    // segments: (key_from, key_to) with points
    let mut segs: Vec<(u64, u64, Pt, Pt)> = Vec::new();
    for j in 0..fh - 1 {
        for i in 0..fw - 1 {
            let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
            let code = ((a >= iso) as u8) | ((b >= iso) as u8) << 1 | ((c >= iso) as u8) << 2 | ((d >= iso) as u8) << 3;
            if code == 0 || code == 15 {
                continue;
            }
            // Edge midpoints with interpolation.
            let top = (hkey(i, j), (ox + i as f64 + lerp(a, b), oy + j as f64));
            let right = (vkey(i + 1, j), (ox + (i + 1) as f64, oy + j as f64 + lerp(b, c)));
            let bottom = (hkey(i, j + 1), (ox + i as f64 + lerp(d, c), oy + (j + 1) as f64));
            let left = (vkey(i, j), (ox + i as f64, oy + j as f64 + lerp(a, d)));
            let mut push = |p: (u64, Pt), q: (u64, Pt)| segs.push((p.0, q.0, p.1, q.1));
            // Orientation: inside on the left when walking from p to q (counter-clockwise in y-down = consistent).
            match code {
                1 => push(left, top),
                2 => push(top, right),
                3 => push(left, right),
                4 => push(right, bottom),
                5 => {
                    let center = (a + b + c + d) * 0.25;
                    if center >= iso { push(left, bottom); push(right, top); } else { push(left, top); push(right, bottom); }
                }
                6 => push(top, bottom),
                7 => push(left, bottom),
                8 => push(bottom, left),
                9 => push(bottom, top),
                10 => {
                    let center = (a + b + c + d) * 0.25;
                    if center >= iso { push(top, left); push(bottom, right); } else { push(top, right); push(bottom, left); }
                }
                11 => push(bottom, right),
                12 => push(right, left),
                13 => push(right, top),
                14 => push(top, left),
                _ => {}
            }
        }
    }
    // Link segments into loops.
    let mut next: HashMap<u64, Vec<usize>> = HashMap::new();
    for (idx, s) in segs.iter().enumerate() {
        next.entry(s.0).or_default().push(idx);
    }
    let mut used = vec![false; segs.len()];
    let mut loops = Vec::new();
    for start in 0..segs.len() {
        if used[start] {
            continue;
        }
        let mut loop_pts: Vec<Pt> = Vec::new();
        let mut cur = start;
        loop {
            used[cur] = true;
            loop_pts.push(segs[cur].2);
            let to = segs[cur].1;
            let Some(cands) = next.get(&to) else { break };
            let Some(&nxt) = cands.iter().find(|&&c| !used[c]) else { break };
            cur = nxt;
            if cur == start {
                break;
            }
        }
        if loop_pts.len() >= 3 {
            loops.push(loop_pts);
        }
    }
    loops
}

fn signed_area(p: &[Pt]) -> f64 {
    let n = p.len();
    let mut a = 0.0;
    for i in 0..n {
        let (x1, y1) = p[i];
        let (x2, y2) = p[(i + 1) % n];
        a += x1 * y2 - x2 * y1;
    }
    a * 0.5
}

// ---------------------------------------------------------------------------
// Contour cleanup: resample, corners, smoothing
// ---------------------------------------------------------------------------

/// Resample a closed polyline so consecutive points are ~`step` apart.
fn resample(p: &[Pt], step: f64) -> Vec<Pt> {
    let n = p.len();
    let mut out = Vec::with_capacity(n);
    let mut carry = 0.0;
    out.push(p[0]);
    for i in 0..n {
        let (x1, y1) = p[i];
        let (x2, y2) = p[(i + 1) % n];
        let len = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
        if len < 1e-9 {
            continue;
        }
        let mut t = step - carry;
        while t < len {
            let u = t / len;
            out.push((x1 + (x2 - x1) * u, y1 + (y2 - y1) * u));
            t += step;
        }
        carry = len - (t - step);
    }
    if out.len() >= 2 {
        let (fx, fy) = out[0];
        let (lx, ly) = *out.last().unwrap();
        if ((fx - lx).powi(2) + (fy - ly).powi(2)).sqrt() < step * 0.5 {
            out.pop();
        }
    }
    out
}

/// Turning angle (degrees) at each point, estimated over a window of `k` points.
fn corner_flags(p: &[Pt], k: usize, angle_deg: f32) -> Vec<bool> {
    let n = p.len();
    if n < 2 * k + 3 {
        return vec![false; n];
    }
    let mut angles = vec![0.0f32; n];
    for i in 0..n {
        let a = p[(i + n - k) % n];
        let b = p[i];
        let c = p[(i + k) % n];
        let (ux, uy) = (b.0 - a.0, b.1 - a.1);
        let (vx, vy) = (c.0 - b.0, c.1 - b.1);
        let lu = (ux * ux + uy * uy).sqrt();
        let lv = (vx * vx + vy * vy).sqrt();
        if lu < 1e-9 || lv < 1e-9 {
            continue;
        }
        let cosang = ((ux * vx + uy * vy) / (lu * lv)).clamp(-1.0, 1.0);
        angles[i] = cosang.acos().to_degrees() as f32;
    }
    // Non-maximum suppression: a corner is a local max above threshold.
    let mut flags = vec![false; n];
    for i in 0..n {
        if angles[i] < angle_deg {
            continue;
        }
        let mut is_max = true;
        for d in 1..=k {
            if angles[(i + d) % n] > angles[i] || angles[(i + n - d) % n] > angles[i] {
                is_max = false;
                break;
            }
        }
        flags[i] = is_max;
    }
    flags
}

/// Moving-average smoothing that does not cross or move corner points.
fn smooth_between_corners(p: &[Pt], corners: &[bool], radius: usize) -> Vec<Pt> {
    let n = p.len();
    if radius == 0 || n < 5 {
        return p.to_vec();
    }
    let mut out = p.to_vec();
    for i in 0..n {
        if corners[i] {
            continue;
        }
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut cnt = 0.0;
        // Expand symmetric window until a corner blocks it.
        let mut r = radius;
        for d in 1..=radius {
            if corners[(i + d) % n] || corners[(i + n - d) % n] {
                r = d - 1;
                break;
            }
        }
        for d in 0..=r {
            let wgt = (r + 1 - d) as f64;
            if d == 0 {
                sx += p[i].0 * wgt;
                sy += p[i].1 * wgt;
                cnt += wgt;
            } else {
                let a = p[(i + d) % n];
                let b = p[(i + n - d) % n];
                sx += (a.0 + b.0) * wgt;
                sy += (a.1 + b.1) * wgt;
                cnt += 2.0 * wgt;
            }
        }
        out[i] = (sx / cnt, sy / cnt);
    }
    out
}

// ---------------------------------------------------------------------------
// Bezier fitting (Schneider, "An Algorithm for Automatically Fitting Digitized Curves")
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Cubic {
    p0: Pt,
    p1: Pt,
    p2: Pt,
    p3: Pt,
}

fn sub(a: Pt, b: Pt) -> Pt { (a.0 - b.0, a.1 - b.1) }
fn add(a: Pt, b: Pt) -> Pt { (a.0 + b.0, a.1 + b.1) }
fn mul(a: Pt, s: f64) -> Pt { (a.0 * s, a.1 * s) }
fn dot(a: Pt, b: Pt) -> f64 { a.0 * b.0 + a.1 * b.1 }
fn norm(a: Pt) -> Pt {
    let l = dot(a, a).sqrt();
    if l < 1e-12 { (0.0, 0.0) } else { (a.0 / l, a.1 / l) }
}
fn dist2(a: Pt, b: Pt) -> f64 { let d = sub(a, b); dot(d, d) }

fn bezier_at(c: &Cubic, t: f64) -> Pt {
    let mt = 1.0 - t;
    let a = mt * mt * mt;
    let b = 3.0 * mt * mt * t;
    let cc = 3.0 * mt * t * t;
    let d = t * t * t;
    (a * c.p0.0 + b * c.p1.0 + cc * c.p2.0 + d * c.p3.0, a * c.p0.1 + b * c.p1.1 + cc * c.p2.1 + d * c.p3.1)
}

fn chord_params(pts: &[Pt]) -> Vec<f64> {
    let mut u = vec![0.0; pts.len()];
    for i in 1..pts.len() {
        u[i] = u[i - 1] + dist2(pts[i], pts[i - 1]).sqrt();
    }
    let total = *u.last().unwrap();
    if total > 1e-12 {
        for v in u.iter_mut() {
            *v /= total;
        }
    }
    u
}

fn generate_bezier(pts: &[Pt], u: &[f64], t0: Pt, t1: Pt) -> Cubic {
    let n = pts.len();
    let first = pts[0];
    let last = pts[n - 1];
    let mut c = [[0.0f64; 2]; 2];
    let mut x = [0.0f64; 2];
    for i in 0..n {
        let t = u[i];
        let mt = 1.0 - t;
        let b1 = 3.0 * mt * mt * t;
        let b2 = 3.0 * mt * t * t;
        let a0 = mul(t0, b1);
        let a1 = mul(t1, b2);
        c[0][0] += dot(a0, a0);
        c[0][1] += dot(a0, a1);
        c[1][1] += dot(a1, a1);
        let b0 = mt * mt * mt;
        let b3 = t * t * t;
        let tmp = sub(pts[i], add(add(mul(first, b0 + b1), mul(last, b2 + b3)), (0.0, 0.0)));
        x[0] += dot(a0, tmp);
        x[1] += dot(a1, tmp);
    }
    c[1][0] = c[0][1];
    let det_c = c[0][0] * c[1][1] - c[1][0] * c[0][1];
    let det_c0x = c[0][0] * x[1] - c[1][0] * x[0];
    let det_xc1 = x[0] * c[1][1] - x[1] * c[0][1];
    let (mut al, mut ar) = if det_c.abs() < 1e-12 {
        (0.0, 0.0)
    } else {
        (det_xc1 / det_c, det_c0x / det_c)
    };
    let seg_len = dist2(first, last).sqrt();
    let eps = 1e-6 * seg_len;
    if al < eps || ar < eps || !al.is_finite() || !ar.is_finite() {
        al = seg_len / 3.0;
        ar = al;
    }
    Cubic { p0: first, p1: add(first, mul(t0, al)), p2: add(last, mul(t1, ar)), p3: last }
}

fn reparameterize(pts: &[Pt], u: &[f64], c: &Cubic) -> Vec<f64> {
    u.iter()
        .zip(pts)
        .map(|(&t, &p)| {
            // Newton-Raphson step on distance.
            let q = bezier_at(c, t);
            let d1 = {
                let mt = 1.0 - t;
                let a = mul(sub(c.p1, c.p0), 3.0 * mt * mt);
                let b = mul(sub(c.p2, c.p1), 6.0 * mt * t);
                let cc = mul(sub(c.p3, c.p2), 3.0 * t * t);
                add(add(a, b), cc)
            };
            let d2 = {
                let mt = 1.0 - t;
                let a = mul(add(sub(c.p2, mul(c.p1, 2.0)), c.p0), 6.0 * mt);
                let b = mul(add(sub(c.p3, mul(c.p2, 2.0)), c.p1), 6.0 * t);
                add(a, b)
            };
            let diff = sub(q, p);
            let num = dot(diff, d1);
            let den = dot(d1, d1) + dot(diff, d2);
            if den.abs() < 1e-12 { t } else { (t - num / den).clamp(0.0, 1.0) }
        })
        .collect()
}

fn max_error(pts: &[Pt], u: &[f64], c: &Cubic) -> (f64, usize) {
    let mut worst = 0.0;
    let mut idx = pts.len() / 2;
    for i in 1..pts.len() - 1 {
        let d = dist2(bezier_at(c, u[i]), pts[i]);
        if d > worst {
            worst = d;
            idx = i;
        }
    }
    (worst, idx)
}

fn fit_cubic(pts: &[Pt], t0: Pt, t1: Pt, err: f64, out: &mut Vec<Cubic>, depth: u32) {
    let n = pts.len();
    if n == 2 {
        let d = dist2(pts[0], pts[1]).sqrt() / 3.0;
        out.push(Cubic { p0: pts[0], p1: add(pts[0], mul(t0, d)), p2: add(pts[1], mul(t1, d)), p3: pts[1] });
        return;
    }
    let mut u = chord_params(pts);
    let mut c = generate_bezier(pts, &u, t0, t1);
    let (mut e, mut split) = max_error(pts, &u, &c);
    let err2 = err * err;
    if e < err2 {
        out.push(c);
        return;
    }
    if e < err2 * 16.0 {
        for _ in 0..4 {
            u = reparameterize(pts, &u, &c);
            c = generate_bezier(pts, &u, t0, t1);
            let (e2, s2) = max_error(pts, &u, &c);
            e = e2;
            split = s2;
            if e < err2 {
                out.push(c);
                return;
            }
        }
    }
    if depth > 24 || n < 4 {
        out.push(c);
        return;
    }
    let split = split.clamp(1, n - 2);
    // Center tangent at the split.
    let tc = norm(sub(pts[split - 1], pts[split + 1]));
    fit_cubic(&pts[..=split], t0, tc, err, out, depth + 1);
    fit_cubic(&pts[split..], mul(tc, -1.0), t1, err, out, depth + 1);
}

/// Fit a closed contour, splitting at corners. Returns cubic segments in order.
fn fit_closed(p: &[Pt], corners: &[bool], err: f64) -> Vec<Cubic> {
    let n = p.len();
    let corner_idx: Vec<usize> = (0..n).filter(|&i| corners[i]).collect();
    let mut out = Vec::new();
    if corner_idx.is_empty() {
        // Closed smooth loop: open it at point 0 with a shared tangent for C1 continuity.
        let mut pts: Vec<Pt> = p.to_vec();
        pts.push(p[0]);
        let t = norm(sub(p[1], p[n - 1]));
        fit_cubic(&pts, t, mul(t, -1.0), err, &mut out, 0);
        return out;
    }
    for k in 0..corner_idx.len() {
        let a = corner_idx[k];
        let b = corner_idx[(k + 1) % corner_idx.len()];
        // Walk from corner a to corner b. With a single corner (a == b, e.g. a
        // leaf with one sharp tip) walk the whole loop back to the same corner.
        let mut pts = Vec::with_capacity(n + 1);
        let mut i = a;
        loop {
            pts.push(p[i]);
            i = (i + 1) % n;
            if i == b {
                pts.push(p[b]);
                break;
            }
            if pts.len() > n + 1 {
                break;
            }
        }
        if pts.len() < 3 {
            continue;
        }
        let t0 = norm(sub(pts[1], pts[0]));
        let t1 = norm(sub(pts[pts.len() - 2], pts[pts.len() - 1]));
        fit_cubic(&pts, t0, t1, err, &mut out, 0);
    }
    out
}

fn fmt(v: f64, precision: u32) -> String {
    let s = format!("{:.*}", precision as usize, v);
    let s = if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s };
    if s == "-0" { "0".into() } else { s }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Trace a quantized image. `src` is the (smoothed) image the quantization was
/// computed from; it supplies the blend fractions.
pub fn trace(quant: &RgbaImage, src: &RgbaImage, palette: &[[u8; 3]], opts: &TraceOptions) -> Vec<TracedShape> {
    let (w, h) = quant.dimensions();
    let labels = label_map(quant, palette);
    let pal_lab: Vec<Lab> = palette.iter().map(|c| rgb_to_lab(c[0], c[1], c[2])).collect();
    let (frac, other) = blend_fractions(src, &labels, &pal_lab, w, h);
    let (all_comps, comp_map) = components(&labels, w, h);
    let depth = nesting_depth(&all_comps, &comp_map, w);
    let mut comps: Vec<(u32, Component)> = all_comps
        .into_iter()
        .enumerate()
        .filter(|(_, c)| c.area >= opts.min_area.max(1))
        .map(|(i, c)| (depth[i], c))
        .collect();
    // Paint order: enclosing regions first, then by size, so that in stacked
    // mode (holes filled) every inner region lands on top of what surrounds it.
    comps.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.area.cmp(&a.1.area)));
    let debug = std::env::var_os("VB_TRACE_DEBUG").is_some();
    let depths: Vec<u32> = comps.iter().map(|(d, _)| *d).collect();
    let comps: Vec<Component> = comps.into_iter().map(|(_, c)| c).collect();
    if debug {
        for (i, c) in comps.iter().enumerate() {
            eprintln!(
                "shape {i}: label {} depth {} area {} bbox [{}..{} x {}..{}]",
                c.label, depths[i], c.area, c.x0, c.x1, c.y0, c.y1
            );
        }
    }

    let step = 1.0;
    let k = 3usize;

    comps
        .par_iter()
        .filter_map(|c| {
            // Field over bbox padded by 2 (one for the blended ring, one for the zero border).
            let pad = 2u32;
            let fx0 = c.x0.saturating_sub(pad);
            let fy0 = c.y0.saturating_sub(pad);
            let fx1 = (c.x1 + pad).min(w - 1);
            let fy1 = (c.y1 + pad).min(h - 1);
            let fw = (fx1 - fx0 + 1) as usize + 2;
            let fh = (fy1 - fy0 + 1) as usize + 2;
            let mut field = vec![0.0f32; fw * fh];
            let mut mark = vec![false; fw * fh];
            let idx = |x: u32, y: u32| ((y - fy0 + 1) as usize) * fw + (x - fx0 + 1) as usize;
            for &i in &c.pixels {
                let (x, y) = (i % w, i / w);
                let fi = idx(x, y);
                // A pixel labeled with this color is at least half this color, so the
                // 0.5 iso-line keeps exactly the label map's topology (no holes lost,
                // no rings broken) while still sliding inside the pixel.
                field[fi] = frac[i as usize].clamp(0.5, 1.0);
                mark[fi] = true;
            }
            // Neighbors blended toward this label contribute 1 - their own fraction.
            for &i in &c.pixels {
                let (x, y) = (i % w, i / w);
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                    if xx < fx0 as i32 || yy < fy0 as i32 || xx > fx1 as i32 || yy > fy1 as i32 {
                        continue;
                    }
                    let j = (yy as u32 * w + xx as u32) as usize;
                    let fj = idx(xx as u32, yy as u32);
                    if mark[fj] || labels[j] == c.label {
                        continue;
                    }
                    if other[j] == c.label {
                        field[fj] = field[fj].max((1.0 - frac[j]).min(0.499));
                    }
                }
            }
            let loops = iso_contours(&field, fw, fh, fx0 as f64 - 1.0 + 0.5, fy0 as f64 - 1.0 + 0.5);
            if loops.is_empty() {
                return None;
            }
            // Loops are consistently oriented, so outer boundaries and holes have
            // opposite signed areas. The largest loop is certainly an outer one.
            let mut loops: Vec<(f64, Vec<Pt>)> = loops.into_iter().map(|l| (signed_area(&l), l)).collect();
            loops.sort_by(|a, b| b.0.abs().partial_cmp(&a.0.abs()).unwrap_or(std::cmp::Ordering::Equal));
            if loops[0].0.abs() < 0.5 {
                return None;
            }
            let outer_sign = loops[0].0.signum();
            if debug {
                let signs: Vec<String> = loops.iter().map(|(a, l)| format!("{:+.0}/{}pts", a, l.len())).collect();
                eprintln!(
                    "  comp label {} area {} bbox [{}..{} x {}..{}] loops: {}",
                    c.label, c.area, c.x0, c.x1, c.y0, c.y1, signs.join(" ")
                );
            }
            let keep: Vec<Vec<Pt>> = loops
                .into_iter()
                .filter(|(a, _)| a.abs() >= 0.5 && (opts.cutout || a.signum() == outer_sign))
                .map(|(_, l)| l)
                .collect();

            let mut d = String::new();
            let mut nodes = 0usize;
            for lp in keep {
                let rs = resample(&lp, step);
                if rs.len() < 4 {
                    continue;
                }
                let corners = if opts.sharp_corners { corner_flags(&rs, k, opts.corner_angle) } else { vec![false; rs.len()] };
                let sm = smooth_between_corners(&rs, &corners, opts.smooth_radius);
                if opts.polygon_only {
                    // Douglas-Peucker-lite: keep corners and points deviating more than fit_error.
                    let pts = simplify_polyline(&sm, &corners, opts.fit_error);
                    if pts.len() < 3 {
                        continue;
                    }
                    d.push_str(&format!("M{} {}", fmt(pts[0].0, opts.precision), fmt(pts[0].1, opts.precision)));
                    for p in &pts[1..] {
                        d.push_str(&format!(" L{} {}", fmt(p.0, opts.precision), fmt(p.1, opts.precision)));
                    }
                    d.push('Z');
                    nodes += pts.len();
                } else {
                    let cubics = fit_closed(&sm, &corners, opts.fit_error);
                    if cubics.is_empty() {
                        continue;
                    }
                    d.push_str(&format!("M{} {}", fmt(cubics[0].p0.0, opts.precision), fmt(cubics[0].p0.1, opts.precision)));
                    for c in &cubics {
                        d.push_str(&format!(
                            " C{} {} {} {} {} {}",
                            fmt(c.p1.0, opts.precision), fmt(c.p1.1, opts.precision),
                            fmt(c.p2.0, opts.precision), fmt(c.p2.1, opts.precision),
                            fmt(c.p3.0, opts.precision), fmt(c.p3.1, opts.precision)
                        ));
                    }
                    d.push('Z');
                    nodes += cubics.len();
                }
            }
            if d.is_empty() {
                return None;
            }
            Some(TracedShape { fill: palette[c.label as usize], area: c.area, d, even_odd: true, node_count: nodes })
        })
        .collect()
}

fn simplify_polyline(p: &[Pt], corners: &[bool], eps: f64) -> Vec<Pt> {
    // Iterative Douglas-Peucker on a closed loop, seeded with corners (or two far points).
    let n = p.len();
    let mut keep = vec![false; n];
    let mut seeds: Vec<usize> = (0..n).filter(|&i| corners[i]).collect();
    if seeds.len() < 2 {
        let mut far = 0;
        let mut best = 0.0;
        for i in 0..n {
            let d = dist2(p[0], p[i]);
            if d > best {
                best = d;
                far = i;
            }
        }
        seeds = vec![0, far];
    }
    for &s in &seeds {
        keep[s] = true;
    }
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for i in 0..seeds.len() {
        stack.push((seeds[i], seeds[(i + 1) % seeds.len()]));
    }
    while let Some((a, b)) = stack.pop() {
        let len = if b >= a { b - a } else { b + n - a };
        if len < 2 {
            continue;
        }
        let (ax, ay) = p[a];
        let (bx, by) = p[b];
        let (dx, dy) = (bx - ax, by - ay);
        let l2 = dx * dx + dy * dy;
        let mut worst = 0.0;
        let mut wi = a;
        for s in 1..len {
            let i = (a + s) % n;
            let (px, py) = p[i];
            let d = if l2 < 1e-12 {
                dist2(p[i], p[a]).sqrt()
            } else {
                ((px - ax) * dy - (py - ay) * dx).abs() / l2.sqrt()
            };
            if d > worst {
                worst = d;
                wi = i;
            }
        }
        if worst > eps {
            keep[wi] = true;
            stack.push((a, wi));
            stack.push((wi, b));
        }
    }
    (0..n).filter(|&i| keep[i]).map(|i| p[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_corner_loop_is_not_dropped() {
        // A teardrop: round bottom, one sharp tip at the top.
        let (w, h) = (80u32, 100u32);
        let mut img = RgbaImage::from_pixel(w, h, image::Rgba([255, 255, 255, 255]));
        for (x, y, p) in img.enumerate_pixels_mut() {
            let fx = x as f64 + 0.5 - 40.0;
            let fy = y as f64 + 0.5;
            let inside_disc = (fx * fx + (fy - 65.0) * (fy - 65.0)).sqrt() < 25.0;
            let inside_cone = fy >= 20.0 && fy <= 65.0 && fx.abs() <= (fy - 20.0) * 25.0 / 45.0;
            if inside_disc || inside_cone {
                *p = image::Rgba([0, 0, 0, 255]);
            }
        }
        let palette = vec![[255, 255, 255], [0, 0, 0]];
        let opts = TraceOptions { min_area: 4, corner_angle: 50.0, sharp_corners: true, smooth_radius: 2, fit_error: 0.4, polygon_only: false, cutout: true, precision: 2 };
        let shapes = trace(&img, &img, &palette, &opts);
        let drop = shapes.iter().find(|s| s.fill == [0, 0, 0]).expect("teardrop shape must be emitted");
        assert!(drop.node_count >= 3);
    }

    #[test]
    fn traces_a_disc_with_subpixel_edges() {
        // Anti-aliased disc: coverage-based alpha blend of black on white.
        let (w, h) = (64u32, 64u32);
        let mut src = RgbaImage::new(w, h);
        let mut quant = RgbaImage::new(w, h);
        for (x, y, p) in src.enumerate_pixels_mut() {
            let dx = x as f64 + 0.5 - 32.0;
            let dy = y as f64 + 0.5 - 32.0;
            let r = (dx * dx + dy * dy).sqrt();
            let cov = (20.5 - r).clamp(0.0, 1.0);
            let v = (255.0 * (1.0 - cov)) as u8;
            *p = image::Rgba([v, v, v, 255]);
            quant.put_pixel(x, y, if cov >= 0.5 { image::Rgba([0, 0, 0, 255]) } else { image::Rgba([255, 255, 255, 255]) });
        }
        let palette = vec![[255, 255, 255], [0, 0, 0]];
        let opts = TraceOptions { min_area: 4, corner_angle: 60.0, sharp_corners: true, smooth_radius: 2, fit_error: 0.3, polygon_only: false, cutout: true, precision: 2 };
        let shapes = trace(&quant, &src, &palette, &opts);
        let disc = shapes.iter().find(|s| s.fill == [0, 0, 0]).expect("disc shape");
        assert!(disc.d.starts_with('M') && disc.d.contains(" C"));
        // A circle should need few nodes, not hundreds.
        assert!(disc.node_count <= 16, "nodes = {}", disc.node_count);
    }
}
