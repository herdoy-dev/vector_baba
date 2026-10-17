//! Palette quantization in OKLab space using k-means++ seeding and Lloyd iterations.
//!
//! Quantizing before tracing is what gives the user a single "number of colors"
//! control: every region in the output SVG is filled with one of the palette
//! colors, so the result is clean and editable instead of a soup of near-identical
//! shades.

use image::RgbaImage;
use rayon::prelude::*;

/// A color in OKLab space. L in 0..1, a and b roughly -0.4..0.4.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Lab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

impl Lab {
    #[inline]
    pub fn dist2(self, o: Lab) -> f32 {
        let dl = self.l - o.l;
        let da = self.a - o.a;
        let db = self.b - o.b;
        dl * dl + da * da + db * db
    }
}

#[inline]
fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_to_srgb(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let v = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (v * 255.0 + 0.5) as u8
}

/// sRGB (8-bit) to OKLab.
pub fn rgb_to_lab(r: u8, g: u8, b: u8) -> Lab {
    let (r, g, b) = (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b));
    let l = 0.412_221_46 * r + 0.536_332_55 * g + 0.051_445_995 * b;
    let m = 0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b;
    let s = 0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b;
    let (l, m, s) = (l.cbrt(), m.cbrt(), s.cbrt());
    Lab {
        l: 0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        a: 1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        b: 0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    }
}

/// OKLab to sRGB (8-bit).
pub fn lab_to_rgb(c: Lab) -> [u8; 3] {
    let l_ = c.l + 0.396_337_78 * c.a + 0.215_803_76 * c.b;
    let m_ = c.l - 0.105_561_346 * c.a - 0.063_854_17 * c.b;
    let s_ = c.l - 0.089_484_18 * c.a - 1.291_485_5 * c.b;
    let (l, m, s) = (l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_);
    let r = 4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s;
    let g = -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s;
    let b = -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s;
    [linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b)]
}

/// Deterministic xorshift so results are reproducible run to run.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn kmeans_pp_seed(samples: &[Lab], k: usize, rng: &mut Rng) -> Vec<Lab> {
    let mut centers = Vec::with_capacity(k);
    centers.push(samples[(rng.next() as usize) % samples.len()]);
    let mut d2: Vec<f32> = samples.iter().map(|s| s.dist2(centers[0])).collect();
    while centers.len() < k {
        let total: f32 = d2.iter().sum();
        if total <= 1e-12 {
            break;
        }
        let mut target = rng.unit() * total;
        let mut idx = samples.len() - 1;
        for (i, d) in d2.iter().enumerate() {
            if target <= *d {
                idx = i;
                break;
            }
            target -= d;
        }
        let c = samples[idx];
        centers.push(c);
        for (i, s) in samples.iter().enumerate() {
            let nd = s.dist2(c);
            if nd < d2[i] {
                d2[i] = nd;
            }
        }
    }
    centers
}

/// Mask of pixels whose 4-neighborhood is nearly uniform (not on an edge).
fn flat_pixels(img: &RgbaImage) -> Vec<bool> {
    let (w, h) = img.dimensions();
    let px = img.as_raw();
    let idx = |x: u32, y: u32| ((y * w + x) * 4) as usize;
    let mut mask = vec![false; (w * h) as usize];
    if w < 3 || h < 3 {
        return mask;
    }
    mask.par_chunks_mut(w as usize).enumerate().for_each(|(y, row)| {
        let y = y as u32;
        if y == 0 || y + 1 >= h {
            return;
        }
        for x in 1..w - 1 {
            let c = idx(x, y);
            if px[c + 3] < 128 {
                continue;
            }
            let mut maxd = 0i32;
            for n in [idx(x - 1, y), idx(x + 1, y), idx(x, y - 1), idx(x, y + 1)] {
                let d = (px[c] as i32 - px[n] as i32).abs()
                    + (px[c + 1] as i32 - px[n + 1] as i32).abs()
                    + (px[c + 2] as i32 - px[n + 2] as i32).abs();
                maxd = maxd.max(d);
            }
            row[x as usize] = maxd < 30;
        }
    });
    mask
}

/// Assign every pixel a palette index. Flat pixels take their nearest palette
/// color. Edge pixels (anti-aliasing, JPEG ringing) are blends of the colors
/// around them, so they may only choose among palette colors already present
/// in their flat neighborhood; otherwise a grey between black and white would
/// snap to some unrelated mid-tone and draw a fringe along every edge.
fn assign_edge_aware(
    labs: &[Option<Lab>],
    flat: &[bool],
    palette_lab: &[Lab],
    palette: &[[u8; 3]],
    w: u32,
    h: u32,
    edge_aware: bool,
) -> Vec<u8> {
    let n = (w * h) as usize;
    let nearest = |l: Lab, allowed: Option<&[bool]>| -> usize {
        let mut best = 0;
        let mut bd = f32::MAX;
        for (i, c) in palette_lab.iter().enumerate() {
            if let Some(a) = allowed {
                if !a[i] {
                    continue;
                }
            }
            let d = l.dist2(*c);
            if d < bd {
                bd = d;
                best = i;
            }
        }
        best
    };

    // Pass 0: flat pixels (or every pixel when the engine is off).
    let mut idx: Vec<i16> = (0..n)
        .into_par_iter()
        .map(|i| match labs[i] {
            Some(l) if flat[i] || !edge_aware => nearest(l, None) as i16,
            Some(_) => -1,
            None => -2,
        })
        .collect();

    // Passes 1..: grow from assigned neighbors with increasing radius.
    let k = palette_lab.len();
    for radius in [2i32, 4, 8] {
        let prev = idx.clone();
        idx = (0..n)
            .into_par_iter()
            .map(|i| {
                if prev[i] != -1 {
                    return prev[i];
                }
                let l = labs[i].unwrap();
                let (x, y) = ((i as u32 % w) as i32, (i as u32 / w) as i32);
                let mut allowed = vec![false; k];
                let mut any = false;
                for dy in -radius..=radius {
                    let yy = y + dy;
                    if yy < 0 || yy >= h as i32 {
                        continue;
                    }
                    for dx in -radius..=radius {
                        let xx = x + dx;
                        if xx < 0 || xx >= w as i32 {
                            continue;
                        }
                        let j = (yy as u32 * w + xx as u32) as usize;
                        if prev[j] >= 0 {
                            allowed[prev[j] as usize] = true;
                            any = true;
                        }
                    }
                }
                if any { nearest(l, Some(&allowed)) as i16 } else { -1 }
            })
            .collect();
    }

    let mut out = vec![0u8; n * 4];
    out.par_chunks_exact_mut(4).enumerate().for_each(|(i, o)| {
        let pi = match idx[i] {
            -2 => return,
            -1 => nearest(labs[i].unwrap(), None),
            v => v as usize,
        };
        let p = palette[pi];
        o[0] = p[0];
        o[1] = p[1];
        o[2] = p[2];
        o[3] = 255;
    });
    out
}

/// Tunables for quantization.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub edge_aware: bool,
    pub merge_similar: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { edge_aware: true, merge_similar: true }
    }
}

/// Result of quantization: the quantized image and the palette (sRGB).
pub struct Quantized {
    pub image: RgbaImage,
    pub palette: Vec<[u8; 3]>,
}

/// Quantize `img` to at most `k` colors. Pixels with alpha < 128 become fully
/// transparent (the tracer keys them out); everything else becomes opaque.
pub fn quantize(img: &RgbaImage, k: usize, iterations: usize) -> Quantized {
    quantize_with(img, k, iterations, Options::default())
}

/// [`quantize`] with explicit options.
pub fn quantize_with(img: &RgbaImage, k: usize, iterations: usize, opts: Options) -> Quantized {
    let k = k.clamp(2, 256);
    let (w, h) = img.dimensions();
    let px = img.as_raw();

    // Convert opaque pixels to Lab.
    let labs: Vec<Option<Lab>> = px
        .par_chunks_exact(4)
        .map(|p| {
            if p[3] < 128 {
                None
            } else {
                Some(rgb_to_lab(p[0], p[1], p[2]))
            }
        })
        .collect();

    let opaque: Vec<Lab> = labs.iter().flatten().copied().collect();
    if opaque.is_empty() {
        return Quantized {
            image: RgbaImage::new(w, h),
            palette: vec![],
        };
    }

    // Fit the palette on flat pixels only. Anti-aliased edge pixels (and JPEG
    // ringing) are blends of two real colors; if they vote, they create muddy
    // in-between clusters that show up as slivers around every edge.
    let flat_mask = flat_pixels(img);
    let flat_count = flat_mask.iter().filter(|f| **f).count();
    let use_flat = flat_count >= (opaque.len() / 20).max(64);
    let pool: Vec<Lab> = if use_flat {
        labs.iter().zip(flat_mask.iter()).filter(|(l, f)| l.is_some() && **f).map(|(l, _)| l.unwrap()).collect()
    } else {
        opaque.clone()
    };

    // Sample up to ~60k points for seeding and iteration speed.
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let max_samples = 60_000;
    let samples: Vec<Lab> = if pool.len() > max_samples {
        let step = pool.len() / max_samples;
        pool.iter().step_by(step.max(1)).copied().collect()
    } else {
        pool
    };

    let mut centers = kmeans_pp_seed(&samples, k, &mut rng);

    for _ in 0..iterations {
        let kc = centers.len();
        let (sums, counts) = samples
            .par_iter()
            .fold(
                || (vec![Lab::default(); kc], vec![0u32; kc]),
                |(mut sums, mut counts), s| {
                    let mut best = 0;
                    let mut bd = f32::MAX;
                    for (i, c) in centers.iter().enumerate() {
                        let d = s.dist2(*c);
                        if d < bd {
                            bd = d;
                            best = i;
                        }
                    }
                    sums[best].l += s.l;
                    sums[best].a += s.a;
                    sums[best].b += s.b;
                    counts[best] += 1;
                    (sums, counts)
                },
            )
            .reduce(
                || (vec![Lab::default(); kc], vec![0u32; kc]),
                |(mut s1, mut c1), (s2, c2)| {
                    for i in 0..kc {
                        s1[i].l += s2[i].l;
                        s1[i].a += s2[i].a;
                        s1[i].b += s2[i].b;
                        c1[i] += c2[i];
                    }
                    (s1, c1)
                },
            );
        let mut moved = 0.0f32;
        for i in 0..kc {
            if counts[i] > 0 {
                let n = counts[i] as f32;
                let nc = Lab {
                    l: sums[i].l / n,
                    a: sums[i].a / n,
                    b: sums[i].b / n,
                };
                moved += nc.dist2(centers[i]);
                centers[i] = nc;
            }
        }
        if moved < 1e-7 {
            break;
        }
    }

    // Merge centers that are perceptually the same color (JPEG noise splits
    // one flat color into several near-identical ones).
    let merge_thresh = 0.003; // ~ΔE 0.055 in OKLab
    let mut palette_lab: Vec<Lab> = Vec::new();
    for c in centers {
        let same = |p: &Lab| {
            if !opts.merge_similar {
                return p.dist2(c) < 1e-6;
            }
            p.dist2(c) < merge_thresh || (p.l < 0.14 && c.l < 0.14 && p.dist2(c) < 0.02)
        };
        if let Some(existing) = palette_lab.iter_mut().find(|p| same(p)) {
            existing.l = (existing.l + c.l) * 0.5;
            existing.a = (existing.a + c.a) * 0.5;
            existing.b = (existing.b + c.b) * 0.5;
        } else {
            palette_lab.push(c);
        }
    }
    let palette: Vec<[u8; 3]> = palette_lab.iter().map(|c| lab_to_rgb(*c)).collect();

    let out = assign_edge_aware(&labs, &flat_mask, &palette_lab, &palette, w, h, opts.edge_aware);

    Quantized {
        image: RgbaImage::from_raw(w, h, out).expect("buffer size"),
        palette,
    }
}

/// Snap every opaque pixel to the nearest color of a fixed palette (OKLab distance).
pub fn quantize_to_palette(img: &RgbaImage, palette: &[[u8; 3]]) -> Quantized {
    quantize_to_palette_with(img, palette, Options::default())
}

/// [`quantize_to_palette`] with explicit options.
pub fn quantize_to_palette_with(img: &RgbaImage, palette: &[[u8; 3]], opts: Options) -> Quantized {
    let (w, h) = img.dimensions();
    let mut pal: Vec<[u8; 3]> = Vec::new();
    for c in palette {
        if !pal.contains(c) {
            pal.push(*c);
        }
    }
    if pal.is_empty() {
        pal.push([0, 0, 0]);
    }
    let pal_lab: Vec<Lab> = pal.iter().map(|c| rgb_to_lab(c[0], c[1], c[2])).collect();
    let labs: Vec<Option<Lab>> = img
        .as_raw()
        .par_chunks_exact(4)
        .map(|p| if p[3] < 128 { None } else { Some(rgb_to_lab(p[0], p[1], p[2])) })
        .collect();
    let flat_mask = flat_pixels(img);
    let out = assign_edge_aware(&labs, &flat_mask, &pal_lab, &pal, w, h, opts.edge_aware);
    Quantized { image: RgbaImage::from_raw(w, h, out).expect("buffer size"), palette: pal }
}

/// Threshold to pure black and white for sketch / line-art tracing.
pub fn binarize(img: &RgbaImage, threshold: u8, invert: bool) -> RgbaImage {
    let (w, h) = img.dimensions();
    let mut out = vec![0u8; (w * h * 4) as usize];
    out.par_chunks_exact_mut(4)
        .zip(img.as_raw().par_chunks_exact(4))
        .for_each(|(o, p)| {
            if p[3] < 128 {
                // Transparent counts as paper: white, which binary tracing leaves empty.
                o[0] = 255;
                o[1] = 255;
                o[2] = 255;
                o[3] = 255;
                return;
            }
            let luma = (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32) as u8;
            let mut dark = luma < threshold;
            if invert {
                dark = !dark;
            }
            let v = if dark { 0 } else { 255 };
            o[0] = v;
            o[1] = v;
            o[2] = v;
            o[3] = 255;
        });
    RgbaImage::from_raw(w, h, out).expect("buffer size")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lab_roundtrip() {
        for &(r, g, b) in &[(0u8, 0u8, 0u8), (255, 255, 255), (200, 30, 90), (12, 200, 140)] {
            let lab = rgb_to_lab(r, g, b);
            let back = lab_to_rgb(lab);
            assert!((back[0] as i32 - r as i32).abs() <= 1);
            assert!((back[1] as i32 - g as i32).abs() <= 1);
            assert!((back[2] as i32 - b as i32).abs() <= 1);
        }
    }

    #[test]
    fn quantize_reduces_colors() {
        let mut img = RgbaImage::new(16, 16);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgba([(x * 16) as u8, (y * 16) as u8, 128, 255]);
        }
        let q = quantize(&img, 4, 10);
        assert!(q.palette.len() <= 4 && q.palette.len() >= 2);
    }
}
