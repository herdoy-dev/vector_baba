//! Image loading and cleanup before tracing.

use anyhow::{Context, Result};
use image::{imageops, DynamicImage, RgbaImage};
use std::path::Path;

/// Load any supported raster file as RGBA8.
pub fn load(path: &Path) -> Result<RgbaImage> {
    let img = image::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    Ok(img.to_rgba8())
}

/// Decode an in-memory file (used for drag and drop of bytes).
pub fn load_bytes(bytes: &[u8]) -> Result<RgbaImage> {
    let img = image::load_from_memory(bytes).context("cannot decode image bytes")?;
    Ok(img.to_rgba8())
}

/// Downscale so the longest side is at most `max_dim`. Returns the image and the
/// scale factor applied (1.0 when untouched).
pub fn limit_size(img: &RgbaImage, max_dim: u32) -> (RgbaImage, f64) {
    let (w, h) = img.dimensions();
    let longest = w.max(h);
    if longest <= max_dim || max_dim == 0 {
        return (img.clone(), 1.0);
    }
    let scale = max_dim as f64 / longest as f64;
    let nw = ((w as f64 * scale).round() as u32).max(1);
    let nh = ((h as f64 * scale).round() as u32).max(1);
    let resized = imageops::resize(img, nw, nh, imageops::FilterType::Lanczos3);
    (resized, scale)
}

/// Gaussian smoothing. `sigma` of 0 is a no-op. Alpha is preserved by blurring
/// premultiplied color so transparent edges do not bleed dark fringes.
pub fn smooth(img: &RgbaImage, sigma: f32) -> RgbaImage {
    if sigma <= 0.01 {
        return img.clone();
    }
    let (w, h) = img.dimensions();
    let mut pre = RgbaImage::new(w, h);
    for (s, d) in img.pixels().zip(pre.pixels_mut()) {
        let a = s[3] as u32;
        d[0] = ((s[0] as u32 * a + 127) / 255) as u8;
        d[1] = ((s[1] as u32 * a + 127) / 255) as u8;
        d[2] = ((s[2] as u32 * a + 127) / 255) as u8;
        d[3] = s[3];
    }
    let blurred = imageops::blur(&pre, sigma);
    let mut out = RgbaImage::new(w, h);
    for (s, d) in blurred.pixels().zip(out.pixels_mut()) {
        let a = s[3] as u32;
        if a == 0 {
            *d = image::Rgba([0, 0, 0, 0]);
        } else {
            d[0] = ((s[0] as u32 * 255 + a / 2) / a).min(255) as u8;
            d[1] = ((s[1] as u32 * 255 + a / 2) / a).min(255) as u8;
            d[2] = ((s[2] as u32 * 255 + a / 2) / a).min(255) as u8;
            d[3] = s[3];
        }
    }
    out
}

/// Flatten semi-transparent pixels against a background color so that the
/// tracer sees hard edges. Fully transparent pixels stay transparent.
pub fn flatten_alpha(img: &RgbaImage, bg: [u8; 3]) -> RgbaImage {
    let mut out = img.clone();
    for p in out.pixels_mut() {
        let a = p[3] as u32;
        if a == 0 || a == 255 {
            continue;
        }
        for i in 0..3 {
            p[i] = ((p[i] as u32 * a + bg[i] as u32 * (255 - a)) / 255) as u8;
        }
        p[3] = 255;
    }
    out
}

/// For images anti-aliased against transparency: make semi-transparent edge
/// pixels either fully opaque (alpha >= 128) or fully transparent, keeping
/// their own color, so the tracer sees a hard edge without a halo.
pub fn unpremultiply_edges(img: &RgbaImage) -> RgbaImage {
    let mut out = img.clone();
    for p in out.pixels_mut() {
        if p[3] == 0 || p[3] == 255 {
            continue;
        }
        p[3] = if p[3] >= 128 { 255 } else { 0 };
    }
    out
}

pub fn to_dynamic(img: RgbaImage) -> DynamicImage {
    DynamicImage::ImageRgba8(img)
}

/// Make the background transparent: find the most common color along the image
/// border, then flood-fill every border-connected pixel of that exact color to
/// alpha 0. Works on an already-quantized image so "exact color" is meaningful.
pub fn remove_border_background(img: &mut RgbaImage) {
    let (w, h) = img.dimensions();
    if w < 2 || h < 2 {
        return;
    }
    let mut counts: std::collections::HashMap<[u8; 4], usize> = std::collections::HashMap::new();
    let mut border = Vec::with_capacity((2 * (w + h)) as usize);
    for x in 0..w {
        border.push((x, 0));
        border.push((x, h - 1));
    }
    for y in 1..h.saturating_sub(1) {
        border.push((0, y));
        border.push((w - 1, y));
    }
    for &(x, y) in &border {
        *counts.entry(img.get_pixel(x, y).0).or_insert(0) += 1;
    }
    let Some((bg, _)) = counts.into_iter().max_by_key(|(_, n)| *n) else { return };
    if bg[3] == 0 {
        return; // already transparent
    }
    let mut visited = vec![false; (w * h) as usize];
    let mut stack: Vec<(u32, u32)> = border.into_iter().filter(|&(x, y)| img.get_pixel(x, y).0 == bg).collect();
    while let Some((x, y)) = stack.pop() {
        let i = (y * w + x) as usize;
        if visited[i] {
            continue;
        }
        visited[i] = true;
        if img.get_pixel(x, y).0 != bg {
            continue;
        }
        img.put_pixel(x, y, image::Rgba([0, 0, 0, 0]));
        if x > 0 { stack.push((x - 1, y)); }
        if x + 1 < w { stack.push((x + 1, y)); }
        if y > 0 { stack.push((x, y - 1)); }
        if y + 1 < h { stack.push((x, y + 1)); }
    }
}
