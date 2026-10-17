//! Conversion settings and the end-to-end raster -> SVG pipeline.

use crate::{preprocess, quantize, svg, trace};
pub use crate::svg::OutPath;
use anyhow::{anyhow, Result};
use image::RgbaImage;
use std::time::Instant;
use visioncortex::{ColorImage, PathSimplifyMode};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Full color output with a quantized palette.
    Color,
    /// Black and white output for sketches, scans and line art.
    BlackWhite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurveMode {
    /// Smooth Bezier curves (best for most art).
    Spline,
    /// Straight segments only (good for pixel art and technical drawings).
    Polygon,
    /// Raw pixel outline, no simplification.
    Pixel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layering {
    /// Shapes stack on top of each other; fewer seams, more overlap.
    Stacked,
    /// Shapes are cut out so nothing overlaps; easier to edit by hand.
    Cutout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    Logo,
    Illustration,
    Photo,
    Sketch,
}

impl Preset {
    pub const ALL: [Preset; 4] = [Preset::Logo, Preset::Illustration, Preset::Photo, Preset::Sketch];

    pub fn label(self) -> &'static str {
        match self {
            Preset::Logo => "Logo",
            Preset::Illustration => "Illustration",
            Preset::Photo => "Photo",
            Preset::Sketch => "Sketch / Line art",
        }
    }

    pub fn describe(self) -> &'static str {
        match self {
            Preset::Logo => "Few flat colors, crisp corners, speckles removed.",
            Preset::Illustration => "Medium palette, smooth curves, moderate detail.",
            Preset::Photo => "Large palette, heavy smoothing, posterized look.",
            Preset::Sketch => "Black and white threshold, keeps fine lines.",
        }
    }
}

/// What to do with semi-transparent pixels before tracing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transparency {
    /// Keep alpha; anti-aliased-to-transparent edges are flattened onto the
    /// shape's own color so they stay crisp.
    Preserve,
    /// Composite onto white (image was anti-aliased against white).
    FlattenWhite,
    /// Composite onto a chosen color (image was anti-aliased against it).
    FlattenCustom([u8; 3]),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    None,
    Moderate,
    Aggressive,
}

impl Level {
    pub const ALL: [Level; 3] = [Level::None, Level::Moderate, Level::Aggressive];
    pub fn label(self) -> &'static str {
        match self {
            Level::None => "None",
            Level::Moderate => "Moderate",
            Level::Aggressive => "Aggressive",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Complexity {
    Low,
    Medium,
    High,
}

impl Complexity {
    pub const ALL: [Complexity; 3] = [Complexity::Low, Complexity::Medium, Complexity::High];
    pub fn label(self) -> &'static str {
        match self {
            Complexity::Low => "Low",
            Complexity::Medium => "Medium",
            Complexity::High => "High",
        }
    }
    pub fn describe(self) -> &'static str {
        match self {
            Complexity::Low => "Low complexity: small file, easier to edit.",
            Complexity::Medium => "Medium complexity.",
            Complexity::High => "High complexity: larger file, closest fit.",
        }
    }
}

/// All user-tunable parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub mode: Mode,
    /// Palette size for color mode (2..=64).
    pub colors: u32,
    /// Gaussian smoothing sigma before tracing (0 = off).
    pub smoothing: f32,
    /// 0 = very simplified, 1 = every wobble preserved.
    pub detail: f32,
    /// Regions smaller than this (in pixels on a side) are dropped.
    pub despeckle: u32,
    pub curves: CurveMode,
    pub layering: Layering,
    /// Longest side is downscaled to this before tracing (0 = never).
    pub max_dimension: u32,
    /// Luma threshold for black/white mode.
    pub threshold: u8,
    /// Swap black and white in black/white mode.
    pub invert: bool,
    /// Decimal places in path coordinates.
    pub precision: u32,
    /// When set, pixels snap to exactly these colors instead of an automatic palette.
    pub custom_palette: Option<Vec<[u8; 3]>>,
    /// Let the analysis decide the palette size ("unlimited colors").
    pub unlimited_colors: bool,
    /// Remove the background: the border color region becomes transparent.
    pub remove_background: bool,
    pub transparency: Transparency,
    /// Edge-aware color assignment: anti-aliased pixels may only take colors
    /// present around them. Off = plain nearest-color.
    pub anti_aliasing_engine: bool,
    /// Removes slivers and specks left over from anti-aliasing.
    pub artifact_rejection: Level,
    /// Merge perceptually identical palette colors.
    pub cluster_similar: bool,
    /// Keep corners sharp instead of rounding them.
    pub sharp_corners: bool,
    /// Extra contour smoothing: 0 = less, 1 = normal, 2 = more.
    pub path_smoothing: u8,
    pub curve_complexity: Complexity,
    /// Also stroke each shape with its fill color to hide hairline seams.
    pub stroke_boundaries: bool,
}

/// What kind of picture this is. Drives the Basic mode defaults.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageType {
    /// Continuous tones, many colors, soft edges.
    Photograph,
    /// Art with anti-aliased or blended edges (most digital illustrations).
    BlendedArt,
    /// Flat art with hard edges and few colors (logos, clip art, pixel art).
    FlatArt,
}

impl ImageType {
    pub const ALL: [ImageType; 3] = [ImageType::Photograph, ImageType::BlendedArt, ImageType::FlatArt];
    pub fn label(self) -> &'static str {
        match self {
            ImageType::Photograph => "Photograph",
            ImageType::BlendedArt => "Artwork with blending",
            ImageType::FlatArt => "Artwork without blending",
        }
    }
    pub fn describe(self) -> &'static str {
        match self {
            ImageType::Photograph => "Most common category. Photos, scans, renders with lighting.",
            ImageType::BlendedArt => "Anti-aliased artwork. Typical of PNG images, illustrations, logos from the web.",
            ImageType::FlatArt => "Hard-edged artwork with no blending. Typical of GIF images, pixel art, clip art.",
        }
    }
}

/// How much to favor fidelity over simplicity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quality {
    Low,
    Medium,
    High,
}

impl Quality {
    pub const ALL: [Quality; 3] = [Quality::Low, Quality::Medium, Quality::High];
    pub fn label(self) -> &'static str {
        match self {
            Quality::Low => "Low",
            Quality::Medium => "Medium",
            Quality::High => "High",
        }
    }
    pub fn describe(self) -> &'static str {
        match self {
            Quality::Low => "Fewest nodes and colors. Smallest file, cleanest shapes.",
            Quality::Medium => "Balanced detail and file size.",
            Quality::High => "Most faithful. More nodes, more colors, bigger file.",
        }
    }
}

/// Result of analyzing an image before tracing. Everything the Fully Automatic
/// mode decides comes from here, and the UI shows it so the choice is explainable.
#[derive(Clone, Debug)]
pub struct Analysis {
    pub image_type: ImageType,
    /// Distinct colors at 5 bits per channel on a 256 px thumbnail.
    pub unique_colors: usize,
    /// Of neighboring pixels that differ, the share that differ only slightly (0..1).
    /// High = anti-aliased / photographic; low = hard edged.
    pub soft_edge_ratio: f32,
    /// High-frequency residual in flat areas, 0..1. High = JPEG noise, grain, dither.
    pub noise_level: f32,
    /// Share of opaque pixels with visible chroma (0 = fully grayscale).
    pub colorfulness: f32,
    /// Luma histogram is strongly two-peaked (ink on paper).
    pub bimodal_luma: bool,
    /// Looks like a line drawing, scan or sketch.
    pub is_line_art: bool,
    pub has_transparency: bool,
    /// Border color if the image has a uniform border, plus how uniform (0..1).
    pub background: Option<[u8; 3]>,
    pub background_uniformity: f32,
    /// Dominant colors with their share of opaque pixels, largest first.
    pub dominant: Vec<([u8; 3], f32)>,
    /// How many palette colors carry real weight.
    pub recommended_colors: u32,
}

impl Analysis {
    /// Settings the Fully Automatic mode uses.
    pub fn recommend(&self, quality: Quality) -> Settings {
        let mut s = Settings::from_basic(self.image_type, quality);
        if self.is_line_art {
            s.mode = Mode::BlackWhite;
            s.threshold = 150;
            s.smoothing = 0.2 + self.noise_level * 1.0;
            s.despeckle = 2 + (self.noise_level * 8.0) as u32;
            return s;
        }
        // Palette size: what the image really contains, nudged by quality.
        let q = match quality {
            Quality::Low => 0.7,
            Quality::Medium => 1.0,
            Quality::High => 1.4,
        };
        let rec = (self.recommended_colors as f32 * q).round() as u32;
        s.colors = match self.image_type {
            ImageType::Photograph => rec.clamp(12, 64),
            ImageType::BlendedArt => rec.clamp(4, 40),
            ImageType::FlatArt => rec.clamp(2, 24),
        };
        // Smoothing follows measured noise, not just the type.
        s.smoothing = (s.smoothing * 0.5 + self.noise_level * 2.2).clamp(0.0, 3.0);
        s.despeckle = (s.despeckle as f32 + self.noise_level * 10.0).round() as u32;
        s
    }

    /// Short human summary for the UI.
    pub fn summary(&self) -> String {
        let mut parts = vec![self.image_type.label().to_string()];
        if self.is_line_art {
            parts.push("line art".into());
        }
        parts.push(format!("{} colors", self.unique_colors));
        parts.push(match self.noise_level {
            n if n < 0.15 => "clean".into(),
            n if n < 0.4 => "some noise".into(),
            _ => "noisy".into(),
        });
        if self.background.is_some() && self.background_uniformity > 0.85 {
            parts.push("solid background".into());
        }
        if self.has_transparency {
            parts.push("transparent".into());
        }
        parts.join(" · ")
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self::from_preset(Preset::Illustration)
    }
}

impl Settings {
    pub fn from_preset(p: Preset) -> Self {
        let base = Settings {
            mode: Mode::Color,
            colors: 16,
            smoothing: 0.6,
            detail: 0.55,
            despeckle: 4,
            curves: CurveMode::Spline,
            layering: Layering::Stacked,
            max_dimension: 1600,
            threshold: 128,
            invert: false,
            precision: 2,
            custom_palette: None,
            unlimited_colors: false,
            remove_background: false,
            transparency: Transparency::Preserve,
            anti_aliasing_engine: true,
            artifact_rejection: Level::Moderate,
            cluster_similar: true,
            sharp_corners: true,
            path_smoothing: 1,
            curve_complexity: Complexity::Medium,
            stroke_boundaries: false,
        };
        match p {
            Preset::Logo => Settings { colors: 8, smoothing: 0.4, detail: 0.5, despeckle: 6, ..base },
            Preset::Illustration => base,
            Preset::Photo => Settings { colors: 48, smoothing: 1.2, detail: 0.35, despeckle: 10, ..base },
            Preset::Sketch => Settings {
                mode: Mode::BlackWhite,
                smoothing: 0.3,
                detail: 0.7,
                despeckle: 3,
                ..base
            },
        }
    }

    /// Build settings from the Basic-mode choices.
    pub fn from_basic(kind: ImageType, quality: Quality) -> Self {
        let q = match quality {
            Quality::Low => 0.0,
            Quality::Medium => 0.5,
            Quality::High => 1.0,
        };
        let lerp = |a: f32, b: f32| a + (b - a) * q;
        let base = Settings::from_preset(Preset::Illustration);
        match kind {
            ImageType::Photograph => Settings {
                colors: lerp(16.0, 64.0).round() as u32,
                smoothing: lerp(1.6, 0.8),
                detail: lerp(0.25, 0.6),
                despeckle: lerp(14.0, 6.0).round() as u32,
                max_dimension: lerp(1200.0, 2400.0).round() as u32,
                ..base
            },
            ImageType::BlendedArt => Settings {
                colors: lerp(8.0, 32.0).round() as u32,
                smoothing: lerp(0.9, 0.4),
                detail: lerp(0.35, 0.75),
                despeckle: lerp(8.0, 3.0).round() as u32,
                max_dimension: lerp(1400.0, 2800.0).round() as u32,
                ..base
            },
            ImageType::FlatArt => Settings {
                colors: lerp(6.0, 24.0).round() as u32,
                smoothing: lerp(0.3, 0.0),
                detail: lerp(0.45, 0.85),
                despeckle: lerp(6.0, 2.0).round() as u32,
                max_dimension: lerp(1600.0, 3200.0).round() as u32,
                ..base
            },
        }
    }

    /// Options for the sub-pixel tracer.
    pub fn trace_options(&self) -> trace::TraceOptions {
        let d = self.detail.clamp(0.0, 1.0);
        let lerp = |a: f32, b: f32| a + (b - a) * d;
        let speck = self.despeckle.max(1)
            + match self.artifact_rejection {
                Level::None => 0,
                Level::Moderate => 2,
                Level::Aggressive => 6,
            };
        let complexity_mul = match self.curve_complexity {
            Complexity::Low => 1.7,
            Complexity::Medium => 1.0,
            Complexity::High => 0.55,
        };
        let smooth_radius = match self.path_smoothing {
            0 => 1,
            1 => 2,
            _ => 4,
        };
        trace::TraceOptions {
            min_area: (speck * speck) as usize,
            corner_angle: lerp(80.0, 45.0),
            sharp_corners: self.sharp_corners,
            smooth_radius,
            fit_error: (lerp(1.4, 0.3) * complexity_mul) as f64,
            polygon_only: self.curves == CurveMode::Polygon,
            cutout: self.layering == Layering::Cutout,
            precision: self.precision,
        }
    }

    fn tracer_config(&self) -> vtracer::Config {
        let d = self.detail.clamp(0.0, 1.0);
        let lerp = |a: f32, b: f32| a + (b - a) * d;
        let smooth_mul = match self.path_smoothing {
            0 => 0.6,
            1 => 1.0,
            _ => 1.8,
        };
        let (complexity_mul, precision) = match self.curve_complexity {
            Complexity::Low => (1.6, self.precision.min(1)),
            Complexity::Medium => (1.0, self.precision),
            Complexity::High => (0.6, self.precision.max(2)),
        };
        let speck = self.despeckle.max(1)
            + match self.artifact_rejection {
                Level::None => 0,
                Level::Moderate => 2,
                Level::Aggressive => 6,
            };
        let corner = if self.sharp_corners { lerp(120.0, 30.0) } else { lerp(170.0, 90.0) };
        vtracer::Config {
            color_mode: match self.mode {
                Mode::Color => vtracer::ColorMode::Color,
                Mode::BlackWhite => vtracer::ColorMode::Binary,
            },
            hierarchical: match self.layering {
                Layering::Stacked => vtracer::Hierarchical::Stacked,
                Layering::Cutout => vtracer::Hierarchical::Cutout,
            },
            filter_speckle: speck as usize,
            // We quantize ourselves, so let the tracer keep colors exact.
            color_precision: 8,
            layer_difference: lerp(48.0, 8.0).round() as i32,
            mode: match self.curves {
                CurveMode::Spline => PathSimplifyMode::Spline,
                CurveMode::Polygon => PathSimplifyMode::Polygon,
                CurveMode::Pixel => PathSimplifyMode::None,
            },
            corner_threshold: corner.round() as i32,
            length_threshold: (lerp(10.0, 2.0) * smooth_mul * complexity_mul) as f64,
            max_iterations: 10,
            splice_threshold: (lerp(60.0, 30.0) * if self.sharp_corners { 1.0 } else { 1.4 }).round() as i32,
            path_precision: Some(precision),
        }
    }
}

/// Everything the conversion produced.
#[derive(Clone, Debug)]
pub struct ConvertOutput {
    pub svg: String,
    /// Original image size.
    pub width: u32,
    pub height: u32,
    /// Size the tracer actually worked at.
    pub traced_width: u32,
    pub traced_height: u32,
    pub palette: Vec<[u8; 3]>,
    pub path_count: usize,
    pub elapsed_ms: u128,
    /// The quantized image the tracer worked from (one flat color per region).
    pub segmentation: RgbaImage,
    /// The traced shapes, in paint order.
    pub traced: Vec<OutPath>,
    pub stroke_boundaries: bool,
    /// Total Bezier/line nodes in the output.
    pub node_count: usize,
}

impl ConvertOutput {
    /// Re-render the SVG without the shapes whose indices are in `removed`.
    pub fn render_svg(&self, removed: &std::collections::HashSet<usize>) -> String {
        svg::serialize_filtered(
            &self.traced, self.traced_width, self.traced_height, self.width, self.height, &self.palette, removed, self.stroke_boundaries,
        )
    }

    /// A copy with the given shapes removed, for export.
    pub fn without(&self, removed: &std::collections::HashSet<usize>) -> ConvertOutput {
        let mut out = self.clone();
        out.svg = self.render_svg(removed);
        out.path_count = self.traced.len().saturating_sub(removed.len());
        out
    }
}

fn to_color_image(img: &RgbaImage) -> ColorImage {
    ColorImage {
        pixels: img.as_raw().clone(),
        width: img.width() as usize,
        height: img.height() as usize,
    }
}

/// Analyze an image. Works on a 256 px thumbnail plus a 512 px copy for noise,
/// so it takes a few milliseconds even for huge inputs.
pub fn analyze(img: &RgbaImage) -> Analysis {
    let (small, _) = preprocess::limit_size(img, 256);
    let (w, h) = small.dimensions();
    let px = small.as_raw();
    let mut has_transparency = false;

    // Unique colors at 5 bits per channel, luma histogram, chroma.
    let mut seen = std::collections::HashSet::with_capacity(4096);
    let mut luma_hist = [0u32; 32];
    let mut colored = 0usize;
    let mut opaque = 0usize;
    for p in px.chunks_exact(4) {
        if p[3] < 255 {
            has_transparency = true;
        }
        if p[3] >= 128 {
            opaque += 1;
            seen.insert(((p[0] >> 3) as u32) << 10 | ((p[1] >> 3) as u32) << 5 | (p[2] >> 3) as u32);
            let luma = (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32) as usize;
            luma_hist[(luma >> 3).min(31)] += 1;
            let lab = quantize::rgb_to_lab(p[0], p[1], p[2]);
            if (lab.a * lab.a + lab.b * lab.b).sqrt() > 0.06 {
                colored += 1;
            }
        }
    }
    let opaque_f = opaque.max(1) as f32;
    let colorfulness = colored as f32 / opaque_f;

    // Bimodal luma: the two biggest histogram bins (far apart) hold most pixels.
    let mut bins: Vec<(usize, u32)> = luma_hist.iter().copied().enumerate().collect();
    bins.sort_by(|a, b| b.1.cmp(&a.1));
    let bimodal_luma = {
        let (i0, c0) = bins[0];
        let second = bins.iter().skip(1).find(|(i, _)| (*i as i32 - i0 as i32).abs() >= 8);
        match second {
            Some((_, c1)) => (c0 + c1) as f32 / opaque_f > 0.75,
            None => false,
        }
    };

    // Edge softness.
    let at = |x: u32, y: u32| {
        let i = ((y * w + x) * 4) as usize;
        (px[i] as i32, px[i + 1] as i32, px[i + 2] as i32)
    };
    let mut differing = 0usize;
    let mut soft = 0usize;
    for y in 0..h {
        for x in 0..w.saturating_sub(1) {
            let (r1, g1, b1) = at(x, y);
            let (r2, g2, b2) = at(x + 1, y);
            let d = (r1 - r2).abs() + (g1 - g2).abs() + (b1 - b2).abs();
            if d > 6 {
                differing += 1;
                if d < 60 {
                    soft += 1;
                }
            }
        }
    }
    let soft_edge_ratio = if differing == 0 { 0.0 } else { soft as f32 / differing as f32 };

    // Noise: residual against a 3x3 box blur, measured only where the local
    // gradient is small (flat areas), so real edges do not count as noise.
    let (mid, _) = preprocess::limit_size(img, 512);
    let (mw, mh) = mid.dimensions();
    let mp = mid.as_raw();
    let lum = |x: u32, y: u32| -> f32 {
        let i = ((y * mw + x) * 4) as usize;
        0.2126 * mp[i] as f32 + 0.7152 * mp[i + 1] as f32 + 0.0722 * mp[i + 2] as f32
    };
    let mut resid = 0.0f32;
    let mut flat = 0usize;
    if mw >= 3 && mh >= 3 {
        for y in 1..mh - 1 {
            for x in 1..mw - 1 {
                let c = lum(x, y);
                let gx = (lum(x + 1, y) - lum(x - 1, y)).abs();
                let gy = (lum(x, y + 1) - lum(x, y - 1)).abs();
                if gx + gy < 24.0 {
                    let mut sum = 0.0;
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            sum += lum((x as i32 + dx) as u32, (y as i32 + dy) as u32);
                        }
                    }
                    resid += (c - sum / 9.0).abs();
                    flat += 1;
                }
            }
        }
    }
    let noise_level = if flat == 0 { 0.0 } else { (resid / flat as f32 / 6.0).clamp(0.0, 1.0) };

    // Border / background.
    let mut border_counts: std::collections::HashMap<u32, (usize, [u8; 3])> = std::collections::HashMap::new();
    let mut border_n = 0usize;
    let mut push = |x: u32, y: u32| {
        let i = ((y * w + x) * 4) as usize;
        if px[i + 3] < 128 {
            return;
        }
        border_n += 1;
        let key = ((px[i] >> 3) as u32) << 10 | ((px[i + 1] >> 3) as u32) << 5 | (px[i + 2] >> 3) as u32;
        let e = border_counts.entry(key).or_insert((0, [px[i], px[i + 1], px[i + 2]]));
        e.0 += 1;
    };
    for x in 0..w {
        push(x, 0);
        push(x, h - 1);
    }
    for y in 1..h.saturating_sub(1) {
        push(0, y);
        push(w - 1, y);
    }
    let (background, background_uniformity) = match border_counts.values().max_by_key(|(n, _)| *n) {
        Some((n, c)) if border_n > 0 => {
            let u = *n as f32 / border_n as f32;
            (if u > 0.6 { Some(*c) } else { None }, u)
        }
        _ => (None, 0.0),
    };

    // Dominant colors: k-means with 24 clusters on a 1024 px copy so thin
    // strokes and small shapes still have flat interiors, then keep the
    // clusters with real weight.
    let (color_src, _) = preprocess::limit_size(img, 1024);
    let q = quantize::quantize(&color_src, 24, 8);
    let opaque_f = color_src.pixels().filter(|p| p[3] >= 128).count().max(1) as f32;
    let mut shares: Vec<([u8; 3], f32)> = q
        .palette
        .iter()
        .map(|c| {
            let n = q.image.pixels().filter(|p| p[3] >= 128 && p[0] == c[0] && p[1] == c[1] && p[2] == c[2]).count();
            (*c, n as f32 / opaque_f)
        })
        .collect();
    shares.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let recommended_colors = shares.iter().filter(|(_, s)| *s > 0.0025).count().clamp(2, 48) as u32;
    let dominant: Vec<([u8; 3], f32)> = shares.into_iter().take(12).collect();

    let unique_colors = seen.len();
    let is_line_art = colorfulness < 0.004 && bimodal_luma && soft_edge_ratio < 0.55;
    let image_type = if is_line_art {
        ImageType::FlatArt
    } else if unique_colors > 1500 && soft_edge_ratio > 0.6 {
        ImageType::Photograph
    } else if unique_colors <= 48 && soft_edge_ratio < 0.35 {
        ImageType::FlatArt
    } else if unique_colors > 3000 {
        ImageType::Photograph
    } else {
        ImageType::BlendedArt
    };

    Analysis {
        image_type,
        unique_colors,
        soft_edge_ratio,
        noise_level,
        colorfulness,
        bimodal_luma,
        is_line_art,
        has_transparency,
        background,
        background_uniformity,
        dominant,
        recommended_colors,
    }
}

/// Run the full pipeline on an RGBA image.
pub fn convert(img: &RgbaImage, settings: &Settings) -> Result<ConvertOutput> {
    convert_with_progress(img, settings, &|_| {})
}

/// Like [`convert`], reporting the current stage by name.
pub fn convert_with_progress(img: &RgbaImage, settings: &Settings, progress: &dyn Fn(&str)) -> Result<ConvertOutput> {
    let start = Instant::now();
    let (ow, oh) = img.dimensions();
    if ow == 0 || oh == 0 {
        return Err(anyhow!("empty image"));
    }

    progress("Preparing image");
    let (small, _scale) = preprocess::limit_size(img, settings.max_dimension);
    let flat = match settings.transparency {
        Transparency::Preserve => preprocess::unpremultiply_edges(&small),
        Transparency::FlattenWhite => preprocess::flatten_alpha(&small, [255, 255, 255]),
        Transparency::FlattenCustom(c) => preprocess::flatten_alpha(&small, c),
    };
    let smoothed = preprocess::smooth(&flat, settings.smoothing);

    let colors = if settings.unlimited_colors {
        progress("Analyzing image colors");
        analyze(img).recommended_colors.max(2)
    } else {
        settings.colors
    };
    let opts = quantize::Options {
        edge_aware: settings.anti_aliasing_engine,
        merge_similar: settings.cluster_similar,
    };

    progress("Segmenting image");
    let (mut prepared, palette) = match settings.mode {
        Mode::Color => match &settings.custom_palette {
            Some(p) if p.len() >= 1 => {
                let q = quantize::quantize_to_palette_with(&smoothed, p, opts);
                (q.image, q.palette)
            }
            _ => {
                let q = quantize::quantize_with(&smoothed, colors as usize, 12, opts);
                (q.image, q.palette)
            }
        },
        Mode::BlackWhite => {
            let b = quantize::binarize(&smoothed, settings.threshold, settings.invert);
            (b, vec![[0, 0, 0], [255, 255, 255]])
        }
    };

    if settings.remove_background {
        preprocess::remove_border_background(&mut prepared);
    }

    let (tw, th) = prepared.dimensions();
    progress("Smoothing result");
    let (traced, node_count): (Vec<OutPath>, usize) = if settings.curves == CurveMode::Pixel {
        let file = vtracer::convert(to_color_image(&prepared), settings.tracer_config())
            .map_err(|e| anyhow!("trace failed: {e}"))?;
        let paths = svg::from_vtracer(&file);
        let nodes = paths.iter().map(|p| p.d.matches(|c| c == 'C' || c == 'L').count()).sum();
        (paths, nodes)
    } else {
        let shapes = trace::trace(&prepared, &smoothed, &palette, &settings.trace_options());
        let nodes = shapes.iter().map(|s| s.node_count).sum();
        let paths = shapes
            .into_iter()
            .map(|sh| OutPath { fill: sh.fill, d: sh.d, translate: (0.0, 0.0), even_odd: sh.even_odd })
            .collect();
        (paths, nodes)
    };

    progress("Writing vectors");
    let none = std::collections::HashSet::new();
    let svg = svg::serialize_filtered(&traced, tw, th, ow, oh, &palette, &none, settings.stroke_boundaries);
    Ok(ConvertOutput {
        segmentation: prepared,
        stroke_boundaries: settings.stroke_boundaries,
        node_count,
        traced: traced.clone(),
        svg,
        width: ow,
        height: oh,
        traced_width: tw,
        traced_height: th,
        palette,
        path_count: traced.len(),
        elapsed_ms: start.elapsed().as_millis(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> RgbaImage {
        let mut img = RgbaImage::from_pixel(96, 96, image::Rgba([255, 255, 255, 255]));
        for (x, y, p) in img.enumerate_pixels_mut() {
            let dx = x as i32 - 48;
            let dy = y as i32 - 48;
            if dx * dx + dy * dy < 30 * 30 {
                *p = image::Rgba([220, 40, 60, 255]);
            }
            if x < 20 && y < 20 {
                *p = image::Rgba([20, 60, 200, 255]);
            }
        }
        img
    }

    #[test]
    fn converts_color() {
        let out = convert(&sample(), &Settings::from_preset(Preset::Logo)).unwrap();
        assert!(out.svg.contains("<svg"));
        assert!(out.path_count >= 2, "got {} paths", out.path_count);
        assert!(out.palette.len() >= 2);
    }

    #[test]
    fn converts_bw() {
        let out = convert(&sample(), &Settings::from_preset(Preset::Sketch)).unwrap();
        assert!(out.svg.contains("<path"));
    }
}
