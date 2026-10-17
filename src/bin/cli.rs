//! Command line vectorizer.
//!   vector-baba-cli logo.png                          # fully automatic, SVG next to the input
//!   vector-baba-cli photo.jpg --type photo --quality high --format pdf
//!   vector-baba-cli art.png --palette "#ffffff,#1a1a1a,#e63946" --remove-background

use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use vector_baba::export::{self, Format};
use vector_baba::{analyze, convert, preprocess, CurveMode, ImageType, Layering, Mode, Quality, Settings};

#[derive(Copy, Clone, ValueEnum)]
enum TypeArg { Auto, Photo, Blended, Flat }
#[derive(Copy, Clone, ValueEnum)]
enum QualityArg { Low, Medium, High }
#[derive(Copy, Clone, ValueEnum)]
enum FormatArg { Svg, Pdf, Eps, Ai, Dxf, Png }
#[derive(Copy, Clone, ValueEnum)]
enum CurvesArg { Spline, Polygon, Pixel }
#[derive(Copy, Clone, ValueEnum)]
enum LayerArg { Stacked, Cutout }

#[derive(Parser)]
#[command(name = "vector-baba-cli", version, about = "Convert raster images to SVG, PDF, EPS, AI, DXF or PNG")]
struct Args {
    /// Input image (png, jpg, gif, bmp, tiff, webp)
    input: PathBuf,
    /// Output path (defaults to the input name with the format's extension)
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Image type. "auto" analyzes the image.
    #[arg(short = 't', long = "type", value_enum, default_value_t = TypeArg::Auto)]
    kind: TypeArg,
    #[arg(short, long, value_enum, default_value_t = QualityArg::Medium)]
    quality: QualityArg,
    #[arg(short, long, value_enum, default_value_t = FormatArg::Svg)]
    format: FormatArg,
    /// Number of colors (2-64), overrides the type/quality default
    #[arg(short, long)]
    colors: Option<u32>,
    /// Fixed palette, comma separated hex colors, e.g. "#fff,#000,#e63946"
    #[arg(long)]
    palette: Option<String>,
    /// Make the border color transparent
    #[arg(long)]
    remove_background: bool,
    /// Black and white output
    #[arg(long)]
    bw: bool,
    /// Luma threshold for black and white (0-255)
    #[arg(long)]
    threshold: Option<u8>,
    #[arg(long)]
    invert: bool,
    /// Noise reduction / smoothing strength (0-3)
    #[arg(long)]
    smoothing: Option<f32>,
    /// Detail level (0-1)
    #[arg(long)]
    detail: Option<f32>,
    /// Drop regions smaller than N px on a side
    #[arg(long)]
    despeckle: Option<u32>,
    #[arg(long, value_enum)]
    curves: Option<CurvesArg>,
    #[arg(long, value_enum)]
    layering: Option<LayerArg>,
    /// Downscale the longest side to this before tracing (0 = never)
    #[arg(long)]
    max_dimension: Option<u32>,
    /// PNG output scale factor
    #[arg(long, default_value_t = 1.0)]
    png_scale: f32,
    /// Flatten PNG onto white instead of keeping transparency
    #[arg(long)]
    png_white: bool,
    /// Also write a PNG preview of the result here (any format)
    #[arg(long)]
    preview: Option<PathBuf>,
    /// Also write the segmentation (quantized image the tracer used) as PNG
    #[arg(long)]
    segmentation: Option<PathBuf>,
    /// Print the largest regions where the rendered result disagrees with the segmentation
    #[arg(long)]
    report: bool,
}

fn parse_palette(s: &str) -> Result<Vec<[u8; 3]>> {
    s.split(',')
        .map(|t| {
            let t = t.trim().trim_start_matches('#');
            let t = if t.len() == 3 {
                t.chars().flat_map(|c| [c, c]).collect::<String>()
            } else {
                t.to_string()
            };
            if t.len() != 6 {
                return Err(anyhow!("bad color '{t}'"));
            }
            let v = u32::from_str_radix(&t, 16)?;
            Ok([(v >> 16) as u8, (v >> 8) as u8, v as u8])
        })
        .collect()
}

fn main() -> Result<()> {
    let a = Args::parse();
    let img = preprocess::load(&a.input)?;

    let quality = match a.quality {
        QualityArg::Low => Quality::Low,
        QualityArg::Medium => Quality::Medium,
        QualityArg::High => Quality::High,
    };
    let mut s = match a.kind {
        TypeArg::Auto => {
            let an = analyze(&img);
            eprintln!(
                "analysis: {}  (useful colors {}, noise {:.0}%, soft edges {:.0}%)",
                an.summary(),
                an.recommended_colors,
                an.noise_level * 100.0,
                an.soft_edge_ratio * 100.0
            );
            let dom: Vec<String> = an
                .dominant
                .iter()
                .map(|(c, sh)| format!("#{:02x}{:02x}{:02x} {:.1}%", c[0], c[1], c[2], sh * 100.0))
                .collect();
            eprintln!("dominant: {}", dom.join("  "));
            an.recommend(quality)
        }
        TypeArg::Photo => Settings::from_basic(ImageType::Photograph, quality),
        TypeArg::Blended => Settings::from_basic(ImageType::BlendedArt, quality),
        TypeArg::Flat => Settings::from_basic(ImageType::FlatArt, quality),
    };
    if a.bw { s.mode = Mode::BlackWhite; }
    if let Some(c) = a.colors { s.colors = c.clamp(2, 64); }
    if let Some(p) = &a.palette { s.custom_palette = Some(parse_palette(p)?); }
    s.remove_background = a.remove_background;
    if let Some(v) = a.threshold { s.threshold = v; }
    if a.invert { s.invert = true; }
    if let Some(v) = a.smoothing { s.smoothing = v.clamp(0.0, 3.0); }
    if let Some(v) = a.detail { s.detail = v.clamp(0.0, 1.0); }
    if let Some(v) = a.despeckle { s.despeckle = v; }
    if let Some(c) = a.curves {
        s.curves = match c { CurvesArg::Spline => CurveMode::Spline, CurvesArg::Polygon => CurveMode::Polygon, CurvesArg::Pixel => CurveMode::Pixel };
    }
    if let Some(l) = a.layering {
        s.layering = match l { LayerArg::Stacked => Layering::Stacked, LayerArg::Cutout => Layering::Cutout };
    }
    if let Some(v) = a.max_dimension { s.max_dimension = v; }

    let out = convert(&img, &s)?;
    let format = match a.format {
        FormatArg::Svg => Format::Svg,
        FormatArg::Pdf => Format::Pdf,
        FormatArg::Eps => Format::Eps,
        FormatArg::Ai => Format::Ai,
        FormatArg::Dxf => Format::Dxf,
        FormatArg::Png => Format::Png,
    };
    let output = a.output.unwrap_or_else(|| a.input.with_extension(format.extension()));
    let bg = if a.png_white { Some([255, 255, 255]) } else { None };
    std::fs::write(&output, export::encode(&out, format, a.png_scale, bg)?)?;
    if let Some(p) = &a.segmentation {
        out.segmentation.save(p)?;
    }
    if a.report {
        fidelity_report(&out)?;
    }
    if let Some(p) = &a.preview {
        std::fs::write(p, export::to_png(&out, 1.0, Some([255, 255, 255]))?)?;
    }
    eprintln!(
        "{} -> {}  ({}x{} traced at {}x{}, {} shapes, {} colors, {} ms)",
        a.input.display(), output.display(), out.width, out.height, out.traced_width, out.traced_height,
        out.path_count, out.palette.len(), out.elapsed_ms
    );
    Ok(())
}

/// Compare the rendered vector (at traced size) against the segmentation and
/// list the biggest connected regions that disagree.
fn fidelity_report(out: &vector_baba::ConvertOutput) -> Result<()> {
    use vector_baba::svg::rasterize;
    let (tw, th) = (out.traced_width, out.traced_height);
    let (w, h, rgba) = rasterize(&out.svg, tw.max(th), Some([255, 255, 255]))?;
    if (w, h) != (tw, th) {
        eprintln!("report: rendered {}x{} vs traced {}x{}", w, h, tw, th);
    }
    let seg = &out.segmentation;
    let n = (tw * th) as usize;
    let mut bad = vec![false; n];
    let mut total_bad = 0usize;
    for y in 0..th.min(h) {
        for x in 0..tw.min(w) {
            let s = seg.get_pixel(x, y).0;
            let i = ((y * w + x) * 4) as usize;
            let r = [rgba[i], rgba[i + 1], rgba[i + 2]];
            let sc = if s[3] < 128 { [255, 255, 255] } else { [s[0], s[1], s[2]] };
            let d = (sc[0] as i32 - r[0] as i32).abs() + (sc[1] as i32 - r[1] as i32).abs() + (sc[2] as i32 - r[2] as i32).abs();
            if d > 90 {
                bad[(y * tw + x) as usize] = true;
                total_bad += 1;
            }
        }
    }
    eprintln!("report: {} of {} pixels disagree ({:.2}%)", total_bad, n, total_bad as f64 * 100.0 / n as f64);
    // Connected regions of disagreement.
    let mut seen = vec![false; n];
    let mut regions: Vec<(usize, u32, u32, u32, u32, [u8; 3], [u8; 3])> = Vec::new();
    let mut stack = Vec::new();
    for start in 0..n {
        if !bad[start] || seen[start] {
            continue;
        }
        let (mut area, mut x0, mut y0, mut x1, mut y1) = (0usize, u32::MAX, u32::MAX, 0u32, 0u32);
        let (sx, sy) = ((start as u32) % tw, (start as u32) / tw);
        let sp = seg.get_pixel(sx, sy).0;
        let si = ((sy * w + sx) * 4) as usize;
        let got = [rgba[si], rgba[si + 1], rgba[si + 2]];
        seen[start] = true;
        stack.push(start);
        while let Some(i) = stack.pop() {
            let (x, y) = ((i as u32) % tw, (i as u32) / tw);
            area += 1;
            x0 = x0.min(x); y0 = y0.min(y); x1 = x1.max(x); y1 = y1.max(y);
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                if xx < 0 || yy < 0 || xx >= tw as i32 || yy >= th as i32 { continue; }
                let j = (yy as u32 * tw + xx as u32) as usize;
                if bad[j] && !seen[j] { seen[j] = true; stack.push(j); }
            }
        }
        regions.push((area, x0, y0, x1, y1, [sp[0], sp[1], sp[2]], got));
    }
    regions.sort_by(|a, b| b.0.cmp(&a.0));
    for (area, x0, y0, x1, y1, exp, got) in regions.iter().take(12) {
        eprintln!(
            "  mismatch area {:>6} bbox [{}..{} x {}..{}] expected #{:02x}{:02x}{:02x} got #{:02x}{:02x}{:02x}",
            area, x0, x1, y0, y1, exp[0], exp[1], exp[2], got[0], got[1], got[2]
        );
    }
    Ok(())
}
