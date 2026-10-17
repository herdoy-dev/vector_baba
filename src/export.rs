//! Export the traced result as PDF, EPS or PNG, in addition to SVG.
//!
//! The PDF and EPS writers parse our own SVG path data (only absolute M, L, C
//! and Z commands are ever emitted) so no extra dependencies are needed.

use crate::pipeline::ConvertOutput;
use crate::svg;
use anyhow::{anyhow, Result};
use std::fmt::Write as _;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Svg,
    Pdf,
    Eps,
    Ai,
    Dxf,
    Png,
}

impl Format {
    pub const ALL: [Format; 6] = [Format::Svg, Format::Pdf, Format::Eps, Format::Ai, Format::Dxf, Format::Png];
    pub fn label(self) -> &'static str {
        match self {
            Format::Svg => "SVG",
            Format::Pdf => "PDF",
            Format::Eps => "EPS",
            Format::Ai => "AI",
            Format::Dxf => "DXF",
            Format::Png => "PNG",
        }
    }
    pub fn extension(self) -> &'static str {
        match self {
            Format::Svg => "svg",
            Format::Pdf => "pdf",
            Format::Eps => "eps",
            Format::Ai => "ai",
            Format::Dxf => "dxf",
            Format::Png => "png",
        }
    }
    pub fn describe(self) -> &'static str {
        match self {
            Format::Svg => "Scalable Vector Graphics. Web, Figma, Inkscape, Illustrator.",
            Format::Pdf => "Portable Document Format. Print, Preview, Illustrator, Affinity.",
            Format::Eps => "Encapsulated PostScript. Legacy print and sign-making tools.",
            Format::Ai => "Adobe Illustrator (PDF-compatible). Opens directly in Illustrator.",
            Format::Dxf => "Drawing Exchange Format. CAD and cutting machines; outlines only, one layer per color.",
            Format::Png => "Portable Network Graphics. Rasterized at a chosen scale.",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Cmd {
    M(f64, f64),
    L(f64, f64),
    C(f64, f64, f64, f64, f64, f64),
    Z,
}

/// One filled shape of the output, in traced (viewBox) coordinates.
#[derive(Clone, Debug)]
pub struct Shape {
    pub fill: [u8; 3],
    pub tx: f64,
    pub ty: f64,
    pub cmds: Vec<Cmd>,
    pub even_odd: bool,
}

impl Shape {
    /// Flatten to polygons (one per subpath), translation applied.
    pub fn polygons(&self) -> Vec<Vec<(f64, f64)>> {
        let mut polys: Vec<Vec<(f64, f64)>> = Vec::new();
        let mut cur: Vec<(f64, f64)> = Vec::new();
        let (tx, ty) = (self.tx, self.ty);
        for c in &self.cmds {
            match *c {
                Cmd::M(x, y) => {
                    if cur.len() > 1 {
                        polys.push(std::mem::take(&mut cur));
                    } else {
                        cur.clear();
                    }
                    cur.push((x + tx, y + ty));
                }
                Cmd::L(x, y) => cur.push((x + tx, y + ty)),
                Cmd::C(x1, y1, x2, y2, x, y) => {
                    let (x0, y0) = cur.last().copied().unwrap_or((x1 + tx, y1 + ty));
                    let (x1, y1, x2, y2, x, y) = (x1 + tx, y1 + ty, x2 + tx, y2 + ty, x + tx, y + ty);
                    let steps = 8;
                    for i in 1..=steps {
                        let t = i as f64 / steps as f64;
                        let mt = 1.0 - t;
                        let px = mt * mt * mt * x0 + 3.0 * mt * mt * t * x1 + 3.0 * mt * t * t * x2 + t * t * t * x;
                        let py = mt * mt * mt * y0 + 3.0 * mt * mt * t * y1 + 3.0 * mt * t * t * y2 + t * t * t * y;
                        cur.push((px, py));
                    }
                }
                Cmd::Z => {
                    if cur.len() > 1 {
                        polys.push(std::mem::take(&mut cur));
                    } else {
                        cur.clear();
                    }
                }
            }
        }
        if cur.len() > 1 {
            polys.push(cur);
        }
        polys
    }

    /// Containment test on the flattened shape (even-odd or non-zero as declared).
    pub fn contains(&self, x: f64, y: f64) -> bool {
        let mut winding = 0i32;
        let mut crossings = 0u32;
        for poly in self.polygons() {
            let n = poly.len();
            for i in 0..n {
                let (x1, y1) = poly[i];
                let (x2, y2) = poly[(i + 1) % n];
                if (y1 > y) != (y2 > y) {
                    let xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1);
                    if xi > x {
                        crossings += 1;
                    }
                }
                if y1 <= y {
                    if y2 > y && (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) > 0.0 {
                        winding += 1;
                    }
                } else if y2 <= y && (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) < 0.0 {
                    winding -= 1;
                }
            }
        }
        if self.even_odd { crossings % 2 == 1 } else { winding != 0 }
    }
}

/// Parse shapes out of our own SVG output (public for hit-testing in the UI).
pub fn parse_shapes(svg: &str) -> Result<Vec<Shape>> {
    shapes(svg)
}

/// Index of the topmost shape under (x, y) in traced coordinates.
pub fn hit_test(shapes: &[Shape], x: f64, y: f64) -> Option<usize> {
    shapes.iter().enumerate().rev().find(|(_, s)| s.contains(x, y)).map(|(i, _)| i)
}

fn parse_d(d: &str) -> Result<Vec<Cmd>> {
    let mut cmds = Vec::new();
    let mut nums: Vec<f64> = Vec::new();
    let mut cur = ' ';
    let mut tok = String::new();
    let flush = |tok: &mut String, nums: &mut Vec<f64>| -> Result<()> {
        if !tok.is_empty() {
            nums.push(tok.parse::<f64>().map_err(|e| anyhow!("bad number {tok}: {e}"))?);
            tok.clear();
        }
        Ok(())
    };
    let emit = |cur: char, nums: &mut Vec<f64>, cmds: &mut Vec<Cmd>| -> Result<()> {
        match cur {
            'M' => {
                if nums.len() >= 2 {
                    cmds.push(Cmd::M(nums[0], nums[1]));
                    for pair in nums[2..].chunks_exact(2) {
                        cmds.push(Cmd::L(pair[0], pair[1]));
                    }
                }
            }
            'L' => {
                for pair in nums.chunks_exact(2) {
                    cmds.push(Cmd::L(pair[0], pair[1]));
                }
            }
            'C' => {
                for c in nums.chunks_exact(6) {
                    cmds.push(Cmd::C(c[0], c[1], c[2], c[3], c[4], c[5]));
                }
            }
            'Z' | 'z' => cmds.push(Cmd::Z),
            ' ' => {}
            other => return Err(anyhow!("unsupported path command {other}")),
        }
        nums.clear();
        Ok(())
    };
    for ch in d.chars() {
        match ch {
            'M' | 'L' | 'C' | 'Z' | 'z' => {
                flush(&mut tok, &mut nums)?;
                emit(cur, &mut nums, &mut cmds)?;
                cur = ch;
            }
            ' ' | ',' | '\n' | '\t' => flush(&mut tok, &mut nums)?,
            '-' if !tok.is_empty() && !tok.ends_with('e') => {
                flush(&mut tok, &mut nums)?;
                tok.push('-');
            }
            _ => tok.push(ch),
        }
    }
    flush(&mut tok, &mut nums)?;
    emit(cur, &mut nums, &mut cmds)?;
    Ok(cmds)
}

fn hex(c: &str) -> Option<[u8; 3]> {
    let c = c.strip_prefix('#')?;
    if c.len() != 6 {
        return None;
    }
    let v = u32::from_str_radix(c, 16).ok()?;
    Some([(v >> 16) as u8, (v >> 8) as u8, v as u8])
}

fn attr<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let key = format!(" {name}=\"");
    let start = tag.find(&key)? + key.len();
    let end = tag[start..].find('"')? + start;
    Some(&tag[start..end])
}

/// Pull shapes out of our own SVG output.
fn shapes(svg: &str) -> Result<Vec<Shape>> {
    let mut out = Vec::new();
    for line in svg.lines() {
        let t = line.trim();
        if !t.starts_with("<path") {
            continue;
        }
        let d = attr(t, "d").ok_or_else(|| anyhow!("path without d"))?;
        let fill = attr(t, "fill").and_then(hex).unwrap_or([0, 0, 0]);
        let (tx, ty) = match attr(t, "transform") {
            Some(tr) => {
                let inner = tr.trim_start_matches("translate(").trim_end_matches(')');
                let mut it = inner.split(',').map(|v| v.trim().parse::<f64>().unwrap_or(0.0));
                (it.next().unwrap_or(0.0), it.next().unwrap_or(0.0))
            }
            None => (0.0, 0.0),
        };
        let even_odd = attr(t, "fill-rule") == Some("evenodd");
        out.push(Shape { fill, tx, ty, cmds: parse_d(d)?, even_odd });
    }
    Ok(out)
}

fn fmt(v: f64) -> String {
    let s = format!("{:.3}", v);
    let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    if s == "-0" { "0".into() } else { s }
}

/// Vector PDF, one page the size of the original image (1 px = 1 pt).
pub fn to_pdf(out: &ConvertOutput) -> Result<Vec<u8>> {
    let shapes = shapes(&out.svg)?;
    let (pw, ph) = (out.width as f64, out.height as f64);
    let sx = pw / out.traced_width as f64;
    let sy = ph / out.traced_height as f64;

    let mut content = String::new();
    // Flip y so that SVG's top-left origin maps to PDF's bottom-left origin.
    writeln!(content, "q {} 0 0 {} 0 {} cm", fmt(sx), fmt(-sy), fmt(ph))?;
    for s in &shapes {
        let [r, g, b] = s.fill;
        writeln!(content, "{} {} {} rg", fmt(r as f64 / 255.0), fmt(g as f64 / 255.0), fmt(b as f64 / 255.0))?;
        if s.tx != 0.0 || s.ty != 0.0 {
            writeln!(content, "q 1 0 0 1 {} {} cm", fmt(s.tx), fmt(s.ty))?;
        }
        for c in &s.cmds {
            match *c {
                Cmd::M(x, y) => writeln!(content, "{} {} m", fmt(x), fmt(y))?,
                Cmd::L(x, y) => writeln!(content, "{} {} l", fmt(x), fmt(y))?,
                Cmd::C(a, b2, c2, d, e, f) => writeln!(
                    content,
                    "{} {} {} {} {} {} c",
                    fmt(a), fmt(b2), fmt(c2), fmt(d), fmt(e), fmt(f)
                )?,
                Cmd::Z => writeln!(content, "h")?,
            }
        }
        writeln!(content, "{}", if s.even_odd { "f*" } else { "f" })?;
        if s.tx != 0.0 || s.ty != 0.0 {
            writeln!(content, "Q")?;
        }
    }
    writeln!(content, "Q")?;

    let mut pdf: Vec<u8> = Vec::new();
    let mut offsets = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n");
    let objs = [
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
        format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {} {}] /Contents 4 0 R /Resources << >> >>", fmt(pw), fmt(ph)),
        format!("<< /Length {} >>\nstream\n{}endstream", content.len(), content),
        "<< /Producer (Vector Baba) >>".to_string(),
    ];
    for (i, o) in objs.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n{}\nendobj\n", i + 1, o).as_bytes());
    }
    let xref = pdf.len();
    pdf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).as_bytes());
    for o in &offsets {
        pdf.extend_from_slice(format!("{:010} 00000 n \n", o).as_bytes());
    }
    pdf.extend_from_slice(
        format!("trailer\n<< /Size {} /Root 1 0 R /Info 5 0 R >>\nstartxref\n{}\n%%EOF\n", objs.len() + 1, xref).as_bytes(),
    );
    Ok(pdf)
}

/// Encapsulated PostScript (level 2), bounding box = original image size.
pub fn to_eps(out: &ConvertOutput) -> Result<String> {
    let shapes = shapes(&out.svg)?;
    let (pw, ph) = (out.width as f64, out.height as f64);
    let sx = pw / out.traced_width as f64;
    let sy = ph / out.traced_height as f64;
    let mut s = String::new();
    writeln!(s, "%!PS-Adobe-3.0 EPSF-3.0")?;
    writeln!(s, "%%Creator: Vector Baba")?;
    writeln!(s, "%%BoundingBox: 0 0 {} {}", pw.ceil() as i64, ph.ceil() as i64)?;
    writeln!(s, "%%HiResBoundingBox: 0 0 {} {}", fmt(pw), fmt(ph))?;
    writeln!(s, "%%LanguageLevel: 2")?;
    writeln!(s, "%%EndComments")?;
    writeln!(s, "/m {{ moveto }} bind def /l {{ lineto }} bind def /c {{ curveto }} bind def")?;
    writeln!(s, "/h {{ closepath }} bind def /f {{ fill }} bind def /rg {{ setrgbcolor }} bind def")?;
    writeln!(s, "gsave")?;
    writeln!(s, "0 {} translate {} {} scale", fmt(ph), fmt(sx), fmt(-sy))?;
    for sh in &shapes {
        let [r, g, b] = sh.fill;
        writeln!(s, "{} {} {} rg", fmt(r as f64 / 255.0), fmt(g as f64 / 255.0), fmt(b as f64 / 255.0))?;
        writeln!(s, "gsave {} {} translate newpath", fmt(sh.tx), fmt(sh.ty))?;
        for c in &sh.cmds {
            match *c {
                Cmd::M(x, y) => writeln!(s, "{} {} m", fmt(x), fmt(y))?,
                Cmd::L(x, y) => writeln!(s, "{} {} l", fmt(x), fmt(y))?,
                Cmd::C(a, b2, c2, d, e, f) => {
                    writeln!(s, "{} {} {} {} {} {} c", fmt(a), fmt(b2), fmt(c2), fmt(d), fmt(e), fmt(f))?
                }
                Cmd::Z => writeln!(s, "h")?,
            }
        }
        writeln!(s, "{} grestore", if sh.even_odd { "eofill" } else { "f" })?;
    }
    writeln!(s, "grestore")?;
    writeln!(s, "%%EOF")?;
    Ok(s)
}

/// DXF (R12-compatible ASCII): each shape outline as closed POLYLINE entities,
/// one layer per fill color, y flipped so the drawing is upright in CAD.
pub fn to_dxf(out: &ConvertOutput) -> Result<String> {
    let shapes = shapes(&out.svg)?;
    let sx = out.width as f64 / out.traced_width as f64;
    let sy = out.height as f64 / out.traced_height as f64;
    let h = out.height as f64;
    let mut s = String::new();
    let mut layers: Vec<String> = Vec::new();
    for sh in &shapes {
        let name = format!("C_{:02X}{:02X}{:02X}", sh.fill[0], sh.fill[1], sh.fill[2]);
        if !layers.contains(&name) {
            layers.push(name);
        }
    }
    writeln!(s, "0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nAC1009\n9\n$EXTMIN\n10\n0\n20\n0\n9\n$EXTMAX\n10\n{}\n20\n{}\n0\nENDSEC", fmt(out.width as f64), fmt(h))?;
    writeln!(s, "0\nSECTION\n2\nTABLES\n0\nTABLE\n2\nLAYER\n70\n{}", layers.len())?;
    for (i, l) in layers.iter().enumerate() {
        writeln!(s, "0\nLAYER\n2\n{}\n70\n0\n62\n{}\n6\nCONTINUOUS", l, (i % 255) + 1)?;
    }
    writeln!(s, "0\nENDTAB\n0\nENDSEC\n0\nSECTION\n2\nENTITIES")?;
    for sh in &shapes {
        let layer = format!("C_{:02X}{:02X}{:02X}", sh.fill[0], sh.fill[1], sh.fill[2]);
        for poly in sh.polygons() {
            writeln!(s, "0\nPOLYLINE\n8\n{}\n66\n1\n70\n1", layer)?;
            for (x, y) in poly {
                writeln!(s, "0\nVERTEX\n8\n{}\n10\n{}\n20\n{}", layer, fmt(x * sx), fmt(h - y * sy))?;
            }
            writeln!(s, "0\nSEQEND")?;
        }
    }
    writeln!(s, "0\nENDSEC\n0\nEOF")?;
    Ok(s)
}

/// Rasterize to PNG bytes at `scale` times the original size, on a transparent
/// or solid background.
pub fn to_png(out: &ConvertOutput, scale: f32, background: Option<[u8; 3]>) -> Result<Vec<u8>> {
    let longest = out.width.max(out.height) as f32 * scale.clamp(0.1, 8.0);
    let (w, h, rgba) = svg::rasterize(&out.svg, longest.round() as u32, background)?;
    let img = image::RgbaImage::from_raw(w, h, rgba).ok_or_else(|| anyhow!("png buffer"))?;
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png)?;
    Ok(buf.into_inner())
}

/// Encode `out` in `format`. Returns bytes ready to write to disk.
pub fn encode(out: &ConvertOutput, format: Format, png_scale: f32, png_background: Option<[u8; 3]>) -> Result<Vec<u8>> {
    Ok(match format {
        Format::Svg => out.svg.clone().into_bytes(),
        Format::Pdf | Format::Ai => to_pdf(out)?,
        Format::Eps => to_eps(out)?.into_bytes(),
        Format::Dxf => to_dxf(out)?.into_bytes(),
        Format::Png => to_png(out, png_scale, png_background)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_our_path_syntax() {
        let c = parse_d("M0 0 C0.93 -0 0.93 -0 5.64 -0.01 L10 10 Z").unwrap();
        assert_eq!(c.len(), 4);
        assert!(matches!(c[1], Cmd::C(..)));
        assert!(matches!(c[3], Cmd::Z));
    }

    #[test]
    fn hit_test_finds_topmost_shape() {
        let svg = "<svg>\n  <path d=\"M0 0 L100 0 L100 100 L0 100 Z\" fill=\"#ffffff\"/>\n  <path d=\"M20 20 L40 20 L40 40 L20 40 Z\" fill=\"#000000\" transform=\"translate(10,10)\"/>\n</svg>\n";
        let shapes = parse_shapes(svg).unwrap();
        assert_eq!(shapes.len(), 2);
        assert_eq!(hit_test(&shapes, 5.0, 5.0), Some(0));
        assert_eq!(hit_test(&shapes, 40.0, 40.0), Some(1)); // inside translated square
        assert_eq!(hit_test(&shapes, 200.0, 200.0), None);
    }

    #[test]
    fn negative_numbers_without_spaces() {
        let c = parse_d("M1-2L-3-4Z").unwrap();
        assert!(matches!(c[0], Cmd::M(x, y) if x == 1.0 && y == -2.0));
        assert!(matches!(c[1], Cmd::L(x, y) if x == -3.0 && y == -4.0));
    }
}
