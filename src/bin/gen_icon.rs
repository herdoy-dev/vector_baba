//! Renders the app icon from an inline SVG to a 1024x1024 PNG.
//! Usage: gen-icon <out.png>

use resvg::{tiny_skia, usvg};

const ICON_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#1b1f3a"/>
      <stop offset="1" stop-color="#0b0d1a"/>
    </linearGradient>
    <linearGradient id="v" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#ff9f43"/>
      <stop offset="0.5" stop-color="#ff5e62"/>
      <stop offset="1" stop-color="#a55eea"/>
    </linearGradient>
  </defs>
  <rect x="64" y="64" width="896" height="896" rx="200" fill="url(#bg)"/>
  <!-- pixel grid hint -->
  <g fill="#ffffff" opacity="0.08">
    <rect x="160" y="160" width="96" height="96"/><rect x="352" y="160" width="96" height="96"/>
    <rect x="256" y="256" width="96" height="96"/><rect x="160" y="352" width="96" height="96"/>
  </g>
  <!-- bezier V -->
  <path d="M 230 300 C 300 300, 330 360, 360 430 L 512 790 L 664 430 C 694 360, 724 300, 794 300 L 794 300 C 740 320, 700 400, 660 500 L 512 870 L 364 500 C 324 400, 284 320, 230 300 Z"
        fill="url(#v)"/>
  <!-- control handles -->
  <g stroke="#ffffff" stroke-width="10" fill="none" opacity="0.9">
    <line x1="230" y1="300" x2="360" y2="430"/>
    <line x1="794" y1="300" x2="664" y2="430"/>
  </g>
  <g fill="#ffffff">
    <circle cx="230" cy="300" r="22"/><circle cx="794" cy="300" r="22"/>
    <circle cx="512" cy="870" r="22"/>
  </g>
</svg>"##;

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "icon_1024.png".into());
    let tree = usvg::Tree::from_str(ICON_SVG, &usvg::Options::default()).expect("parse icon svg");
    let size = 1024u32;
    let mut pixmap = tiny_skia::Pixmap::new(size, size).expect("pixmap");
    let scale = size as f32 / tree.size().width();
    resvg::render(
        &tree,
        tiny_skia::Transform::from_scale(scale, scale),
        &mut pixmap.as_mut(),
    );
    pixmap.save_png(&out).expect("write png");
    println!("wrote {out}");
}
