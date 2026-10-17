# Vector Baba

Raster to SVG vectorizer for macOS (Apple Silicon), written in Rust. Drop a
PNG/JPEG, tune a handful of controls, export a clean, editable SVG.

Original work. Not derived from any commercial vectorizer's code.

## Build

```sh
cargo build --release               # vector-baba (GUI), vector-baba-cli, gen-icon
scripts/build_dmg.sh                # dist/VectorBaba-<version>-arm64.dmg
```

## Workflow

Load Image → Choose Mode → Review Result → Remove Background → Save Result, with
Batch Processing from the first page. Modes:

- **Fully automatic**: the analyzer (color clustering on flat pixels, edge
  softness, noise estimate, line-art and background detection) picks everything.
- **Basic**: Image Category (Photograph / Artwork with blending / Artwork without
  blending) and Image Quality (Low / Medium / High).
- **Advanced**: palette mode (unlimited, number of colors, pick the colors with an
  eyedropper, two colors), cluster similar colors, transparency mode (preserve,
  flatten to white, flatten to custom color), anti-aliasing engine, artifact
  rejection, noise reduction, detect sharp corners, path smoothing, curve
  complexity, lines-only output, stacked or cut-out layering, stroked boundaries.

Review shows Bitmap, Vector, Segmentation or side by side. Remove Background
lets you click shapes to drop them from the export. Export: SVG, PDF, EPS, AI
(PDF-compatible), DXF (outlines, one layer per color), PNG.

## CLI

```sh
vector-baba-cli input.png                       # fully automatic, writes input.svg
vector-baba-cli photo.jpg -t photo -q high -f pdf
vector-baba-cli art.png --palette "#fff,#000,#e63946" --remove-background -f dxf
vector-baba-cli --help
```

## How it works

1. **Load and limit size.** Large images are downscaled (Lanczos) to the work
   size; the SVG keeps the original dimensions through `viewBox`.
2. **Smooth.** Premultiplied Gaussian blur removes JPEG noise and anti-aliasing
   fringes without dark halos at transparent edges.
3. **Quantize.** k-means++ in OKLab space reduces the image to the chosen palette
   so every region gets one flat, perceptually chosen color. Black-and-white mode
   thresholds on luma instead.
4. **Trace.** Regions are clustered hierarchically and their outlines fitted with
   Bezier splines (via the visioncortex tracer), with corner, length and speckle
   thresholds driven by the Detail and Despeckle controls.
5. **Serialize.** Paths are written as plain `<path d="…" fill="#rrggbb"/>` so the
   file opens in Illustrator, Inkscape, Figma and browsers without surprises.

## Shortcuts

⌘O open · ⌘S export SVG · ⌘+ / ⌘− / ⌘0 zoom

## Installing the DMG

The app is ad-hoc signed, not notarized. On first launch macOS will say it
cannot verify the developer: right-click the app, choose Open, then Open again.
Or run `xattr -dr com.apple.quarantine "/Applications/Vector Baba.app"`.

## License

MIT.
