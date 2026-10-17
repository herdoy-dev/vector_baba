#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
//! Vector Baba desktop app.
//!
//! Layout: image canvas on the left, a titled step panel on the right with
//! Reset / Back / Next along its bottom, and a status bar. Steps:
//! Load Image → Choose Mode → Review Result → Save Result, plus Batch Processing.

use eframe::egui;
use image::RgbaImage;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashSet;
use vector_baba::export::{self, Format, Shape};
use vector_baba::{
    analyze, convert, convert_with_progress, preprocess, svg, Analysis, Complexity, ConvertOutput, CurveMode,
    ImageType, Layering, Level, Mode, Quality, Settings, Transparency,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const PREVIEW_MAX: u32 = 2048;
const DEBOUNCE: Duration = Duration::from_millis(250);
const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp"];
const ACCENT: egui::Color32 = egui::Color32::from_rgb(58, 110, 200);
const PANEL_W: f32 = 372.0;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Vector Baba")
            .with_inner_size([1240.0, 820.0])
            .with_min_inner_size([960.0, 620.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    let initial: Option<PathBuf> = std::env::args().nth(1).map(PathBuf::from).filter(|p| p.is_file());
    eframe::run_native(
        "Vector Baba",
        options,
        Box::new(move |cc| {
            let mut app = App::new(cc);
            if let Some(p) = initial {
                app.load_path(&cc.egui_ctx, p);
                app.step = Step::Review;
            }
            Ok(Box::new(app))
        }),
    )
}

// ---------------------------------------------------------------------------

enum JobMsg {
    Stage { id: u64, name: &'static str },
    Traced { id: u64, result: anyhow::Result<(ConvertOutput, Vec<Shape>, (u32, u32, Vec<u8>))> },
    Rendered { id: u64, result: anyhow::Result<(u32, u32, Vec<u8>)> },
}

enum BatchMsg {
    Progress { done: usize, total: usize, last: String },
    Finished { ok: usize, failed: usize },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    Load,
    Mode,
    Review,
    Background,
    Save,
    Batch,
}

impl Step {
    fn title(self) -> &'static str {
        match self {
            Step::Load => "Load Image",
            Step::Mode => "Choose Mode",
            Step::Review => "Review Result",
            Step::Background => "Remove Background",
            Step::Save => "Save Result",
            Step::Batch => "Batch Processing",
        }
    }
    fn help(self) -> &'static str {
        match self {
            Step::Load => "Open an image from disk, paste one from the clipboard, or drop it on the canvas.",
            Step::Mode => "Fully automatic analyzes the image and picks settings. Basic lets you say what the image is. Advanced exposes every control.",
            Step::Review => "Compare the bitmap and the vector. Zoom with ⌘+ and ⌘−. Segmentation shows the flat color regions behind the shapes.",
            Step::Background => "Click shapes in the vector view to remove them. Removed shapes are outlined in red and are not exported. Click again to restore.",
            Step::Save => "Choose a format and save. SVG, PDF and EPS are true vectors. PNG rasterizes at the chosen scale.",
            Step::Batch => "Apply the current mode and settings to many files at once, writing results to one folder.",
        }
    }
    fn next(self) -> Option<Step> {
        match self {
            Step::Load => Some(Step::Mode),
            Step::Mode => Some(Step::Review),
            Step::Review => Some(Step::Background),
            Step::Background => Some(Step::Save),
            Step::Save | Step::Batch => None,
        }
    }
    fn back(self) -> Option<Step> {
        match self {
            Step::Load => None,
            Step::Mode | Step::Batch => Some(Step::Load),
            Step::Review => Some(Step::Mode),
            Step::Background => Some(Step::Review),
            Step::Save => Some(Step::Background),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ModeChoice {
    Automatic,
    Basic,
    Advanced,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum View {
    Bitmap,
    Vector,
    Segmentation,
    Compare,
}

struct App {
    step: Step,
    mode_choice: ModeChoice,
    image_type: ImageType,
    quality: Quality,
    remove_background: bool,
    adv: Settings,
    palette_custom: bool,
    custom_palette: Vec<[u8; 3]>,
    picking: bool,
    flatten_color: [u8; 3],
    removed: HashSet<usize>,
    shapes: Vec<Shape>,
    render_job: Option<u64>,
    stage: Option<&'static str>,

    analysis: Option<Analysis>,
    image_path: Option<PathBuf>,
    image_name: String,
    image: Option<Arc<RgbaImage>>,
    bitmap_tex: Option<egui::TextureHandle>,
    vector_tex: Option<egui::TextureHandle>,
    seg_tex: Option<egui::TextureHandle>,
    output: Option<ConvertOutput>,
    recent: Vec<PathBuf>,

    export_format: Format,
    png_scale: f32,
    png_transparent: bool,

    batch_files: Vec<PathBuf>,
    batch_out_dir: Option<PathBuf>,
    batch_format: Format,
    batch_running: bool,
    batch_progress: (usize, usize),
    batch_log: Vec<String>,
    batch_tx: Sender<BatchMsg>,
    batch_rx: Receiver<BatchMsg>,

    view: View,
    zoom: f32,
    status: String,
    error: Option<String>,

    last_sent: Option<Settings>,
    dirty_since: Option<Instant>,
    job_counter: u64,
    running_job: Option<u64>,
    tx: Sender<JobMsg>,
    rx: Receiver<JobMsg>,
}

fn recent_file() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let dir = PathBuf::from(home).join("Library/Application Support/VectorBaba");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join("recent.txt"))
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::light());
        cc.egui_ctx.all_styles_mut(|style| {
            style.spacing.item_spacing = egui::vec2(8.0, 7.0);
            style.spacing.slider_width = 170.0;
            style.spacing.button_padding = egui::vec2(10.0, 5.0);
            style.visuals.selection.bg_fill = ACCENT.gamma_multiply(0.85);
            style.visuals.selection.stroke = egui::Stroke::new(1.0, ACCENT);
            style.visuals.hyperlink_color = ACCENT;
        });
        let (tx, rx) = channel();
        let (batch_tx, batch_rx) = channel();
        let recent = recent_file()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .map(|s| s.lines().filter(|l| !l.is_empty()).map(PathBuf::from).filter(|p| p.exists()).collect())
            .unwrap_or_default();
        App {
            step: Step::Load,
            mode_choice: ModeChoice::Automatic,
            image_type: ImageType::BlendedArt,
            quality: Quality::Medium,
            remove_background: false,
            adv: Settings::default(),
            palette_custom: false,
            custom_palette: Vec::new(),
            picking: false,
            flatten_color: [255, 255, 255],
            removed: HashSet::new(),
            shapes: Vec::new(),
            render_job: None,
            stage: None,
            analysis: None,
            image_path: None,
            image_name: String::new(),
            image: None,
            bitmap_tex: None,
            vector_tex: None,
            seg_tex: None,
            output: None,
            recent,
            export_format: Format::Svg,
            png_scale: 1.0,
            png_transparent: true,
            batch_files: Vec::new(),
            batch_out_dir: None,
            batch_format: Format::Svg,
            batch_running: false,
            batch_progress: (0, 0),
            batch_log: Vec::new(),
            batch_tx,
            batch_rx,
            view: View::Compare,
            zoom: 1.0,
            status: "Ready".to_owned(),
            error: None,
            last_sent: None,
            dirty_since: None,
            job_counter: 0,
            running_job: None,
            tx,
            rx,
        }
    }

    // ----- settings ------------------------------------------------------------

    fn effective_settings(&self) -> Settings {
        let mut s = match self.mode_choice {
            ModeChoice::Automatic => match &self.analysis {
                Some(a) => a.recommend(self.quality),
                None => Settings::from_basic(ImageType::BlendedArt, self.quality),
            },
            ModeChoice::Basic => Settings::from_basic(self.image_type, self.quality),
            ModeChoice::Advanced => {
                let mut s = self.adv.clone();
                s.custom_palette = if self.palette_custom && !self.custom_palette.is_empty() {
                    Some(self.custom_palette.clone())
                } else {
                    None
                };
                s
            }
        };
        s.remove_background = self.remove_background;
        s
    }

    fn reset_step(&mut self) {
        match self.step {
            Step::Mode => {
                self.mode_choice = ModeChoice::Automatic;
                self.quality = Quality::Medium;
                self.remove_background = false;
                self.adv = Settings::default();
                self.palette_custom = false;
                self.custom_palette.clear();
                self.picking = false;
                if let Some(a) = &self.analysis {
                    self.image_type = a.image_type;
                }
            }
            Step::Review => {
                self.zoom = 1.0;
                self.view = View::Compare;
            }
            Step::Background => {
                self.removed.clear();
                self.remove_background = false;
                self.rerender(&egui::Context::default());
            }
            Step::Save => {
                self.export_format = Format::Svg;
                self.png_scale = 1.0;
                self.png_transparent = true;
            }
            Step::Batch => {
                if !self.batch_running {
                    self.batch_files.clear();
                    self.batch_log.clear();
                    self.batch_progress = (0, 0);
                }
            }
            Step::Load => {}
        }
    }

    // ----- loading -------------------------------------------------------------

    fn open_dialog(&mut self, ctx: &egui::Context) {
        if let Some(p) = rfd::FileDialog::new().add_filter("Images", IMAGE_EXTS).pick_file() {
            self.load_path(ctx, p);
        }
    }

    fn load_path(&mut self, ctx: &egui::Context, path: PathBuf) {
        match preprocess::load(&path) {
            Ok(img) => {
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                self.recent.retain(|p| p != &path);
                self.recent.insert(0, path.clone());
                self.recent.truncate(8);
                if let Some(f) = recent_file() {
                    let body: Vec<String> = self.recent.iter().map(|p| p.display().to_string()).collect();
                    let _ = std::fs::write(f, body.join("\n"));
                }
                self.image_path = Some(path);
                self.install_image(ctx, img, name);
            }
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }

    fn paste_clipboard(&mut self, ctx: &egui::Context) {
        let result = arboard::Clipboard::new().and_then(|mut c| c.get_image());
        match result {
            Ok(data) => {
                let (w, h) = (data.width as u32, data.height as u32);
                match RgbaImage::from_raw(w, h, data.bytes.into_owned()) {
                    Some(img) => {
                        self.image_path = None;
                        self.install_image(ctx, img, "Clipboard image".to_owned());
                    }
                    None => self.error = Some("clipboard image has an unexpected layout".into()),
                }
            }
            Err(e) => self.error = Some(format!("No image on the clipboard ({e})")),
        }
    }

    fn install_image(&mut self, ctx: &egui::Context, img: RgbaImage, name: String) {
        let (w, h) = img.dimensions();
        self.bitmap_tex = Some(ctx.load_texture(
            "bitmap",
            egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], img.as_raw()),
            egui::TextureOptions::LINEAR,
        ));
        let analysis = analyze(&img);
        self.image_type = analysis.image_type;
        self.remove_background = false;
        self.analysis = Some(analysis);
        self.vector_tex = None;
        self.seg_tex = None;
        self.output = None;
        self.removed.clear();
        self.shapes.clear();
        self.image = Some(Arc::new(img));
        self.image_name = name.clone();
        self.error = None;
        self.status = format!("{name} · {w}×{h} px");
        self.last_sent = None;
        self.dirty_since = Some(Instant::now() - DEBOUNCE);
        self.step = Step::Mode;
        self.zoom = 1.0;
    }

    // ----- conversion ----------------------------------------------------------

    fn start_job(&mut self, ctx: &egui::Context) {
        let Some(img) = self.image.clone() else { return };
        let settings = self.effective_settings();
        self.job_counter += 1;
        let id = self.job_counter;
        self.running_job = Some(id);
        self.last_sent = Some(settings.clone());
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        self.removed.clear();
        std::thread::spawn(move || {
            let progress = |name: &str| {
                let name: &'static str = match name {
                    "Preparing image" => "Preparing image",
                    "Analyzing image colors" => "Analyzing image colors",
                    "Segmenting image" => "Segmenting image",
                    "Smoothing result" => "Smoothing result",
                    _ => "Writing vectors",
                };
                let _ = tx.send(JobMsg::Stage { id, name });
                ctx.request_repaint();
            };
            let result = convert_with_progress(&img, &settings, &progress).and_then(|out| {
                let shapes = export::parse_shapes(&out.svg)?;
                let preview = svg::rasterize(&out.svg, PREVIEW_MAX, None)?;
                Ok((out, shapes, preview))
            });
            let _ = tx.send(JobMsg::Traced { id, result });
            ctx.request_repaint();
        });
    }

    /// Re-rasterize the vector preview after shapes were removed or restored.
    fn rerender(&mut self, ctx: &egui::Context) {
        let Some(out) = &self.output else { return };
        let svg_text = out.render_svg(&self.removed);
        self.job_counter += 1;
        let id = self.job_counter;
        self.render_job = Some(id);
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = svg::rasterize(&svg_text, PREVIEW_MAX, None);
            let _ = tx.send(JobMsg::Rendered { id, result });
            ctx.request_repaint();
        });
    }

    fn poll(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                JobMsg::Stage { id, name } => {
                    if Some(id) == self.running_job {
                        self.stage = Some(name);
                    }
                    continue;
                }
                JobMsg::Rendered { id, result } => {
                    if Some(id) != self.render_job {
                        continue;
                    }
                    self.render_job = None;
                    if let Ok((w, h, rgba)) = result {
                        self.vector_tex = Some(ctx.load_texture(
                            "vector",
                            egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba),
                            egui::TextureOptions::LINEAR,
                        ));
                    }
                    continue;
                }
                JobMsg::Traced { .. } => {}
            }
            let JobMsg::Traced { id, result } = msg else { continue };
            if Some(id) != self.running_job {
                continue;
            }
            self.running_job = None;
            self.stage = None;
            match result {
                Ok((out, shapes, (w, h, rgba))) => {
                    self.shapes = shapes;
                    self.vector_tex = Some(ctx.load_texture(
                        "vector",
                        egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba),
                        egui::TextureOptions::LINEAR,
                    ));
                    let seg = &out.segmentation;
                    self.seg_tex = Some(ctx.load_texture(
                        "segmentation",
                        egui::ColorImage::from_rgba_unmultiplied([seg.width() as usize, seg.height() as usize], seg.as_raw()),
                        egui::TextureOptions::NEAREST,
                    ));
                    self.status = format!(
                        "{} · {} shapes · {} colors · {} ms",
                        self.image_name, out.path_count, out.palette.len(), out.elapsed_ms
                    );
                    self.error = None;
                    self.output = Some(out);
                }
                Err(e) => self.error = Some(format!("{e:#}")),
            }
        }
        while let Ok(msg) = self.batch_rx.try_recv() {
            match msg {
                BatchMsg::Progress { done, total, last } => {
                    self.batch_progress = (done, total);
                    self.batch_log.push(last);
                    self.status = format!("Batch: {done}/{total}");
                }
                BatchMsg::Finished { ok, failed } => {
                    self.batch_running = false;
                    self.status = format!("Batch finished: {ok} converted, {failed} failed");
                }
            }
        }
    }

    fn start_batch(&mut self, ctx: &egui::Context) {
        let Some(out_dir) = self.batch_out_dir.clone() else { return };
        if self.batch_files.is_empty() || self.batch_running {
            return;
        }
        let files = self.batch_files.clone();
        let settings = self.effective_settings();
        let format = self.batch_format;
        let png_scale = self.png_scale;
        let bg = if self.png_transparent { None } else { Some([255, 255, 255]) };
        let tx = self.batch_tx.clone();
        let ctx = ctx.clone();
        self.batch_running = true;
        self.batch_log.clear();
        self.batch_progress = (0, files.len());
        std::thread::spawn(move || {
            let total = files.len();
            let (mut ok, mut failed) = (0, 0);
            for (i, f) in files.iter().enumerate() {
                let stem = f.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| format!("image{i}"));
                let target = out_dir.join(format!("{stem}.{}", format.extension()));
                let res = preprocess::load(f)
                    .and_then(|img| convert(&img, &settings))
                    .and_then(|out| export::encode(&out, format, png_scale, bg))
                    .and_then(|bytes| Ok(std::fs::write(&target, bytes)?));
                let line = match res {
                    Ok(()) => {
                        ok += 1;
                        format!("✓ {}", target.file_name().unwrap_or_default().to_string_lossy())
                    }
                    Err(e) => {
                        failed += 1;
                        format!("✗ {}: {e:#}", f.file_name().unwrap_or_default().to_string_lossy())
                    }
                };
                let _ = tx.send(BatchMsg::Progress { done: i + 1, total, last: line });
                ctx.request_repaint();
            }
            let _ = tx.send(BatchMsg::Finished { ok, failed });
            ctx.request_repaint();
        });
    }

    fn save_dialog(&mut self) {
        let Some(out) = &self.output else { return };
        let fmt = self.export_format;
        let stem = self
            .image_path
            .as_ref()
            .and_then(|p| p.file_stem().map(|s| s.to_string_lossy().to_string()))
            .unwrap_or_else(|| "vector".to_owned());
        let mut dlg = rfd::FileDialog::new()
            .add_filter(fmt.label(), &[fmt.extension()])
            .set_file_name(format!("{stem}.{}", fmt.extension()));
        if let Some(dir) = self.image_path.as_ref().and_then(|p| p.parent()) {
            dlg = dlg.set_directory(dir);
        }
        let Some(path) = dlg.save_file() else { return };
        let bg = if self.png_transparent { None } else { Some([255, 255, 255]) };
        let out = out.without(&self.removed);
        match export::encode(&out, fmt, self.png_scale, bg).and_then(|b| Ok(std::fs::write(&path, b)?)) {
            Ok(()) => self.status = format!("Saved {}", path.display()),
            Err(e) => self.error = Some(format!("export failed: {e:#}")),
        }
    }

    // ----- right panel ---------------------------------------------------------

    fn header(&mut self, ui: &mut egui::Ui) {
        let frame = egui::Frame::new()
            .fill(ACCENT.gamma_multiply(0.18))
            .stroke(egui::Stroke::new(1.0, ACCENT.gamma_multiply(0.5)))
            .corner_radius(6.0)
            .inner_margin(egui::Margin::symmetric(10, 8));
        frame.show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(self.step.title()).size(16.0).strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.button("?  Help").on_hover_text(self.step.help());
                });
            });
        });
    }

    fn action_card(ui: &mut egui::Ui, enabled: bool, glyph: &str, title: &str, lines: &[&str]) -> bool {
        let frame = egui::Frame::group(ui.style())
            .fill(if enabled { ui.visuals().extreme_bg_color } else { ui.visuals().faint_bg_color })
            .inner_margin(8.0);
        let resp = frame
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(56.0, 56.0), egui::Sense::hover());
                    ui.painter().rect_filled(rect, 8.0, if enabled { ACCENT.gamma_multiply(0.15) } else { egui::Color32::from_gray(225) });
                    ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, glyph, egui::FontId::proportional(28.0),
                        if enabled { ACCENT } else { egui::Color32::from_gray(160) });
                    ui.vertical(|ui| {
                        ui.add(egui::Label::new(egui::RichText::new(title).strong().color(
                            if enabled { ui.visuals().text_color() } else { egui::Color32::from_gray(150) })).wrap());
                        for l in lines {
                            ui.add(egui::Label::new(egui::RichText::new(format!("• {l}")).small().color(
                                if enabled { ui.visuals().text_color() } else { egui::Color32::from_gray(160) })).wrap());
                        }
                    });
                });
            })
            .response;
        enabled && ui.interact(resp.rect, resp.id.with(title), egui::Sense::click()).clicked()
    }

    fn panel_load(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if Self::action_card(ui, true, "📂", "Import from file",
            &["PNG, JPEG, GIF, BMP, TIFF, WebP.", "Shortcut: ⌘O"]) {
            self.open_dialog(ctx);
        }
        if Self::action_card(ui, true, "📋", "Paste from clipboard",
            &["Copy an image in any app, then paste it here.", "Shortcut: ⌘V"]) {
            self.paste_clipboard(ctx);
        }
        if Self::action_card(ui, true, "⚙", "Batch processing",
            &["Convert many images with the same settings."]) {
            self.step = Step::Batch;
        }
        ui.add_space(4.0);
        egui::CollapsingHeader::new("Recent images").default_open(!self.recent.is_empty()).show(ui, |ui| {
            if self.recent.is_empty() {
                ui.label(egui::RichText::new("Nothing yet.").weak());
            }
            let mut open: Option<PathBuf> = None;
            for p in &self.recent {
                let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
                if ui.link(name).on_hover_text(p.display().to_string()).clicked() {
                    open = Some(p.clone());
                }
            }
            if let Some(p) = open {
                self.load_path(ctx, p);
            }
        });
        if let (Some(img), Some(a)) = (&self.image, self.analysis.clone()) {
            ui.separator();
            ui.label(egui::RichText::new(&self.image_name).strong());
            ui.label(format!("{} × {} px", img.width(), img.height()));
            analysis_panel(ui, &a);
        }
    }

    fn panel_mode(&mut self, ui: &mut egui::Ui) {
        let card = |ui: &mut egui::Ui, selected: bool, title: &str, body: &str| -> bool {
            let frame = egui::Frame::group(ui.style())
                .fill(if selected { ACCENT.gamma_multiply(0.14) } else { ui.visuals().extreme_bg_color })
                .stroke(if selected { egui::Stroke::new(1.5, ACCENT) } else { ui.visuals().widgets.noninteractive.bg_stroke })
                .inner_margin(8.0);
            let r = frame
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        let _ = ui.radio(selected, "");
                        ui.vertical(|ui| {
                            ui.label(egui::RichText::new(title).strong());
                            ui.add(egui::Label::new(egui::RichText::new(body).small().weak()).wrap());
                        });
                    });
                })
                .response;
            ui.interact(r.rect, r.id.with(title), egui::Sense::click()).clicked()
        };
        if card(ui, self.mode_choice == ModeChoice::Automatic, "Fully automatic",
            "Analyzes the image and picks all settings. Best first try.") {
            self.mode_choice = ModeChoice::Automatic;
        }
        if card(ui, self.mode_choice == ModeChoice::Basic, "Basic",
            "Tell it what kind of image this is and how much detail to keep.") {
            self.mode_choice = ModeChoice::Basic;
        }
        if card(ui, self.mode_choice == ModeChoice::Advanced, "Advanced",
            "Full control: palette, eyedropper, noise, curve fitting.") {
            self.mode_choice = ModeChoice::Advanced;
        }
        ui.separator();
        match self.mode_choice {
            ModeChoice::Automatic => {
                if let Some(a) = self.analysis.clone() {
                    ui.label(egui::RichText::new("Analysis").strong());
                    ui.add(egui::Label::new(a.summary()).wrap());
                    let rec = a.recommend(self.quality);
                    ui.add(egui::Label::new(egui::RichText::new(if rec.mode == Mode::BlackWhite {
                        format!("Plan: black & white trace, noise reduction {:.1}", rec.smoothing)
                    } else {
                        format!("Plan: {} colors · noise reduction {:.1} · detail {:.0}%", rec.colors, rec.smoothing, rec.detail * 100.0)
                    }).small().weak()).wrap());
                    if a.background.is_some() && a.background_uniformity > 0.85 && !self.remove_background {
                        ui.add(egui::Label::new(egui::RichText::new("Tip: the background is a solid color. You can remove it.").small().color(ACCENT)).wrap());
                    }
                }
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Quality").strong());
                ui.horizontal(|ui| {
                    for q in Quality::ALL {
                        ui.selectable_value(&mut self.quality, q, q.label()).on_hover_text(q.describe());
                    }
                });
                ui.checkbox(&mut self.remove_background, "Remove background");
            }
            ModeChoice::Basic => {
                ui.label(egui::RichText::new("Image type").strong());
                for t in ImageType::ALL {
                    ui.radio_value(&mut self.image_type, t, t.label()).on_hover_text(t.describe());
                }
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Quality").strong());
                ui.horizontal(|ui| {
                    for q in Quality::ALL {
                        ui.selectable_value(&mut self.quality, q, q.label()).on_hover_text(q.describe());
                    }
                });
                ui.add(egui::Label::new(egui::RichText::new(self.quality.describe()).small().weak()).wrap());
                ui.add_space(4.0);
                ui.checkbox(&mut self.remove_background, "Remove background");
            }
            ModeChoice::Advanced => self.panel_advanced(ui),
        }
    }

    fn panel_advanced(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Start from").strong());
        ui.horizontal_wrapped(|ui| {
            if ui.small_button("Analysis").clicked() {
                if let Some(a) = &self.analysis {
                    let keep = self.remove_background;
                    self.adv = a.recommend(self.quality);
                    self.remove_background = keep;
                }
            }
            for t in ImageType::ALL {
                if ui.small_button(t.label()).clicked() {
                    let keep = self.remove_background;
                    self.adv = Settings::from_basic(t, self.quality);
                    self.remove_background = keep;
                }
            }
        });
        ui.separator();

        // ----- Palette -----
        ui.label(egui::RichText::new("Palette").strong());
        #[derive(PartialEq, Clone, Copy)]
        enum Pal { Unlimited, Number, Pick, Two }
        let mut pal = if self.adv.mode == Mode::BlackWhite {
            Pal::Two
        } else if self.palette_custom {
            Pal::Pick
        } else if self.adv.unlimited_colors {
            Pal::Unlimited
        } else {
            Pal::Number
        };
        ui.radio_value(&mut pal, Pal::Unlimited, "Unlimited colors").on_hover_text("Use as many colors as the analysis sees fit.");
        ui.radio_value(&mut pal, Pal::Number, "Number of colors").on_hover_text("Use at most this many colors.");
        ui.radio_value(&mut pal, Pal::Pick, "Pick the colors").on_hover_text("Use only colors you specify.");
        ui.radio_value(&mut pal, Pal::Two, "Two colors (black & white)").on_hover_text("Use only two colors.");
        match pal {
            Pal::Unlimited => { self.adv.mode = Mode::Color; self.adv.unlimited_colors = true; self.palette_custom = false; }
            Pal::Number => { self.adv.mode = Mode::Color; self.adv.unlimited_colors = false; self.palette_custom = false; }
            Pal::Pick => { self.adv.mode = Mode::Color; self.adv.unlimited_colors = false; self.palette_custom = true; }
            Pal::Two => { self.adv.mode = Mode::BlackWhite; self.palette_custom = false; }
        }
        match pal {
            Pal::Number => {
                ui.add(egui::Slider::new(&mut self.adv.colors, 2..=64).text("Colors"));
            }
            Pal::Pick => {
                ui.horizontal(|ui| {
                    if ui.selectable_label(self.picking, "✚ Pick from image").on_hover_text("Click in the bitmap to add a color. Esc to stop.").clicked() {
                        self.picking = !self.picking;
                        if self.picking {
                            self.view = View::Bitmap;
                        }
                    }
                    if ui.small_button("Suggested").on_hover_text("Start from the colors found by the analysis.").clicked() {
                        if let Some(a) = &self.analysis {
                            self.custom_palette = a.dominant.iter().map(|(c, _)| *c).collect();
                        }
                    }
                    if ui.small_button("Clear").clicked() {
                        self.custom_palette.clear();
                    }
                });
                let mut remove = None;
                ui.horizontal_wrapped(|ui| {
                    for (i, c) in self.custom_palette.iter().enumerate() {
                        let (rect, resp) = ui.allocate_exact_size(egui::vec2(24.0, 24.0), egui::Sense::click());
                        ui.painter().rect_filled(rect, 4.0, egui::Color32::from_rgb(c[0], c[1], c[2]));
                        ui.painter().rect_stroke(rect, 4.0, egui::Stroke::new(1.0, egui::Color32::from_gray(120)), egui::StrokeKind::Inside);
                        if resp.on_hover_text(format!("#{:02x}{:02x}{:02x} (click to remove)", c[0], c[1], c[2])).clicked() {
                            remove = Some(i);
                        }
                    }
                    if self.custom_palette.is_empty() {
                        ui.label(egui::RichText::new("No colors yet.").weak().small());
                    }
                });
                if let Some(i) = remove {
                    self.custom_palette.remove(i);
                }
            }
            Pal::Two => {
                ui.add(egui::Slider::new(&mut self.adv.threshold, 1..=254).text("Threshold"));
                ui.checkbox(&mut self.adv.invert, "Invert");
            }
            Pal::Unlimited => {}
        }
        if pal != Pal::Two {
            ui.checkbox(&mut self.adv.cluster_similar, "Cluster similar colors")
                .on_hover_text("Makes similar colors identical, thereby using fewer colors.");
        }
        ui.separator();

        // ----- Transparency -----
        ui.label(egui::RichText::new("Transparency mode").strong());
        #[derive(PartialEq, Clone, Copy)]
        enum Tr { Preserve, White, Custom }
        let mut tr = match self.adv.transparency {
            Transparency::Preserve => Tr::Preserve,
            Transparency::FlattenWhite => Tr::White,
            Transparency::FlattenCustom(_) => Tr::Custom,
        };
        ui.radio_value(&mut tr, Tr::Preserve, "Preserve transparency").on_hover_text("Suitable if the image is anti-aliased to transparent.");
        ui.radio_value(&mut tr, Tr::White, "Flatten to white").on_hover_text("Suitable if the image was anti-aliased to white.");
        ui.horizontal(|ui| {
            ui.radio_value(&mut tr, Tr::Custom, "Flatten to custom color").on_hover_text("Suitable if the image was anti-aliased to some other color.");
            let mut c = egui::Color32::from_rgb(self.flatten_color[0], self.flatten_color[1], self.flatten_color[2]);
            if ui.color_edit_button_srgba(&mut c).changed() {
                self.flatten_color = [c.r(), c.g(), c.b()];
            }
        });
        self.adv.transparency = match tr {
            Tr::Preserve => Transparency::Preserve,
            Tr::White => Transparency::FlattenWhite,
            Tr::Custom => Transparency::FlattenCustom(self.flatten_color),
        };
        ui.checkbox(&mut self.remove_background, "Remove background").on_hover_text("Makes the border color transparent.");
        ui.separator();

        // ----- Segmentation -----
        ui.label(egui::RichText::new("Segmentation").strong());
        ui.checkbox(&mut self.adv.anti_aliasing_engine, "Use anti-aliasing engine")
            .on_hover_text("Enable if your image has blended edges (is anti-aliased). This has a big impact on the shapes produced.");
        ui.horizontal(|ui| {
            ui.label("Artifact rejection").on_hover_text("Anti-aliasing can leave slivers between shapes. Artifact rejection removes them.");
            for l in Level::ALL {
                ui.selectable_value(&mut self.adv.artifact_rejection, l, l.label());
            }
        });
        ui.add(egui::Slider::new(&mut self.adv.smoothing, 0.0..=3.0).text("Noise reduction"))
            .on_hover_text("Strong smoothing loses detail but is robust to blurry and low-quality sources.");
        ui.add(egui::Slider::new(&mut self.adv.despeckle, 1..=40).text("Minimum pixels"))
            .on_hover_text("Shapes smaller than this many pixels on a side are dropped.");
        ui.separator();

        // ----- Curves -----
        ui.label(egui::RichText::new("Smoothing & curves").strong());
        ui.add(egui::Slider::new(&mut self.adv.detail, 0.0..=1.0).text("Detail"))
            .on_hover_text("Low = fewer nodes and smoother shapes. High = preserve as many details as possible.");
        ui.checkbox(&mut self.adv.sharp_corners, "Detect sharp corners")
            .on_hover_text("Produces crisper results for images containing shapes with sharp corners.");
        ui.horizontal(|ui| {
            ui.label("Path smoothing");
            ui.selectable_value(&mut self.adv.path_smoothing, 0u8, "Less");
            ui.selectable_value(&mut self.adv.path_smoothing, 1u8, "Normal");
            ui.selectable_value(&mut self.adv.path_smoothing, 2u8, "More");
        });
        ui.horizontal(|ui| {
            ui.label("Curve complexity");
            for c in Complexity::ALL {
                ui.selectable_value(&mut self.adv.curve_complexity, c, c.label()).on_hover_text(c.describe());
            }
        });
        ui.label(egui::RichText::new("Output curves").strong());
        ui.radio_value(&mut self.adv.curves, CurveMode::Spline, "Lines and spline curves");
        ui.radio_value(&mut self.adv.curves, CurveMode::Polygon, "Lines only (smaller file)");
        ui.radio_value(&mut self.adv.curves, CurveMode::Pixel, "Pixel-exact outlines");
        ui.separator();

        // ----- Shapes -----
        ui.label(egui::RichText::new("Shape layering").strong());
        ui.radio_value(&mut self.adv.layering, Layering::Stacked, "Stack shapes on top of each other");
        ui.radio_value(&mut self.adv.layering, Layering::Cutout, "Place shapes in cut-outs in shapes below");
        ui.checkbox(&mut self.adv.stroke_boundaries, "Stroke shape boundaries")
            .on_hover_text("Hides hairline seams between cut-out shapes in some viewers.");
        egui::CollapsingHeader::new("Expert").show(ui, |ui| {
            ui.add(egui::Slider::new(&mut self.adv.max_dimension, 400..=4000).text("Work size (px)"));
            ui.add(egui::Slider::new(&mut self.adv.precision, 0..=4).text("Coordinate decimals"));
        });
    }

    fn panel_background(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.add(egui::Label::new("The purpose of this page is to let you remove background shapes from the result. Click a shape in the vector view to remove it; click again to restore it. Removed shapes are outlined in red and are not exported.").wrap());
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.button("Remove border background").on_hover_text("Drops every shape touching the image border that has the border color.").clicked() {
                self.remove_background = true;
            }
            if ui.add_enabled(!self.removed.is_empty(), egui::Button::new("Restore all")).clicked() {
                self.removed.clear();
                self.rerender(ctx);
            }
        });
        ui.label(egui::RichText::new(format!("{} of {} shapes removed", self.removed.len(), self.shapes.len())).weak());
        if self.view != View::Vector && self.view != View::Compare {
            self.view = View::Compare;
        }
    }

    fn panel_review(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("View").strong());
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.view, View::Compare, "Side by side");
            ui.selectable_value(&mut self.view, View::Bitmap, "Bitmap");
            ui.selectable_value(&mut self.view, View::Vector, "Vector");
            ui.selectable_value(&mut self.view, View::Segmentation, "Segmentation");
        });
        ui.separator();
        if let Some(out) = &self.output {
            ui.label(egui::RichText::new("Result").strong());
            egui::Grid::new("stats").num_columns(2).spacing([16.0, 3.0]).show(ui, |ui| {
                ui.label("Shapes"); ui.label(out.path_count.to_string()); ui.end_row();
                ui.label("Colors"); ui.label(out.palette.len().to_string()); ui.end_row();
                ui.label("Traced at"); ui.label(format!("{}×{}", out.traced_width, out.traced_height)); ui.end_row();
                ui.label("Time"); ui.label(format!("{} ms", out.elapsed_ms)); ui.end_row();
                ui.label("SVG size"); ui.label(format!("{:.0} KB", out.svg.len() as f32 / 1024.0)); ui.end_row();
            });
            ui.label(egui::RichText::new("Palette").strong());
            ui.horizontal_wrapped(|ui| {
                for c in &out.palette {
                    let (rect, r) = ui.allocate_exact_size(egui::vec2(20.0, 20.0), egui::Sense::hover());
                    ui.painter().rect_filled(rect, 3.0, egui::Color32::from_rgb(c[0], c[1], c[2]));
                    ui.painter().rect_stroke(rect, 3.0, egui::Stroke::new(1.0, egui::Color32::from_gray(150)), egui::StrokeKind::Inside);
                    r.on_hover_text(format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2]));
                }
            });
        } else if self.running_job.is_some() {
            ui.horizontal(|ui| { ui.spinner(); ui.label("Tracing…"); });
        }
        ui.separator();
        ui.label(egui::RichText::new("Fine-tune").strong());
        match self.mode_choice {
            ModeChoice::Advanced => {
                ui.add(egui::Slider::new(&mut self.adv.detail, 0.0..=1.0).text("Detail"));
                ui.add(egui::Slider::new(&mut self.adv.smoothing, 0.0..=3.0).text("Noise reduction"));
                if self.adv.mode == Mode::Color && !self.palette_custom {
                    ui.add(egui::Slider::new(&mut self.adv.colors, 2..=64).text("Colors"));
                }
            }
            _ => {
                ui.horizontal(|ui| {
                    ui.label("Quality");
                    for q in Quality::ALL {
                        ui.selectable_value(&mut self.quality, q, q.label());
                    }
                });
                if ui.button("Hand-pick settings (Advanced)").clicked() {
                    let cur = self.effective_settings();
                    self.adv = cur;
                    self.adv.custom_palette = None;
                    self.mode_choice = ModeChoice::Advanced;
                    self.step = Step::Mode;
                }
            }
        }
    }

    fn panel_save(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Format").strong());
        for f in Format::ALL {
            ui.radio_value(&mut self.export_format, f, f.label()).on_hover_text(f.describe());
        }
        ui.add(egui::Label::new(egui::RichText::new(self.export_format.describe()).small().weak()).wrap());
        if self.export_format == Format::Png {
            ui.add(egui::Slider::new(&mut self.png_scale, 0.25..=8.0).logarithmic(true).text("Scale"));
            if let Some(out) = &self.output {
                ui.label(egui::RichText::new(format!("{} × {} px",
                    (out.width as f32 * self.png_scale).round() as u32,
                    (out.height as f32 * self.png_scale).round() as u32)).small().weak());
            }
            ui.checkbox(&mut self.png_transparent, "Transparent background");
        }
        ui.add_space(8.0);
        let ready = self.output.is_some() && self.running_job.is_none();
        if ui.add_enabled(ready, egui::Button::new(egui::RichText::new("Save as…   ⌘S").strong()).min_size(egui::vec2(200.0, 34.0))).clicked() {
            self.save_dialog();
        }
        if !ready {
            ui.label(egui::RichText::new("Waiting for the trace to finish.").small().weak());
        }
    }

    fn panel_batch(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.label(egui::RichText::new("Files").strong());
        ui.horizontal(|ui| {
            if ui.add_enabled(!self.batch_running, egui::Button::new("Add images…")).clicked() {
                if let Some(mut files) = rfd::FileDialog::new().add_filter("Images", IMAGE_EXTS).pick_files() {
                    files.retain(|f| !self.batch_files.contains(f));
                    self.batch_files.append(&mut files);
                }
            }
            if ui.add_enabled(!self.batch_running && !self.batch_files.is_empty(), egui::Button::new("Clear")).clicked() {
                self.batch_files.clear();
            }
        });
        egui::ScrollArea::vertical().max_height(140.0).id_salt("batchfiles").show(ui, |ui| {
            if self.batch_files.is_empty() {
                ui.label(egui::RichText::new("No files added.").weak());
            }
            for f in &self.batch_files {
                ui.label(f.file_name().unwrap_or_default().to_string_lossy().to_string());
            }
        });
        ui.separator();
        ui.label(egui::RichText::new("Output folder").strong());
        ui.horizontal(|ui| {
            if ui.add_enabled(!self.batch_running, egui::Button::new("Choose…")).clicked() {
                if let Some(d) = rfd::FileDialog::new().pick_folder() {
                    self.batch_out_dir = Some(d);
                }
            }
            ui.add(egui::Label::new(self.batch_out_dir.as_ref().map(|d| d.display().to_string()).unwrap_or_else(|| "not set".into())).truncate());
        });
        ui.label(egui::RichText::new("Format").strong());
        ui.horizontal(|ui| {
            for f in Format::ALL {
                ui.selectable_value(&mut self.batch_format, f, f.label());
            }
        });
        let mode = match self.mode_choice { ModeChoice::Automatic => "Fully automatic", ModeChoice::Basic => "Basic", ModeChoice::Advanced => "Advanced" };
        ui.add(egui::Label::new(egui::RichText::new(format!("Uses the current mode ({mode}) and its settings. Change them in Choose Mode.")).small().weak()).wrap());
        ui.add_space(6.0);
        let can_run = !self.batch_running && !self.batch_files.is_empty() && self.batch_out_dir.is_some();
        if ui.add_enabled(can_run, egui::Button::new(egui::RichText::new("Run batch").strong()).min_size(egui::vec2(200.0, 34.0))).clicked() {
            self.start_batch(ctx);
        }
        if self.batch_progress.1 > 0 {
            let (d, t) = self.batch_progress;
            ui.add(egui::ProgressBar::new(d as f32 / t as f32).text(format!("{d} / {t}")));
        }
        egui::ScrollArea::vertical().max_height(160.0).id_salt("batchlog").stick_to_bottom(true).show(ui, |ui| {
            for l in &self.batch_log {
                ui.add(egui::Label::new(egui::RichText::new(l).small()).wrap());
            }
        });
    }

    fn nav_bar(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.horizontal(|ui| {
            let w = (ui.available_width() - 16.0) / 3.0;
            let size = egui::vec2(w, 34.0);
            let reset_ok = self.step != Step::Load;
            if ui.add_enabled(reset_ok, egui::Button::new("↺  Reset").min_size(size)).clicked() {
                self.reset_step();
            }
            let back = self.step.back();
            if ui.add_enabled(back.is_some(), egui::Button::new("◀  Back").min_size(size)).clicked() {
                self.step = back.unwrap();
            }
            let next = self.step.next().filter(|_| self.image.is_some());
            if ui.add_enabled(next.is_some(), egui::Button::new("Next  ▶").min_size(size)).clicked() {
                self.step = next.unwrap();
            }
        });
    }

    // ----- canvas --------------------------------------------------------------

    fn canvas(&mut self, ui: &mut egui::Ui) -> Option<(u32, u32)> {
        let editing_bg = self.step == Step::Background;
        let avail = ui.available_rect_before_wrap();
        ui.painter().rect_filled(avail, 0.0, egui::Color32::WHITE);
        let Some(bitmap) = self.bitmap_tex.clone() else {
            // Empty state: dashed drop zone.
            let zone = avail.shrink2(egui::vec2(avail.width() * 0.12, avail.height() * 0.18));
            let stroke = egui::Stroke::new(3.0, egui::Color32::from_gray(190));
            let c = [zone.left_top(), zone.right_top(), zone.right_bottom(), zone.left_bottom(), zone.left_top()];
            for pair in c.windows(2) {
                ui.painter().add(egui::Shape::dashed_line(pair, stroke, 14.0, 10.0));
            }
            ui.painter().text(zone.center() - egui::vec2(0.0, 28.0), egui::Align2::CENTER_CENTER, "Drag and drop",
                egui::FontId::proportional(30.0), egui::Color32::from_gray(170));
            ui.painter().text(zone.center() + egui::vec2(0.0, 14.0), egui::Align2::CENTER_CENTER, "your image here",
                egui::FontId::proportional(30.0), egui::Color32::from_gray(170));
            ui.painter().text(zone.center() + egui::vec2(0.0, 60.0), egui::Align2::CENTER_CENTER, "or press ⌘O to open, ⌘V to paste",
                egui::FontId::proportional(15.0), egui::Color32::from_gray(185));
            return None;
        };
        draw_checker(ui.painter(), avail);
        let img_size = bitmap.size_vec2();
        let aspect = img_size.x / img_size.y;
        let pad = 12.0;
        let panes: Vec<(&str, Option<egui::TextureHandle>)> = match self.view {
            View::Compare => vec![("Bitmap", Some(bitmap)), ("Vector", self.vector_tex.clone())],
            View::Bitmap => vec![("Bitmap", Some(bitmap))],
            View::Vector => vec![("Vector", self.vector_tex.clone())],
            View::Segmentation => vec![("Segmentation", self.seg_tex.clone())],
        };
        let n = panes.len() as f32;
        let pane_w = (avail.width() - pad * (n + 1.0)) / n;
        let pane_h = avail.height() - pad * 2.0 - 22.0;
        let fit = (pane_w / aspect).min(pane_h);
        let draw_h = fit * self.zoom;
        let draw_w = draw_h * aspect;
        let mut picked = None;
        for (i, (label, tex)) in panes.into_iter().enumerate() {
            let x0 = avail.left() + pad + i as f32 * (pane_w + pad);
            let pane = egui::Rect::from_min_size(egui::pos2(x0, avail.top() + pad), egui::vec2(pane_w, pane_h + 22.0));
            let picking_here = self.picking && label == "Bitmap";
            let removing_here = editing_bg && label == "Vector";
            let title = if picking_here { "Bitmap  —  click a color to add it to the palette" }
                else if removing_here { "Vector  —  click a shape to remove or restore it" } else { label };
            ui.painter().text(pane.left_top() + egui::vec2(4.0, 0.0), egui::Align2::LEFT_TOP, title,
                egui::FontId::proportional(14.0), if picking_here || removing_here { ACCENT } else { egui::Color32::from_gray(110) });
            let inner = egui::Rect::from_min_size(pane.left_top() + egui::vec2(0.0, 22.0), egui::vec2(pane_w, pane_h));
            let mut child = ui.new_child(egui::UiBuilder::new().max_rect(inner));
            child.set_clip_rect(inner);
            let running = self.running_job.is_some();
            let traced_size = self.output.as_ref().map(|o| (o.traced_width as f32, o.traced_height as f32));
            let removed_outlines: Vec<Vec<(f64, f64)>> = if removing_here {
                self.removed.iter().filter_map(|i| self.shapes.get(*i)).flat_map(|s| s.polygons()).collect()
            } else {
                Vec::new()
            };
            let mut toggle: Option<usize> = None;
            egui::ScrollArea::both().id_salt(label).auto_shrink([false, false]).show(&mut child, |ui| {
                let sense = if picking_here || removing_here { egui::Sense::click() } else { egui::Sense::hover() };
                let (rect, resp) = ui.allocate_exact_size(egui::vec2(draw_w.max(inner.width()), draw_h.max(inner.height())), sense);
                let img_rect = egui::Rect::from_center_size(rect.center(), egui::vec2(draw_w, draw_h));
                match tex {
                    Some(t) => {
                        ui.painter().image(t.id(), img_rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
                    }
                    None => {
                        let msg = if running { "Tracing…" } else { "No output yet" };
                        ui.painter().text(img_rect.center(), egui::Align2::CENTER_CENTER, msg, egui::FontId::proportional(18.0), egui::Color32::from_gray(120));
                    }
                }
                if removing_here {
                    if let Some((tw, th)) = traced_size {
                        let sx = img_rect.width() / tw;
                        let sy = img_rect.height() / th;
                        let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(220, 30, 30));
                        for poly in &removed_outlines {
                            let pts: Vec<egui::Pos2> = poly.iter().map(|(x, y)| egui::pos2(img_rect.left() + *x as f32 * sx, img_rect.top() + *y as f32 * sy)).collect();
                            if pts.len() > 1 {
                                let mut closed = pts.clone();
                                closed.push(pts[0]);
                                ui.painter().add(egui::Shape::dashed_line(&closed, stroke, 4.0, 3.0));
                            }
                        }
                        if resp.hover_pos().is_some_and(|p| img_rect.contains(p)) {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                        }
                        if resp.clicked() {
                            if let Some(pos) = resp.interact_pointer_pos() {
                                if img_rect.contains(pos) {
                                    let x = ((pos.x - img_rect.left()) / sx) as f64;
                                    let y = ((pos.y - img_rect.top()) / sy) as f64;
                                    toggle = export::hit_test(&self.shapes, x, y);
                                }
                            }
                        }
                    }
                }
                if picking_here {
                    if resp.hover_pos().is_some_and(|p| img_rect.contains(p)) {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
                    }
                    if resp.clicked() {
                        if let Some(pos) = resp.interact_pointer_pos() {
                            if img_rect.contains(pos) {
                                let u = (pos.x - img_rect.left()) / img_rect.width();
                                let v = (pos.y - img_rect.top()) / img_rect.height();
                                picked = Some((
                                    ((u * img_size.x) as u32).min(img_size.x as u32 - 1),
                                    ((v * img_size.y) as u32).min(img_size.y as u32 - 1),
                                ));
                            }
                        }
                    }
                }
            });
            if let Some(i) = toggle {
                if !self.removed.remove(&i) {
                    self.removed.insert(i);
                }
                let ctx = ui.ctx().clone();
                self.rerender(&ctx);
            }
        }
        picked
    }
}

fn analysis_panel(ui: &mut egui::Ui, a: &Analysis) {
    ui.label(egui::RichText::new("Analysis").strong());
    ui.add(egui::Label::new(a.summary()).wrap());
    egui::Grid::new("analysis").num_columns(2).spacing([14.0, 2.0]).show(ui, |ui| {
        ui.label("Type"); ui.label(a.image_type.label()); ui.end_row();
        ui.label("Distinct colors"); ui.label(a.unique_colors.to_string()); ui.end_row();
        ui.label("Useful colors"); ui.label(a.recommended_colors.to_string()); ui.end_row();
        ui.label("Soft edges"); ui.label(format!("{:.0}%", a.soft_edge_ratio * 100.0)); ui.end_row();
        ui.label("Noise"); ui.label(format!("{:.0}%", a.noise_level * 100.0)); ui.end_row();
        ui.label("Colored pixels"); ui.label(format!("{:.1}%", a.colorfulness * 100.0)); ui.end_row();
        ui.label("Line art"); ui.label(if a.is_line_art { "yes" } else { "no" }); ui.end_row();
        ui.label("Background");
        match a.background {
            Some(c) => {
                ui.horizontal(|ui| {
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::hover());
                    ui.painter().rect_filled(rect, 3.0, egui::Color32::from_rgb(c[0], c[1], c[2]));
                    ui.painter().rect_stroke(rect, 3.0, egui::Stroke::new(1.0, egui::Color32::from_gray(150)), egui::StrokeKind::Inside);
                    ui.label(format!("{:.0}% uniform", a.background_uniformity * 100.0));
                });
            }
            None => { ui.label("none detected"); }
        }
        ui.end_row();
    });
    ui.label(egui::RichText::new("Dominant colors").strong());
    ui.horizontal_wrapped(|ui| {
        for (c, share) in &a.dominant {
            let (rect, r) = ui.allocate_exact_size(egui::vec2(20.0, 20.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 3.0, egui::Color32::from_rgb(c[0], c[1], c[2]));
            ui.painter().rect_stroke(rect, 3.0, egui::Stroke::new(1.0, egui::Color32::from_gray(150)), egui::StrokeKind::Inside);
            r.on_hover_text(format!("#{:02x}{:02x}{:02x}  {:.1}%", c[0], c[1], c[2], share * 100.0));
        }
    });
}

fn draw_checker(p: &egui::Painter, rect: egui::Rect) {
    let a = egui::Color32::from_gray(236);
    let b = egui::Color32::from_gray(246);
    p.rect_filled(rect, 0.0, a);
    let s = 14.0;
    let cols = (rect.width() / s).ceil() as i32;
    let rows = (rect.height() / s).ceil() as i32;
    for r in 0..rows {
        for c in 0..cols {
            if (r + c) % 2 == 0 {
                let min = egui::pos2(rect.left() + c as f32 * s, rect.top() + r as f32 * s);
                p.rect_filled(egui::Rect::from_min_size(min, egui::vec2(s, s)).intersect(rect), 0.0, b);
            }
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, root: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = root.ctx().clone();
        let ctx = &ctx;
        self.poll(ctx);

        let dropped: Option<PathBuf> = ctx.input(|i| i.raw.dropped_files.first().map(|f| f.path().to_path_buf()));
        if let Some(p) = dropped {
            self.load_path(ctx, p);
        }
        let hovering = ctx.input(|i| !i.raw.hovered_files.is_empty());

        let (open, save, paste, zin, zout, zreset, esc) = ctx.input(|i| {
            let m = i.modifiers.command;
            (
                m && i.key_pressed(egui::Key::O),
                m && i.key_pressed(egui::Key::S),
                m && i.key_pressed(egui::Key::V),
                m && (i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals)),
                m && i.key_pressed(egui::Key::Minus),
                m && i.key_pressed(egui::Key::Num0),
                i.key_pressed(egui::Key::Escape),
            )
        });
        if open { self.open_dialog(ctx); }
        if save && self.output.is_some() { self.save_dialog(); }
        if paste { self.paste_clipboard(ctx); }
        if zin { self.zoom = (self.zoom * 1.25).min(8.0); }
        if zout { self.zoom = (self.zoom / 1.25).max(0.25); }
        if zreset { self.zoom = 1.0; }
        if esc { self.picking = false; }

        if self.image.is_some() {
            let eff = self.effective_settings();
            if self.last_sent.as_ref() != Some(&eff) {
                if self.dirty_since.is_none() {
                    self.dirty_since = Some(Instant::now());
                }
                if let Some(t) = self.dirty_since {
                    if t.elapsed() >= DEBOUNCE {
                        self.dirty_since = None;
                        self.start_job(ctx);
                    } else {
                        ctx.request_repaint_after(DEBOUNCE - t.elapsed());
                    }
                }
            }
        }

        egui::Panel::bottom("status").show(root, |ui| {
            ui.horizontal(|ui| {
                if self.running_job.is_some() || self.batch_running { ui.spinner(); }
                if let Some(e) = &self.error {
                    ui.colored_label(egui::Color32::from_rgb(190, 40, 40), e);
                } else if let (Some(stage), true) = (self.stage, self.running_job.is_some()) {
                    ui.label(format!("{stage}…"));
                } else {
                    ui.label(&self.status);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new(format!("v{VERSION}")).weak());
                    ui.separator();
                    if ui.button("+").clicked() { self.zoom = (self.zoom * 1.25).min(8.0); }
                    if ui.button(format!("{:.0}%", self.zoom * 100.0)).on_hover_text("Reset zoom (⌘0)").clicked() { self.zoom = 1.0; }
                    if ui.button("−").clicked() { self.zoom = (self.zoom / 1.25).max(0.25); }
                    ui.separator();
                    ui.selectable_value(&mut self.view, View::Segmentation, "Segmentation");
                    ui.selectable_value(&mut self.view, View::Vector, "Vector");
                    ui.selectable_value(&mut self.view, View::Bitmap, "Bitmap");
                    ui.selectable_value(&mut self.view, View::Compare, "Side by side");
                });
            });
        });

        egui::Panel::right("controls").resizable(false).exact_size(PANEL_W).show(root, |ui| {
            ui.add_space(6.0);
            self.header(ui);
            ui.add_space(4.0);
            // Content scrolls; nav bar stays pinned at the bottom.
            let nav_h = 52.0;
            let content_h = (ui.available_height() - nav_h).max(100.0);
            egui::ScrollArea::vertical().id_salt("content").max_height(content_h).auto_shrink([false, false]).show(ui, |ui| {
                match self.step {
                    Step::Load => self.panel_load(ui, ctx),
                    Step::Mode => self.panel_mode(ui),
                    Step::Review => self.panel_review(ui),
                    Step::Background => self.panel_background(ui, ctx),
                    Step::Save => self.panel_save(ui),
                    Step::Batch => self.panel_batch(ui, ctx),
                }
                ui.add_space(8.0);
            });
            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.add_space(4.0);
                self.nav_bar(ui);
            });
        });

        egui::CentralPanel::no_frame().show(root, |ui| {
            if let Some((x, y)) = self.canvas(ui) {
                if let Some(img) = &self.image {
                    let p = img.get_pixel(x, y).0;
                    if p[3] >= 128 {
                        let c = [p[0], p[1], p[2]];
                        if !self.custom_palette.contains(&c) {
                            self.custom_palette.push(c);
                        }
                        self.palette_custom = true;
                    }
                }
            }
            if hovering {
                let r = ui.max_rect();
                ui.painter().rect_filled(r, 0.0, ACCENT.gamma_multiply(0.12));
                ui.painter().rect_stroke(r.shrink(6.0), 8.0, egui::Stroke::new(2.0, ACCENT), egui::StrokeKind::Inside);
                ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, "Drop to open", egui::FontId::proportional(28.0), ACCENT);
            }
        });
    }
}
