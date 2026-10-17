//! Vector Baba core: turns raster images into clean, editable SVG.
//!
//! Pipeline: load -> limit size -> smooth -> quantize (or binarize) ->
//! trace regions into Bezier paths -> serialize SVG.

pub mod export;
pub mod pipeline;
pub mod preprocess;
pub mod quantize;
pub mod svg;
pub mod trace;

pub use pipeline::{
    analyze, convert, convert_with_progress, Analysis, Complexity, ConvertOutput, CurveMode, ImageType, Layering,
    Level, Mode, OutPath, Preset, Quality, Settings, Transparency,
};
