//! Small CPU renderer for the disposable 4 km H/C/G discriminator.
//!
//! Physical elevation selects the base colour on one shared linear scale.
//! Hillshade is a presentation-only modulation derived from the unmodified
//! physical surface; its modest vertical gain is reported in the metadata.

use std::fmt;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use serde::Serialize;

use super::{
    reconstruct_mean_surface_gradient, LandscapeMesh, LinkedResolutionInputV0, SurfaceGradientError,
};

pub const THIN_HCG_RENDER_SCHEMA_V0: &str = "orogen-owner-thin-hcg-render-v0";
pub const THIN_HCG_DIAGNOSTIC_SCHEMA_V0: &str = "orogen-owner-thin-hcg-diagnostic-sheet-v0";
pub const THIN_HCG_PANEL_ORDER_V0: [&str; 3] = ["H", "C", "G"];
const ACCEPTED_SPACING_KM: f64 = 4.0;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ThinHcgRenderConfigV0 {
    /// Width of each panel. Height follows the shared physical aspect ratio.
    pub panel_width_px: u32,
    pub gap_px: u32,
    /// Optional explicit shared physical-elevation range. When absent, the
    /// finite extrema across all three surfaces are used.
    pub physical_elevation_range_km: Option<[f64; 2]>,
    /// Presentation-only gain applied to physical dz/dx and dz/dy.
    pub hillshade_vertical_gain: f64,
    /// Strength of the hillshade modulation of the physical colour ramp.
    pub hillshade_strength: f64,
}

impl Default for ThinHcgRenderConfigV0 {
    fn default() -> Self {
        Self {
            panel_width_px: 960,
            gap_px: 8,
            physical_elevation_range_km: None,
            hillshade_vertical_gain: 3.0,
            hillshade_strength: 0.55,
        }
    }
}

/// Compact, renderer-only settings for the morphology-review sheet.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ThinHcgDiagnosticConfigV0 {
    pub panel_width_px: u32,
    pub gap_px: u32,
    /// Padding outside the union of positive accepted forcing stencils. This
    /// changes only framing, never physical state.
    pub crop_padding_km: f64,
    /// Height of the longitudinal and transverse profile plots together.
    pub profiles_height_px: u32,
    pub robust_lower_quantile: f64,
    pub robust_upper_quantile: f64,
}

impl Default for ThinHcgDiagnosticConfigV0 {
    fn default() -> Self {
        Self {
            panel_width_px: 720,
            gap_px: 8,
            crop_padding_km: 48.0,
            profiles_height_px: 240,
            robust_lower_quantile: 0.02,
            robust_upper_quantile: 0.98,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ThinProfileAxisV0 {
    pub start_km: [f64; 2],
    pub end_km: [f64; 2],
    pub sample_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ThinNamedProfileV0 {
    pub source: String,
    pub orientation: String,
    pub axis: ThinProfileAxisV0,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ThinHcgDiagnosticMetadataV0 {
    pub schema_version: String,
    pub panel_order: [String; 3],
    pub image_width_px: u32,
    pub image_height_px: u32,
    pub map_height_px: u32,
    pub profiles_height_px: u32,
    pub planar_crop_bounds_km: [f64; 4],
    pub crop_source: String,
    pub physical_elevation_range_km: [f64; 2],
    pub diagnostic_robust_ranges_km: [[f64; 2]; 3],
    pub row_semantics: [String; 3],
    /// Two unsmoothed axes (longitudinal, transverse) per accepted forcing
    /// stencil, in stencil order. Sample locations are shared by H/C/G.
    pub profiles: Vec<ThinNamedProfileV0>,
    pub physical_state_modified: bool,
    pub panel_labels_embedded_as_png_text: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ThinHcgRenderMetadataV0 {
    pub schema_version: String,
    pub panel_order: [String; 3],
    pub image_width_px: u32,
    pub image_height_px: u32,
    pub panel_width_px: u32,
    pub panel_height_px: u32,
    pub gap_px: u32,
    pub planar_bounds_km: [f64; 4],
    pub physical_elevation_range_km: [f64; 2],
    pub elevation_colour_scale: String,
    pub hillshade_vertical_gain: f64,
    pub hillshade_strength: f64,
    pub panel_labels_embedded_as_png_text: bool,
}

#[derive(Debug)]
pub enum ThinHcgRenderErrorV0 {
    InvalidInput(String),
    Gradient(SurfaceGradientError),
    Io(std::io::Error),
    Png(png::EncodingError),
}

impl fmt::Display for ThinHcgRenderErrorV0 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => formatter.write_str(message),
            Self::Gradient(error) => write!(
                formatter,
                "physical-gradient reconstruction failed: {error}"
            ),
            Self::Io(error) => write!(formatter, "image I/O failed: {error}"),
            Self::Png(error) => write!(formatter, "PNG encoding failed: {error}"),
        }
    }
}

impl std::error::Error for ThinHcgRenderErrorV0 {}

impl From<SurfaceGradientError> for ThinHcgRenderErrorV0 {
    fn from(error: SurfaceGradientError) -> Self {
        Self::Gradient(error)
    }
}

impl From<std::io::Error> for ThinHcgRenderErrorV0 {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<png::EncodingError> for ThinHcgRenderErrorV0 {
    fn from(error: png::EncodingError) -> Self {
        Self::Png(error)
    }
}

/// Render H, C, and G in that order into one matched RGBA8 image.
///
/// The accepted linked input supplies geometry only. The three slices are
/// authoritative physical elevation in kilometres and are never exaggerated.
pub fn render_thin_hcg_matched_rgba_v0(
    input: &LinkedResolutionInputV0,
    h_elevation_km: &[f64],
    c_elevation_km: &[f64],
    g_elevation_km: &[f64],
    config: ThinHcgRenderConfigV0,
) -> Result<(Vec<u8>, ThinHcgRenderMetadataV0), ThinHcgRenderErrorV0> {
    if input.nominal_spacing_km.to_bits() != ACCEPTED_SPACING_KM.to_bits() {
        return Err(invalid("renderer requires the accepted exact 4 km input"));
    }
    render_mesh(
        &input.mesh,
        h_elevation_km,
        c_elevation_km,
        g_elevation_km,
        config,
    )
}

/// Render and write the matched image as an RGBA8 PNG.
pub fn write_thin_hcg_matched_png_v0(
    path: impl AsRef<Path>,
    input: &LinkedResolutionInputV0,
    h_elevation_km: &[f64],
    c_elevation_km: &[f64],
    g_elevation_km: &[f64],
    config: ThinHcgRenderConfigV0,
) -> Result<ThinHcgRenderMetadataV0, ThinHcgRenderErrorV0> {
    let (rgba, mut metadata) = render_thin_hcg_matched_rgba_v0(
        input,
        h_elevation_km,
        c_elevation_km,
        g_elevation_km,
        config,
    )?;
    let file = File::create(path)?;
    let mut encoder = png::Encoder::new(
        BufWriter::new(file),
        metadata.image_width_px,
        metadata.image_height_px,
    );
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.add_text_chunk(
        "Title".into(),
        "Thin H/C/G physical surface comparison".into(),
    )?;
    encoder.add_text_chunk("Panel order".into(), THIN_HCG_PANEL_ORDER_V0.join(","))?;
    encoder.add_text_chunk(
        "Physical elevation range km".into(),
        format!(
            "{:.17},{:.17}",
            metadata.physical_elevation_range_km[0], metadata.physical_elevation_range_km[1]
        ),
    )?;
    encoder.add_text_chunk(
        "Hillshade".into(),
        format!(
            "presentation-only; vertical gain {:.6}; strength {:.6}",
            metadata.hillshade_vertical_gain, metadata.hillshade_strength
        ),
    )?;
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&rgba)?;
    metadata.panel_labels_embedded_as_png_text = true;
    Ok(metadata)
}

/// Render a sharp, shared-crop morphology sheet.
///
/// Row one uses one physical kilometre scale for all arms. Row two uses a
/// separately reported robust colour range per arm solely to expose internal
/// form. Row three contains matched, unsmoothed longitudinal and transverse
/// physical-elevation bands for every forcing stencil on one vertical scale.
pub fn render_thin_hcg_diagnostic_rgba_v0(
    input: &LinkedResolutionInputV0,
    h_elevation_km: &[f64],
    c_elevation_km: &[f64],
    g_elevation_km: &[f64],
    config: ThinHcgDiagnosticConfigV0,
) -> Result<(Vec<u8>, ThinHcgDiagnosticMetadataV0), ThinHcgRenderErrorV0> {
    if input.nominal_spacing_km.to_bits() != ACCEPTED_SPACING_KM.to_bits() {
        return Err(invalid(
            "diagnostic renderer requires the accepted exact 4 km input",
        ));
    }
    if input
        .compiled_stencils
        .iter()
        .any(|stencil| stencil.weight_per_km2.len() != input.mesh.cell_count())
    {
        return Err(invalid("diagnostic forcing stencil length mismatch"));
    }
    let forcing_support = (0..input.mesh.cell_count())
        .map(|cell| {
            input
                .compiled_stencils
                .iter()
                .any(|stencil| stencil.weight_per_km2[cell] > 0.0)
        })
        .collect::<Vec<_>>();
    let profile_forcing = input
        .compiled_stencils
        .iter()
        .map(|stencil| stencil.weight_per_km2.as_slice())
        .collect::<Vec<_>>();
    render_diagnostic_mesh(
        &input.mesh,
        &forcing_support,
        &input.cumulative_rock_displacement_km,
        &profile_forcing,
        h_elevation_km,
        c_elevation_km,
        g_elevation_km,
        config,
    )
}

pub fn write_thin_hcg_diagnostic_png_v0(
    path: impl AsRef<Path>,
    input: &LinkedResolutionInputV0,
    h_elevation_km: &[f64],
    c_elevation_km: &[f64],
    g_elevation_km: &[f64],
    config: ThinHcgDiagnosticConfigV0,
) -> Result<ThinHcgDiagnosticMetadataV0, ThinHcgRenderErrorV0> {
    let (rgba, mut metadata) = render_thin_hcg_diagnostic_rgba_v0(
        input,
        h_elevation_km,
        c_elevation_km,
        g_elevation_km,
        config,
    )?;
    let file = File::create(path)?;
    let mut encoder = png::Encoder::new(
        BufWriter::new(file),
        metadata.image_width_px,
        metadata.image_height_px,
    );
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.add_text_chunk("Title".into(), "Thin H/C/G morphology diagnostic".into())?;
    encoder.add_text_chunk("Panel order".into(), THIN_HCG_PANEL_ORDER_V0.join(","))?;
    encoder.add_text_chunk(
        "Rows".into(),
        "shared physical elevation; per-arm robust diagnostic contrast; matched physical profiles"
            .into(),
    )?;
    encoder.add_text_chunk(
        "Physical elevation range km".into(),
        format!(
            "{:.17},{:.17}",
            metadata.physical_elevation_range_km[0], metadata.physical_elevation_range_km[1]
        ),
    )?;
    encoder.add_text_chunk(
        "Diagnostic warning".into(),
        "row two normalization is renderer-only and differs by arm".into(),
    )?;
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&rgba)?;
    metadata.panel_labels_embedded_as_png_text = true;
    Ok(metadata)
}

#[allow(clippy::too_many_arguments)]
fn render_diagnostic_mesh(
    mesh: &LandscapeMesh,
    candidate: &[bool],
    forcing: &[f64],
    profile_forcing: &[&[f64]],
    h_elevation_km: &[f64],
    c_elevation_km: &[f64],
    g_elevation_km: &[f64],
    config: ThinHcgDiagnosticConfigV0,
) -> Result<(Vec<u8>, ThinHcgDiagnosticMetadataV0), ThinHcgRenderErrorV0> {
    mesh.validate()
        .map_err(|error| invalid(format!("invalid landscape mesh: {error}")))?;
    let n = mesh.cell_count();
    if candidate.len() != n
        || forcing.len() != n
        || profile_forcing.is_empty()
        || profile_forcing.iter().any(|weights| weights.len() != n)
    {
        return Err(invalid("diagnostic crop field length mismatch"));
    }
    let surfaces = [h_elevation_km, c_elevation_km, g_elevation_km];
    for (label, surface) in THIN_HCG_PANEL_ORDER_V0.into_iter().zip(surfaces) {
        if surface.len() != n || surface.iter().any(|value| !value.is_finite()) {
            return Err(invalid(format!("{label} diagnostic surface is invalid")));
        }
    }
    validate_diagnostic_config(config)?;
    let spacing = regular_spacing(mesh)?;
    let bounds = diagnostic_crop_bounds(mesh, candidate, forcing, config.crop_padding_km, spacing)?;
    let span_x = bounds[1] - bounds[0];
    let span_y = bounds[3] - bounds[2];
    let map_height = ((f64::from(config.panel_width_px) * span_y / span_x).round() as u32).max(1);
    let image_width = config
        .panel_width_px
        .checked_mul(3)
        .and_then(|v| v.checked_add(config.gap_px.checked_mul(2)?))
        .ok_or_else(|| invalid("diagnostic dimensions overflow"))?;
    let image_height = map_height
        .checked_mul(2)
        .and_then(|v| v.checked_add(config.gap_px.checked_mul(2)?))
        .and_then(|v| v.checked_add(config.profiles_height_px))
        .ok_or_else(|| invalid("diagnostic dimensions overflow"))?;
    let pixels = image_width as usize * image_height as usize;
    let mut rgba = vec![0_u8; pixels * 4];
    for pixel in rgba.chunks_exact_mut(4) {
        pixel.copy_from_slice(&[15, 17, 21, 255]);
    }
    let owner = raster_owners(mesh, bounds, spacing, config.panel_width_px, map_height)?;
    let shared = physical_range(surfaces, None)?;
    let robust = [
        quantile_range(h_elevation_km, candidate, config)?,
        quantile_range(c_elevation_km, candidate, config)?,
        quantile_range(g_elevation_km, candidate, config)?,
    ];
    let mut profiles = Vec::with_capacity(profile_forcing.len() * 2);
    for (stencil, weights) in profile_forcing.iter().enumerate() {
        let (center, axis) = profile_frame(mesh, candidate, weights)?;
        profiles.push(ThinNamedProfileV0 {
            source: format!("compiled-stencil-{stencil}"),
            orientation: "longitudinal".into(),
            axis: clipped_axis(center, axis, bounds, config.panel_width_px),
        });
        profiles.push(ThinNamedProfileV0 {
            source: format!("compiled-stencil-{stencil}"),
            orientation: "transverse".into(),
            axis: clipped_axis(center, [-axis[1], axis[0]], bounds, config.panel_width_px),
        });
    }
    if config.profiles_height_px < profiles.len() as u32 * 8 {
        return Err(invalid(
            "profile region is too short for the accepted forcing stencils",
        ));
    }

    for panel in 0..3_usize {
        let x0 = panel as u32 * (config.panel_width_px + config.gap_px);
        paint_sharp_map(
            &mut rgba,
            image_width,
            x0,
            0,
            config.panel_width_px,
            map_height,
            &owner,
            surfaces[panel],
            shared,
        );
        let diagnostic_y = map_height + config.gap_px;
        paint_sharp_map(
            &mut rgba,
            image_width,
            x0,
            diagnostic_y,
            config.panel_width_px,
            map_height,
            &owner,
            surfaces[panel],
            robust[panel],
        );
        for profile in &profiles {
            let colour = if profile.orientation == "longitudinal" {
                [255, 196, 62, 255]
            } else {
                [68, 210, 255, 255]
            };
            draw_axis_on_map(
                &mut rgba,
                image_width,
                x0,
                diagnostic_y,
                config.panel_width_px,
                map_height,
                bounds,
                &profile.axis,
                colour,
            );
        }
        let profile_y = 2 * map_height + 2 * config.gap_px;
        let band = config.profiles_height_px / profiles.len() as u32;
        for (index, profile) in profiles.iter().enumerate() {
            let y = profile_y + index as u32 * band;
            let height = if index + 1 == profiles.len() {
                config.profiles_height_px - index as u32 * band
            } else {
                band
            };
            let colour = if profile.orientation == "longitudinal" {
                [255, 196, 62, 255]
            } else {
                [68, 210, 255, 255]
            };
            paint_profile(
                &mut rgba,
                image_width,
                x0,
                y,
                config.panel_width_px,
                height,
                mesh,
                surfaces[panel],
                &profile.axis,
                shared,
                colour,
            );
        }
    }
    Ok((
        rgba,
        ThinHcgDiagnosticMetadataV0 {
            schema_version: THIN_HCG_DIAGNOSTIC_SCHEMA_V0.into(),
            panel_order: ["H".into(), "C".into(), "G".into()],
            image_width_px: image_width,
            image_height_px: image_height,
            map_height_px: map_height,
            profiles_height_px: config.profiles_height_px,
            planar_crop_bounds_km: bounds,
            crop_source:
                "positive-compiled-stencil-union-plus-padding; forcing-weighted-profile-axis".into(),
            physical_elevation_range_km: shared,
            diagnostic_robust_ranges_km: robust,
            row_semantics: [
                "sharp nearest-cell; shared linear physical elevation km".into(),
                "sharp nearest-cell; per-arm robust renderer-only contrast".into(),
                "longitudinal then transverse physical elevation; shared vertical km scale".into(),
            ],
            profiles,
            physical_state_modified: false,
            panel_labels_embedded_as_png_text: false,
        },
    ))
}

fn validate_diagnostic_config(
    config: ThinHcgDiagnosticConfigV0,
) -> Result<(), ThinHcgRenderErrorV0> {
    if config.panel_width_px < 16 || config.profiles_height_px < 32 {
        return Err(invalid("diagnostic panel/profile dimensions are too small"));
    }
    if !config.crop_padding_km.is_finite() || config.crop_padding_km < 0.0 {
        return Err(invalid(
            "diagnostic crop padding must be finite and nonnegative",
        ));
    }
    if !config.robust_lower_quantile.is_finite()
        || !config.robust_upper_quantile.is_finite()
        || config.robust_lower_quantile < 0.0
        || config.robust_upper_quantile > 1.0
        || config.robust_lower_quantile >= config.robust_upper_quantile
    {
        return Err(invalid("diagnostic robust quantiles are invalid"));
    }
    Ok(())
}

fn diagnostic_crop_bounds(
    mesh: &LandscapeMesh,
    candidate: &[bool],
    forcing: &[f64],
    padding: f64,
    spacing: f64,
) -> Result<[f64; 4], ThinHcgRenderErrorV0> {
    let mut bounds = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    for (cell, center) in mesh.cell_center_km.iter().enumerate() {
        if candidate[cell] || forcing[cell] != 0.0 {
            bounds[0] = bounds[0].min(center.x);
            bounds[1] = bounds[1].max(center.x);
            bounds[2] = bounds[2].min(center.y);
            bounds[3] = bounds[3].max(center.y);
        }
    }
    if !bounds.iter().all(|v| v.is_finite()) {
        return Err(invalid("candidate/forcing crop is empty"));
    }
    let full = mesh_bounds(mesh, 0.5 * spacing, spacing / 3.0_f64.sqrt())?;
    bounds[0] = (bounds[0] - 0.5 * spacing - padding).max(full[0]);
    bounds[1] = (bounds[1] + 0.5 * spacing + padding).min(full[1]);
    bounds[2] = (bounds[2] - spacing / 3.0_f64.sqrt() - padding).max(full[2]);
    bounds[3] = (bounds[3] + spacing / 3.0_f64.sqrt() + padding).min(full[3]);
    Ok(bounds)
}

fn quantile_range(
    elevation: &[f64],
    candidate: &[bool],
    config: ThinHcgDiagnosticConfigV0,
) -> Result<[f64; 2], ThinHcgRenderErrorV0> {
    let mut values = elevation
        .iter()
        .zip(candidate)
        .filter_map(|(&z, &selected)| selected.then_some(z))
        .collect::<Vec<_>>();
    if values.is_empty() {
        values.extend_from_slice(elevation);
    }
    values.sort_by(f64::total_cmp);
    let last = values.len() - 1;
    let lower = values[((last as f64 * config.robust_lower_quantile).round() as usize).min(last)];
    let upper = values[((last as f64 * config.robust_upper_quantile).round() as usize).min(last)];
    if lower < upper {
        Ok([lower, upper])
    } else {
        let pad = lower.abs().max(1.0) * 0.5;
        Ok([lower - pad, upper + pad])
    }
}

fn profile_frame(
    mesh: &LandscapeMesh,
    candidate: &[bool],
    forcing: &[f64],
) -> Result<([f64; 2], [f64; 2]), ThinHcgRenderErrorV0> {
    let mut total = 0.0;
    let mut center = [0.0, 0.0];
    for (cell, point) in mesh.cell_center_km.iter().enumerate() {
        let weight = forcing[cell].abs();
        if weight > 0.0 {
            total += weight;
            center[0] += weight * point.x;
            center[1] += weight * point.y;
        }
    }
    if total == 0.0 {
        for (cell, point) in mesh.cell_center_km.iter().enumerate() {
            if candidate[cell] {
                total += 1.0;
                center[0] += point.x;
                center[1] += point.y;
            }
        }
    }
    if total == 0.0 {
        return Err(invalid("cannot derive profile frame from empty crop"));
    }
    center[0] /= total;
    center[1] /= total;
    let mut xx = 0.0;
    let mut xy = 0.0;
    let mut yy = 0.0;
    for (cell, point) in mesh.cell_center_km.iter().enumerate() {
        let weight = if forcing[cell] != 0.0 {
            forcing[cell].abs()
        } else if candidate[cell] {
            f64::EPSILON
        } else {
            0.0
        };
        let x = point.x - center[0];
        let y = point.y - center[1];
        xx += weight * x * x;
        xy += weight * x * y;
        yy += weight * y * y;
    }
    let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
    let mut axis = [angle.cos(), angle.sin()];
    if axis[0] < 0.0 {
        axis = [-axis[0], -axis[1]];
    }
    Ok((center, axis))
}

fn clipped_axis(
    center: [f64; 2],
    axis: [f64; 2],
    bounds: [f64; 4],
    samples: u32,
) -> ThinProfileAxisV0 {
    let mut low: f64 = -f64::INFINITY;
    let mut high: f64 = f64::INFINITY;
    for (coordinate, direction, minimum, maximum) in [
        (center[0], axis[0], bounds[0], bounds[1]),
        (center[1], axis[1], bounds[2], bounds[3]),
    ] {
        if direction.abs() > 1.0e-15 {
            let a = (minimum - coordinate) / direction;
            let b = (maximum - coordinate) / direction;
            low = low.max(a.min(b));
            high = high.min(a.max(b));
        }
    }
    ThinProfileAxisV0 {
        start_km: [center[0] + low * axis[0], center[1] + low * axis[1]],
        end_km: [center[0] + high * axis[0], center[1] + high * axis[1]],
        sample_count: samples,
    }
}

#[allow(clippy::too_many_arguments)]
fn paint_sharp_map(
    rgba: &mut [u8],
    image_width: u32,
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
    owner: &[Option<usize>],
    elevation: &[f64],
    range: [f64; 2],
) {
    for y in 0..height {
        for x in 0..width {
            let source = (y * width + x) as usize;
            let Some(cell) = owner[source] else { continue };
            let mut colour = elevation_colour(elevation[cell], range);
            let boundary = (x > 0 && owner[source - 1] != Some(cell))
                || (y > 0 && owner[source - width as usize] != Some(cell));
            if boundary {
                colour = colour.map(|channel| ((u16::from(channel) * 3) / 5) as u8);
            }
            put_pixel(
                rgba,
                image_width,
                x0 + x,
                y0 + y,
                [colour[0], colour[1], colour[2], 255],
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_axis_on_map(
    rgba: &mut [u8],
    image_width: u32,
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
    bounds: [f64; 4],
    axis: &ThinProfileAxisV0,
    colour: [u8; 4],
) {
    let pixel = |point: [f64; 2]| -> (i32, i32) {
        (
            ((point[0] - bounds[0]) / (bounds[1] - bounds[0]) * f64::from(width - 1)).round()
                as i32,
            ((bounds[3] - point[1]) / (bounds[3] - bounds[2]) * f64::from(height - 1)).round()
                as i32,
        )
    };
    let a = pixel(axis.start_km);
    let b = pixel(axis.end_km);
    draw_line(
        rgba,
        image_width,
        x0 as i32 + a.0,
        y0 as i32 + a.1,
        x0 as i32 + b.0,
        y0 as i32 + b.1,
        colour,
    );
}

#[allow(clippy::too_many_arguments)]
fn paint_profile(
    rgba: &mut [u8],
    image_width: u32,
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
    mesh: &LandscapeMesh,
    elevation: &[f64],
    axis: &ThinProfileAxisV0,
    range: [f64; 2],
    colour: [u8; 4],
) {
    if height < 2 {
        return;
    }
    for fraction in [0.25, 0.5, 0.75] {
        let y = y0 + (fraction * f64::from(height - 1)).round() as u32;
        draw_line(
            rgba,
            image_width,
            x0 as i32,
            y as i32,
            (x0 + width - 1) as i32,
            y as i32,
            [48, 52, 60, 255],
        );
    }
    let mut previous = None;
    for sample in 0..width {
        let t = if width > 1 {
            f64::from(sample) / f64::from(width - 1)
        } else {
            0.0
        };
        let point = [
            axis.start_km[0] + t * (axis.end_km[0] - axis.start_km[0]),
            axis.start_km[1] + t * (axis.end_km[1] - axis.start_km[1]),
        ];
        let cell = nearest_cell(mesh, point);
        let normalized = ((elevation[cell] - range[0]) / (range[1] - range[0])).clamp(0.0, 1.0);
        let current = (
            (x0 + sample) as i32,
            (y0 + height - 1 - (normalized * f64::from(height - 1)).round() as u32) as i32,
        );
        if let Some((px, py)) = previous {
            draw_line(rgba, image_width, px, py, current.0, current.1, colour);
        }
        previous = Some(current);
    }
}

fn nearest_cell(mesh: &LandscapeMesh, point: [f64; 2]) -> usize {
    mesh.cell_center_km
        .iter()
        .enumerate()
        .min_by(|(ia, a), (ib, b)| {
            let da = (a.x - point[0]).powi(2) + (a.y - point[1]).powi(2);
            let db = (b.x - point[0]).powi(2) + (b.y - point[1]).powi(2);
            da.total_cmp(&db).then_with(|| ia.cmp(ib))
        })
        .map(|(cell, _)| cell)
        .unwrap_or(0)
}

fn put_pixel(rgba: &mut [u8], width: u32, x: u32, y: u32, colour: [u8; 4]) {
    let offset = (y as usize * width as usize + x as usize) * 4;
    if offset + 4 <= rgba.len() {
        rgba[offset..offset + 4].copy_from_slice(&colour);
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_line(
    rgba: &mut [u8],
    width: u32,
    mut x0: i32,
    mut y0: i32,
    x1: i32,
    y1: i32,
    colour: [u8; 4],
) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        if x0 >= 0 && y0 >= 0 {
            put_pixel(rgba, width, x0 as u32, y0 as u32, colour);
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let twice = 2 * error;
        if twice >= dy {
            error += dy;
            x0 += sx;
        }
        if twice <= dx {
            error += dx;
            y0 += sy;
        }
    }
}

fn render_mesh(
    mesh: &LandscapeMesh,
    h_elevation_km: &[f64],
    c_elevation_km: &[f64],
    g_elevation_km: &[f64],
    config: ThinHcgRenderConfigV0,
) -> Result<(Vec<u8>, ThinHcgRenderMetadataV0), ThinHcgRenderErrorV0> {
    mesh.validate()
        .map_err(|error| invalid(format!("invalid landscape mesh: {error}")))?;
    validate_config(config)?;
    let surfaces = [h_elevation_km, c_elevation_km, g_elevation_km];
    let n = mesh.cell_count();
    for (label, surface) in THIN_HCG_PANEL_ORDER_V0.into_iter().zip(surfaces) {
        if surface.len() != n {
            return Err(invalid(format!(
                "{label} physical surface has {} cells, expected {n}",
                surface.len()
            )));
        }
        if let Some(cell) = surface.iter().position(|value| !value.is_finite()) {
            return Err(invalid(format!(
                "{label} physical surface is non-finite at cell {cell}"
            )));
        }
    }

    let spacing_km = regular_spacing(mesh)?;
    let half_width = 0.5 * spacing_km;
    let half_height = spacing_km / 3.0_f64.sqrt();
    let bounds = mesh_bounds(mesh, half_width, half_height)?;
    let span_x = bounds[1] - bounds[0];
    let span_y = bounds[3] - bounds[2];
    let panel_height_px =
        ((f64::from(config.panel_width_px) * span_y / span_x).round() as u32).max(1);
    let image_width_px = config
        .panel_width_px
        .checked_mul(3)
        .and_then(|value| value.checked_add(config.gap_px.checked_mul(2)?))
        .ok_or_else(|| invalid("render dimensions overflow u32"))?;
    let pixel_count = usize::try_from(image_width_px)
        .ok()
        .and_then(|width| {
            usize::try_from(panel_height_px)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| invalid("render dimensions overflow address space"))?;

    let elevation_range = physical_range(surfaces, config.physical_elevation_range_km)?;
    let gradients = [
        reconstruct_mean_surface_gradient(mesh, h_elevation_km)?,
        reconstruct_mean_surface_gradient(mesh, c_elevation_km)?,
        reconstruct_mean_surface_gradient(mesh, g_elevation_km)?,
    ];
    let owner = raster_owners(
        mesh,
        bounds,
        spacing_km,
        config.panel_width_px,
        panel_height_px,
    )?;
    let mut rgba = vec![0_u8; pixel_count * 4];
    for pixel in rgba.chunks_exact_mut(4) {
        pixel.copy_from_slice(&[18, 20, 24, 255]);
    }
    for panel in 0..3_usize {
        let x_offset = panel as u32 * (config.panel_width_px + config.gap_px);
        for y in 0..panel_height_px {
            for x in 0..config.panel_width_px {
                let source = (y * config.panel_width_px + x) as usize;
                let target = (y * image_width_px + x_offset + x) as usize * 4;
                let Some(cell) = owner[source] else {
                    rgba[target..target + 4].copy_from_slice(&[18, 20, 24, 255]);
                    continue;
                };
                let base = elevation_colour(surfaces[panel][cell], elevation_range);
                let gradient = gradients[panel].vector[cell];
                let shade = hillshade_factor(
                    gradient.x,
                    gradient.y,
                    config.hillshade_vertical_gain,
                    config.hillshade_strength,
                );
                rgba[target] = shade_channel(base[0], shade);
                rgba[target + 1] = shade_channel(base[1], shade);
                rgba[target + 2] = shade_channel(base[2], shade);
                rgba[target + 3] = 255;
            }
        }
    }

    let metadata = ThinHcgRenderMetadataV0 {
        schema_version: THIN_HCG_RENDER_SCHEMA_V0.into(),
        panel_order: ["H".into(), "C".into(), "G".into()],
        image_width_px,
        image_height_px: panel_height_px,
        panel_width_px: config.panel_width_px,
        panel_height_px,
        gap_px: config.gap_px,
        planar_bounds_km: bounds,
        physical_elevation_range_km: elevation_range,
        elevation_colour_scale: "shared-linear-physical-km/olive-tan-rock-snow-v0".into(),
        hillshade_vertical_gain: config.hillshade_vertical_gain,
        hillshade_strength: config.hillshade_strength,
        panel_labels_embedded_as_png_text: false,
    };
    Ok((rgba, metadata))
}

fn validate_config(config: ThinHcgRenderConfigV0) -> Result<(), ThinHcgRenderErrorV0> {
    if config.panel_width_px == 0 {
        return Err(invalid("panel width must be positive"));
    }
    if !config.hillshade_vertical_gain.is_finite() || config.hillshade_vertical_gain < 0.0 {
        return Err(invalid(
            "hillshade vertical gain must be finite and nonnegative",
        ));
    }
    if !config.hillshade_strength.is_finite() || !(0.0..=1.0).contains(&config.hillshade_strength) {
        return Err(invalid("hillshade strength must be finite and in [0, 1]"));
    }
    if let Some([minimum, maximum]) = config.physical_elevation_range_km {
        if !minimum.is_finite() || !maximum.is_finite() || minimum >= maximum {
            return Err(invalid(
                "explicit physical-elevation range must be finite and increasing",
            ));
        }
    }
    Ok(())
}

fn regular_spacing(mesh: &LandscapeMesh) -> Result<f64, ThinHcgRenderErrorV0> {
    let spacing = mesh
        .edge_distance_km
        .iter()
        .copied()
        .map(f64::from)
        .find(|value| *value > 0.0 && value.is_finite())
        .ok_or_else(|| invalid("mesh has no finite positive neighbor spacing"))?;
    let tolerance = 64.0 * f64::EPSILON.max(f64::from(f32::EPSILON)) * spacing;
    if mesh
        .edge_distance_km
        .iter()
        .map(|value| f64::from(*value))
        .any(|value| (value - spacing).abs() > tolerance)
    {
        return Err(invalid(
            "renderer currently requires a uniform planar hex mesh",
        ));
    }
    Ok(spacing)
}

fn mesh_bounds(
    mesh: &LandscapeMesh,
    half_width: f64,
    half_height: f64,
) -> Result<[f64; 4], ThinHcgRenderErrorV0> {
    let first = mesh
        .cell_center_km
        .first()
        .ok_or_else(|| invalid("cannot render an empty mesh"))?;
    let mut bounds = [first.x, first.x, first.y, first.y];
    for center in &mesh.cell_center_km {
        bounds[0] = bounds[0].min(center.x);
        bounds[1] = bounds[1].max(center.x);
        bounds[2] = bounds[2].min(center.y);
        bounds[3] = bounds[3].max(center.y);
    }
    bounds[0] -= half_width;
    bounds[1] += half_width;
    bounds[2] -= half_height;
    bounds[3] += half_height;
    if bounds.iter().any(|value| !value.is_finite())
        || bounds[0] >= bounds[1]
        || bounds[2] >= bounds[3]
    {
        return Err(invalid("mesh has invalid planar bounds"));
    }
    Ok(bounds)
}

fn physical_range(
    surfaces: [&[f64]; 3],
    explicit: Option<[f64; 2]>,
) -> Result<[f64; 2], ThinHcgRenderErrorV0> {
    if let Some(range) = explicit {
        return Ok(range);
    }
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for value in surfaces.into_iter().flatten().copied() {
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    if !minimum.is_finite() || !maximum.is_finite() {
        return Err(invalid("cannot derive a finite physical-elevation range"));
    }
    if minimum == maximum {
        let padding = minimum.abs().max(1.0) * 0.5;
        minimum -= padding;
        maximum += padding;
    }
    Ok([minimum, maximum])
}

fn raster_owners(
    mesh: &LandscapeMesh,
    bounds: [f64; 4],
    spacing: f64,
    width: u32,
    height: u32,
) -> Result<Vec<Option<usize>>, ThinHcgRenderErrorV0> {
    let bucket_size = spacing;
    // Keep one halo bucket because a cropped image can cut through cells whose
    // centres lie just outside its framing bounds.
    let bucket_min_x = bounds[0] - bucket_size;
    let bucket_min_y = bounds[2] - bucket_size;
    let bucket_cols = (((bounds[1] - bounds[0]) / bucket_size).ceil() as usize).max(1) + 3;
    let bucket_rows = (((bounds[3] - bounds[2]) / bucket_size).ceil() as usize).max(1) + 3;
    let bucket_count = bucket_cols
        .checked_mul(bucket_rows)
        .ok_or_else(|| invalid("spatial-index dimensions overflow"))?;
    let mut buckets = vec![Vec::<usize>::new(); bucket_count];
    for (cell, center) in mesh.cell_center_km.iter().enumerate() {
        let bx = ((center.x - bucket_min_x) / bucket_size).floor() as isize;
        let by = ((center.y - bucket_min_y) / bucket_size).floor() as isize;
        if bx >= 0 && by >= 0 && bx < bucket_cols as isize && by < bucket_rows as isize {
            buckets[by as usize * bucket_cols + bx as usize].push(cell);
        }
    }

    let count = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .ok_or_else(|| invalid("panel dimensions overflow address space"))?;
    let mut owner = vec![None; count];
    let span_x = bounds[1] - bounds[0];
    let span_y = bounds[3] - bounds[2];
    for py in 0..height {
        let y = bounds[3] - (f64::from(py) + 0.5) * span_y / f64::from(height);
        let by = ((y - bucket_min_y) / bucket_size).floor() as isize;
        for px in 0..width {
            let x = bounds[0] + (f64::from(px) + 0.5) * span_x / f64::from(width);
            let bx = ((x - bucket_min_x) / bucket_size).floor() as isize;
            let mut nearest = None;
            let mut nearest_distance_squared = f64::INFINITY;
            for candidate_by in by - 1..=by + 1 {
                for candidate_bx in bx - 1..=bx + 1 {
                    if candidate_bx < 0
                        || candidate_by < 0
                        || candidate_bx >= bucket_cols as isize
                        || candidate_by >= bucket_rows as isize
                    {
                        continue;
                    }
                    for &cell in
                        &buckets[candidate_by as usize * bucket_cols + candidate_bx as usize]
                    {
                        let center = mesh.cell_center_km[cell];
                        let distance_squared = (center.x - x).powi(2) + (center.y - y).powi(2);
                        if distance_squared < nearest_distance_squared
                            || (distance_squared == nearest_distance_squared
                                && nearest.is_none_or(|current| cell < current))
                        {
                            nearest = Some(cell);
                            nearest_distance_squared = distance_squared;
                        }
                    }
                }
            }
            if let Some(cell) = nearest {
                let center = mesh.cell_center_km[cell];
                if point_in_registered_hex(x - center.x, y - center.y, spacing) {
                    owner[(py * width + px) as usize] = Some(cell);
                }
            }
        }
    }
    Ok(owner)
}

fn point_in_registered_hex(dx: f64, dy: f64, spacing: f64) -> bool {
    let x = dx.abs();
    let y = dy.abs();
    x <= 0.5 * spacing && y <= (spacing - x) / 3.0_f64.sqrt()
}

fn elevation_colour(elevation: f64, range: [f64; 2]) -> [u8; 3] {
    const STOPS: [[u8; 3]; 5] = [
        [31, 55, 43],
        [78, 103, 59],
        [139, 130, 77],
        [170, 143, 111],
        [235, 237, 233],
    ];
    let t = ((elevation - range[0]) / (range[1] - range[0])).clamp(0.0, 1.0);
    let scaled = t * (STOPS.len() - 1) as f64;
    let lower = (scaled.floor() as usize).min(STOPS.len() - 2);
    let local = scaled - lower as f64;
    let mut result = [0; 3];
    for channel in 0..3 {
        result[channel] = ((1.0 - local) * f64::from(STOPS[lower][channel])
            + local * f64::from(STOPS[lower + 1][channel]))
        .round() as u8;
    }
    result
}

fn hillshade_factor(dx: f64, dy: f64, vertical_gain: f64, strength: f64) -> f64 {
    let nx = -vertical_gain * dx;
    let ny = -vertical_gain * dy;
    let nz = 1.0;
    let inverse_length = 1.0 / (nx * nx + ny * ny + nz * nz).sqrt();
    // Northwest light at 45 degrees altitude.
    let light = [-0.5_f64, 0.5_f64, 0.5_f64.sqrt()];
    let lambert = (nx * light[0] + ny * light[1] + nz * light[2]) * inverse_length;
    (1.0 + strength * (lambert - light[2])).clamp(0.72, 1.20)
}

fn shade_channel(channel: u8, factor: f64) -> u8 {
    (f64::from(channel) * factor).round().clamp(0.0, 255.0) as u8
}

fn invalid(message: impl Into<String>) -> ThinHcgRenderErrorV0 {
    ThinHcgRenderErrorV0::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matched_render_is_deterministic_and_uses_one_scale() {
        let mesh = LandscapeMesh::uniform_planar_hex(48.0, 32.0, 4.0).unwrap();
        let h: Vec<_> = mesh
            .cell_center_km
            .iter()
            .map(|center| 0.2 + 0.01 * center.x)
            .collect();
        let c: Vec<_> = mesh
            .cell_center_km
            .iter()
            .map(|center| 0.3 + 0.01 * center.y)
            .collect();
        let g: Vec<_> = h.iter().zip(&c).map(|(a, b)| 0.5 * (a + b)).collect();
        let config = ThinHcgRenderConfigV0 {
            panel_width_px: 64,
            gap_px: 3,
            ..ThinHcgRenderConfigV0::default()
        };

        let first = render_mesh(&mesh, &h, &c, &g, config).unwrap();
        let second = render_mesh(&mesh, &h, &c, &g, config).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.1.panel_order, ["H", "C", "G"]);
        assert_eq!(first.1.image_width_px, 3 * 64 + 2 * 3);
        assert_eq!(
            first.0.len(),
            first.1.image_width_px as usize * first.1.image_height_px as usize * 4
        );
        assert!(first.1.physical_elevation_range_km[0] < first.1.physical_elevation_range_km[1]);
    }

    #[test]
    fn diagnostic_sheet_is_sharp_cropped_and_deterministic() {
        let mesh = LandscapeMesh::uniform_planar_hex(80.0, 48.0, 4.0).unwrap();
        let selected = mesh
            .cell_center_km
            .iter()
            .map(|center| center.x.abs() <= 20.0 && center.y.abs() <= 8.0)
            .collect::<Vec<_>>();
        let forcing = mesh
            .cell_center_km
            .iter()
            .map(|center| {
                if center.x.abs() <= 18.0 && center.y.abs() <= 6.0 {
                    1.0 - center.x.abs() / 20.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let h = mesh
            .cell_center_km
            .iter()
            .map(|center| 1.0 + 0.01 * center.x)
            .collect::<Vec<_>>();
        let c = mesh
            .cell_center_km
            .iter()
            .map(|center| 1.2 + 0.02 * center.y)
            .collect::<Vec<_>>();
        let g = h.iter().zip(&c).map(|(a, b)| a.max(*b)).collect::<Vec<_>>();
        let config = ThinHcgDiagnosticConfigV0 {
            panel_width_px: 64,
            gap_px: 2,
            crop_padding_km: 4.0,
            profiles_height_px: 40,
            ..ThinHcgDiagnosticConfigV0::default()
        };
        let profile_forcing = [forcing.as_slice()];
        let first = render_diagnostic_mesh(
            &mesh,
            &selected,
            &forcing,
            &profile_forcing,
            &h,
            &c,
            &g,
            config,
        )
        .unwrap();
        let second = render_diagnostic_mesh(
            &mesh,
            &selected,
            &forcing,
            &profile_forcing,
            &h,
            &c,
            &g,
            config,
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.1.schema_version, THIN_HCG_DIAGNOSTIC_SCHEMA_V0);
        assert!(!first.1.physical_state_modified);
        assert_eq!(first.1.image_width_px, 3 * 64 + 2 * 2);
        assert_eq!(first.1.profiles_height_px, 40);
        assert_eq!(first.1.profiles.len(), 2);
        assert!(first.1.planar_crop_bounds_km[0] > -41.0);
        assert!(
            first.1.diagnostic_robust_ranges_km[0][0] < first.1.diagnostic_robust_ranges_km[0][1]
        );
        assert_eq!(
            first.0.len(),
            first.1.image_width_px as usize * first.1.image_height_px as usize * 4
        );
    }
}
