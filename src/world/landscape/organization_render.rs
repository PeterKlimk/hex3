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
    let bucket_cols = (((bounds[1] - bounds[0]) / bucket_size).ceil() as usize).max(1) + 1;
    let bucket_rows = (((bounds[3] - bounds[2]) / bucket_size).ceil() as usize).max(1) + 1;
    let bucket_count = bucket_cols
        .checked_mul(bucket_rows)
        .ok_or_else(|| invalid("spatial-index dimensions overflow"))?;
    let mut buckets = vec![Vec::<usize>::new(); bucket_count];
    for (cell, center) in mesh.cell_center_km.iter().enumerate() {
        let bx = ((center.x - bounds[0]) / bucket_size).floor() as usize;
        let by = ((center.y - bounds[2]) / bucket_size).floor() as usize;
        buckets[by * bucket_cols + bx].push(cell);
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
        let by = ((y - bounds[2]) / bucket_size).floor() as isize;
        for px in 0..width {
            let x = bounds[0] + (f64::from(px) + 0.5) * span_x / f64::from(width);
            let bx = ((x - bounds[0]) / bucket_size).floor() as isize;
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
}
