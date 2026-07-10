/// Named presentation choices. World elevations remain physical; these only
/// control radial displacement in the renderer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReliefPreset {
    Flat,
    Physical,
    Authentic,
    Dramatic,
    Custom(f32),
}

impl ReliefPreset {
    pub const PHYSICAL_SCALE: f32 = 10.0 / hex3::world::PLANET_RADIUS_KM;
    pub const AUTHENTIC_SCALE: f32 = hex3::world::RELIEF_SCALE;
    pub const DRAMATIC_SCALE: f32 = 0.08;

    pub fn scale(self) -> f32 {
        match self {
            Self::Flat => 0.0,
            Self::Physical => Self::PHYSICAL_SCALE,
            Self::Authentic => Self::AUTHENTIC_SCALE,
            Self::Dramatic => Self::DRAMATIC_SCALE,
            Self::Custom(scale) => scale.max(0.0),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Flat => "Flat",
            Self::Physical => "Physical 1x",
            Self::Authentic => "Authentic",
            Self::Dramatic => "Dramatic",
            Self::Custom(_) => "Custom",
        }
    }

    pub fn from_scale(scale: f32) -> Self {
        let close = |a: f32, b: f32| (a - b).abs() <= 1e-6;
        if close(scale, 0.0) {
            Self::Flat
        } else if close(scale, Self::PHYSICAL_SCALE) {
            Self::Physical
        } else if close(scale, Self::AUTHENTIC_SCALE) {
            Self::Authentic
        } else if close(scale, Self::DRAMATIC_SCALE) {
            Self::Dramatic
        } else {
            Self::Custom(scale.max(0.0))
        }
    }

    pub fn cycle(self) -> Self {
        match self {
            Self::Flat => Self::Physical,
            Self::Physical => Self::Authentic,
            Self::Authentic => Self::Dramatic,
            Self::Dramatic | Self::Custom(_) => Self::Flat,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewMode {
    Globe,
    Map,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RiverMode {
    /// No rivers displayed
    #[default]
    Off,
    /// Only major rivers (high flow threshold, more opaque)
    Major,
    /// All rivers with flow-based transparency
    All,
}

impl RiverMode {
    /// Cycle to the next river mode.
    pub fn cycle(self) -> Self {
        match self {
            Self::Off => Self::Major,
            Self::Major => Self::All,
            Self::All => Self::Off,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Major => "Major",
            Self::All => "All",
        }
    }
}

/// Which noise layer to visualize in Noise mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NoiseLayer {
    /// All layers combined
    #[default]
    Combined,
    /// Macro layer (continental tilt)
    Macro,
    /// Arc shape noise (oceanic arc coastline variation)
    ArcShape,
}

impl NoiseLayer {
    /// Cycle to the next noise layer view.
    pub fn cycle(self) -> Self {
        match self {
            Self::Combined => Self::Macro,
            Self::Macro => Self::ArcShape,
            Self::ArcShape => Self::Combined,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Combined => "Combined",
            Self::Macro => "Macro",
            Self::ArcShape => "ArcShape",
        }
    }
}

/// Which tectonic feature field to visualize in Features mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FeatureLayer {
    /// Trench depth (subduction)
    #[default]
    Trench,
    /// Volcanic arc uplift
    Arc,
    /// Mid-ocean ridge uplift
    Ridge,
    /// Continental collision uplift
    Collision,
    /// Tectonic activity (noise modulator)
    Activity,
}

/// Which climate/atmosphere layer to visualize in Climate mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ClimateLayer {
    /// Temperature (latitude + elevation)
    #[default]
    Temperature,
    /// Surface wind (terrain-influenced)
    Wind,
    /// Upper wind (terrain-unaware, free-flowing)
    UpperWind,
    /// Uplift (convergence + orographic proxy)
    Uplift,
    /// Precipitation (moisture transport rain field)
    Precipitation,
}

impl FeatureLayer {
    /// Cycle to the next feature layer view.
    pub fn cycle(self) -> Self {
        match self {
            Self::Trench => Self::Arc,
            Self::Arc => Self::Ridge,
            Self::Ridge => Self::Collision,
            Self::Collision => Self::Activity,
            Self::Activity => Self::Trench,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Trench => "Trench",
            Self::Arc => "Arc",
            Self::Ridge => "Ridge",
            Self::Collision => "Collision",
            Self::Activity => "Activity",
        }
    }
}

impl ClimateLayer {
    /// Cycle to the next climate layer view.
    pub fn cycle(self) -> Self {
        match self {
            Self::Temperature => Self::Wind,
            Self::Wind => Self::UpperWind,
            Self::UpperWind => Self::Uplift,
            Self::Uplift => Self::Precipitation,
            Self::Precipitation => Self::Temperature,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Temperature => "Temperature",
            Self::Wind => "Wind (Surface)",
            Self::UpperWind => "Wind (Upper)",
            Self::Uplift => "Uplift",
            Self::Precipitation => "Precipitation",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderMode {
    /// 3D relief with terrain coloring + lakes (default)
    Relief,
    /// Flat terrain with elevation coloring + lakes
    Terrain,
    /// Raw elevation coloring only (no lakes)
    Elevation,
    /// Plate boundaries and velocities
    Plates,
    /// fBm noise contribution
    Noise,
    /// Flow accumulation and drainage
    Hydrology,
    /// Tectonic feature fields (trench, arc, ridge, collision, activity)
    Features,
    /// Climate data (temperature, future: precipitation)
    Climate,
}

impl RenderMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Relief => "Relief",
            Self::Terrain => "Terrain",
            Self::Elevation => "Elevation",
            Self::Plates => "Plates",
            Self::Noise => "Noise",
            Self::Hydrology => "Hydrology",
            Self::Features => "Features",
            Self::Climate => "Climate",
        }
    }

    /// Whether this mode uses 3D relief displacement.
    pub fn is_relief(self) -> bool {
        matches!(self, Self::Relief)
    }
}

#[cfg(test)]
mod tests {
    use super::ReliefPreset;

    #[test]
    fn relief_presets_have_stable_scales_and_cycle() {
        assert!((ReliefPreset::Physical.scale() - 10.0 / 6371.0).abs() < 1e-6);
        assert_eq!(ReliefPreset::Authentic.scale(), 0.04);
        assert_eq!(ReliefPreset::Dramatic.cycle(), ReliefPreset::Flat);
        assert_eq!(
            ReliefPreset::from_scale(ReliefPreset::PHYSICAL_SCALE),
            ReliefPreset::Physical
        );
    }
}
