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
    /// Hills layer (regional terrain)
    Hills,
    /// Ridge layer (drainage divides)
    Ridges,
    /// Micro layer (surface texture)
    Micro,
    /// Arc shape noise (oceanic arc coastline variation)
    ArcShape,
}

impl NoiseLayer {
    /// Cycle to the next noise layer view.
    pub fn cycle(self) -> Self {
        match self {
            Self::Combined => Self::Macro,
            Self::Macro => Self::Hills,
            Self::Hills => Self::Ridges,
            Self::Ridges => Self::Micro,
            Self::Micro => Self::ArcShape,
            Self::ArcShape => Self::Combined,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Combined => "Combined",
            Self::Macro => "Macro",
            Self::Hills => "Hills",
            Self::Ridges => "Ridges",
            Self::Micro => "Micro",
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
    /// Wind speed
    Wind,
    /// Uplift (from projection solver)
    Uplift,
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
            Self::Wind => Self::Uplift,
            Self::Uplift => Self::Temperature,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Temperature => "Temperature",
            Self::Wind => "Wind",
            Self::Uplift => "Uplift",
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
