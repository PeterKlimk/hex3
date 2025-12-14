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
}

impl NoiseLayer {
    /// Cycle to the next noise layer view.
    pub fn cycle(self) -> Self {
        match self {
            Self::Combined => Self::Macro,
            Self::Macro => Self::Hills,
            Self::Hills => Self::Ridges,
            Self::Ridges => Self::Micro,
            Self::Micro => Self::Combined,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Combined => "Combined",
            Self::Macro => "Macro",
            Self::Hills => "Hills",
            Self::Ridges => "Ridges",
            Self::Micro => "Micro",
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
    /// Stress field visualization
    Stress,
    /// fBm noise contribution
    Noise,
    /// Flow accumulation and drainage
    Hydrology,
}

impl RenderMode {
    pub const COUNT: usize = 7;

    pub fn idx(self) -> usize {
        match self {
            Self::Relief => 0,
            Self::Terrain => 1,
            Self::Elevation => 2,
            Self::Plates => 3,
            Self::Stress => 4,
            Self::Noise => 5,
            Self::Hydrology => 6,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Relief => "Relief",
            Self::Terrain => "Terrain",
            Self::Elevation => "Elevation",
            Self::Plates => "Plates",
            Self::Stress => "Stress",
            Self::Noise => "Noise",
            Self::Hydrology => "Hydrology",
        }
    }

    /// Whether this mode uses 3D relief displacement.
    pub fn is_relief(self) -> bool {
        matches!(self, Self::Relief)
    }
}
