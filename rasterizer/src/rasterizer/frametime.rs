use std::time::Duration;

#[derive(Debug, Default)]
pub struct FrameTime {
    time_vertex_shader: Duration,
    time_rasterization: Duration,
    time_fragment_shader: Duration,
}

impl FrameTime {
    pub fn new(vertex: Duration, raster: Duration, frag: Duration) -> Self {
        Self {
            time_vertex_shader: vertex,
            time_rasterization: raster,
            time_fragment_shader: frag,
        }
    }
    pub fn zero() -> Self {
        Self::default()
    }

    pub fn geometry_stage(&self) -> &Duration {
        &self.time_vertex_shader
    }

    pub fn rasterization_stage(&self) -> &Duration {
        &self.time_rasterization
    }

    pub fn fragment_stage(&self) -> &Duration {
        &self.time_fragment_shader
    }

    pub fn total(&self) -> Duration {
        *self.geometry_stage() + *self.rasterization_stage() + *self.fragment_stage()
    }
}
