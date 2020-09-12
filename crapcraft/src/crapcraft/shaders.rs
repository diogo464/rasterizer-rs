use glm::{Mat4, Vec3, Vec4};
use nalgebra_glm as glm;
use rasterizer::{FragmentShader, Interpolate, VertexShader};

use super::chunk::ChunkVertex;

#[derive(Debug, Default)]
pub struct ProjViewModel {
    pub projection: Mat4,
    pub view: Mat4,
    pub model: Mat4,
}

impl ProjViewModel {
    pub fn new(projection: &Mat4, view: &Mat4, model: &Mat4) -> Self {
        Self {
            projection: *projection,
            view: *view,
            model: *model,
        }
    }
}

#[derive(Interpolate)]
pub struct ChunkShareData {}

#[derive(Debug, Default)]
pub struct ChunkVertexShader {}
impl VertexShader for ChunkVertexShader {
    type VertexData = ChunkVertex;
    type Uniform = ProjViewModel;
    type SharedData = ChunkShareData;

    fn vertex(
        &self,
        vertex: &Self::VertexData,
        uniform: &Self::Uniform,
    ) -> (Vec4, Self::SharedData) {
        let vpos = Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0);
        let glpos = uniform.projection * uniform.view * uniform.model * vpos;
        let data = ChunkShareData {};
        (glpos, data)
    }
}

#[derive(Debug, Default)]
pub struct ChunkFragmentShader {}
impl FragmentShader for ChunkFragmentShader {
    type Uniform = ProjViewModel;
    type SharedData = ChunkShareData;

    fn fragment(&self, shared: &Self::SharedData, uniform: &Self::Uniform) -> Vec4 {
        Vec4::new(0.0, 1.0, 0.0, 1.0)
    }
}
