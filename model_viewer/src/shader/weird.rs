use glam::Vec4;
use rasterizer::VertexShader;

use crate::model::ModelVertex;

use super::{ProjViewModel, StandardShaderData};

pub struct WeirdSinVertexShader {
    pub t: f32,
}
impl VertexShader for WeirdSinVertexShader {
    type VertexData = ModelVertex;
    type Uniform = ProjViewModel;
    type SharedData = StandardShaderData;

    fn vertex(
        &self,
        vertex: &Self::VertexData,
        uniform: &Self::Uniform,
    ) -> (Vec4, Self::SharedData) {
        let position =
            vertex.position + (self.t + vertex.position.y * 3.0).sin() * vertex.normal * 0.01;
        let position = Vec4::new(position.x, position.y, position.z, 1.0);
        let pre_proj_pos = uniform.view * uniform.model * position;
        let glpos = uniform.projection * pre_proj_pos;
        let data = Self::SharedData {
            position: position.truncate(),
            texture_coords: vertex.texture,
            normal: vertex.normal,
        };
        (glpos, data)
    }
}
