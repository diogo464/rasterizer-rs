use glam::{Mat4, Vec2, Vec3, Vec4};
use rasterizer::{Interpolate, VertexShader};

mod diablo3;
mod flat;
mod normal;
mod pbr;
mod weird;

pub use diablo3::*;
pub use flat::*;
pub use normal::*;
pub use pbr::*;
pub use weird::*;

use crate::model::ModelVertex;

#[derive(Debug, Default)]
pub struct ProjViewModel {
    pub projection: Mat4,
    pub view: Mat4,
    pub model: Mat4,
}

#[derive(Interpolate)]
pub struct StandardShaderData {
    position: Vec3,
    texture_coords: Vec2,
    normal: Vec3,
}

pub struct StandardVertexShader;
impl VertexShader for StandardVertexShader {
    type VertexData = ModelVertex;
    type Uniform = ProjViewModel;
    type SharedData = StandardShaderData;
    fn vertex(
        &self,
        vertex: &Self::VertexData,
        uniform: &Self::Uniform,
    ) -> (Vec4, Self::SharedData) {
        let position = Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0);
        let world_position = uniform.model * position;
        let glpos = uniform.projection * uniform.view * world_position;
        let data = Self::SharedData {
            position: world_position.truncate(),
            texture_coords: vertex.texture,
            normal: vertex.normal,
        };
        (glpos, data)
    }
}

fn reflect(dir: Vec3, normal: Vec3) -> Vec3 {
    let nlen = normal.dot(dir);
    let normal_portion = normal * nlen;
    normal_portion * 2.0 - dir
}
