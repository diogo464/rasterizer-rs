use glam::{Vec2, Vec3, Vec4, Vec4Swizzles as _};
use rasterizer::{FragmentShader, Interpolate, VertexShader};

use crate::{model::ModelVertex, texture::Texture};

use super::{reflect, ProjViewModel};

pub struct D3Uniform {
    pub pvm: ProjViewModel,
    pub light_position: Vec3,
    pub light_color: Vec3,
    pub light_intensity: f32,
    pub ambient_color: Vec3,
    pub ambient_intensity: f32,
    pub view_pos: Vec3,
    pub diffuse: Texture,
    pub specular: Texture,
    pub normals: Texture,
    pub glow: Texture,
}

#[derive(Interpolate)]
pub struct D3ShareData {
    texture_coords: Vec2,
    normal: Vec3,
    vertex_pos: Vec3,
}

pub struct D3VertexShader;
impl VertexShader for D3VertexShader {
    type VertexData = ModelVertex;
    type Uniform = D3Uniform;
    type SharedData = D3ShareData;

    fn vertex(
        &self,
        vertex: &Self::VertexData,
        uniform: &Self::Uniform,
    ) -> (Vec4, Self::SharedData) {
        let vpos = Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0);
        let vertex_pos = (uniform.pvm.model * vpos).xyz();
        let final_pos = uniform.pvm.projection * uniform.pvm.view * uniform.pvm.model * vpos;
        let data = Self::SharedData {
            texture_coords: vertex.texture,
            normal: vertex.normal,
            vertex_pos,
        };
        (final_pos, data)
    }
}

pub struct D3FragmentShader;
impl FragmentShader for D3FragmentShader {
    type Uniform = D3Uniform;
    type SharedData = D3ShareData;

    fn fragment(&self, shared: &Self::SharedData, uniform: &Self::Uniform) -> Vec4 {
        let vertex_to_light = (uniform.light_position - shared.vertex_pos).normalize();
        let vertex_to_view = (uniform.view_pos - shared.vertex_pos).normalize();
        let tc = &shared.texture_coords;

        let glow = uniform.glow.sample(*tc);
        let normal = uniform.normals.sample(*tc).xyz().normalize();
        let ambient = uniform.ambient_color * uniform.ambient_intensity;
        let diffuse =
            vertex_to_light.dot(normal).max(0.0) * uniform.light_color * uniform.light_intensity;
        let specular = (uniform.specular.sample(shared.texture_coords).xyz() * uniform.light_color)
            * reflect(vertex_to_light, normal)
                .dot(vertex_to_view)
                .max(0.0)
                .powf(32.0)
            * 32.0;
        let light = ambient + diffuse + specular;

        let color = uniform.diffuse.sample(shared.texture_coords);
        let final_color = Vec4::new(light.x, light.y, light.z, 1.0) * color + glow;
        final_color
    }
}
