use crate::{model::ModelVertex, texture::Texture};
use rasterizer::math_prelude::*;
use rasterizer::{FragmentShader, Interpolate, VertexShader};




pub struct BasicUniform {
    pub model: Mat4,
    pub view: Mat4,
    pub projection: Mat4,
    pub viewpos: Vec3,
    pub light_pos: Vec3,
    pub ambient: Vec4,
    pub diffuse: Texture,
    pub specular: Texture,
    pub specular_factor: f32,
    pub normal: Texture,
    pub glow: Texture,
}

pub struct BasicVertexShader;
impl VertexShader for BasicVertexShader {
    type VertexData = ModelVertex;
    type Uniform = BasicUniform;
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

pub struct BasicFragmentShader;

impl FragmentShader for BasicFragmentShader {
    type Uniform = BasicUniform;

    type SharedData = StandardShaderData;

    fn fragment(&self, shared: &Self::SharedData, uniform: &Self::Uniform) -> Vec4 {
        let diffuse_sample = uniform.diffuse.sample(shared.texture_coords);
        let specular_sample = uniform.specular.sample(shared.texture_coords);
        let glow_sample = uniform.glow.sample(shared.texture_coords);
        let normal = uniform.normal.sample(shared.texture_coords).xyz();

        let (light_dir, light_dist) = {
            let v = uniform.light_pos - shared.position;
            let d = v.length();
            (v.normalize(), d)
        };
        let ambient = uniform.ambient * diffuse_sample;
        let diffuse = normal.dot(light_dir).max(0.0) * diffuse_sample;
        let specular_strength = uniform.specular_factor * 0.0;
        let specular = uniform.specular_factor * uniform.specular.sample(shared.texture_coords);

        ambient + diffuse + specular + glow_sample
    }
}




