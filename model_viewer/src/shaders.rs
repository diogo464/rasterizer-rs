use crate::{model::ModelVertex, texture::Texture};
use glm::{Mat4, Vec2, Vec3, Vec4};
use image::DynamicImage;
use nalgebra_glm as glm;
use rasterizer::{FragmentShader, Interpolate, ShaderData, VertexShader};

fn reflect(dir: &Vec3, normal: &Vec3) -> Vec3 {
    let nlen = normal.dot(&dir);
    let normal_portion = *normal * nlen;
    //let offset = *dir - normal_portion;
    normal_portion * 2.0 - *dir
    //offset - normal_portion
    //normal_portion - offset | normal_portion - *dir + normal_portion
}

#[derive(Debug, Default)]
pub struct ProjViewModel {
    pub projection: Mat4,
    pub view: Mat4,
    pub model: Mat4,
}

pub struct StandardShaderData {
    texture_coords: Vec2,
    normal: Vec3,
}
impl Interpolate for StandardShaderData {
    fn interpolate(v0: &Self, v1: &Self, v2: &Self, r0: f32, r1: f32, r2: f32) -> Self {
        Self {
            texture_coords: Vec2::interpolate(
                &v0.texture_coords,
                &v1.texture_coords,
                &v2.texture_coords,
                r0,
                r1,
                r2,
            ),
            normal: Vec3::interpolate(&v0.normal, &v1.normal, &v2.normal, r0, r1, r2),
        }
    }
}
impl ShaderData for StandardShaderData {}

pub struct StandardVertexShader {}
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
        let glpos = uniform.projection * uniform.view * uniform.model * position;
        let data = Self::SharedData {
            texture_coords: vertex.texture.unwrap(),
            normal: vertex.normal.unwrap(),
        };
        (glpos, data)
    }
}

pub struct FlatTextureFragmentShader {
    image: DynamicImage,
}
impl FlatTextureFragmentShader {
    pub fn new(img: DynamicImage) -> Self {
        Self { image: img }
    }
}
impl FragmentShader for FlatTextureFragmentShader {
    type Uniform = ProjViewModel;
    type SharedData = StandardShaderData;

    fn fragment(&self, shared: &Self::SharedData, _uniform: &Self::Uniform) -> Vec4 {
        let shade = (0.3 + shared.normal.dot(&Vec3::new(0.5, 1.0, 0.5))).min(1.0);
        <DynamicImage as Texture>::color_at(
            &self.image,
            shared.texture_coords.x,
            shared.texture_coords.y,
        ) * shade
    }
}

#[derive(Default)]
pub struct NormalShadingFragmentShader {}
impl NormalShadingFragmentShader {
    pub fn new() -> Self {
        Self::default()
    }
}
impl FragmentShader for NormalShadingFragmentShader {
    type Uniform = ProjViewModel;
    type SharedData = StandardShaderData;

    fn fragment(&self, shared: &Self::SharedData, _uniform: &Self::Uniform) -> Vec4 {
        let shading = shared.normal.dot(&Vec3::new(0.0, 1.0, 0.0)).min(0.8);
        Vec4::new(shading, shading, shading, 1.0)
    }
}

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
        let position = vertex.position
            + (self.t + vertex.position.y * 3.0).sin() * vertex.normal.unwrap() * 0.01;
        let position = Vec4::new(position.x, position.y, position.z, 1.0);
        let pre_proj_pos = uniform.view * uniform.model * position;
        let glpos = uniform.projection * pre_proj_pos;
        let data = Self::SharedData {
            texture_coords: vertex.texture.unwrap(),
            normal: vertex.normal.unwrap(),
        };
        (glpos, data)
    }
}

pub struct PhongUniform {
    pub pvm: ProjViewModel,
    pub light_pos: Vec3,
    pub light_color: Vec3,
    pub light_intensity: f32,
    pub ambient_color: Vec3,
    pub ambient_intensity: f32,
    pub viewpos: Vec3,
    pub texture: DynamicImage,
}

pub struct PhongShareData {
    texture_coords: Vec2,
    normal: Vec3,
    vertex_pos: Vec3,
}
impl Interpolate for PhongShareData {
    fn interpolate(v0: &Self, v1: &Self, v2: &Self, r0: f32, r1: f32, r2: f32) -> Self {
        Self {
            texture_coords: Vec2::interpolate(
                &v0.texture_coords,
                &v1.texture_coords,
                &v2.texture_coords,
                r0,
                r1,
                r2,
            ),
            normal: Vec3::interpolate(&v0.normal, &v1.normal, &v2.normal, r0, r1, r2),
            vertex_pos: Vec3::interpolate(
                &v0.vertex_pos,
                &v1.vertex_pos,
                &v2.vertex_pos,
                r0,
                r1,
                r2,
            ),
        }
    }
}
impl ShaderData for PhongShareData {}

pub struct PhongVertexShader {}
impl VertexShader for PhongVertexShader {
    type VertexData = ModelVertex;
    type Uniform = PhongUniform;
    type SharedData = PhongShareData;

    fn vertex(
        &self,
        vertex: &Self::VertexData,
        uniform: &Self::Uniform,
    ) -> (Vec4, Self::SharedData) {
        let vpos = glm::Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0);
        let vertex_pos = (uniform.pvm.model * vpos).xyz();
        let final_pos = uniform.pvm.projection * uniform.pvm.view * uniform.pvm.model * vpos;
        let data = Self::SharedData {
            texture_coords: vertex.texture.unwrap(),
            normal: vertex.normal.unwrap(),
            vertex_pos,
        };
        (final_pos, data)
    }
}

pub struct PhongFragmentShader {}
impl FragmentShader for PhongFragmentShader {
    type Uniform = PhongUniform;
    type SharedData = PhongShareData;

    fn fragment(&self, shared: &Self::SharedData, uniform: &Self::Uniform) -> Vec4 {
        let vertex_to_light = glm::normalize(&(uniform.light_pos - shared.vertex_pos));
        let vertex_to_view = glm::normalize(&(uniform.viewpos - shared.vertex_pos));

        let ambient = uniform.ambient_color * uniform.ambient_intensity;
        let diffuse = glm::dot(&vertex_to_light, &shared.normal).max(0.0)
            * uniform.light_color
            * uniform.light_intensity;
        let specular = glm::dot(&reflect(&vertex_to_light, &shared.normal), &vertex_to_view)
            .max(0.0)
            .powf(32.0)
            * uniform.light_color
            * 1.3
            * uniform.light_intensity;
        let light = ambient + diffuse + specular;

        let color = uniform
            .texture
            .color_at(shared.texture_coords.x, shared.texture_coords.y);
        let final_color = Vec4::new(light.x, light.y, light.z, 1.0).component_mul(&color);
        final_color
    }
}