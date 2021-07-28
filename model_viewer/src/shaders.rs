use crate::{model::ModelVertex, texture::Texture};
use rasterizer::math_prelude::*;
use rasterizer::{FragmentShader, Interpolate, VertexShader};

fn reflect(dir: Vec3, normal: Vec3) -> Vec3 {
    let nlen = normal.dot(dir);
    let normal_portion = normal * nlen;
    normal_portion * 2.0 - dir
}

#[derive(Debug, Default)]
pub struct ProjViewModel {
    pub projection: Mat4,
    pub view: Mat4,
    pub model: Mat4,
}

#[derive(Interpolate)]
pub struct StandardShaderData {
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
        let glpos = uniform.projection * uniform.view * uniform.model * position;
        let data = Self::SharedData {
            texture_coords: vertex.texture,
            normal: vertex.normal,
        };
        (glpos, data)
    }
}

pub struct FlatTextureFragmentShader {
    image: Texture,
}
impl FlatTextureFragmentShader {
    pub fn new(img: Texture) -> Self {
        Self { image: img }
    }
}
impl FragmentShader for FlatTextureFragmentShader {
    type Uniform = ProjViewModel;
    type SharedData = StandardShaderData;

    fn fragment(&self, shared: &Self::SharedData, _uniform: &Self::Uniform) -> Vec4 {
        self.image.sample(shared.texture_coords)
    }
}

pub struct PBRUniform {
    pub model: Mat4,
    pub view: Mat4,
    pub projection: Mat4,
    pub viewpos: Vec3,
    pub light_pos: Vec3,
    pub albedo_tex: Texture,
    pub normal_tex: Texture,
    pub ao_tex: Texture,
    pub roughness_tex: Texture,
    pub metallic_tex: Texture,
}

#[derive(Interpolate)]
pub struct PBRData {
    // world position
    pub position: Vec3,
    pub normal: Vec3,
    pub texcoords: Vec2,
    pub tangent: Vec3,
    pub bitangent: Vec3,
    pub tangent_position: Vec3,
    pub tangent_light_pos: Vec3,
    pub tangent_view_pos: Vec3,
}

pub struct PBRVertexShader;
impl VertexShader for PBRVertexShader {
    type VertexData = ModelVertex;
    type Uniform = PBRUniform;
    type SharedData = PBRData;

    fn vertex(
        &self,
        vertex: &Self::VertexData,
        uniform: &Self::Uniform,
    ) -> (Vec4, Self::SharedData) {
        let vpos = Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0);
        let vertex_pos = uniform.model * vpos;
        let final_pos = uniform.projection * uniform.view * vertex_pos;

        let t = (uniform.model * vertex.tangent.extend(0.0))
            .normalize()
            .xyz();
        let b = (uniform.model * vertex.bitangent.extend(0.0))
            .normalize()
            .xyz();
        let n = (uniform.model * vertex.normal.extend(0.0))
            .normalize()
            .xyz();
        let tbn = Mat3::from_cols(t, b, n);

        let shared = PBRData {
            position: vertex_pos.xyz(),
            normal: vertex.normal,
            texcoords: vertex.texture,
            tangent: vertex.tangent,
            bitangent: vertex.bitangent,
            tangent_position: tbn * vertex_pos.xyz(),
            tangent_light_pos: tbn * uniform.light_pos,
            tangent_view_pos: tbn * uniform.viewpos,
        };

        (final_pos, shared)
    }
}

pub struct PBRFragmentShader;
impl PBRFragmentShader {
    fn normal_dist_ggx(normal: Vec3, halfway: Vec3, roughness: f32) -> f32 {
        let a2 = roughness * roughness;
        let ndot_h = normal.dot(halfway).max(0.0);
        let ndot_h2 = ndot_h * ndot_h;

        let nom = a2;
        let mut denom = ndot_h2 * (a2 - 1.0) + 1.0;
        denom = std::f32::consts::PI * denom * denom;

        nom / denom

        //let r2 = roughness * roughness;
        //let nh2 = (normal.dot(halfway)).powi(2);
        //let denom = std::f32::consts::PI * (nh2 * (r2 - 1.0) + 1.0).powi(2);
        //r2 / denom
    }

    fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
        n_dot_v / (n_dot_v * (1.0 - roughness) + roughness)
    }

    fn geometry_smith(normal: Vec3, view: Vec3, light: Vec3, roughness: f32) -> f32 {
        let n_dot_v = normal.dot(view).max(0.0);
        let n_dot_l = normal.dot(light).max(0.0);
        let ggx1 = Self::geometry_schlick_ggx(n_dot_v, roughness);
        let ggx2 = Self::geometry_schlick_ggx(n_dot_l, roughness);
        ggx1 * ggx2
    }

    fn fresnel_schlick(cos_theta: f32, f0: Vec3) -> Vec3 {
        f0 + (1.0 - f0) * (1.0 - cos_theta).powi(5)
    }
}
impl FragmentShader for PBRFragmentShader {
    type Uniform = PBRUniform;
    type SharedData = PBRData;

    #[allow(non_snake_case)]
    fn fragment(&self, shared: &Self::SharedData, uniform: &Self::Uniform) -> Vec4 {
        const PI: f32 = std::f32::consts::PI;

        let normal = uniform
            .normal_tex
            .sample(shared.texcoords)
            .xyz()
            .normalize();
        let normal = (normal - 0.5) * 2.0;
        let tbn = Mat3::from_cols(shared.tangent, shared.bitangent, shared.normal);

        let albedo = uniform.albedo_tex.sample(shared.texcoords).xyz();
        let normal = tbn * normal;
        let ao = uniform.ao_tex.sample(shared.texcoords).xyz();
        let roughness = uniform.roughness_tex.sample(shared.texcoords).x;
        let metallic = uniform.metallic_tex.sample(shared.texcoords).x;

        let N = normal.normalize();
        let V = (uniform.viewpos - shared.position).normalize();

        let F0 = Vec3::splat(0.04) * (1.0 - metallic) + albedo * metallic;

        let lightPositions = [uniform.light_pos];
        let lightColors = [Vec3::ONE];

        // reflectance equation
        let mut Lo = Vec3::ZERO;
        for i in 0..1 {
            // calculate per-light radiance
            let L = (lightPositions[i] - shared.position).normalize();
            let H = (V + L).normalize();
            let distance = (lightPositions[i] - shared.position).length();
            let attenuation = 1.0 / (distance * distance);
            let radiance = lightColors[i] * attenuation;

            // cook-torrance brdf
            let NDF = Self::normal_dist_ggx(N, H, roughness);
            let G = Self::geometry_smith(N, V, L, roughness);
            let F = Self::fresnel_schlick(H.dot(V).max(0.0), F0);

            let kS = F;
            let mut kD = Vec3::ONE - kS;
            kD *= 1.0 - metallic;

            let numerator = NDF * G * F;
            let denominator = 4.0 * N.dot(V).max(0.0) * N.dot(L).max(0.0);
            let specular = numerator / denominator.max(0.001);

            // add to outgoing radiance Lo
            let NdotL = N.dot(L).max(0.0);
            Lo += (kD * albedo / PI + specular) * radiance * NdotL;
        }

        let ambient = Vec3::splat(0.03) * albedo * ao;
        let mut color = ambient + Lo;

        color = color / (color + Vec3::splat(1.0));
        color = color.powf(1.0 / 2.2);

        color.extend(1.0)
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
        let shading = shared.normal.dot(Vec3::new(0.0, 1.0, 0.0)).min(0.8);
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
        let position =
            vertex.position + (self.t + vertex.position.y * 3.0).sin() * vertex.normal * 0.01;
        let position = Vec4::new(position.x, position.y, position.z, 1.0);
        let pre_proj_pos = uniform.view * uniform.model * position;
        let glpos = uniform.projection * pre_proj_pos;
        let data = Self::SharedData {
            texture_coords: vertex.texture,
            normal: vertex.normal,
        };
        (glpos, data)
    }
}

pub struct D3Uniform {
    pub pvm: ProjViewModel,
    pub light_pos: Vec3,
    pub light_color: Vec3,
    pub light_intensity: f32,
    pub ambient_color: Vec3,
    pub ambient_intensity: f32,
    pub viewpos: Vec3,
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
        let vertex_to_light = (uniform.light_pos - shared.vertex_pos).normalize();
        let vertex_to_view = (uniform.viewpos - shared.vertex_pos).normalize();
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
