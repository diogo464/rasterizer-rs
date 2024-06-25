use glam::{Mat3, Mat4, Vec2, Vec3, Vec4, Vec4Swizzles as _};
use rasterizer::{FragmentShader, Interpolate, VertexShader};

use crate::{model::ModelVertex, texture::Texture};

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

        let ambient = Vec3::splat(0.3) * albedo * ao;
        let mut color = ambient + Lo;

        color = color / (color + Vec3::splat(1.0));
        color = color.powf(1.0 / 2.2);

        color.extend(1.0)
    }
}
