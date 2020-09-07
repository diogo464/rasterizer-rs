use super::vertex_data::VertexData;
use glm::{Vec2, Vec3, Vec4};
use nalgebra_glm as glm;

pub trait Interpolate {
    fn interpolate(v0: &Self, v1: &Self, v2: &Self, r0: f32, r1: f32, r2: f32) -> Self;
}
pub trait ShaderData: Interpolate + Send + Sync {}

pub trait Shader: Send + Sync {
    type Data: ShaderData;
    fn vertex(&self, vertex: &VertexData) -> (Vec4, Self::Data);
    fn fragment(&self, data: &Self::Data) -> Vec4;
}

pub trait Texture {
    fn color(&self, u: f32, v: f32) -> Vec4;
}

macro_rules! impl_interpolate {
    ($ty:ident) => {
        impl Interpolate for $ty {
            fn interpolate(v0: &Self, v1: &Self, v2: &Self, r0: f32, r1: f32, r2: f32) -> Self {
                v0 * r0 + v1 * r1 + v2 * r2
            }
        }
    };
}

impl_interpolate!(f32);
impl_interpolate!(Vec2);
impl_interpolate!(Vec3);
impl_interpolate!(Vec4);

impl Interpolate for f64 {
    fn interpolate(v0: &Self, v1: &Self, v2: &Self, r0: f32, r1: f32, r2: f32) -> Self {
        v0 * r0 as f64 + v1 * r1 as f64 + v2 * r2 as f64
    }
}
