use crate::math_prelude::*;

pub trait Interpolate {
    fn interpolate(v0: &Self, v1: &Self, v2: &Self, r0: f32, r1: f32, r2: f32) -> Self;
}

pub trait ShaderData: Interpolate + Send + Sync {}
impl<T: Interpolate + Send + Sync> ShaderData for T {}

pub trait Shader: Send + Sync {
    type VertexData: Send + Sync;
    type Data: ShaderData;
    fn vertex(&self, vertex: &Self::VertexData) -> (Vec4, Self::Data);
    fn fragment(&self, data: &Self::Data) -> Vec4;
}

pub trait VertexShader: Send + Sync {
    type VertexData: Send + Sync;
    type Uniform: Send + Sync;
    type SharedData: ShaderData;

    fn vertex(
        &self,
        vertex: &Self::VertexData,
        uniform: &Self::Uniform,
    ) -> (Vec4, Self::SharedData);
}

pub trait FragmentShader: Send + Sync {
    type Uniform: Send + Sync;
    type SharedData: ShaderData;

    fn fragment(&self, shared: &Self::SharedData, uniform: &Self::Uniform) -> Vec4;
}

macro_rules! impl_interpolate {
    ($ty:ident) => {
        impl Interpolate for $ty {
            fn interpolate(v0: &Self, v1: &Self, v2: &Self, r0: f32, r1: f32, r2: f32) -> Self {
                *v0 * r0 + *v1 * r1 + *v2 * r2
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
